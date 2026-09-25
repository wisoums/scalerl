"""Tests for the predictive autoscaler: linear-trend forecast (#14) plus backlog recovery (#63)."""

import math
from typing import Any

import numpy as np
import pytest

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers import (
    Controller,
    ForecastRecord,
    PredictiveController,
    ThresholdController,
    backlog_recovery_rate,
    effective_sizing_demand,
    run_episode,
)
from scalerl.environment import (
    AutoscalingEnv,
    ReplicaConfig,
    ReplicaPool,
    SimulatorConfig,
    TimingConfig,
)
from scalerl.evaluation import evaluate_controller_episode
from scalerl.workloads import WorkloadTrace, ramp_workload

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
OBSERVATION = np.zeros(12, dtype=np.float32)


_PARAMS: dict[str, Any] = {
    "min_replicas": 1,
    "max_replicas": 20,
    "service_capacity_rps": 50.0,
    "startup_delay_seconds": 60.0,
    "control_interval_seconds": 30.0,
}


def make_controller(**overrides: Any) -> PredictiveController:
    return PredictiveController(**{**_PARAMS, **overrides})


def step_info(
    tick: int, rate: float, active: int = 1, pending: int = 0, queued: float = 0.0
) -> dict[str, Any]:
    return {
        "tick": tick,
        "request_rate": rate,
        "active_replicas": active,
        "pending_replicas": pending,
        "queued_requests": queued,
    }


def feed(
    controller: PredictiveController, rates: list[float], queued: float = 0.0, **replicas: int
) -> int:
    action = HOLD
    for tick, rate in enumerate(rates):
        action = controller.act(OBSERVATION, step_info(tick, rate, queued=queued, **replicas))
    return action


class ForecastOnlyReference(PredictiveController):
    """Test-only #14 reference: sizes for forecast arrivals and ignores the queue."""

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        if "tick" in info:
            info = {**info, "queued_requests": 0.0}
        return super().act(observation, info)


def env_for(rates: list[float], **replicas: Any) -> AutoscalingEnv:
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 10,
        "initial_replicas": 2,
        "startup_delay_seconds": 60.0,
        "service_capacity_rps": 50.0,
    }
    config = SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=30.0, episode_duration_seconds=30.0 * len(rates)
        ),
        replicas=ReplicaConfig(**{**params, **replicas}),
    )
    return AutoscalingEnv(config, WorkloadTrace(rates, control_interval_seconds=30.0))


# --- configuration ----------------------------------------------------------------------


def test_defaults_are_the_predeclared_v1_settings() -> None:
    controller = make_controller()

    assert controller.history_window_ticks == 4
    assert controller.target_utilization == 0.8
    assert isinstance(controller, Controller)


@pytest.mark.parametrize("window", [1, 4, 10])
def test_valid_history_windows_are_accepted(window: int) -> None:
    assert make_controller(history_window_ticks=window).history_window_ticks == window


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("history_window_ticks", 0, ValueError),
        ("history_window_ticks", -2, ValueError),
        ("history_window_ticks", True, TypeError),
        ("history_window_ticks", 4.0, TypeError),
        ("target_utilization", 0, ValueError),
        ("target_utilization", -0.5, ValueError),
        ("target_utilization", 1.2, ValueError),
        ("target_utilization", math.nan, ValueError),
        ("target_utilization", math.inf, ValueError),
        ("target_utilization", True, TypeError),
        ("service_capacity_rps", 0.0, ValueError),
        ("service_capacity_rps", math.inf, ValueError),
        ("startup_delay_seconds", -1.0, ValueError),
        ("startup_delay_seconds", math.nan, ValueError),
        ("control_interval_seconds", 0.0, ValueError),
        ("min_replicas", 0, ValueError),
        ("max_replicas", True, TypeError),
    ],
)
def test_invalid_settings_are_rejected(field: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error, match=field):
        make_controller(**{field: value})


def test_max_below_min_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_replicas must be at least min_replicas"):
        make_controller(min_replicas=5, max_replicas=3)


@pytest.mark.parametrize("utilization", [0.5, 0.8, 1.0])
def test_valid_target_utilization_is_accepted(utilization: float) -> None:
    assert make_controller(target_utilization=utilization).target_utilization == utilization


def test_from_config_takes_bounds_capacity_and_timing_from_the_simulator() -> None:
    config = SimulatorConfig(
        replicas=ReplicaConfig(
            min_replicas=2, max_replicas=7, initial_replicas=2, startup_delay_seconds=90.0
        )
    )

    controller = PredictiveController.from_config(config)

    assert controller.forecast_horizon_ticks == 1 + 3
    feed(controller, [10_000.0])
    assert controller.last_decision is not None
    assert controller.last_decision.desired_replicas == 7


# --- forecasting ----------------------------------------------------------------------


def test_first_decision_without_a_sample_holds() -> None:
    controller = make_controller()

    action = controller.act(OBSERVATION, {"tick": 0, "active_replicas": 3, "pending_replicas": 0})

    decision = controller.last_decision
    assert action == HOLD
    assert decision is not None
    assert (decision.reason, decision.forecast_rps, decision.desired_replicas) == (
        "no_sample",
        None,
        None,
    )
    assert controller.forecasts == ()


def test_one_sample_uses_persistence() -> None:
    controller = make_controller()

    feed(controller, [100.0])

    assert controller.last_decision is not None
    assert controller.last_decision.forecast_rps == 100.0
    assert controller.forecasts[0].sample_count == 1


def test_constant_demand_forecasts_the_same_demand() -> None:
    controller = make_controller()

    feed(controller, [100.0, 100.0, 100.0, 100.0])

    assert controller.last_decision is not None
    assert controller.last_decision.forecast_rps == pytest.approx(100.0)


def test_rising_trend_is_extrapolated_to_the_horizon() -> None:
    controller = make_controller()  # 60 s startup / 30 s ticks -> horizon 3

    feed(controller, [50.0, 80.0, 110.0, 140.0])

    decision = controller.last_decision
    assert decision is not None
    # Least squares through (0,50) ... (3,140): slope 30/tick; tick 3 + 3 = 6 -> 230.
    assert decision.forecast_rps == pytest.approx(230.0)
    assert controller.forecasts[-1].target_tick == 6
    assert decision.desired_replicas == math.ceil(230.0 / (50.0 * 0.8))


def test_falling_trend_never_forecasts_negative_demand() -> None:
    controller = make_controller()

    feed(controller, [140.0, 100.0, 60.0, 20.0])  # the line would reach -100 at the horizon

    assert controller.last_decision is not None
    assert controller.last_decision.forecast_rps == 0.0
    assert controller.last_decision.desired_replicas == 1  # clamped to min


def test_only_the_newest_window_influences_the_forecast() -> None:
    rolled = make_controller(history_window_ticks=4)
    feed(rolled, [500.0, 0.0, 50.0, 80.0, 110.0, 140.0])

    fresh = make_controller(history_window_ticks=4)
    for tick, rate in zip(range(2, 6), [50.0, 80.0, 110.0, 140.0], strict=True):
        fresh.act(OBSERVATION, step_info(tick, rate))

    assert rolled.request_history == (50.0, 80.0, 110.0, 140.0)
    assert rolled.forecasts[-1] == fresh.forecasts[-1]


def test_rereading_the_same_tick_does_not_add_a_sample() -> None:
    controller = make_controller()
    info = step_info(0, 100.0)

    controller.act(OBSERVATION, info)
    controller.act(OBSERVATION, info)

    assert controller.request_history == (100.0,)


# --- horizon ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay", "interval", "horizon"),
    [(0.0, 30.0, 1), (30.0, 30.0, 2), (45.0, 30.0, 3), (60.0, 30.0, 3), (90.0, 30.0, 4)],
)
def test_horizon_is_one_plus_startup_ticks(delay: float, interval: float, horizon: int) -> None:
    controller = make_controller(startup_delay_seconds=delay, control_interval_seconds=interval)

    assert controller.forecast_horizon_ticks == horizon


@pytest.mark.parametrize("delay", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_horizon_targets_the_first_tick_new_capacity_serves(delay: float) -> None:
    env = env_for([1.0] * 10, initial_replicas=1, startup_delay_seconds=delay)
    env.reset(seed=0)
    env.step(HOLD)  # tick 0 completes; a decision is made now
    serving = [
        env.step(SCALE_UP if tick == 1 else HOLD)[4]["active_replicas"] for tick in range(1, 8)
    ]
    first_serving_tick = 1 + serving.index(2)

    horizon = PredictiveController.from_config(env.config).forecast_horizon_ticks

    assert first_serving_tick == 0 + horizon


# --- no future leakage -------------------------------------------------------------------


def test_forecast_uses_only_completed_demand() -> None:
    spiky = env_for([10.0, 20.0, 30.0, 999.0, 40.0])
    calm = env_for([10.0, 20.0, 30.0, 35.0, 40.0])  # differs only at the unseen tick 3
    states = {}
    for name, env in (("spiky", spiky), ("calm", calm)):
        controller = PredictiveController.from_config(env.config)
        observation, info = env.reset(seed=0)
        controller.reset()
        for _ in range(3):  # ticks 0, 1, 2 complete
            observation, _, _, _, info = env.step(controller.act(observation, info))
        controller.act(observation, info)  # the decision before tick 3 runs
        states[name] = controller

    spiky_controller, calm_controller = states["spiky"], states["calm"]
    assert spiky_controller.request_history == (10.0, 20.0, 30.0)
    assert spiky_controller.forecasts == calm_controller.forecasts
    assert spiky_controller.last_decision == calm_controller.last_decision
    recorded = [f.forecast_rps for f in spiky_controller.forecasts]
    assert 999.0 not in spiky_controller.request_history
    assert all(abs(value - 999.0) > 1e-6 for value in recorded)
    assert all(f.source_tick <= 2 for f in spiky_controller.forecasts)


# --- capacity decisions ------------------------------------------------------------------


@pytest.mark.parametrize(("rate", "desired"), [(160.0, 4), (160.5, 5), (40.0, 1), (0.0, 1)])
def test_desired_replicas_use_capacity_with_headroom(rate: float, desired: int) -> None:
    controller = make_controller()  # 50 RPS x 0.8 = 40 RPS per replica

    feed(controller, [rate])

    assert controller.last_decision is not None
    assert controller.last_decision.desired_replicas == desired


def test_desired_replicas_stay_within_bounds() -> None:
    high = make_controller(min_replicas=2, max_replicas=5)
    low = make_controller(min_replicas=2, max_replicas=5)

    assert feed(high, [10_000.0], active=5) == HOLD
    assert feed(low, [0.0], active=2) == HOLD

    assert high.last_decision is not None and low.last_decision is not None
    assert (high.last_decision.desired_replicas, high.last_decision.reason) == (5, "at_max")
    assert (low.last_decision.desired_replicas, low.last_decision.reason) == (2, "at_min")


@pytest.mark.parametrize(
    ("active", "pending", "action", "reason"),
    [
        (2, 0, SCALE_UP, "scale_up"),  # desired 3 > committed 2
        (2, 1, HOLD, "at_target"),  # pending counts: committed 3 == desired 3
        (3, 0, HOLD, "at_target"),
        (2, 2, SCALE_DOWN, "scale_down"),  # committed 4 > desired 3
        (5, 0, SCALE_DOWN, "scale_down"),
    ],
)
def test_action_moves_committed_capacity_toward_desired(
    active: int, pending: int, action: int, reason: str
) -> None:
    controller = make_controller()

    result = feed(controller, [100.0], active=active, pending=pending)  # desired ceil(100/40) = 3

    assert result == action
    decision = controller.last_decision
    assert decision is not None
    assert (decision.desired_replicas, decision.active_replicas, decision.pending_replicas) == (
        3,
        active,
        pending,
    )
    assert decision.reason == reason


def test_scale_down_with_pending_capacity_cancels_the_pending_replica() -> None:
    pool = ReplicaPool(ReplicaConfig(min_replicas=1, max_replicas=10, initial_replicas=2))
    pool.scale_up(2)
    controller = make_controller()

    action = feed(controller, [100.0], active=pool.active_count, pending=pool.pending_count)
    assert action == SCALE_DOWN
    pool.scale_down()

    assert (pool.active_count, pool.pending_count) == (2, 1)  # pending cancelled first


# --- episodes -----------------------------------------------------------------------------


def test_reset_clears_history_forecasts_and_diagnostics() -> None:
    controller = make_controller()
    feed(controller, [50.0, 80.0, 110.0])

    controller.reset(seed=3)

    assert controller.request_history == ()
    assert controller.forecasts == ()
    assert controller.last_decision is None


def test_episodes_are_deterministic_across_resets() -> None:
    env = env_for([float(x) for x in range(20, 200, 6)])
    controller = PredictiveController.from_config(env.config)

    def run() -> tuple[list[int], tuple[Any, ...]]:
        infos = run_episode(env, controller, seed=0)
        return [info["requested_action"] for info in infos], controller.forecasts

    assert run() == run()


def ramp_env() -> AutoscalingEnv:
    config = SimulatorConfig(
        replicas=ReplicaConfig(min_replicas=1, max_replicas=10, initial_replicas=2)
    )  # 50 RPS per replica, 60 s startup, 30 s ticks
    trace = ramp_workload(
        duration_seconds=3600.0, control_interval_seconds=30.0, start_rate=40.0, end_rate=400.0
    )
    return AutoscalingEnv(config, trace)


def first_scale_up(infos: list[dict[str, Any]]) -> int:
    return next(info["tick"] for info in infos if info["requested_action"] == SCALE_UP)


def test_predictive_requests_capacity_earlier_than_threshold_on_a_ramp() -> None:
    env = ramp_env()
    threshold = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, cooldown_ticks=3, min_replicas=1, max_replicas=10
    )
    predictive = PredictiveController.from_config(env.config)

    threshold_infos = run_episode(env, threshold, seed=0)
    predictive_infos = run_episode(env, predictive, seed=0)

    # The forecaster sees the rising trend and extrapolates above observed demand.
    trending = [f for f in predictive.forecasts if f.sample_count == 4]
    assert all(f.forecast_rps > predictive_infos[f.source_tick]["request_rate"] for f in trending)
    # ... so it asks for more capacity before utilization reaches the threshold.
    assert first_scale_up(predictive_infos) < first_scale_up(threshold_infos)


def test_predictive_cannot_anticipate_an_abrupt_spike() -> None:
    rates = [50.0] * 20 + [400.0] * 20
    env = env_for(rates)
    controller = PredictiveController.from_config(env.config)

    infos = run_episode(env, controller, seed=0)

    # Every decision made before the spike was observed held steady capacity.
    before_spike = [info for info in infos if info["tick"] <= 20]
    assert all(info["requested_action"] == HOLD for info in before_spike[1:])
    assert all(
        f.forecast_rps == pytest.approx(50.0) for f in controller.forecasts if f.source_tick < 20
    )
    # Only once the spike is part of the completed history does it react.
    assert any(info["requested_action"] == SCALE_UP for info in infos if info["tick"] > 20)


# --- backlog recovery (#63) ---------------------------------------------------------------


def test_backlog_recovery_clears_the_queue_in_one_control_interval() -> None:
    assert backlog_recovery_rate(3000.0, 30.0) == pytest.approx(100.0)
    assert backlog_recovery_rate(0, 30.0) == 0.0


def test_effective_demand_is_forecast_plus_backlog_recovery() -> None:
    assert effective_sizing_demand(80.0, 3000.0, 30.0) == pytest.approx(180.0)
    assert effective_sizing_demand(80.0, 0.0, 30.0) == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("queued", "error"),
    [
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-1.0, ValueError),
        (True, TypeError),
        ("10", TypeError),
    ],
)
def test_malformed_queue_is_rejected(queued: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="queued_requests"):
        backlog_recovery_rate(queued, 30.0)  # type: ignore[arg-type]
    with pytest.raises(error, match="queued_requests"):
        feed(make_controller(), [50.0], queued=queued)  # type: ignore[arg-type]


@pytest.mark.parametrize("interval", [0.0, -30.0, math.nan, math.inf])
def test_backlog_recovery_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError, match="control_interval_seconds"):
        backlog_recovery_rate(100.0, interval)


def test_missing_queue_is_rejected() -> None:
    info = step_info(0, 50.0)
    del info["queued_requests"]

    with pytest.raises(ValueError, match="queued_requests"):
        make_controller().act(OBSERVATION, info)


def test_empty_queue_reproduces_forecast_only_sizing() -> None:
    rates = [40.0, 90.0, 60.0, 150.0, 20.0, 0.0, 300.0]
    for active, pending in [(1, 0), (3, 1), (8, 0)]:
        queue_aware, reference = make_controller(), ForecastOnlyReference(**_PARAMS)
        for tick, rate in enumerate(rates):
            info = step_info(tick, rate, active=active, pending=pending)
            assert queue_aware.act(OBSERVATION, info) == reference.act(OBSERVATION, info)
            assert queue_aware.last_decision == reference.last_decision
        decision = queue_aware.last_decision
        assert decision is not None
        assert decision.queued_requests == 0.0
        assert decision.backlog_recovery_rps == 0.0
        assert decision.effective_demand_rps == decision.forecast_rps


def test_queue_empty_workloads_behave_exactly_like_forecast_only() -> None:
    env = ramp_env()  # capacity keeps up, so the queue stays empty

    queue_aware = run_episode(env, PredictiveController.from_config(env.config), seed=0)
    reference = run_episode(env, ForecastOnlyReference.from_config(env.config), seed=0)

    assert all(info["queued_requests"] == 0 for info in queue_aware)
    assert queue_aware == reference


def test_forecast_is_identical_whatever_the_queue() -> None:
    rates = [40.0, 60.0, 90.0, 130.0, 110.0]
    empty, backlogged = make_controller(), make_controller()

    feed(empty, rates, queued=0.0)
    feed(backlogged, rates, queued=10_000.0)

    assert empty.forecasts == backlogged.forecasts
    assert empty.request_history == backlogged.request_history
    assert empty.last_decision is not None and backlogged.last_decision is not None
    assert empty.last_decision.forecast_rps == backlogged.last_decision.forecast_rps
    assert backlogged.last_decision.effective_demand_rps == pytest.approx(
        backlogged.last_decision.forecast_rps + 10_000.0 / 30.0  # type: ignore[operator]
    )


def test_forecast_record_is_unchanged() -> None:
    assert ForecastRecord.__slots__ == (
        "source_tick",
        "target_tick",
        "forecast_rps",
        "sample_count",
    )


def test_large_queue_scales_up_for_recovery() -> None:
    controller = make_controller()

    action = feed(controller, [40.0], queued=3000.0)  # forecast 40 + recovery 100 = 140 RPS

    decision = controller.last_decision
    assert action == SCALE_UP
    assert decision is not None
    assert decision.forecast_rps == pytest.approx(40.0)  # the forecast alone needs 1 replica
    assert decision.queued_requests == 3000.0
    assert decision.backlog_recovery_rps == pytest.approx(100.0)
    assert decision.effective_demand_rps == pytest.approx(140.0)
    assert decision.desired_replicas == 4  # ceil(140 / 40)
    assert decision.reason == "queue_recovery"


def test_scale_up_needed_by_the_forecast_alone_keeps_the_scale_up_reason() -> None:
    controller = make_controller()

    assert feed(controller, [100.0], queued=3000.0) == SCALE_UP  # forecast alone needs 3

    assert controller.last_decision is not None
    assert controller.last_decision.reason == "scale_up"
    assert controller.last_decision.desired_replicas == 5  # ceil(200 / 40)


def test_positive_queue_blocks_scale_down() -> None:
    controller = make_controller()

    action = feed(controller, [10.0], queued=1.0, active=5)

    decision = controller.last_decision
    assert action == HOLD
    assert decision is not None
    assert decision.desired_replicas == 1  # sizing alone would shrink
    assert decision.reason == "backlog_hold"


def test_empty_queue_allows_scale_down() -> None:
    controller = make_controller()

    assert feed(controller, [10.0], queued=0.0, active=5) == SCALE_DOWN
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "scale_down"


def test_pending_capacity_counts_toward_backlog_recovery() -> None:
    controller = make_controller()

    # desired ceil(140 / 40) = 4 == committed 1 active + 3 pending
    assert feed(controller, [40.0], queued=3000.0, active=1, pending=3) == HOLD
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "at_target"

    # desired 4 < committed 5, but requests are still queued: no cancellation
    assert feed(make_controller(), [40.0], queued=3000.0, active=1, pending=4) == HOLD


def test_backlog_recovery_stays_within_bounds() -> None:
    controller = make_controller(max_replicas=5)

    assert feed(controller, [40.0], queued=1_000_000.0, active=5) == HOLD

    assert controller.last_decision is not None
    assert (controller.last_decision.desired_replicas, controller.last_decision.reason) == (
        5,
        "at_max",
    )


def test_reset_clears_backlog_diagnostics() -> None:
    controller = make_controller()
    feed(controller, [40.0], queued=3000.0)

    controller.reset()

    assert controller.last_decision is None
    assert feed(controller, [40.0]) == HOLD
    assert controller.last_decision is not None
    assert controller.last_decision.queued_requests == 0.0


def test_no_sample_decision_has_no_backlog_diagnostics() -> None:
    controller = make_controller()

    controller.act(OBSERVATION, {"active_replicas": 1, "pending_replicas": 0})

    decision = controller.last_decision
    assert decision is not None
    assert decision.reason == "no_sample"
    assert (
        decision.queued_requests,
        decision.backlog_recovery_rps,
        decision.effective_demand_rps,
    ) == (None, None, None)


def bursty_episode(controller: PredictiveController) -> list[dict[str, Any]]:
    config = SimulatorConfig()
    entry = load_benchmark_manifest().get("syn-train-bursty")  # development split only
    trace = build_workloads([entry])[entry.id]
    return evaluate_controller_episode(AutoscalingEnv(config, trace), controller).infos


def scale_downs_into_backlog(infos: list[dict[str, Any]]) -> int:
    # The action logged at tick t was decided from the queue left by tick t - 1.
    return sum(
        1
        for previous, info in zip(infos, infos[1:], strict=False)
        if previous["queued_requests"] > 0 and info["requested_action"] == SCALE_DOWN
    )


def test_burst_falling_edge_never_scales_down_into_a_backlog() -> None:
    config = SimulatorConfig()
    reference = bursty_episode(ForecastOnlyReference.from_config(config))
    queue_aware = bursty_episode(PredictiveController.from_config(config))

    # Forecast-only (#14) removes capacity on falling edges while requests still wait ...
    assert scale_downs_into_backlog(reference) > 0
    # ... queue-aware sizing never does, and so waits less and violates the SLA less.
    assert scale_downs_into_backlog(queue_aware) == 0

    def mean(infos: list[dict[str, Any]], key: str) -> float:
        return sum(float(info[key]) for info in infos) / len(infos)

    assert mean(queue_aware, "queued_requests") < mean(reference, "queued_requests")
    assert mean(queue_aware, "sla_violated") < mean(reference, "sla_violated")
