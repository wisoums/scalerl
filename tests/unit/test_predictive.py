"""Tests for the predictive (linear-trend forecasting) autoscaler (#14)."""

import math
from typing import Any

import numpy as np
import pytest

from scalerl.controllers import (
    Controller,
    PredictiveController,
    ThresholdController,
    run_episode,
)
from scalerl.environment import (
    AutoscalingEnv,
    ReplicaConfig,
    ReplicaPool,
    SimulatorConfig,
    TimingConfig,
)
from scalerl.workloads import WorkloadTrace, ramp_workload

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
OBSERVATION = np.zeros(12, dtype=np.float32)


def make_controller(**overrides: Any) -> PredictiveController:
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 20,
        "service_capacity_rps": 50.0,
        "startup_delay_seconds": 60.0,
        "control_interval_seconds": 30.0,
    }
    return PredictiveController(**{**params, **overrides})


def step_info(tick: int, rate: float, active: int = 1, pending: int = 0) -> dict[str, Any]:
    return {
        "tick": tick,
        "request_rate": rate,
        "active_replicas": active,
        "pending_replicas": pending,
    }


def feed(controller: PredictiveController, rates: list[float], **replicas: int) -> int:
    action = HOLD
    for tick, rate in enumerate(rates):
        action = controller.act(OBSERVATION, step_info(tick, rate, **replicas))
    return action


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
