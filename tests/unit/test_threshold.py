"""Tests for the reactive threshold controller."""

import math
from typing import Any

import numpy as np
import pytest

from scalerl.controllers import Controller, ThresholdController, ThresholdDecision, run_episode
from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.workloads import WorkloadTrace, steady_workload

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
INTERVAL = 30.0
TICKS = 40
OBSERVATION = np.zeros(9, dtype=np.float32)


def make_controller(**overrides: Any) -> ThresholdController:
    params: dict[str, Any] = {
        "low_threshold": 0.3,
        "high_threshold": 0.8,
        "min_replicas": 2,
        "max_replicas": 6,
    }
    return ThresholdController(**{**params, **overrides})


def tick_info(utilization: float | None, active: int, pending: int = 0) -> dict[str, Any]:
    info: dict[str, Any] = {"active_replicas": active, "pending_replicas": pending}
    if utilization is not None:
        info["utilization"] = utilization
    return info


def make_env(rate: float, **replicas: Any) -> AutoscalingEnv:
    """Replicas serve 10 rps each; 30 s ticks."""
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 8,
        "initial_replicas": 2,
        "startup_delay_seconds": 60.0,
        "service_capacity_rps": 10.0,
    }
    config = SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * TICKS
        ),
        replicas=ReplicaConfig(**{**params, **replicas}),
    )
    trace: WorkloadTrace = steady_workload(
        duration_seconds=INTERVAL * TICKS, control_interval_seconds=INTERVAL, rate=rate
    )
    return AutoscalingEnv(config, trace)


def desired(info: dict[str, Any]) -> int:
    return int(info["active_replicas"] + info["pending_replicas"])


# --- configuration ----------------------------------------------------------


def test_valid_configuration_is_exposed() -> None:
    controller = make_controller(low_threshold=0, high_threshold=0.95)

    assert (controller.low_threshold, controller.high_threshold) == (0.0, 0.95)
    assert (controller.min_replicas, controller.max_replicas) == (2, 6)


@pytest.mark.parametrize(
    ("low", "high"),
    [(0.5, 0.5), (0.8, 0.3), (-0.1, 0.8), (0.3, 1.0), (0.3, 1.5)],
    ids=["equal", "inverted", "negative_low", "high_is_one", "high_above_one"],
)
def test_invalid_threshold_ranges_are_rejected(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="0 <= low_threshold < high_threshold < 1"):
        make_controller(low_threshold=low, high_threshold=high)


@pytest.mark.parametrize("field", ["low_threshold", "high_threshold"])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_thresholds_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        make_controller(**{field: value})


@pytest.mark.parametrize("value", ["0.5", True, None])
def test_non_numeric_thresholds_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="low_threshold must be a number"):
        make_controller(low_threshold=value)


@pytest.mark.parametrize(
    ("bounds", "error", "message"),
    [
        ({"min_replicas": 0}, ValueError, "min_replicas must be at least 1"),
        ({"min_replicas": 4, "max_replicas": 3}, ValueError, "max_replicas must be at least"),
        ({"min_replicas": 2.0}, TypeError, "min_replicas must be an integer"),
        ({"max_replicas": True}, TypeError, "max_replicas must be an integer"),
    ],
)
def test_invalid_replica_bounds_are_rejected(
    bounds: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_controller(**bounds)


def test_min_equal_to_max_is_allowed() -> None:
    controller = make_controller(min_replicas=3, max_replicas=3)

    assert controller.act(OBSERVATION, tick_info(1.0, active=3)) == HOLD
    assert controller.act(OBSERVATION, tick_info(0.0, active=3)) == HOLD


# --- decision logic ---------------------------------------------------------


def test_first_decision_without_utilization_holds() -> None:
    controller = make_controller()

    # Would be a scale-down if the missing sample were read as 0% utilization.
    assert controller.act(OBSERVATION, tick_info(None, active=5)) == HOLD
    assert controller.last_decision == ThresholdDecision(None, 5, HOLD, "no_sample")


@pytest.mark.parametrize(
    ("utilization", "active", "pending", "action", "reason"),
    [
        (0.9, 3, 0, SCALE_UP, "above_high"),
        (1.0, 5, 0, SCALE_UP, "above_high"),
        (1.0, 6, 0, HOLD, "at_max"),
        (1.0, 3, 3, HOLD, "at_max"),  # pending replicas fill the remaining headroom
        (1.0, 3, 2, SCALE_UP, "above_high"),
        (0.1, 4, 0, SCALE_DOWN, "below_low"),
        (0.0, 3, 0, SCALE_DOWN, "below_low"),
        (0.1, 2, 0, HOLD, "at_min"),
        (0.1, 1, 1, HOLD, "at_min"),  # pending replicas count toward the minimum too
        (0.5, 4, 0, HOLD, "within_band"),
        (0.8, 4, 0, HOLD, "within_band"),  # exactly high: strict crossing required
        (0.3, 4, 0, HOLD, "within_band"),  # exactly low: strict crossing required
        (1.0, 8, 0, SCALE_DOWN, "above_max"),  # outside bounds wins over high utilization
        (0.5, 5, 2, SCALE_DOWN, "above_max"),
        (0.5, 1, 0, SCALE_UP, "below_min"),
        (0.0, 1, 0, SCALE_UP, "below_min"),  # ...and over low utilization
    ],
)
def test_threshold_decisions(
    utilization: float, active: int, pending: int, action: int, reason: str
) -> None:
    controller = make_controller()

    assert controller.act(OBSERVATION, tick_info(utilization, active, pending)) == action
    assert controller.last_decision == ThresholdDecision(
        utilization,
        active + pending,
        action,
        reason,  # type: ignore[arg-type]
    )


def test_decisions_ignore_the_observation() -> None:
    controller = make_controller()
    saturated = np.ones(9, dtype=np.float32)

    assert controller.act(saturated, tick_info(0.5, active=4)) == HOLD
    assert controller.act(OBSERVATION, tick_info(0.9, active=4)) == SCALE_UP


def test_reset_clears_diagnostics() -> None:
    controller = make_controller()
    controller.act(OBSERVATION, tick_info(0.9, active=3))

    controller.reset(seed=1)

    assert controller.last_decision is None


# --- determinism ------------------------------------------------------------


def test_fixed_input_sequence_gives_identical_actions_regardless_of_seed() -> None:
    inputs = [
        tick_info(None, 2),
        tick_info(0.95, 2),
        tick_info(0.95, 2, 1),
        tick_info(0.6, 3, 1),
        tick_info(0.1, 4),
        tick_info(0.25, 3),
        tick_info(0.3, 2),
        tick_info(0.8, 2),
    ]

    def run(controller: ThresholdController, seed: int | None) -> list[int]:
        controller.reset(seed=seed)
        return [controller.act(OBSERVATION, info) for info in inputs]

    controller = make_controller()
    first = run(controller, seed=None)

    assert first == [HOLD, SCALE_UP, SCALE_UP, HOLD, SCALE_DOWN, SCALE_DOWN, HOLD, HOLD]
    assert run(controller, seed=1) == first
    assert run(make_controller(), seed=99) == first


# --- integration ------------------------------------------------------------


def test_threshold_controller_satisfies_the_controller_protocol() -> None:
    assert isinstance(make_controller(), Controller)


def test_reset_state_holds_despite_zero_filled_observation() -> None:
    env = make_env(rate=1.0, initial_replicas=5)
    observation, info = env.reset(seed=0)
    controller = make_controller()

    utilization = observation[env.observation_features.index("utilization")]
    assert utilization == 0.0  # utilization feature is zero-filled at reset
    assert controller.act(observation, info) == HOLD


def test_sustained_load_scales_up_then_settles() -> None:
    env = make_env(rate=45.0, initial_replicas=2)  # needs ~5-6 replicas at 10 rps each
    controller = make_controller(max_replicas=8)

    infos = run_episode(env, controller)

    actions = [info["requested_action"] for info in infos]
    assert all(env.action_space.contains(action) for action in actions)
    assert actions[0] == HOLD  # no utilization sample yet
    assert actions[1] == SCALE_UP
    assert all(desired(info) <= 8 for info in infos)
    assert actions[-10:] == [HOLD] * 10  # settled once the backlog drained
    assert 0.3 <= infos[-1]["utilization"] <= 0.8
    assert infos[-1]["queued_requests"] == 0.0


def test_startup_delay_never_pushes_desired_capacity_past_max() -> None:
    env = make_env(rate=500.0, initial_replicas=2, startup_delay_seconds=150)  # 5-tick delay
    controller = make_controller(max_replicas=6)

    infos = run_episode(env, controller)

    ups = [info for info in infos if info["requested_action"] == SCALE_UP]
    # Overload keeps utilization saturated while replicas start, so scale-ups continue,
    # but pending replicas count toward max: exactly 4 requests, all applied.
    assert len(ups) == 4
    assert all(info["applied_replica_change"] == 1 for info in ups)
    assert max(desired(info) for info in infos) == 6
    assert infos[-1]["active_replicas"] == 6


def test_sustained_low_load_scales_down_to_min_and_holds() -> None:
    env = make_env(rate=1.0, initial_replicas=6)
    controller = make_controller(min_replicas=2)

    infos = run_episode(env, controller)

    actions = [info["requested_action"] for info in infos]
    assert actions[:5] == [HOLD] + [SCALE_DOWN] * 4
    assert actions[5:] == [HOLD] * (TICKS - 5)
    assert all(desired(info) >= 2 for info in infos)
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "at_min"


def test_fleet_above_controller_max_is_brought_back_under_overload() -> None:
    env = make_env(rate=500.0, initial_replicas=8)  # env allows 8; controller max is 6
    controller = make_controller(max_replicas=6)

    infos = run_episode(env, controller)

    actions = [info["requested_action"] for info in infos]
    assert actions[:3] == [HOLD, SCALE_DOWN, SCALE_DOWN]
    assert actions[3:] == [HOLD] * (TICKS - 3)
    assert infos[-1]["active_replicas"] == 6
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "at_max"


def test_fleet_below_controller_min_is_raised_within_the_band() -> None:
    env = make_env(rate=5.0, initial_replicas=1, startup_delay_seconds=0)  # env allows 1
    controller = make_controller(low_threshold=0.1, min_replicas=3)

    infos = run_episode(env, controller)

    actions = [info["requested_action"] for info in infos]
    assert actions[:3] == [HOLD, SCALE_UP, SCALE_UP]
    assert infos[0]["utilization"] == 0.5  # within the band, so thresholds alone would hold
    assert actions[3:] == [HOLD] * (TICKS - 3)
    assert infos[-1]["active_replicas"] == 3


def test_complete_episodes_are_deterministic() -> None:
    env = make_env(rate=45.0)

    assert run_episode(env, make_controller(), seed=1) == run_episode(
        env, make_controller(), seed=2
    )


# --- cooldown -------------------------------------------------------------------


def step_info(
    utilization: float, active: int, *, tick: int, applied: int = 0, pending: int = 0
) -> dict[str, Any]:
    """Decision info after a completed step, as run_episode passes it."""
    return {
        **tick_info(utilization, active, pending),
        "tick": tick,
        "applied_replica_change": applied,
    }


@pytest.mark.parametrize("cooldown", [0, 1, 10])
def test_valid_cooldown_is_accepted(cooldown: int) -> None:
    assert make_controller(cooldown_ticks=cooldown).cooldown_ticks == cooldown


def test_cooldown_defaults_to_disabled() -> None:
    assert make_controller().cooldown_ticks == 0


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (-1, ValueError, "cooldown_ticks must be non-negative"),
        (True, TypeError, "cooldown_ticks must be an integer"),
        (2.0, TypeError, "cooldown_ticks must be an integer"),
        ("3", TypeError, "cooldown_ticks must be an integer"),
    ],
)
def test_invalid_cooldown_is_rejected(value: object, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        make_controller(cooldown_ticks=value)


@pytest.mark.parametrize(
    ("utilization", "applied", "action"),
    [(0.9, 1, SCALE_UP), (0.1, -1, SCALE_DOWN)],
    ids=["after_scale_up", "after_scale_down"],
)
def test_applied_change_blocks_the_next_n_decisions(
    utilization: float, applied: int, action: int
) -> None:
    controller = make_controller(cooldown_ticks=3)

    # Tick 4 completed with a real one-replica change; cooldown covers ticks 5-7.
    decisions = []
    for tick, change in [(4, applied), (5, 0), (6, 0), (7, 0)]:
        decisions.append(
            controller.act(OBSERVATION, step_info(utilization, 4, tick=tick, applied=change))
        )
        decisions.append(controller.last_decision)

    actions, details = decisions[::2], decisions[1::2]
    assert actions == [HOLD, HOLD, HOLD, action]
    assert [d.reason for d in details] == ["cooldown", "cooldown", "cooldown", details[-1].reason]
    assert details[-1].reason in {"above_high", "below_low"}
    assert [d.cooldown_remaining for d in details] == [2, 1, 0, 0]


def test_bound_clipped_request_does_not_start_cooldown() -> None:
    controller = make_controller(cooldown_ticks=3)

    # The previous SCALE_UP was clipped by the environment: applied change 0.
    action = controller.act(OBSERVATION, step_info(0.95, 5, tick=8, applied=0))

    assert action == SCALE_UP
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "above_high"
    assert controller.last_decision.cooldown_remaining == 0


def test_rereading_the_same_tick_does_not_restart_cooldown() -> None:
    controller = make_controller(cooldown_ticks=2)
    same = step_info(0.95, 4, tick=10, applied=1)

    first = controller.act(OBSERVATION, same)
    second = controller.act(OBSERVATION, same)  # e.g. a re-render with stale info
    third = controller.act(OBSERVATION, same)

    assert (first, second) == (HOLD, HOLD)
    assert third == SCALE_UP  # the tick-10 change was consumed once, not restarted
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "above_high"


def test_reset_clears_cooldown() -> None:
    controller = make_controller(cooldown_ticks=5)
    controller.act(OBSERVATION, step_info(0.95, 4, tick=3, applied=1))

    controller.reset()

    assert controller.act(OBSERVATION, step_info(0.95, 4, tick=3, applied=0)) == SCALE_UP
    # After reset the same tick can be consumed again (new episode).
    controller.reset()
    assert controller.act(OBSERVATION, step_info(0.95, 4, tick=3, applied=1)) == HOLD


def test_bound_correction_overrides_cooldown() -> None:
    controller = make_controller(cooldown_ticks=3, max_replicas=6)

    action = controller.act(OBSERVATION, step_info(0.95, 8, tick=2, applied=1))

    assert action == SCALE_DOWN
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "above_max"
    assert controller.last_decision.cooldown_remaining == 2  # the correction used a slot


def test_threshold_equality_and_pending_rules_hold_after_cooldown() -> None:
    controller = make_controller(cooldown_ticks=1)
    controller.act(OBSERVATION, step_info(0.9, 3, tick=0, applied=1))

    assert controller.act(OBSERVATION, step_info(0.8, 3, tick=1)) == HOLD  # equal to high
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "within_band"
    assert controller.act(OBSERVATION, step_info(0.9, 3, tick=2, pending=3)) == HOLD
    assert controller.last_decision.reason == "at_max"  # pending counts toward desired


def test_zero_cooldown_reproduces_the_uncooled_controller() -> None:
    def actions(controller: ThresholdController) -> list[int]:
        infos = run_episode(make_env(rate=45.0, initial_replicas=2), controller)
        return [info["requested_action"] for info in infos]

    plain = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=8
    )
    zero = make_controller(max_replicas=8, min_replicas=1, cooldown_ticks=0)

    assert actions(zero) == actions(plain)


def oscillating_env() -> AutoscalingEnv:
    """Demand alternates across both thresholds every tick, provoking thrashing."""
    config = SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * 60
        ),
        replicas=ReplicaConfig(
            min_replicas=1,
            max_replicas=6,
            initial_replicas=2,
            startup_delay_seconds=0.0,
            service_capacity_rps=10.0,
        ),
    )
    trace = WorkloadTrace([28.0 if tick % 2 else 4.0 for tick in range(60)], INTERVAL)
    return AutoscalingEnv(config, trace)


def test_cooldown_reduces_applied_scaling_churn_on_oscillating_demand() -> None:
    def applied_changes(cooldown: int) -> int:
        controller = make_controller(min_replicas=1, max_replicas=6, cooldown_ticks=cooldown)
        infos = run_episode(oscillating_env(), controller)
        return sum(1 for info in infos if info["applied_replica_change"] != 0)

    uncooled, cooled = applied_changes(0), applied_changes(3)

    assert uncooled >= 30  # scales on almost every tick
    assert cooled < uncooled / 2
