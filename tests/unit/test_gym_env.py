"""Tests for the Gymnasium autoscaling environment."""

import warnings
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from scalerl.environment import (
    MAX_REWARD_WEIGHT,
    AutoscalingEnv,
    ReplicaConfig,
    RewardWeights,
    SimulatorConfig,
    SlaConfig,
    TimingConfig,
)
from scalerl.workloads import WorkloadTrace, bursty_workload

HOLD, SCALE_DOWN, SCALE_UP = 1, 0, 2
INTERVAL = 30.0
TICKS = 5


def make_config(**replica_overrides: Any) -> SimulatorConfig:
    """1-4 replicas, 10 rps each (300 requests per replica per 30 s tick), 60 s startup."""
    replicas: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 4,
        "initial_replicas": 1,
        "startup_delay_seconds": 60.0,
        "service_capacity_rps": 10.0,
        "cost_per_hour": 0.36,
    }
    return SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * TICKS
        ),
        replicas=ReplicaConfig(**{**replicas, **replica_overrides}),
        sla=SlaConfig(latency_target_seconds=0.5),
    )


def make_env(
    rates: list[float] | None = None,
    reward_weights: RewardWeights | None = None,
    **replica_overrides: Any,
) -> AutoscalingEnv:
    trace = WorkloadTrace(rates or [5.0] * TICKS, control_interval_seconds=INTERVAL)
    return AutoscalingEnv(make_config(**replica_overrides), trace, reward_weights)


def feature(env: AutoscalingEnv, observation: np.ndarray, name: str) -> float:
    return float(observation[env.observation_features.index(name)])


def pending(env: AutoscalingEnv, observation: np.ndarray) -> np.ndarray:
    start = env.observation_features.index("episode_progress") + 1
    return observation[start:]


def reset(env: AutoscalingEnv) -> np.ndarray:
    observation, _ = env.reset(seed=0)
    return observation


# --- construction -----------------------------------------------------------


def test_spaces_are_discrete_actions_and_unit_box_observations() -> None:
    env = make_env()

    assert env.action_space == spaces.Discrete(3)
    assert isinstance(env.observation_space, spaces.Box)
    # 4 traffic-history ticks + 6 state features + 2 pending buckets (60 s delay / 30 s ticks)
    assert env.observation_space.shape == (4 + 6 + 2,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == 0)
    assert np.all(env.observation_space.high == 1)
    assert env.episode_ticks == TICKS


def test_passes_gymnasium_env_checker() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(make_env(), skip_render_check=True)


def test_mismatched_trace_interval_is_rejected() -> None:
    config = make_config()
    trace = WorkloadTrace([5.0] * 10, control_interval_seconds=15)

    with pytest.raises(ValueError, match="does not match configured control interval"):
        AutoscalingEnv(config, trace)


@pytest.mark.parametrize("length", [TICKS - 1, TICKS + 1])
def test_trace_must_span_exactly_one_episode(length: int) -> None:
    config = make_config()
    trace = WorkloadTrace([5.0] * length, control_interval_seconds=INTERVAL)

    with pytest.raises(ValueError, match="not truncated or resampled"):
        AutoscalingEnv(config, trace)


def test_episode_must_be_whole_number_of_intervals() -> None:
    config = SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=INTERVAL, episode_duration_seconds=100)
    )
    trace = WorkloadTrace([5.0] * 3, control_interval_seconds=INTERVAL)

    with pytest.raises(ValueError, match="whole number of control intervals"):
        AutoscalingEnv(config, trace)


# --- reset ------------------------------------------------------------------


def test_reset_returns_initial_observation_and_info() -> None:
    env = make_env()

    observation, info = env.reset(seed=0)

    assert observation in env.observation_space
    # history (4), utilization, queue, latency, active 1/4, cost, progress, pending (2)
    np.testing.assert_array_equal(observation, [0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0])
    assert info == {
        "tick": 0,
        "time_seconds": 0.0,
        "active_replicas": 1,
        "pending_replicas": 0,
        "terminating_replicas": 0,
    }


def test_reset_restores_every_component() -> None:
    env = make_env(rates=[100.0] * TICKS)
    initial = reset(env)
    for action in (SCALE_UP, SCALE_UP, HOLD):
        env.step(action)

    observation, info = env.reset(seed=0)

    np.testing.assert_array_equal(observation, initial)
    assert (info["active_replicas"], info["pending_replicas"]) == (1, 0)
    _, _, _, _, step_info = env.step(HOLD)
    assert step_info["tick"] == 0
    assert step_info["arrived_requests"] == 3000.0  # queue and replay restarted
    assert step_info["processed_requests"] == 300.0


def test_step_before_reset_is_rejected() -> None:
    env = make_env()

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(HOLD)


# --- one step ---------------------------------------------------------------


def test_one_step_composes_workload_queue_metrics_and_reward() -> None:
    env = make_env()  # 5 rps against one 10 rps replica
    reset(env)

    observation, reward, terminated, truncated, info = env.step(HOLD)

    assert (terminated, truncated) == (False, False)
    assert info["tick"] == 0
    assert info["time_seconds"] == INTERVAL
    assert info["request_rate"] == 5.0
    assert (info["requested_action"], info["applied_replica_change"]) == (HOLD, 0)
    assert (info["active_replicas"], info["pending_replicas"], info["terminating_replicas"]) == (
        1,
        0,
        0,
    )
    assert info["arrived_requests"] == 150.0
    assert info["processed_requests"] == 150.0
    assert info["queued_requests"] == 0.0
    assert info["dropped_requests"] == 0.0
    assert info["utilization"] == pytest.approx(0.5)
    assert info["p95_latency_seconds"] == pytest.approx(0.1 * 2)  # service time x 1/(1-0.5)
    assert info["sla_violated"] is False
    assert info["infrastructure_cost"] == pytest.approx(0.36 * INTERVAL / 3600)
    assert info["reward"] == reward

    expected_observation = [
        5 / (5 + 40),  # latest demand vs. 4 replicas x 10 rps
        0.0,  # no older completed ticks yet
        0.0,
        0.0,
        0.5,
        0.0,
        0.2 / (0.2 + 0.5),
        0.25,
        0.25,  # 1 of 4 billable replicas
        1 / TICKS,
        0.0,  # no pending replicas
        0.0,
    ]
    np.testing.assert_allclose(observation, expected_observation, rtol=1e-6)
    assert observation in env.observation_space


# --- actions ----------------------------------------------------------------


def test_actions_map_to_scale_down_hold_and_scale_up() -> None:
    env = make_env(initial_replicas=2)
    reset(env)

    up = env.step(SCALE_UP)[4]
    hold = env.step(HOLD)[4]
    down = env.step(SCALE_DOWN)[4]

    assert (up["applied_replica_change"], up["pending_replicas"]) == (1, 1)
    assert (hold["applied_replica_change"], hold["pending_replicas"]) == (0, 1)
    assert (down["applied_replica_change"], down["pending_replicas"]) == (-1, 0)
    assert down["active_replicas"] == 2  # pending was cancelled before any active replica


def test_scale_down_terminates_an_active_replica() -> None:
    env = make_env(initial_replicas=3)
    reset(env)

    info = env.step(SCALE_DOWN)[4]

    assert info["applied_replica_change"] == -1
    assert (info["active_replicas"], info["terminating_replicas"]) == (2, 1)
    assert info["processed_requests"] == 150.0  # terminating replica served nothing


@pytest.mark.parametrize(
    ("action", "overrides"),
    [(SCALE_DOWN, {"initial_replicas": 1}), (SCALE_UP, {"initial_replicas": 4})],
)
def test_scaling_at_bounds_is_a_reported_no_op(action: int, overrides: dict[str, Any]) -> None:
    env = make_env(**overrides)
    reset(env)

    info = env.step(action)[4]

    assert info["requested_action"] == action
    assert info["applied_replica_change"] == 0
    assert info["active_replicas"] == overrides["initial_replicas"]
    assert info["reward_components"]["churn_penalty"] == 0.0


def test_numpy_integer_actions_are_accepted() -> None:
    env = make_env()
    reset(env)

    assert env.step(np.int64(SCALE_UP))[4]["applied_replica_change"] == 1


@pytest.mark.parametrize("action", [-1, 3, 1.0, "1", True, None])
def test_invalid_actions_are_rejected(action: object) -> None:
    env = make_env()
    reset(env)

    with pytest.raises(ValueError, match="invalid action"):
        env.step(action)  # type: ignore[arg-type]


# --- startup delay ----------------------------------------------------------


def test_scale_up_serves_traffic_only_after_startup_delay() -> None:
    env = make_env(rates=[100.0] * TICKS, startup_delay_seconds=60)  # 2 ticks of delay
    reset(env)

    ticks = [env.step(action)[4] for action in (SCALE_UP, HOLD, HOLD)]

    assert [t["active_replicas"] for t in ticks] == [1, 1, 2]
    assert [t["pending_replicas"] for t in ticks] == [1, 1, 0]
    assert [t["processed_requests"] for t in ticks] == [300.0, 300.0, 600.0]


def test_observation_distinguishes_pending_replicas_by_readiness() -> None:
    # Same tick and pending count, but one replica is a tick closer to active.
    early = make_env(startup_delay_seconds=90)  # 3 ticks of delay
    late = make_env(startup_delay_seconds=90)
    reset(early)
    reset(late)
    for early_action, late_action in [(SCALE_UP, HOLD), (HOLD, SCALE_UP)]:
        early_observation = early.step(early_action)[0]
        late_observation = late.step(late_action)[0]

    np.testing.assert_allclose(pending(early, early_observation), [0.25, 0.0, 0.0])
    np.testing.assert_allclose(pending(late, late_observation), [0.0, 0.25, 0.0])
    # After one more tick, only the earlier request has become active.
    assert feature(early, early.step(HOLD)[0], "active_replicas_fraction") == 0.5
    assert feature(late, late.step(HOLD)[0], "active_replicas_fraction") == 0.25


@pytest.mark.parametrize(("delay", "buckets"), [(0, 0), (30, 1), (45, 2), (90, 3)])
def test_pending_bucket_count_follows_startup_delay(delay: float, buckets: int) -> None:
    env = make_env(startup_delay_seconds=delay)

    assert env.observation_space.shape == (4 + 6 + buckets,)
    assert reset(env).shape == (4 + 6 + buckets,)


def test_replica_counts_reflect_state_after_lifecycle_advance() -> None:
    env = make_env(startup_delay_seconds=30)  # activates after one tick
    _, reset_info = env.reset(seed=0)
    assert env.replica_counts == {
        key: reset_info[key]
        for key in ("active_replicas", "pending_replicas", "terminating_replicas")
    }

    info = env.step(SCALE_UP)[4]

    assert (info["active_replicas"], info["pending_replicas"]) == (1, 1)  # served the tick
    assert env.replica_counts == {
        "active_replicas": 2,
        "pending_replicas": 0,
        "terminating_replicas": 0,
    }


def test_pending_replica_is_billed_but_does_not_serve() -> None:
    env = make_env(rates=[100.0] * TICKS)
    reset(env)

    info = env.step(SCALE_UP)[4]

    assert info["processed_requests"] == 300.0
    assert info["infrastructure_cost"] == pytest.approx(2 * 0.36 * INTERVAL / 3600)


def test_zero_startup_delay_serves_in_the_same_tick() -> None:
    env = make_env(rates=[100.0] * TICKS, startup_delay_seconds=0)
    reset(env)

    info = env.step(SCALE_UP)[4]

    assert (info["active_replicas"], info["pending_replicas"]) == (2, 0)
    assert info["processed_requests"] == 600.0


# --- reward -----------------------------------------------------------------


def test_reward_is_negative_sum_of_reported_components() -> None:
    env = make_env(rates=[100.0] * TICKS)  # overloaded: SLA violated, queue grows
    reset(env)

    _, reward, _, _, info = env.step(SCALE_UP)
    components = info["reward_components"]
    weights = RewardWeights()

    assert reward == pytest.approx(-sum(components.values()))
    assert components["sla_penalty"] == weights.sla
    assert components["churn_penalty"] == weights.churn
    assert 0 < components["queue_penalty"] < weights.queue
    assert 0 < components["latency_penalty"] < weights.latency
    assert components["cost_penalty"] == pytest.approx(weights.cost * 2 / 4)


def test_reward_weights_scale_components() -> None:
    zero = RewardWeights(latency=0, cost=0, sla=0, queue=0, churn=0)
    doubled = RewardWeights(latency=2, cost=2, sla=2, queue=2, churn=0.2)
    results = {}
    for name, weights in {"default": None, "zero": zero, "doubled": doubled}.items():
        env = make_env(rates=[100.0] * TICKS, reward_weights=weights)
        reset(env)
        results[name] = env.step(SCALE_UP)[1:5:3]

    assert results["zero"][0] == 0.0
    assert results["doubled"][0] == pytest.approx(2 * float(results["default"][0]))


def test_reward_weights_are_validated_and_immutable() -> None:
    with pytest.raises(ValueError):
        RewardWeights(latency=-1)
    with pytest.raises(ValueError):
        RewardWeights(cost=float("nan"))
    with pytest.raises(ValueError):
        RewardWeights().sla = 5  # type: ignore[misc]


def test_reward_weights_are_capped() -> None:
    RewardWeights(latency=MAX_REWARD_WEIGHT)
    with pytest.raises(ValueError):
        RewardWeights(latency=MAX_REWARD_WEIGHT * 2)


def test_maximum_reward_weights_keep_reward_finite() -> None:
    weights = RewardWeights(
        latency=MAX_REWARD_WEIGHT,
        cost=MAX_REWARD_WEIGHT,
        sla=MAX_REWARD_WEIGHT,
        queue=MAX_REWARD_WEIGHT,
        churn=MAX_REWARD_WEIGHT,
    )
    env = make_env(rates=[1000.0] * TICKS, reward_weights=weights)
    reset(env)

    reward = float(env.step(SCALE_UP)[1])

    assert np.isfinite(reward)
    assert -5 * MAX_REWARD_WEIGHT <= reward < 0


def test_zero_price_config_has_zero_cost_penalty_and_observation() -> None:
    env = make_env(cost_per_hour=0.0)
    reset(env)

    observation, _, _, _, info = env.step(HOLD)

    assert info["reward_components"]["cost_penalty"] == 0.0
    assert feature(env, observation, "tick_cost_fraction") == 0.0


# --- episode end ------------------------------------------------------------


def test_episode_is_truncated_when_trace_is_exhausted() -> None:
    env = make_env()
    reset(env)

    outcomes = [env.step(HOLD) for _ in range(TICKS)]

    assert [o[2] for o in outcomes] == [False] * TICKS
    assert [o[3] for o in outcomes] == [False] * (TICKS - 1) + [True]
    assert feature(env, outcomes[-1][0], "episode_progress") == 1.0
    assert outcomes[-1][4]["time_seconds"] == INTERVAL * TICKS


def test_step_after_episode_end_fails_until_reset() -> None:
    env = make_env()
    reset(env)
    for _ in range(TICKS):
        env.step(HOLD)

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(HOLD)

    reset(env)
    assert env.step(HOLD)[4]["tick"] == 0


# --- observation bounds and determinism -------------------------------------


def test_observations_stay_in_bounds_under_extreme_bursty_traffic() -> None:
    config = SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=30, episode_duration_seconds=3000),
        replicas=ReplicaConfig(min_replicas=1, max_replicas=3, initial_replicas=1),
    )
    trace = bursty_workload(
        duration_seconds=3000,
        control_interval_seconds=30,
        baseline=20,
        burst_rate=5000,
        burst_probability=0.2,
        burst_duration_seconds=90,
        noise_std=50,
        seed=11,
    )
    env = AutoscalingEnv(config, trace)
    observation, _ = env.reset(seed=3)
    env.action_space.seed(3)

    truncated = False
    while not truncated:
        assert observation in env.observation_space
        observation, reward, _, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(float(reward))
    assert observation in env.observation_space


def test_same_seed_and_actions_reproduce_the_episode() -> None:
    actions = [SCALE_UP, HOLD, SCALE_UP, SCALE_DOWN, HOLD]

    def run(env: AutoscalingEnv) -> list[Any]:
        history: list[Any] = [env.reset(seed=42)[0].tolist()]
        for action in actions:
            observation, reward, _, truncated, info = env.step(action)
            history.append((observation.tolist(), reward, truncated, info))
        return history

    env = make_env(rates=[40.0, 80.0, 10.0, 120.0, 60.0])

    assert run(env) == run(env) == run(make_env(rates=[40.0, 80.0, 10.0, 120.0, 60.0]))
