"""Tests for the recent-traffic window in the AutoscalingEnv observation (#58)."""

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import RandomController, StaticController, ThresholdController, run_episode
from scalerl.environment import (
    AutoscalingEnv,
    ObservationConfig,
    ReplicaConfig,
    SimulatorConfig,
    TimingConfig,
)
from scalerl.mlops import EnvironmentCompatibility
from scalerl.workloads import WorkloadTrace, bursty_workload

HOLD, SCALE_UP = 1, 2
INTERVAL = 30.0
MAX_RATE = 4 * 10.0  # 4 replicas x 10 rps, the demand-pressure denominator below


def make_env(rates: list[float], history: int = 4, **replicas: Any) -> AutoscalingEnv:
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 4,
        "initial_replicas": 1,
        "startup_delay_seconds": 60.0,
        "service_capacity_rps": 10.0,
    }
    config = SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * len(rates)
        ),
        replicas=ReplicaConfig(**{**params, **replicas}),
        observation=ObservationConfig(traffic_history_ticks=history),
    )
    return AutoscalingEnv(config, WorkloadTrace(rates, control_interval_seconds=INTERVAL))


def pressure(rate: float) -> float:
    return rate / (rate + MAX_RATE)


def history_block(env: AutoscalingEnv, observation: np.ndarray) -> list[float]:
    return observation[: env.config.observation.traffic_history_ticks].tolist()


def pending_buckets(delay: float, interval: float) -> int:
    return math.ceil(delay / interval) if delay > 0 else 0


# --- configuration ------------------------------------------------------------


def test_v1_default_is_four_ticks() -> None:
    assert SimulatorConfig().observation.traffic_history_ticks == 4
    assert ObservationConfig().traffic_history_ticks == 4


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "4", None])
def test_invalid_history_lengths_are_rejected(value: object) -> None:
    with pytest.raises(ValidationError, match="traffic_history_ticks"):
        ObservationConfig(traffic_history_ticks=value)  # type: ignore[arg-type]


def test_history_length_is_frozen_and_serialized() -> None:
    config = SimulatorConfig(observation=ObservationConfig(traffic_history_ticks=6))

    with pytest.raises(ValidationError):
        config.observation.traffic_history_ticks = 2  # type: ignore[misc]
    assert config.model_dump()["observation"] == {"traffic_history_ticks": 6}
    assert SimulatorConfig.model_validate_json(config.model_dump_json()) == config


# --- history contents -----------------------------------------------------------


def test_reset_zero_fills_the_history() -> None:
    env = make_env([50.0, 80.0])

    observation, _ = env.reset(seed=0)

    assert history_block(env, observation) == [0.0, 0.0, 0.0, 0.0]


def test_history_fills_newest_first_then_rolls() -> None:
    rates = [50.0, 80.0, 120.0, 180.0, 30.0, 70.0]
    env = make_env(rates)
    env.reset(seed=0)

    blocks = [history_block(env, env.step(HOLD)[0]) for _ in rates]

    expected = [
        [pressure(50), 0, 0, 0],
        [pressure(80), pressure(50), 0, 0],
        [pressure(120), pressure(80), pressure(50), 0],
        [pressure(180), pressure(120), pressure(80), pressure(50)],
        [pressure(30), pressure(180), pressure(120), pressure(80)],  # 50 has rolled out
        [pressure(70), pressure(30), pressure(180), pressure(120)],
    ]
    np.testing.assert_allclose(blocks, expected, rtol=1e-6)


def test_history_matches_the_demand_consumed_each_tick() -> None:
    env = make_env([12.0, 0.0, 400.0, 7.5, 7.5])
    env.reset(seed=0)

    for _ in range(5):
        observation, _, _, _, info = env.step(SCALE_UP)
        assert observation[0] == pytest.approx(pressure(info["request_rate"]), rel=1e-6)


def test_reset_clears_the_history() -> None:
    env = make_env([50.0, 80.0, 120.0])
    env.reset(seed=0)
    env.step(HOLD)
    env.step(HOLD)

    observation, _ = env.reset(seed=0)

    assert history_block(env, observation) == [0.0, 0.0, 0.0, 0.0]


def test_feature_names_describe_the_layout() -> None:
    env = make_env([1.0, 2.0])

    assert env.observation_features == (
        "demand_pressure_t-0",
        "demand_pressure_t-1",
        "demand_pressure_t-2",
        "demand_pressure_t-3",
        "utilization",
        "queue_pressure",
        "latency_pressure",
        "active_replicas_fraction",
        "tick_cost_fraction",
        "episode_progress",
        "pending_ready_in_1",
        "pending_ready_in_2",
    )
    assert len(env.observation_features) == env.observation_space.shape[0]


# --- no look-ahead ----------------------------------------------------------------


def test_the_next_workload_value_never_reaches_the_observation() -> None:
    spiky = make_env([10.0, 20.0, 999.0, 5.0])
    calm = make_env([10.0, 20.0, 30.0, 5.0])  # identical except the not-yet-consumed tick 3

    observations = {}
    for name, env in (("spiky", spiky), ("calm", calm)):
        observation, _ = env.reset(seed=0)
        seen = [observation]
        for _ in range(2):
            seen.append(env.step(HOLD)[0])
        observations[name] = seen

    # Up to the decision before tick 3, both environments look exactly the same.
    for spiky_obs, calm_obs in zip(observations["spiky"], observations["calm"], strict=True):
        np.testing.assert_array_equal(spiky_obs, calm_obs)
    after_second_tick = observations["spiky"][-1]
    np.testing.assert_allclose(
        history_block(spiky, after_second_tick), [pressure(20), pressure(10), 0, 0], rtol=1e-6
    )
    assert not np.any(np.isclose(after_second_tick, pressure(999.0)))

    # Only after tick 3 is consumed does it appear, as the newest entry.
    third = spiky.step(HOLD)[0]
    assert third[0] == pytest.approx(pressure(999.0), rel=1e-6)


# --- shapes and backward semantics -------------------------------------------------


@pytest.mark.parametrize(("delay", "history"), [(60, 4), (0, 4), (90, 4), (60, 1), (0, 1), (60, 7)])
def test_shape_is_history_plus_six_plus_pending_buckets(delay: float, history: int) -> None:
    env = make_env([5.0, 5.0], history=history, startup_delay_seconds=delay)

    expected = history + 6 + pending_buckets(delay, INTERVAL)
    assert env.observation_space.shape == (expected,)
    assert env.reset(seed=0)[0].shape == (expected,)


def test_default_config_shape_is_twelve() -> None:
    config = SimulatorConfig()  # 4 history ticks, 60 s delay / 30 s ticks -> 2 buckets
    trace = WorkloadTrace([1.0] * 120, control_interval_seconds=30.0)

    assert AutoscalingEnv(config, trace).observation_space.shape == (12,)


def test_zero_startup_delay_with_v1_history_has_ten_features() -> None:
    env = make_env([5.0, 5.0], startup_delay_seconds=0.0)

    assert env.observation_space.shape == (10,)


def test_single_tick_history_reproduces_the_previous_layout() -> None:
    rates = [50.0, 80.0, 120.0, 180.0, 30.0]
    old = make_env(rates, history=1)
    new = make_env(rates, history=4)
    assert old.observation_space.shape == (7 + 2,)  # the pre-#58 formula 7 + k

    old_obs, _ = old.reset(seed=0)
    new_obs, _ = new.reset(seed=0)
    steps = [(old_obs, new_obs)]
    for action in (SCALE_UP, HOLD, HOLD, HOLD, HOLD):
        steps.append((old.step(action)[0], new.step(action)[0]))

    for old_obs, new_obs in steps:
        # Same latest demand, and every non-traffic feature is identical.
        assert old_obs[0] == new_obs[0]
        np.testing.assert_array_equal(old_obs[1:], new_obs[4:])
    assert old.observation_features[1:] == new.observation_features[4:]


# --- bounds, determinism, and integration ----------------------------------------------


@pytest.mark.parametrize("history", [1, 4, 9])
def test_observations_stay_finite_and_in_bounds(history: int) -> None:
    config = SimulatorConfig(
        replicas=ReplicaConfig(min_replicas=1, max_replicas=3, initial_replicas=1),
        observation=ObservationConfig(traffic_history_ticks=history),
    )
    trace = bursty_workload(
        duration_seconds=3600,
        control_interval_seconds=30,
        baseline=5,
        burst_rate=1e6,
        burst_probability=0.3,
        burst_duration_seconds=60,
        noise_std=50,
        seed=3,
    )
    env = AutoscalingEnv(config, trace)
    controller = RandomController(seed=history)
    observation, info = env.reset(seed=0)
    controller.reset(seed=history)

    truncated = False
    while not truncated:
        assert observation.dtype == np.float32
        assert observation.shape == env.observation_space.shape
        assert np.all(np.isfinite(observation))
        assert observation in env.observation_space
        observation, _, _, truncated, info = env.step(controller.act(observation, info))


def test_replay_after_reset_reproduces_the_history_exactly() -> None:
    env = make_env([50.0, 80.0, 120.0, 180.0, 30.0, 70.0, 10.0])
    actions = [SCALE_UP, HOLD, SCALE_UP, HOLD, HOLD, HOLD, HOLD]

    def rollout() -> list[list[float]]:
        env.reset(seed=0)
        return [env.step(action)[0].tolist() for action in actions]

    assert rollout() == rollout()


@pytest.mark.parametrize("controller_kind", ["random", "static", "threshold"])
def test_baseline_controllers_run_full_v1_benchmark_episodes(controller_kind: str) -> None:
    config = SimulatorConfig()
    trace = build_workload(load_benchmark_manifest().get("syn-train-spike"))
    env = AutoscalingEnv(config, trace)
    controller: Any = {
        "random": RandomController(seed=0),
        "static": StaticController(3, config.replicas),
        "threshold": ThresholdController(
            low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=10
        ),
    }[controller_kind]

    infos = run_episode(env, controller, seed=0)

    assert len(infos) == 120
    assert env.observation_space.shape == (12,)


# --- model compatibility ------------------------------------------------------------------


def test_history_length_is_part_of_model_compatibility() -> None:
    v1 = EnvironmentCompatibility.from_config(SimulatorConfig())
    single = EnvironmentCompatibility.from_config(
        SimulatorConfig(observation=ObservationConfig(traffic_history_ticks=1))
    )

    assert v1.traffic_history_ticks == 4
    assert v1.mismatches(single)["traffic_history_ticks"] == (4, 1)
    with pytest.raises(ValueError, match="traffic_history_ticks: 4 != 1"):
        v1.require_compatible(single)
    v1.require_compatible(EnvironmentCompatibility.from_config(SimulatorConfig()))


def test_history_mismatch_is_rejected_even_when_shapes_coincide() -> None:
    # 4 history + 2 buckets (60 s delay) and 5 history + 1 bucket (30 s delay) are both 12.
    four = EnvironmentCompatibility.from_config(SimulatorConfig())
    five = EnvironmentCompatibility.from_config(
        SimulatorConfig(
            replicas=ReplicaConfig(startup_delay_seconds=30.0),
            observation=ObservationConfig(traffic_history_ticks=5),
        )
    )

    assert four.observation_shape == five.observation_shape == (12,)
    with pytest.raises(ValueError, match="traffic_history_ticks: 4 != 5"):
        four.require_compatible(five)
