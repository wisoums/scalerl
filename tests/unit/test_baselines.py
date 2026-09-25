"""Tests for the controller interface and the random/static baselines."""

import random
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from scalerl.controllers import (
    Controller,
    RandomController,
    StaticController,
    decision_info,
    run_episode,
)
from scalerl.environment import (
    AutoscalingEnv,
    ReplicaConfig,
    SimulatorConfig,
    TimingConfig,
)
from scalerl.workloads import WorkloadTrace, bursty_workload, steady_workload

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
INTERVAL = 30.0
TICKS = 40
OBSERVATION = np.zeros(9, dtype=np.float32)


def make_config(**replicas: Any) -> SimulatorConfig:
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 6,
        "initial_replicas": 2,
        "startup_delay_seconds": 60.0,
    }
    return SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * TICKS
        ),
        replicas=ReplicaConfig(**{**params, **replicas}),
    )


def make_env(trace: WorkloadTrace | None = None, **replicas: Any) -> AutoscalingEnv:
    config = make_config(**replicas)
    trace = trace or steady_workload(
        duration_seconds=INTERVAL * TICKS, control_interval_seconds=INTERVAL, rate=40.0
    )
    return AutoscalingEnv(config, trace)


def random_actions(controller: RandomController, count: int = 200) -> list[int]:
    return [controller.act(OBSERVATION, {}) for _ in range(count)]


def requested_actions(infos: list[dict[str, Any]]) -> list[int]:
    return [info["requested_action"] for info in infos]


# --- common interface -------------------------------------------------------


def make_controllers() -> list[Controller]:
    return [RandomController(seed=0), StaticController(4, make_config().replicas)]


@pytest.mark.parametrize("controller", make_controllers(), ids=["random", "static"])
def test_controllers_satisfy_the_common_interface(controller: Controller) -> None:
    assert isinstance(controller, Controller)


@pytest.mark.parametrize("controller", make_controllers(), ids=["random", "static"])
def test_controllers_drive_a_complete_episode_through_the_shared_runner(
    controller: Controller,
) -> None:
    env = make_env()

    infos = run_episode(env, controller, seed=0)

    assert len(infos) == env.episode_ticks
    assert all(env.action_space.contains(action) for action in requested_actions(infos))
    assert [info["tick"] for info in infos] == list(range(env.episode_ticks))
    assert {"p95_latency_seconds", "infrastructure_cost", "reward"} <= infos[-1].keys()


class ActiveTargetController:
    """Scales up while *active* replicas are below target, ignoring pending ones."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.seen: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None) -> None:
        self.seen = []

    def act(self, observation: np.ndarray, info: Mapping[str, Any]) -> int:
        self.seen.append(dict(info))
        return SCALE_UP if info["active_replicas"] < self.target else HOLD


def test_controllers_see_replica_counts_at_decision_time() -> None:
    env = make_env(initial_replicas=2, startup_delay_seconds=30)  # activates after one tick
    controller = ActiveTargetController(target=3)

    infos = run_episode(env, controller)

    # The replica requested on tick 0 activates at the end of that tick, so the
    # next decision already sees it and does not request a redundant scale-up.
    assert requested_actions(infos)[:3] == [SCALE_UP, HOLD, HOLD]
    assert (controller.seen[1]["active_replicas"], controller.seen[1]["pending_replicas"]) == (3, 0)
    # Returned infos stay raw: tick 0 was served with the new replica still pending.
    assert (infos[0]["active_replicas"], infos[0]["pending_replicas"]) == (2, 1)
    # Tick metrics are passed through unchanged.
    assert controller.seen[1]["processed_requests"] == infos[0]["processed_requests"]


def test_decision_info_matches_the_observation_every_tick() -> None:
    env = make_env(startup_delay_seconds=90)
    max_replicas = env.config.replicas.max_replicas
    observation, info = env.reset(seed=0)
    controller = RandomController(seed=3)

    for _ in range(TICKS):
        current = decision_info(env, info)
        names = env.observation_features
        active = observation[names.index("active_replicas_fraction")]
        pending = observation[names.index("episode_progress") + 1 :]
        assert current["active_replicas"] == round(active * max_replicas)
        assert current["pending_replicas"] == round(pending.sum() * max_replicas)
        observation, _, _, _, info = env.step(controller.act(observation, current))


# --- random -----------------------------------------------------------------


def test_random_controller_samples_all_actions_uniformly() -> None:
    counts = Counter(random_actions(RandomController(seed=1), count=3000))

    assert set(counts) == {SCALE_DOWN, HOLD, SCALE_UP}
    assert all(900 < count < 1100 for count in counts.values())


def test_random_same_seed_reproduces_the_action_sequence() -> None:
    assert random_actions(RandomController(seed=7)) == random_actions(RandomController(seed=7))


def test_random_reset_with_seed_restarts_the_sequence() -> None:
    controller = RandomController(seed=7)
    first = random_actions(controller)

    controller.reset(seed=7)

    assert random_actions(controller) == first


def test_random_reset_without_seed_continues_the_sequence() -> None:
    continued = RandomController(seed=7)
    random_actions(continued, count=50)
    continued.reset()

    expected = random_actions(RandomController(seed=7), count=100)[50:]

    assert random_actions(continued, count=50) == expected


def test_random_different_seeds_produce_different_sequences() -> None:
    assert random_actions(RandomController(seed=1)) != random_actions(RandomController(seed=2))


def test_random_controller_does_not_touch_global_random_state() -> None:
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.random())

    random.seed(123)
    np.random.seed(123)
    controller = RandomController(seed=5)
    random_actions(controller)
    controller.reset(seed=6)
    random_actions(controller)

    assert (random.random(), np.random.random()) == expected


def test_random_episodes_are_reproducible_through_the_runner() -> None:
    env = make_env()

    first = run_episode(env, RandomController(), seed=11)
    second = run_episode(env, RandomController(), seed=11)
    other = run_episode(env, RandomController(), seed=12)

    assert first == second
    assert requested_actions(first) != requested_actions(other)


@pytest.mark.parametrize("seed", [1.5, "1", True])
def test_random_controller_rejects_non_integer_seeds(seed: object) -> None:
    controller = RandomController()

    with pytest.raises(TypeError, match="seed must be an integer or None"):
        RandomController(seed=seed)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        controller.reset(seed=seed)  # type: ignore[arg-type]


# --- static -----------------------------------------------------------------


def test_static_target_equal_to_initial_capacity_holds_immediately() -> None:
    env = make_env(initial_replicas=3)

    infos = run_episode(env, StaticController(3, env.config.replicas))

    assert requested_actions(infos) == [HOLD] * TICKS
    assert all(info["applied_replica_change"] == 0 for info in infos)


def test_static_scales_up_one_replica_per_decision_until_target() -> None:
    env = make_env(initial_replicas=1, startup_delay_seconds=0)

    infos = run_episode(env, StaticController(4, env.config.replicas))

    assert requested_actions(infos) == [SCALE_UP] * 3 + [HOLD] * (TICKS - 3)
    assert [info["active_replicas"] for info in infos[:4]] == [2, 3, 4, 4]


def test_static_scales_down_until_target() -> None:
    env = make_env(initial_replicas=6)

    infos = run_episode(env, StaticController(2, env.config.replicas))

    assert requested_actions(infos) == [SCALE_DOWN] * 4 + [HOLD] * (TICKS - 4)
    assert infos[-1]["active_replicas"] == 2


def test_static_counts_pending_replicas_toward_target() -> None:
    env = make_env(initial_replicas=1, startup_delay_seconds=150)  # 5 ticks to activate

    infos = run_episode(env, StaticController(4, env.config.replicas))

    # Exactly three scale-ups even though no new replica is active for 5 ticks.
    assert requested_actions(infos) == [SCALE_UP] * 3 + [HOLD] * (TICKS - 3)
    assert [info["active_replicas"] for info in infos[:8]] == [1, 1, 1, 1, 1, 2, 3, 4]
    assert (infos[-1]["active_replicas"], infos[-1]["pending_replicas"]) == (4, 0)


def test_static_decision_uses_active_plus_pending() -> None:
    controller = StaticController(4, make_config().replicas)

    assert controller.act(OBSERVATION, {"active_replicas": 1, "pending_replicas": 3}) == HOLD
    assert controller.act(OBSERVATION, {"active_replicas": 1, "pending_replicas": 2}) == SCALE_UP
    assert controller.act(OBSERVATION, {"active_replicas": 4, "pending_replicas": 1}) == SCALE_DOWN


def test_static_holds_at_target_regardless_of_traffic() -> None:
    trace = bursty_workload(
        duration_seconds=INTERVAL * TICKS,
        control_interval_seconds=INTERVAL,
        baseline=5,
        burst_rate=5000,
        burst_probability=0.3,
        burst_duration_seconds=90,
        noise_std=20,
        seed=4,
    )
    env = make_env(trace=trace, initial_replicas=3)

    infos = run_episode(env, StaticController(3, env.config.replicas))

    assert any(info["sla_violated"] for info in infos)
    assert len({info["queued_requests"] for info in infos}) > 1
    assert requested_actions(infos) == [HOLD] * TICKS


@pytest.mark.parametrize("target", [0, 7, -1])
def test_static_rejects_targets_outside_replica_bounds(target: int) -> None:
    replicas = make_config().replicas

    with pytest.raises(ValueError, match=r"outside replica bounds \[1, 6\]"):
        StaticController(target, replicas)


@pytest.mark.parametrize("target", [2.0, "2", True])
def test_static_rejects_non_integer_targets(target: object) -> None:
    replicas = make_config().replicas

    with pytest.raises(TypeError, match="target_replicas must be an integer"):
        StaticController(target, replicas)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [1, 6])
def test_static_accepts_targets_at_the_bounds(target: int) -> None:
    env = make_env(initial_replicas=3, startup_delay_seconds=0)

    infos = run_episode(env, StaticController(target, env.config.replicas))

    assert infos[-1]["active_replicas"] == target
    assert all(
        info["applied_replica_change"] != 0 for info in infos if info["requested_action"] != HOLD
    )
