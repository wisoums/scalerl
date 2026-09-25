"""Validation suite: Gymnasium compliance, determinism, and simulator invariants.

Every scenario runs complete episodes under several deterministic action
policies, and every step is checked against the physical invariants the
simulator promises.
"""

import math
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from scalerl.environment import (
    AutoscalingEnv,
    ReplicaConfig,
    RewardWeights,
    SimulatorConfig,
    SlaConfig,
    TimingConfig,
)
from scalerl.workloads import (
    WorkloadTrace,
    bursty_workload,
    spike_workload,
    steady_workload,
)

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
TOLERANCE = 1e-9


# --- scenarios --------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    config: SimulatorConfig
    trace: WorkloadTrace

    def make_env(self) -> AutoscalingEnv:
        return AutoscalingEnv(self.config, self.trace)


def _config(
    *,
    interval: float = 30.0,
    ticks: int = 120,
    target: float = 0.5,
    **replicas: Any,
) -> SimulatorConfig:
    replica_params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 6,
        "initial_replicas": 2,
        "startup_delay_seconds": 60.0,
        "service_capacity_rps": 10.0,
        "cost_per_hour": 0.36,
    }
    return SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=interval, episode_duration_seconds=interval * ticks
        ),
        replicas=ReplicaConfig(**{**replica_params, **replicas}),
        sla=SlaConfig(latency_target_seconds=target),
    )


def _timing(config: SimulatorConfig) -> dict[str, float]:
    return {
        "duration_seconds": config.timing.episode_duration_seconds,
        "control_interval_seconds": config.timing.control_interval_seconds,
    }


def _scenario(
    name: str, config: SimulatorConfig, make_trace: Callable[..., WorkloadTrace], **params: Any
) -> Scenario:
    return Scenario(name, config, make_trace(**_timing(config), **params))


def _build_scenarios() -> list[Scenario]:
    default = _config()
    return [
        _scenario("steady", default, steady_workload, rate=25.0),
        _scenario(
            "bursty",
            default,
            bursty_workload,
            baseline=15.0,
            burst_rate=80.0,
            burst_probability=0.1,
            burst_duration_seconds=90.0,
            noise_std=5.0,
            seed=7,
        ),
        _scenario(
            "overload_recovery",
            default,
            spike_workload,
            baseline=10.0,
            spike_rate=100.0,  # well beyond the 60 rps max fleet, but recoverable
            spike_start_seconds=600.0,
            spike_duration_seconds=300.0,
        ),
        _scenario(
            "extreme_bursty",
            default,
            bursty_workload,
            baseline=5.0,
            burst_rate=1e6,
            burst_probability=0.3,
            burst_duration_seconds=60.0,
            noise_std=50.0,
            seed=3,
        ),
        _scenario("long_delay", _config(startup_delay_seconds=150.0), steady_workload, rate=40.0),
        _scenario("zero_delay", _config(startup_delay_seconds=0.0), steady_workload, rate=40.0),
        _scenario(
            "fixed_fleet",
            _config(min_replicas=3, max_replicas=3, initial_replicas=3),
            steady_workload,
            rate=40.0,
        ),
        _scenario("zero_cost", _config(cost_per_hour=0.0), steady_workload, rate=25.0),
        _scenario(
            "short_interval",
            _config(interval=5.0, ticks=240, startup_delay_seconds=12.0),
            steady_workload,
            rate=30.0,
        ),
        _scenario(
            "long_interval",
            _config(interval=120.0, ticks=48, startup_delay_seconds=60.0),
            steady_workload,
            rate=30.0,
        ),
    ]


SCENARIOS = _build_scenarios()
SCENARIO_IDS = [s.name for s in SCENARIOS]


# --- action policies ----------------------------------------------------------


def _random_actions(seed: int) -> Callable[[int], int]:
    rng = random.Random(seed)
    return lambda _tick: rng.randrange(3)


POLICIES: dict[str, Callable[[], Callable[[int], int]]] = {
    "random": lambda: _random_actions(0),
    "always_up": lambda: lambda _tick: SCALE_UP,
    "always_down": lambda: lambda _tick: SCALE_DOWN,
    "thrash": lambda: lambda tick: SCALE_UP if tick % 2 == 0 else SCALE_DOWN,
    "surge_then_drain": lambda: lambda tick: SCALE_UP if tick % 20 < 10 else SCALE_DOWN,
}


# --- invariant checker ----------------------------------------------------------


@dataclass
class EpisodeChecker:
    """Checks every step of one episode against the simulator's invariants."""

    env: AutoscalingEnv
    desired: int = 0
    queue_depth: float = 0.0
    arrived_total: float = 0.0
    processed_total: float = 0.0
    steps: int = 0
    shape: tuple[int, ...] = field(default=())

    def reset(self, seed: int = 0) -> np.ndarray:
        observation, info = self.env.reset(seed=seed)
        replicas = self.env.config.replicas
        self.shape = observation.shape
        self.desired = replicas.initial_replicas
        self._check_observation(observation)
        assert info["active_replicas"] == replicas.initial_replicas
        assert (info["pending_replicas"], info["terminating_replicas"]) == (0, 0)
        return observation

    def step(self, action: int) -> tuple[bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        config = self.env.config
        replicas = config.replicas
        interval = config.timing.control_interval_seconds

        self._check_observation(observation)
        assert terminated is False
        assert truncated is (self.steps + 1 == self.env.episode_ticks)
        assert info["tick"] == self.steps
        assert info["time_seconds"] == pytest.approx((self.steps + 1) * interval)

        # Actions: bound-limited scaling is a no-op, otherwise exactly one replica.
        change = info["applied_replica_change"]
        assert info["requested_action"] == action
        if action == SCALE_UP:
            assert change == (1 if self.desired < replicas.max_replicas else 0)
        elif action == SCALE_DOWN:
            assert change == (-1 if self.desired > replicas.min_replicas else 0)
        else:
            assert change == 0

        # Replica lifecycle and bounds.
        active, pending = info["active_replicas"], info["pending_replicas"]
        terminating = info["terminating_replicas"]
        assert min(active, pending, terminating) >= 0
        self.desired += change
        assert active + pending == self.desired
        assert replicas.min_replicas <= self.desired <= replicas.max_replicas
        if replicas.startup_delay_seconds == 0:
            assert pending == 0

        # Request accounting under the unbounded queue.
        arrived, processed = info["arrived_requests"], info["processed_requests"]
        queued, dropped = info["queued_requests"], info["dropped_requests"]
        for value in (arrived, processed, queued):
            assert math.isfinite(value) and value >= 0
        assert dropped == 0.0
        assert arrived == pytest.approx(info["request_rate"] * interval)
        capacity = active * replicas.service_capacity_rps * interval
        assert processed <= capacity * (1 + TOLERANCE)
        assert processed + queued == pytest.approx(self.queue_depth + arrived, rel=TOLERANCE)
        self.queue_depth = queued
        self.arrived_total += arrived
        self.processed_total += processed

        # Metrics.
        latency, cost = info["p95_latency_seconds"], info["infrastructure_cost"]
        assert 0.0 <= info["utilization"] <= 1.0
        assert math.isfinite(latency) and latency >= 1 / replicas.service_capacity_rps - TOLERANCE
        assert info["sla_violated"] is (latency > config.sla.latency_target_seconds)
        assert math.isfinite(cost) and cost >= 0
        assert cost == pytest.approx(
            (active + pending) * replicas.cost_per_hour * interval / 3600, abs=1e-15
        )

        # Reward decomposition.
        weights = self.env.reward_weights.model_dump()
        components = info["reward_components"]
        assert set(components) == {f"{name}_penalty" for name in weights}
        for name, weight in weights.items():
            value = components[f"{name}_penalty"]
            assert math.isfinite(value) and 0.0 <= value <= weight
        assert math.isfinite(float(reward))
        assert reward == info["reward"] == pytest.approx(-sum(components.values()))
        assert components["churn_penalty"] == (weights["churn"] if change else 0.0)

        # Observation replica features agree with post-lifecycle replica state:
        # advancing moves pending to active but never changes the desired count.
        max_replicas = replicas.max_replicas
        names = self.env.observation_features
        active = observation[names.index("active_replicas_fraction")]
        pending = observation[names.index("episode_progress") + 1 :]
        observed_desired = (active + pending.sum()) * max_replicas
        assert observed_desired == pytest.approx(self.desired, abs=1e-5)
        progress = observation[names.index("episode_progress")]
        assert progress == pytest.approx((self.steps + 1) / self.env.episode_ticks)
        # The latest traffic-history slot is this tick's consumed demand.
        rate = info["request_rate"]
        max_rate = replicas.max_replicas * replicas.service_capacity_rps
        assert observation[0] == pytest.approx(rate / (rate + max_rate), rel=1e-6)

        self.steps += 1
        return truncated, info

    def finish(self) -> None:
        assert self.steps == self.env.episode_ticks
        assert self.arrived_total == pytest.approx(
            self.processed_total + self.queue_depth, rel=TOLERANCE
        )
        with pytest.raises(RuntimeError, match="call reset"):
            self.env.step(HOLD)

    def _check_observation(self, observation: np.ndarray) -> None:
        assert observation.shape == self.shape
        assert observation.dtype == np.float32
        assert np.all(np.isfinite(observation))
        assert observation in self.env.observation_space


def run_checked_episode(env: AutoscalingEnv, policy: Callable[[int], int], seed: int = 0) -> None:
    checker = EpisodeChecker(env)
    checker.reset(seed)
    truncated = False
    while not truncated:
        truncated, _ = checker.step(policy(checker.steps))
    checker.finish()


# --- Gymnasium compliance -----------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_gymnasium_env_checker_passes(scenario: Scenario) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(scenario.make_env(), skip_render_check=True)


# --- invariants -----------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES, ids=list(POLICIES))
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_invariants_hold_over_full_episodes(scenario: Scenario, policy: str) -> None:
    run_checked_episode(scenario.make_env(), POLICIES[policy]())


def test_invariants_hold_across_consecutive_episodes() -> None:
    env = SCENARIOS[1].make_env()

    for episode in range(3):
        run_checked_episode(env, _random_actions(episode), seed=episode)


def test_overload_builds_queue_then_recovers() -> None:
    scenario = next(s for s in SCENARIOS if s.name == "overload_recovery")
    env = scenario.make_env()
    checker = EpisodeChecker(env)
    checker.reset()

    history = []
    truncated = False
    while not truncated:
        # Scale up while the queue grows, otherwise hold.
        action = SCALE_UP if checker.queue_depth > 0 else HOLD
        truncated, info = checker.step(action)
        history.append(info)
    checker.finish()

    queued = [info["queued_requests"] for info in history]
    peak = int(np.argmax(queued))
    assert 20 <= peak < 40  # grows through the 10-tick spike starting at tick 20
    assert all(info["sla_violated"] for info in history[20:peak])
    assert queued[-1] == 0.0  # the backlog drains once demand falls back
    assert not history[-1]["sla_violated"]


# --- startup delay --------------------------------------------------------------


@pytest.mark.parametrize(
    ("delay", "interval", "ticks_to_active"),
    [(0, 30, 0), (1, 30, 1), (30, 30, 1), (45, 30, 2), (60, 30, 2), (150, 30, 5), (12, 5, 3)],
)
def test_scale_up_activates_after_configured_delay(
    delay: float, interval: float, ticks_to_active: int
) -> None:
    config = _config(interval=interval, ticks=12, startup_delay_seconds=delay, initial_replicas=1)
    env = AutoscalingEnv(config, steady_workload(**_timing(config), rate=5.0))
    env.reset(seed=0)

    serving = [env.step(SCALE_UP if tick == 0 else HOLD)[4]["active_replicas"] for tick in range(8)]

    # info reports replicas serving each tick; the new one serves from tick `ticks_to_active`.
    assert serving == [1] * ticks_to_active + [2] * (8 - ticks_to_active)


def test_repeated_scale_up_forms_a_startup_pipeline() -> None:
    config = _config(ticks=12, startup_delay_seconds=90.0, initial_replicas=1, max_replicas=10)
    env = AutoscalingEnv(config, steady_workload(**_timing(config), rate=5.0))
    env.reset(seed=0)

    infos = [env.step(SCALE_UP)[4] for _ in range(6)]

    # One replica requested per tick, each serving 3 ticks after its request.
    assert [i["active_replicas"] for i in infos] == [1, 1, 1, 2, 3, 4]
    assert [i["pending_replicas"] for i in infos] == [1, 2, 3, 3, 3, 3]


# --- determinism ----------------------------------------------------------------


def _trajectory(env: AutoscalingEnv, actions: list[int], seed: int) -> list[Any]:
    observation, info = env.reset(seed=seed)
    trajectory: list[Any] = [(observation.tolist(), info)]
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append((observation.tolist(), reward, terminated, truncated, info))
    return trajectory


def _episode_actions(env: AutoscalingEnv, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(3) for _ in range(env.episode_ticks)]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_independent_environments_produce_identical_trajectories(scenario: Scenario) -> None:
    first, second = scenario.make_env(), scenario.make_env()
    actions = _episode_actions(first, seed=123)

    assert _trajectory(first, actions, seed=5) == _trajectory(second, actions, seed=5)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_replay_after_reset_reproduces_the_trajectory(scenario: Scenario) -> None:
    env = scenario.make_env()
    actions = _episode_actions(env, seed=321)

    first = _trajectory(env, actions, seed=9)
    _trajectory(env, _episode_actions(env, seed=1), seed=0)  # an unrelated episode in between

    assert _trajectory(env, actions, seed=9) == first


def test_regenerated_seeded_trace_reproduces_the_trajectory() -> None:
    config = _config()

    def bursty(seed: int) -> WorkloadTrace:
        return bursty_workload(
            **_timing(config),
            baseline=15.0,
            burst_rate=80.0,
            burst_probability=0.1,
            burst_duration_seconds=90.0,
            seed=seed,
        )

    env = AutoscalingEnv(config, bursty(42))
    actions = _episode_actions(env, seed=7)

    same = _trajectory(AutoscalingEnv(config, bursty(42)), actions, seed=0)
    other = _trajectory(AutoscalingEnv(config, bursty(43)), actions, seed=0)

    assert _trajectory(env, actions, seed=0) == same
    assert _trajectory(env, actions, seed=0) != other


def test_reset_seed_does_not_change_the_simulation() -> None:
    env = SCENARIOS[1].make_env()
    actions = _episode_actions(env, seed=11)

    assert _trajectory(env, actions, seed=1) == _trajectory(env, actions, seed=2)
    assert _trajectory(env, actions, seed=None)[1:] == _trajectory(env, actions, seed=3)[1:]


def test_reset_after_completed_episode_restores_initial_state() -> None:
    env = SCENARIOS[2].make_env()
    initial_observation, initial_info = env.reset(seed=0)
    for action in _episode_actions(env, seed=4):
        env.step(action)

    observation, info = env.reset(seed=0)

    np.testing.assert_array_equal(observation, initial_observation)
    assert info == initial_info


# --- observation-space contract -------------------------------------------------


def expected_observation_size(delay: float, interval: float, history: int = 4) -> int:
    """The documented contract: h traffic ticks + 6 state features + one per startup tick."""
    return history + 6 + (math.ceil(delay / interval) if delay > 0 else 0)


@pytest.mark.parametrize(
    ("delay", "interval"),
    [(0, 30), (1, 30), (30, 30), (45, 30), (60, 30), (90, 30), (150, 30), (12, 5), (60, 120)],
)
def test_observation_size_follows_documented_contract(delay: float, interval: float) -> None:
    config = _config(interval=interval, ticks=10, startup_delay_seconds=delay)
    env = AutoscalingEnv(config, steady_workload(**_timing(config), rate=5.0))

    assert env.observation_space.shape == (expected_observation_size(delay, interval),)


def test_observation_shape_is_stable_for_a_fixed_config() -> None:
    config = _config(startup_delay_seconds=90.0)
    traces = [
        steady_workload(**_timing(config), rate=0.0),
        steady_workload(**_timing(config), rate=1e5),
        bursty_workload(
            **_timing(config),
            baseline=10,
            burst_rate=500,
            burst_probability=0.2,
            burst_duration_seconds=60,
            seed=2,
        ),
    ]
    expected = (expected_observation_size(90.0, 30.0),)

    for trace in traces:
        env = AutoscalingEnv(config, trace)
        assert env.observation_space.shape == expected
        for seed, policy in enumerate(POLICIES.values()):
            observation, _ = env.reset(seed=seed)
            assert observation.shape == expected
            actor = policy()
            truncated = False
            while not truncated:
                observation, _, _, truncated, _ = env.step(actor(0))
                assert observation.shape == expected


def test_different_timing_configs_yield_incompatible_observation_spaces() -> None:
    """A policy trained on one env only fits envs with matching observation/action spaces."""

    def env_for(delay: float, interval: float) -> AutoscalingEnv:
        config = _config(interval=interval, ticks=10, startup_delay_seconds=delay)
        return AutoscalingEnv(config, steady_workload(**_timing(config), rate=5.0))

    trained = env_for(60, 30)

    assert env_for(60, 30).observation_space == trained.observation_space
    assert env_for(30, 15).observation_space == trained.observation_space  # same ratio
    assert env_for(90, 30).observation_space != trained.observation_space
    assert env_for(0, 30).observation_space != trained.observation_space
    assert env_for(90, 30).action_space == trained.action_space


def test_zero_reward_weights_give_zero_reward_everywhere() -> None:
    zero = RewardWeights(latency=0, cost=0, sla=0, queue=0, churn=0)
    scenario = next(s for s in SCENARIOS if s.name == "extreme_bursty")
    env = AutoscalingEnv(scenario.config, scenario.trace, zero)
    env.reset(seed=0)

    rewards = [env.step(action)[1] for action in _episode_actions(env, seed=0)]

    assert set(rewards) == {0.0}
