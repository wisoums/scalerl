"""Tests for #65: seeded capacity jitter, delayed controller telemetry, and scenarios.

Synthetic train/validation workloads and hand-built traces only.
"""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlflow import MlflowClient
from pydantic import ValidationError
from stable_baselines3.common.monitor import Monitor

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import (
    PredictiveController,
    StaticController,
    ThresholdController,
    decision_info,
    run_episode,
)
from scalerl.environment import (
    CAPACITY_JITTER_MODEL,
    AutoscalingEnv,
    DynamicsConfig,
    ReplicaConfig,
    SimulatorConfig,
    TimingConfig,
    compute_tick_metrics,
)
from scalerl.environment.queue import QueueStepResult
from scalerl.evaluation.robustness import (
    CAPACITY_JITTER,
    COMBINED_ROBUSTNESS,
    DELAYED_TELEMETRY,
    JITTER_SENSITIVITY_LEVELS,
    NOMINAL,
    ROBUSTNESS_SCENARIO_VERSION,
    ROBUSTNESS_SCENARIOS,
    apply_scenario,
    evaluate_robustness,
    evaluate_robustness_tracked,
    get_scenario,
    robustness_run_spec,
    summarize_dynamics,
)
from scalerl.mlops import EnvironmentCompatibility, RunSpec
from scalerl.rl import ModelMetadata, load_sb3_controller, save_model_bundle
from scalerl.training.dqn import DQNHyperparameters, build_dqn
from scalerl.training.ppo import PPOHyperparameters, build_ppo
from scalerl.workloads import WorkloadTrace

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
INTERVAL = 10.0
CAPACITY = 10.0  # RPS per replica
MAX_REPLICAS = 4
MAX_RATE = MAX_REPLICAS * CAPACITY  # demand pressure = rate / (rate + 40)


def config(jitter: float = 0.0, delay: int = 0, seed: int = 0, ticks: int = 8) -> SimulatorConfig:
    return SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL, episode_duration_seconds=INTERVAL * ticks
        ),
        replicas=ReplicaConfig(
            min_replicas=1,
            max_replicas=MAX_REPLICAS,
            initial_replicas=1,
            startup_delay_seconds=INTERVAL,
            service_capacity_rps=CAPACITY,
        ),
        dynamics=DynamicsConfig(
            capacity_jitter_fraction=jitter, telemetry_delay_ticks=delay, dynamics_seed=seed
        ),
    )


def env_for(rates: list[float], **dynamics: Any) -> AutoscalingEnv:
    cfg = config(ticks=len(rates), **dynamics)
    return AutoscalingEnv(cfg, WorkloadTrace(rates, control_interval_seconds=INTERVAL))


def run_actions(env: AutoscalingEnv, actions: list[int]) -> list[dict[str, Any]]:
    env.reset(seed=0)
    return [env.step(action)[4] for action in actions]


def pressure(rate: float) -> float:
    return rate / (rate + MAX_RATE)


def benchmark_env(workload: str, cfg: SimulatorConfig | None = None) -> AutoscalingEnv:
    return AutoscalingEnv(
        cfg or SimulatorConfig(), build_workload(load_benchmark_manifest().get(workload))
    )


# --- config ---------------------------------------------------------------------------------


def test_dynamics_defaults_are_nominal() -> None:
    dynamics = SimulatorConfig().dynamics

    assert dynamics == DynamicsConfig()
    assert (dynamics.capacity_jitter_fraction, dynamics.dynamics_seed) == (0.0, 0)
    assert dynamics.telemetry_delay_ticks == 0
    assert CAPACITY_JITTER_MODEL == "uniform-multiplicative-v1"


@pytest.mark.parametrize("jitter", [0.0, 0.05, 0.10, 0.2, 0.999])
def test_valid_jitter_is_accepted(jitter: float) -> None:
    assert DynamicsConfig(capacity_jitter_fraction=jitter).capacity_jitter_fraction == jitter


@pytest.mark.parametrize(
    "fields",
    [
        {"capacity_jitter_fraction": 1.0},
        {"capacity_jitter_fraction": 1.5},
        {"capacity_jitter_fraction": -0.01},
        {"capacity_jitter_fraction": float("nan")},
        {"capacity_jitter_fraction": float("inf")},
        {"dynamics_seed": True},
        {"dynamics_seed": 1.0},
        {"dynamics_seed": "1"},
        {"dynamics_seed": -1},
        {"telemetry_delay_ticks": -1},
        {"telemetry_delay_ticks": False},
        {"telemetry_delay_ticks": 1.0},
        {"capacity_noise": 0.1},
    ],
)
def test_invalid_dynamics_are_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DynamicsConfig(**fields)


def test_dynamics_config_is_frozen() -> None:
    dynamics = DynamicsConfig()
    with pytest.raises(ValidationError):
        dynamics.telemetry_delay_ticks = 1  # type: ignore[misc]


def test_config_json_without_dynamics_loads_as_nominal() -> None:
    payload = json.loads(SimulatorConfig().model_dump_json())
    del payload["dynamics"]

    assert SimulatorConfig.model_validate_json(json.dumps(payload)) == SimulatorConfig()


# --- nominal regression ----------------------------------------------------------------------


def test_nominal_dynamics_never_touch_the_rng_and_multiply_by_exactly_one() -> None:
    env = env_for([30.0] * 8, seed=123)
    env.reset(seed=0)

    class Forbidden:
        def uniform(self, *args: Any) -> float:
            raise AssertionError("nominal dynamics must not draw")

    env._dynamics_rng = Forbidden()  # type: ignore[assignment]
    infos = [env.step(HOLD)[4] for _ in range(8)]

    assert all(info["capacity_multiplier"] == 1.0 for info in infos)
    assert all(info["effective_service_capacity_rps_per_replica"] == CAPACITY for info in infos)


def test_dynamics_seed_is_irrelevant_without_jitter() -> None:
    actions = [SCALE_UP, HOLD, SCALE_UP, HOLD, SCALE_DOWN, HOLD, HOLD, HOLD]
    rates = [10.0, 45.0, 80.0, 20.0, 60.0, 5.0, 90.0, 30.0]

    reference = run_actions(env_for(rates), actions)
    for seed in (1, 42, 2**31):
        assert run_actions(env_for(rates, seed=seed), actions) == reference


def test_nominal_decision_info_is_the_step_info_with_current_replicas() -> None:
    env = env_for([30.0, 60.0, 90.0])
    _, reset_info = env.reset(seed=0)
    assert decision_info(env, reset_info) == reset_info | env.replica_counts

    for action in (SCALE_UP, HOLD, HOLD):
        info = env.step(action)[4]
        assert decision_info(env, info) == info | env.replica_counts | {
            "telemetry_tick": info["tick"]
        }


def test_nominal_observation_matches_the_pre_65_formula() -> None:
    rates = [100.0, 200.0, 300.0]
    env = env_for(rates)
    env.reset(seed=0)
    for _ in rates:
        observation, _, _, _, info = env.step(HOLD)

    queued = info["queued_requests"]
    max_tick_capacity = MAX_RATE * INTERVAL
    latency = info["p95_latency_seconds"]
    cost_fraction = info["infrastructure_cost"] / (MAX_REPLICAS * 0.10 * INTERVAL / 3600)
    np.testing.assert_allclose(
        observation[:9],
        [
            pressure(300),
            pressure(200),
            pressure(100),
            0.0,
            info["utilization"],
            queued / (queued + max_tick_capacity),
            latency / (latency + 0.5),
            1 / MAX_REPLICAS,
            cost_fraction,
        ],
        rtol=1e-6,
    )


# --- capacity jitter -------------------------------------------------------------------------


def multipliers(env: AutoscalingEnv, actions: list[int] | None = None) -> list[float]:
    actions = actions or [HOLD] * env.episode_ticks
    return [info["capacity_multiplier"] for info in run_actions(env, actions)]


def test_same_seed_reproduces_the_multiplier_sequence_across_envs_and_resets() -> None:
    first = env_for([30.0] * 8, jitter=0.1, seed=7)
    second = env_for([30.0] * 8, jitter=0.1, seed=7)

    sequence = multipliers(first)

    assert multipliers(second) == sequence
    assert multipliers(first) == sequence  # reset restarts the realization
    assert len(set(sequence)) == len(sequence)


def test_different_seeds_give_different_sequences() -> None:
    sequences = {
        tuple(multipliers(env_for([30.0] * 8, jitter=0.1, seed=seed))) for seed in range(5)
    }

    assert len(sequences) == 5


@pytest.mark.parametrize("jitter", [0.05, 0.1, 0.2, 0.9])
def test_multipliers_stay_within_bounds(jitter: float) -> None:
    for seed in range(20):
        values = multipliers(env_for([30.0] * 8, jitter=jitter, seed=seed))
        assert all(1 - jitter <= value <= 1 + jitter for value in values)


def test_jitter_is_exogenous_to_actions() -> None:
    env = env_for([30.0] * 8, jitter=0.1, seed=3)

    holding = multipliers(env, [HOLD] * 8)
    scaling = multipliers(
        env, [SCALE_UP, SCALE_UP, SCALE_UP, SCALE_DOWN, HOLD, SCALE_UP, HOLD, HOLD]
    )

    assert holding == scaling


def test_jitter_changes_real_service_queue_and_latency_consistently() -> None:
    rates = [60.0] * 8  # 600 requests/tick arrive; 1 replica serves ~100
    nominal = run_actions(env_for(rates), [HOLD] * 8)
    jittered = run_actions(env_for(rates, jitter=0.1, seed=5), [HOLD] * 8)

    assert [i["processed_requests"] for i in jittered] != [i["processed_requests"] for i in nominal]
    assert [i["queued_requests"] for i in jittered] != [i["queued_requests"] for i in nominal]
    assert [i["p95_latency_seconds"] for i in jittered] != [
        i["p95_latency_seconds"] for i in nominal
    ]
    cfg = config(jitter=0.1, seed=5)
    backlog = 0.0
    for info in jittered:
        m = info["capacity_multiplier"]
        per_replica = CAPACITY * m
        assert info["effective_service_capacity_rps_per_replica"] == pytest.approx(per_replica)
        assert info["effective_total_service_capacity_rps"] == pytest.approx(
            info["active_replicas"] * per_replica
        )
        # The queue served exactly the realized capacity (saturated demand) ...
        assert info["processed_requests"] == pytest.approx(per_replica * INTERVAL)
        assert info["queued_requests"] == pytest.approx(backlog + 600 - per_replica * INTERVAL)
        backlog = info["queued_requests"]
        # ... and latency/utilization come from the same realization.
        metrics = compute_tick_metrics(
            QueueStepResult(info["arrived_requests"], info["processed_requests"], backlog, 0.0),
            active_replicas=info["active_replicas"],
            pending_replicas=info["pending_replicas"],
            config=cfg,
            capacity_multiplier=m,
        )
        assert info["p95_latency_seconds"] == metrics.p95_latency_seconds
        assert info["utilization"] == metrics.utilization
        # Price does not jitter.
        assert info["infrastructure_cost"] == nominal[0]["infrastructure_cost"]


def test_jitter_uses_no_global_rng() -> None:
    np.random.seed(0)
    random.seed(0)
    np_state, py_state = np.random.get_state(), random.getstate()

    first = multipliers(env_for([30.0] * 8, jitter=0.1, seed=11))
    np.random.seed(999)
    random.seed(999)
    second = multipliers(env_for([30.0] * 8, jitter=0.1, seed=11))

    assert first == second
    np.random.seed(0)
    random.seed(0)
    assert np.random.get_state()[1].tolist() == np_state[1].tolist()
    assert random.getstate() == py_state


def test_controllers_get_the_same_realization_regardless_of_order() -> None:
    workload = "syn-train-spike"
    cfg = apply_scenario(SimulatorConfig(), CAPACITY_JITTER, dynamics_seed=4)

    def threshold() -> ThresholdController:
        return ThresholdController(
            low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=10
        )

    def static() -> StaticController:
        return StaticController(4, cfg.replicas)

    def sequence(controller: Any) -> list[float]:
        return [
            i["capacity_multiplier"]
            for i in run_episode(benchmark_env(workload, cfg), controller, seed=0)
        ]

    order_ab = (sequence(threshold()), sequence(static()))
    order_ba = (sequence(static()), sequence(threshold()))

    assert order_ab[0] == order_ab[1] == order_ba[0] == order_ba[1]


# --- telemetry delay ------------------------------------------------------------------------


def test_exact_delayed_history_example() -> None:
    history = {}
    for delay in (0, 1):
        env = env_for([100.0, 200.0, 300.0, 400.0], delay=delay)
        env.reset(seed=0)
        for _ in range(3):
            observation = env.step(HOLD)[0]
        history[delay] = observation[:4]

    np.testing.assert_allclose(
        history[0], [pressure(300), pressure(200), pressure(100), 0.0], rtol=1e-6
    )
    np.testing.assert_allclose(history[1], [pressure(200), pressure(100), 0.0, 0.0], rtol=1e-6)


@pytest.mark.parametrize("delay", [1, 2, 3])
def test_observation_telemetry_comes_from_the_delayed_tick(delay: int) -> None:
    rates = [5.0, 25.0, 55.0, 15.0, 70.0, 35.0, 45.0, 10.0]
    delayed = env_for(rates, delay=delay)
    fresh = env_for(rates)
    delayed.reset(seed=0)
    fresh.reset(seed=0)
    fresh_observations = []
    actions = [SCALE_UP, HOLD, SCALE_UP, HOLD, HOLD, SCALE_DOWN, HOLD, HOLD]
    for tick, action in enumerate(actions):
        observation = delayed.step(action)[0]
        fresh_observations.append(fresh.step(action)[0])
        if tick >= delay:
            # Measurements (history, utilization, queue, latency, cost) lag by `delay` ticks.
            np.testing.assert_array_equal(observation[:7], fresh_observations[tick - delay][:7])
            np.testing.assert_array_equal(observation[8], fresh_observations[tick - delay][8])
        else:
            assert not observation[:7].any() and observation[8] == 0  # zero: no sample yet
        # Control-plane features stay current: active, progress, pending readiness.
        np.testing.assert_array_equal(observation[7], fresh_observations[tick][7])
        np.testing.assert_array_equal(observation[9:], fresh_observations[tick][9:])


def test_physical_info_stays_true_while_controllers_see_the_previous_tick() -> None:
    # Tick 0: 5 RPS on 1 replica (half load, no queue). Tick 1: 30 RPS on 2 replicas
    # (saturated, queue builds).
    rates = [5.0, 30.0, 20.0, 70.0]
    env = env_for(rates, delay=1)
    env.reset(seed=0)
    physical = [env.step(action)[4] for action in (SCALE_UP, HOLD)]
    now, before = physical[-1], physical[-2]
    view = decision_info(env, now)

    # Raw step info: tick 1 as it physically happened.
    assert (now["tick"], now["request_rate"]) == (1, 30.0)
    assert (before["utilization"], before["queued_requests"]) == (0.5, 0.0)
    assert (now["utilization"], now["queued_requests"]) == (1.0, 100.0)
    assert now["queued_requests"] != before["queued_requests"]
    assert now["utilization"] != before["utilization"]
    # Controller view: tick 0's measurements, current control-plane facts.
    assert view["telemetry_tick"] == 0
    for key in ("request_rate", "queued_requests", "utilization", "p95_latency_seconds"):
        assert view[key] == before[key]
    assert (view["tick"], view["applied_replica_change"]) == (1, now["applied_replica_change"])
    assert (view["active_replicas"], view["pending_replicas"]) == (
        env.replica_counts["active_replicas"],
        env.replica_counts["pending_replicas"],
    )
    observation = env._observation()
    assert observation[0] == pytest.approx(pressure(5.0))  # tick 0, not tick 1


def test_missing_delayed_telemetry_is_no_sample_not_zero() -> None:
    env = env_for([30.0] * 4, delay=1)
    env.reset(seed=0)
    info = env.step(SCALE_UP)[4]

    view = decision_info(env, info)

    for key in ("request_rate", "utilization", "queued_requests", "p95_latency_seconds"):
        assert key not in view
    assert "telemetry_tick" not in view
    assert view["tick"] == 0 and view["applied_replica_change"] == 1
    assert view["pending_replicas"] == env.replica_counts["pending_replicas"]


def test_threshold_sees_delayed_utilization_and_never_scales_down_on_a_missing_sample() -> None:
    env = env_for([5.0, 38.0, 38.0, 38.0], delay=1)
    controller = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=4
    )
    _, info = env.reset(seed=0)
    controller.reset()

    physical = []
    for _ in range(3):
        action = controller.act(np.zeros(1, dtype=np.float32), decision_info(env, info))
        decision = controller.last_decision
        assert decision is not None
        if not physical:
            assert decision.reason == "no_sample" and action == HOLD
        elif len(physical) == 1:
            assert decision.utilization is None and action == HOLD  # tick 0 not yet visible
        else:
            assert decision.utilization == physical[-2]["utilization"]  # one tick stale
        info = env.step(action)[4]
        physical.append(info)


def test_threshold_cooldown_is_not_delayed() -> None:
    env = env_for([80.0] * 8, delay=2)
    controller = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=4, cooldown_ticks=3
    )
    _, info = env.reset(seed=0)
    controller.reset()
    reasons = []
    for _ in range(6):
        action = controller.act(np.zeros(1, dtype=np.float32), decision_info(env, info))
        assert controller.last_decision is not None
        reasons.append(controller.last_decision.reason)
        info = env.step(action)[4]

    first_scale = reasons.index("above_high")
    # The very next decision after a real change is in cooldown: no telemetry lag.
    assert reasons[first_scale + 1] == "cooldown"


def test_predictive_sees_delayed_rate_and_queue() -> None:
    rates = [20.0, 60.0, 90.0, 30.0, 50.0, 70.0]
    env = env_for(rates, delay=1)
    controller = PredictiveController.from_config(env.config)
    _, info = env.reset(seed=0)
    controller.reset()
    physical: list[dict[str, Any]] = []
    for _ in rates:
        action = controller.act(np.zeros(1, dtype=np.float32), decision_info(env, info))
        decision = controller.last_decision
        assert decision is not None
        if len(physical) >= 2:
            assert decision.latest_request_rate == physical[-2]["request_rate"]
            assert decision.queued_requests == physical[-2]["queued_requests"]
        else:
            assert decision.reason == "no_sample"  # no fabricated measurement
        info = env.step(action)[4]
        physical.append(info)
    # Its forecast history holds only the stale samples it received: the last
    # decision (before tick 5) could see tick 3 at most.
    assert list(controller.request_history) == [info["request_rate"] for info in physical[:-2]]
    assert list(controller.request_history) == [20.0, 60.0, 90.0, 30.0]


def test_learned_policies_receive_the_delayed_observation_through_the_runner() -> None:
    rates = [100.0, 200.0, 300.0, 50.0]
    seen: list[np.ndarray] = []

    class Recorder:
        def reset(self, seed: int | None = None) -> None:
            pass

        def act(self, observation: Any, info: Any) -> int:
            seen.append(observation.copy())
            return HOLD

    infos = run_episode(env_for(rates, delay=1), Recorder(), seed=0)

    assert [info["request_rate"] for info in infos] == rates  # physical infos returned
    assert seen[3][0] == pytest.approx(pressure(200.0))  # after tick 2: tick 1 is newest visible
    assert seen[3][1] == pytest.approx(pressure(100.0))


def test_static_actions_give_identical_physics_under_any_delay() -> None:
    # Static reads only current replica counts, so delay cannot change its actions.
    rates = [30.0, 90.0, 20.0, 70.0, 10.0, 60.0, 40.0, 80.0]
    results = [
        run_episode(env_for(rates, delay=delay), StaticController(3, config().replicas), seed=0)
        for delay in (0, 1, 3)
    ]

    assert results[0] == results[1] == results[2]


def test_reset_clears_telemetry_and_the_buffer_is_bounded() -> None:
    env = env_for([30.0] * 8, delay=2)
    env.reset(seed=0)
    for _ in range(8):
        env.step(HOLD)
        assert len(env._telemetry) <= env.telemetry_buffer_size == 2 + 4

    observation, info = env.reset(seed=0)

    assert len(env._telemetry) == 0
    assert not observation[:7].any()
    assert "request_rate" not in decision_info(env, info)


def test_no_future_measurement_is_ever_visible() -> None:
    env = env_for([10.0 * (i + 1) for i in range(8)], delay=1)
    env.reset(seed=0)
    for tick in range(8):
        env.step(HOLD)
        visible = env.visible_telemetry()
        assert all(snapshot.tick <= tick - 1 for snapshot in visible)
        assert [s.tick for s in visible] == list(range(tick - 1, max(tick - 1 - 4, -1), -1))


# --- compatibility --------------------------------------------------------------------------


def test_pre_65_compatibility_json_loads_as_nominal() -> None:
    payload = EnvironmentCompatibility.from_config(SimulatorConfig(), "v1").model_dump(mode="json")
    del payload["telemetry_delay_ticks"], payload["capacity_jitter_model"]

    old = EnvironmentCompatibility.model_validate_json(json.dumps(payload))

    assert old.telemetry_delay_ticks == 0 and old.capacity_jitter_model == CAPACITY_JITTER_MODEL
    old.require_compatible(EnvironmentCompatibility.from_config(SimulatorConfig(), "v1"))


def test_jitter_fraction_and_seed_are_evaluation_conditions_not_contract() -> None:
    nominal = EnvironmentCompatibility.from_config(SimulatorConfig(), "v1")
    for scenario in (CAPACITY_JITTER,):
        for seed in (0, 7, 99):
            nominal.require_compatible(
                EnvironmentCompatibility.from_config(
                    apply_scenario(SimulatorConfig(), scenario, dynamics_seed=seed), "v1"
                )
            )


def test_delay_and_jitter_model_changes_are_detected() -> None:
    nominal = EnvironmentCompatibility.from_config(SimulatorConfig(), "v1")
    delayed = EnvironmentCompatibility.from_config(
        apply_scenario(SimulatorConfig(), DELAYED_TELEMETRY), "v1"
    )
    other_model = nominal.model_copy(update={"capacity_jitter_model": "gaussian-v9"})

    with pytest.raises(ValueError, match="telemetry_delay_ticks: 0 != 1"):
        nominal.require_compatible(delayed)
    with pytest.raises(ValueError, match="capacity_jitter_model"):
        nominal.require_compatible(other_model)
    with pytest.raises(ValueError, match="capacity_jitter_model"):
        nominal.require_compatible_for_robustness(other_model)
    assert nominal.require_compatible_for_robustness(delayed) == ("telemetry_delay_ticks",)
    assert nominal.require_compatible_for_robustness(nominal) == ()


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    env = benchmark_env("syn-train-spike")
    compatibility = EnvironmentCompatibility.from_env(env, "v1")
    models = {
        "dqn": (
            build_dqn(
                Monitor(env),
                DQNHyperparameters(
                    buffer_size=200, learning_starts=16, batch_size=16, net_arch=(8,)
                ),
                seed=0,
            ),
            DQNHyperparameters(),
        ),
        "ppo": (
            build_ppo(
                Monitor(env),
                PPOHyperparameters(n_steps=32, batch_size=16, n_epochs=1, net_arch=(8,)),
                seed=0,
            ),
            PPOHyperparameters(),
        ),
    }
    paths = {}
    for name, (model, hp) in models.items():
        model.learn(total_timesteps=32)
        metadata = ModelMetadata(
            algorithm=name,  # type: ignore[arg-type]
            config_version=f"{name}-v1",
            benchmark_version="v1",
            training_workload_id="syn-train-spike",
            training_workload_split="train",
            seed=0,
            total_timesteps=32,
            hyperparameters=hp.as_params(),
            scalerl_version="test",
        )
        directory = tmp_path_factory.mktemp(name)
        save_model_bundle(model, directory, metadata=metadata, compatibility=compatibility)
        # Rewrite the contract as a pre-#65 bundle would have saved it.
        payload = json.loads((directory / "compatibility.json").read_text())
        del payload["telemetry_delay_ticks"], payload["capacity_jitter_model"]
        (directory / "compatibility.json").write_text(json.dumps(payload))
        paths[name] = directory
    return paths


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_pre_65_bundles_load_nominally_and_under_jitter(
    bundles: dict[str, Path], algorithm: str
) -> None:
    for scenario in (NOMINAL, CAPACITY_JITTER):
        env = benchmark_env(
            "syn-val-bursty", apply_scenario(SimulatorConfig(), scenario, dynamics_seed=5)
        )
        controller = load_sb3_controller(bundles[algorithm], env)
        assert controller.perturbed_compatibility == ()
        result = evaluate_robustness(controller, env.trace, scenario=scenario, dynamics_seed=5)
        assert result.metrics.sla_violation_rate >= 0


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_delay_needs_the_explicit_robustness_path(bundles: dict[str, Path], algorithm: str) -> None:
    delayed = benchmark_env(
        "syn-val-bursty", apply_scenario(SimulatorConfig(), COMBINED_ROBUSTNESS, dynamics_seed=1)
    )

    with pytest.raises(ValueError, match="telemetry_delay_ticks"):
        load_sb3_controller(bundles[algorithm], delayed)
    controller = load_sb3_controller(bundles[algorithm], delayed, robustness_evaluation=True)
    assert controller.perturbed_compatibility == ("telemetry_delay_ticks",)

    # The robustness path still rejects unrelated mismatches.
    other = SimulatorConfig(replicas=ReplicaConfig(max_replicas=12))
    wrong = benchmark_env("syn-val-bursty", apply_scenario(other, DELAYED_TELEMETRY))
    with pytest.raises(ValueError, match="max_replicas"):
        load_sb3_controller(bundles[algorithm], wrong, robustness_evaluation=True)


# --- scenarios and tracking -----------------------------------------------------------------


def test_robustness_v1_scenarios_are_frozen() -> None:
    assert ROBUSTNESS_SCENARIO_VERSION == "robustness-v1"
    assert {
        name: (s.capacity_jitter_fraction, s.telemetry_delay_ticks, s.version)
        for name, s in ROBUSTNESS_SCENARIOS.items()
    } == {
        "nominal": (0.0, 0, "robustness-v1"),
        "capacity-jitter": (0.10, 0, "robustness-v1"),
        "delayed-telemetry": (0.0, 1, "robustness-v1"),
        "combined-robustness": (0.10, 1, "robustness-v1"),
    }
    assert JITTER_SENSITIVITY_LEVELS == (0.0, 0.05, 0.10, 0.20)
    assert get_scenario("combined-robustness") is COMBINED_ROBUSTNESS
    with pytest.raises(ValueError, match="unknown robustness scenario"):
        get_scenario("chaos")


def test_apply_scenario_copies_without_mutating() -> None:
    base = SimulatorConfig(replicas=ReplicaConfig(max_replicas=6))

    combined = apply_scenario(base, COMBINED_ROBUSTNESS, dynamics_seed=9)

    assert base.dynamics == DynamicsConfig()
    assert combined.replicas == base.replicas
    assert combined.dynamics == DynamicsConfig(
        capacity_jitter_fraction=0.10, telemetry_delay_ticks=1, dynamics_seed=9
    )
    assert apply_scenario(SimulatorConfig(), NOMINAL) == SimulatorConfig()


def test_dynamics_summary_reports_realized_multipliers() -> None:
    infos = run_actions(env_for([30.0] * 8, jitter=0.1, seed=2), [HOLD] * 8)

    summary = summarize_dynamics(infos)
    values = [info["capacity_multiplier"] for info in infos]

    assert summary.min_capacity_multiplier == min(values)
    assert summary.max_capacity_multiplier == max(values)
    assert 0.9 <= summary.mean_capacity_multiplier <= 1.1
    assert (
        summarize_dynamics(run_actions(env_for([30.0] * 8), [HOLD] * 8)).mean_capacity_multiplier
        == 1.0
    )


def test_run_spec_requires_robustness_name_and_version_together() -> None:
    entry = load_benchmark_manifest().get("syn-val-bursty")
    spec = robustness_run_spec(
        controller="threshold", entry=entry, scenario=DELAYED_TELEMETRY, dynamics_seed=0
    )
    assert spec.simulator_config_source == "predeclared"  # not the default simulator
    assert (
        robustness_run_spec(
            controller="threshold", entry=entry, scenario=NOMINAL, dynamics_seed=0
        ).simulator_config_source
        == "default"
    )
    with pytest.raises(ValidationError, match="set together"):
        RunSpec(**{**dict(spec), "robustness_version": None})
    with pytest.raises(ValidationError, match="default"):
        RunSpec(**{**dict(spec), "simulator_config_source": "default"})


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_tracked_robustness_evaluation_logs_scenario_lineage(
    tracking_uri: str, tmp_path: Path
) -> None:
    entry = load_benchmark_manifest().get("syn-val-bursty")
    controller = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=10
    )

    result, run_id = evaluate_robustness_tracked(
        controller,
        build_workload(entry),
        controller_name="threshold",
        entry=entry,
        scenario=COMBINED_ROBUSTNESS,
        dynamics_seed=3,
        hyperparameters={"low_threshold": 0.3, "high_threshold": 0.8},
        tracking_uri=tracking_uri,
        raw_infos_path=tmp_path / "infos.json",
    )
    run = MlflowClient(tracking_uri).get_run(run_id)
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics

    assert run.info.status == "FINISHED"
    assert tags["scalerl.run_kind"] == "evaluate" and tags["scalerl.controller"] == "threshold"
    assert tags["scalerl.workload_split"] == "validation"
    assert (
        tags["scalerl.robustness_scenario"]
        == params["robustness_scenario"]
        == "combined-robustness"
    )
    assert tags["scalerl.robustness_version"] == params["robustness_version"] == "robustness-v1"
    assert tags["scalerl.dynamics_seed"] == params["sim.dynamics.dynamics_seed"] == "3"
    assert params["sim.dynamics.capacity_jitter_fraction"] == "0.1"
    assert params["sim.dynamics.telemetry_delay_ticks"] == "1"
    assert (
        params["compat.capacity_jitter_model"]
        == tags["scalerl.capacity_jitter_model"]
        == CAPACITY_JITTER_MODEL
    )
    assert params["compat.telemetry_delay_ticks"] == "1"
    assert params["simulator_config_source"] == "predeclared"
    for name in (
        "sla_violation_rate",
        "normalized_cost",
        "queue_pressure",
        "churn_rate",
        "episode_reward",
    ):
        assert metrics[name] == pytest.approx(getattr(result.metrics, name))
    assert metrics["dynamics.mean_capacity_multiplier"] == pytest.approx(
        result.dynamics.mean_capacity_multiplier
    )
    assert {a.path for a in MlflowClient(tracking_uri).list_artifacts(run_id, "robustness")} == {
        "robustness/infos.json"
    }
    assert len(result.infos) == 120


def test_robustness_evaluation_is_reproducible_per_seed() -> None:
    trace = build_workload(load_benchmark_manifest().get("syn-train-bursty"))

    def run(seed: int) -> Any:
        controller = PredictiveController.from_config(SimulatorConfig())
        return evaluate_robustness(
            controller, trace, scenario=COMBINED_ROBUSTNESS, dynamics_seed=seed
        )

    first, again, other = run(1), run(1), run(2)

    assert first.infos == again.infos and first.metrics == again.metrics
    assert first.dynamics != other.dynamics
