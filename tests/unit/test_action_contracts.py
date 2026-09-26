"""Tests for the versioned action contracts (#79): delta-v1 and desired-replicas-v1.

Environment semantics, controllers under both contracts, reward/metric
invariance, compatibility, and tiny DQN/PPO smoke runs (no performance claims).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from pydantic import ValidationError
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import (
    PredictiveController,
    RandomController,
    StaticController,
    ThresholdController,
    run_episode,
)
from scalerl.controllers.predictive import CAPACITY_POLICY, FORECAST_METHOD
from scalerl.environment import (
    DELTA_V1,
    DESIRED_REPLICAS_V1,
    HOLD,
    SCALE_DOWN,
    SCALE_UP,
    ActionConfig,
    ActionContract,
    AutoscalingEnv,
    DynamicsConfig,
    ReplicaConfig,
    SimulatorConfig,
)
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import summarize_action_magnitude, summarize_episode
from scalerl.mlops import EnvironmentCompatibility
from scalerl.mlops.spec import ROBUSTNESS_PERTURBABLE_FIELDS
from scalerl.rl import ModelMetadata, load_sb3_controller, read_model_bundle, save_model_bundle
from scalerl.training.dqn import DQNHyperparameters, build_dqn
from scalerl.training.ppo import PPOHyperparameters, build_ppo
from scalerl.workloads import WorkloadTrace, steady_workload

DESIRED = ActionConfig(semantics=DESIRED_REPLICAS_V1)


def desired_config(**replicas: Any) -> SimulatorConfig:
    return SimulatorConfig(action=DESIRED, replicas=ReplicaConfig(**replicas))


def idle(config: SimulatorConfig, rate: float = 0.0) -> WorkloadTrace:
    return steady_workload(
        duration_seconds=config.timing.episode_duration_seconds,
        control_interval_seconds=config.timing.control_interval_seconds,
        rate=rate,
    )


def benchmark_trace(workload_id: str) -> WorkloadTrace:
    return build_workload(load_benchmark_manifest().get(workload_id))


def env_of(config: SimulatorConfig, rate: float = 0.0) -> AutoscalingEnv:
    env = AutoscalingEnv(config, idle(config, rate))
    env.reset(seed=0)
    return env


# --- A. delta-v1 historical behavior ---------------------------------------------------------


def test_delta_v1_is_the_default_discrete_three_contract() -> None:
    env = env_of(SimulatorConfig())
    assert SimulatorConfig().action.semantics == DELTA_V1 == env.action_semantics
    assert env.action_space == spaces.Discrete(3)
    assert (SCALE_DOWN, HOLD, SCALE_UP) == (0, 1, 2)
    contract = env.action_contract
    assert [contract.target_for(code, 5) - 5 for code in (0, 1, 2)] == [-1, 0, 1]


@pytest.mark.parametrize("code", [-1, 3, True, 1.0, np.float32(1.0)])
def test_delta_v1_rejects_effects_and_non_codes(code: Any) -> None:
    env = env_of(SimulatorConfig())
    with pytest.raises(ValueError, match="invalid action"):
        env.step(code)


def test_delta_v1_codes_step_one_replica() -> None:
    env = env_of(SimulatorConfig(replicas=ReplicaConfig(initial_replicas=5)))
    up = env.step(SCALE_UP)[4]
    hold = env.step(HOLD)[4]
    down = env.step(SCALE_DOWN)[4]
    assert [i["applied_replica_change"] for i in (up, hold, down)] == [1, 0, -1]
    assert [i["requested_action"] for i in (up, hold, down)] == [2, 1, 0]
    assert [i["requested_replica_target"] for i in (up, hold, down)] == [6, 6, 5]


def test_delta_v1_bound_clipping_is_a_no_op() -> None:
    env = env_of(SimulatorConfig())  # starts at min_replicas = 1
    info = env.step(SCALE_DOWN)[4]
    assert info["applied_replica_change"] == 0 and info["requested_replica_target"] == 1


def _fingerprint(infos: list[dict[str, Any]]) -> str:
    clean = [{k: v for k, v in i.items() if k != "requested_replica_target"} for i in infos]
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=float).encode()).hexdigest()[
        :16
    ]


# Recorded with the pre-#79 environment and controllers (commit c69c672) on the
# same workloads, dynamics, and seeds; the new requested_replica_target key is excluded.
PRE_79_FINGERPRINTS = {
    ("syn-val-bursty", 0, "threshold"): "3ae83b309977e774",
    ("syn-val-bursty", 0, "predictive"): "bceb8ac768748a16",
    ("syn-val-bursty", 0, "random"): "370ffda22055493b",
    ("syn-val-bursty", 0, "static"): "509e42202555bc61",
    ("syn-val-bursty", 1, "threshold"): "97a1ebccaff3ddfc",
    ("syn-val-bursty", 1, "predictive"): "baf034b802d038ed",
    ("syn-train-bursty", 0, "threshold"): "e92574e37cba0509",
    ("syn-train-bursty", 0, "predictive"): "5f5cceddad422297",
    ("syn-val-ramp-down", 1, "random"): "b6a9fd8ce866041e",
    ("syn-val-ramp-down", 0, "static"): "db34a30e2f5b4497",
}


def _pre_79_controllers(config: SimulatorConfig, contract: ActionContract | None) -> dict[str, Any]:
    return {
        "threshold": ThresholdController(
            low_threshold=0.2,
            high_threshold=0.6,
            min_replicas=1,
            max_replicas=10,
            cooldown_ticks=3,
            action_contract=contract,
        ),
        "predictive": PredictiveController.from_config(config, action_contract=contract),
        "random": RandomController(seed=0, action_contract=contract),
        "static": StaticController(5, config.replicas, action_contract=contract),
    }


@pytest.mark.parametrize("explicit_contract", [False, True])
@pytest.mark.parametrize(("workload_id", "delay", "name"), list(PRE_79_FINGERPRINTS))
def test_delta_v1_episodes_are_bit_identical_to_pre_79(
    workload_id: str, delay: int, name: str, explicit_contract: bool
) -> None:
    dynamics = (
        DynamicsConfig()
        if delay == 0
        else DynamicsConfig(capacity_jitter_fraction=0.1, dynamics_seed=3, telemetry_delay_ticks=1)
    )
    config = SimulatorConfig(dynamics=dynamics)
    contract = ActionContract.from_config(config) if explicit_contract else None
    controller = _pre_79_controllers(config, contract)[name]
    infos = run_episode(AutoscalingEnv(config, benchmark_trace(workload_id)), controller, seed=0)
    assert _fingerprint(infos) == PRE_79_FINGERPRINTS[(workload_id, delay, name)]


def test_default_config_serialization_is_unchanged() -> None:
    # Hashes of configs (Optuna study identities, #19 plan IDs) must not move.
    dumped = SimulatorConfig().model_dump(mode="json")
    assert "action" not in dumped
    assert SimulatorConfig.model_validate(dumped).action.semantics == DELTA_V1
    desired = SimulatorConfig(action=DESIRED)
    assert desired.model_dump(mode="json")["action"] == {"semantics": DESIRED_REPLICAS_V1}
    assert SimulatorConfig.model_validate_json(desired.model_dump_json()) == desired


def test_action_semantics_is_a_closed_version_set() -> None:
    with pytest.raises(ValidationError):
        ActionConfig(semantics="desired-replicas-v2")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SimulatorConfig.model_validate({"action": {"use_desired_actions": True}})


# --- B. desired-replicas-v1 encoding ------------------------------------------------------------


def test_desired_default_bounds_map_codes_zero_to_nine_onto_one_to_ten() -> None:
    env = env_of(desired_config())
    assert env.action_space == spaces.Discrete(10)
    assert [env.action_contract.target_for(code, 4) for code in range(10)] == list(range(1, 11))


@pytest.mark.parametrize(("low", "high"), [(2, 5), (1, 1), (3, 12)])
def test_desired_arbitrary_bounds(low: int, high: int) -> None:
    contract = ActionContract(DESIRED_REPLICAS_V1, low, high)
    assert contract.action_count == high - low + 1
    targets = [contract.target_for(code, low) for code in range(contract.action_count)]
    assert targets == list(range(low, high + 1))
    assert all(isinstance(target, int) for target in targets)
    config = desired_config(min_replicas=low, max_replicas=high, initial_replicas=low)
    assert AutoscalingEnv(config, idle(config)).action_space == spaces.Discrete(high - low + 1)


@pytest.mark.parametrize("code", [-1, 10, True, np.bool_(True), 2.0, 2.5, "3"])
def test_desired_rejects_invalid_codes(code: Any) -> None:
    env = env_of(desired_config())
    with pytest.raises(ValueError, match="invalid action"):
        env.step(code)


def test_numpy_integer_codes_are_accepted() -> None:
    env = env_of(desired_config())
    assert env.step(np.int64(4))[4]["requested_replica_target"] == 5


def test_code_for_target_round_trips_and_clips() -> None:
    contract = ActionContract(DESIRED_REPLICAS_V1, 2, 5)
    for target in range(2, 6):
        assert contract.target_for(contract.code_for_target(target, 3), 3) == target
    assert contract.code_for_target(99, 3) == 3 and contract.code_for_target(-4, 3) == 0
    with pytest.raises(TypeError):
        contract.code_for_target(2.5, 3)  # type: ignore[arg-type]


# --- C-F. multi-replica lifecycle ------------------------------------------------------------


def test_multi_replica_scale_up_goes_pending_and_waits_out_startup() -> None:
    env = env_of(desired_config(initial_replicas=2))
    info = env.step(7)[4]  # target 8
    assert info["requested_replica_target"] == 8 and info["applied_replica_change"] == 6
    assert (info["active_replicas"], info["pending_replicas"]) == (2, 6)
    assert env.replica_counts["active_replicas"] == 2  # 30 of 60 s elapsed
    assert env.observation_features[-2:] == ("pending_ready_in_1", "pending_ready_in_2")
    info = env.step(7)[4]
    assert info["applied_replica_change"] == 0 and info["active_replicas"] == 2
    assert env.replica_counts == {
        "active_replicas": 8,
        "pending_replicas": 0,
        "terminating_replicas": 0,
    }


def test_multi_replica_scale_down_cancels_newest_pending_then_active() -> None:
    env = env_of(desired_config(initial_replicas=4))
    env.step(8)  # target 9: +5 pending
    info = env.step(2)[4]  # target 3 from 9 committed: cancel 5 pending, terminate 1 active
    assert info["applied_replica_change"] == -6
    assert (info["active_replicas"], info["pending_replicas"]) == (3, 0)
    assert info["terminating_replicas"] == 1


def test_scale_down_cancels_pending_before_active() -> None:
    env = env_of(desired_config(initial_replicas=2))
    env.step(7)  # 2 active + 6 pending
    info = env.step(4)[4]  # target 5: cancel three newest pending, keep both active
    assert info["applied_replica_change"] == -3
    assert (info["active_replicas"], info["pending_replicas"], info["terminating_replicas"]) == (
        2,
        3,
        0,
    )


def test_min_bound_holds_under_desired_semantics() -> None:
    env = env_of(desired_config(min_replicas=2, max_replicas=6, initial_replicas=4))
    info = env.step(0)[4]  # code 0 = min_replicas = 2
    assert info["requested_replica_target"] == 2 and info["applied_replica_change"] == -2
    assert env.replica_counts["active_replicas"] == 2


def test_zero_startup_delay_activates_multi_replica_request_at_once() -> None:
    env = env_of(desired_config(initial_replicas=2, startup_delay_seconds=0.0))
    info = env.step(7)[4]
    assert info["applied_replica_change"] == 6
    assert (info["active_replicas"], info["pending_replicas"]) == (8, 0)


def test_target_equal_to_committed_is_a_no_op() -> None:
    env = env_of(desired_config(initial_replicas=3))
    env.step(5)  # target 6: 3 active + 3 pending
    info = env.step(5)[4]
    assert info["applied_replica_change"] == 0 and info["requested_replica_target"] == 6
    assert info["reward_components"]["churn_penalty"] == 0.0


# --- G. info and provenance ----------------------------------------------------------------------


def test_info_records_code_target_and_applied_change() -> None:
    env = env_of(desired_config(initial_replicas=2))
    info = env.step(np.int64(7))[4]
    assert info["requested_action"] == 7 and isinstance(info["requested_action"], int)
    assert info["requested_replica_target"] == 8 and info["applied_replica_change"] == 6
    assert env.action_semantics == DESIRED_REPLICAS_V1


def test_requested_target_is_current_control_plane_info_under_delay() -> None:
    config = desired_config(initial_replicas=2).model_copy(
        update={"dynamics": DynamicsConfig(telemetry_delay_ticks=2)}
    )
    env = AutoscalingEnv(config, idle(config, rate=40.0))
    env.reset(seed=0)
    info = env.step(7)[4]
    view = env.decision_info(info)
    assert view["requested_replica_target"] == 8 and view["applied_replica_change"] == 6
    assert "request_rate" not in view  # no delayed measurement exists yet: nothing leaks
    later = env.step(4)[4]
    view = env.decision_info(later)
    assert view["requested_replica_target"] == 5 and view["tick"] == 1
    assert "request_rate" not in view


def test_run_tags_record_action_semantics(tmp_path: Path) -> None:
    from scalerl.mlops.spec import RunSpec
    from scalerl.mlops.tracking import _tags

    spec = RunSpec(
        run_kind="evaluate",
        controller="threshold",
        workload_id="syn-val-bursty",
        workload_split="validation",
        simulator_config=SimulatorConfig(action=DESIRED),
        simulator_config_source="predeclared",
    )
    tags = _tags(spec, {"git_sha": "abc", "git_dirty": False, "scalerl_version": "t"})
    assert tags["scalerl.action_semantics"] == DESIRED_REPLICAS_V1
    with pytest.raises(ValidationError, match="default SimulatorConfig"):
        RunSpec(**{**spec.model_dump(), "simulator_config_source": "default"})


# --- controllers --------------------------------------------------------------------------------


def threshold(contract: ActionContract | None, cooldown: int = 0) -> ThresholdController:
    return ThresholdController(
        low_threshold=0.2,
        high_threshold=0.6,
        min_replicas=1,
        max_replicas=10,
        cooldown_ticks=cooldown,
        action_contract=contract,
    )


def decision_info(
    utilization: float, active: int, pending: int = 0, **extra: Any
) -> dict[str, Any]:
    return {
        "utilization": utilization,
        "active_replicas": active,
        "pending_replicas": pending,
        **extra,
    }


DELTA = ActionContract(DELTA_V1, 1, 10)
DESIRED_1_10 = ActionContract(DESIRED_REPLICAS_V1, 1, 10)


@pytest.mark.parametrize(
    ("utilization", "active", "pending", "delta_code", "target"),
    [
        (0.9, 3, 1, SCALE_UP, 5),  # above high: committed 4 -> 5
        (0.1, 3, 1, SCALE_DOWN, 3),  # below low: 4 -> 3
        (0.4, 3, 1, HOLD, 4),  # within band: hold committed
        (0.9, 10, 0, HOLD, 10),  # at max
        (0.1, 1, 0, HOLD, 1),  # at min
    ],
)
def test_threshold_same_law_under_both_contracts(
    utilization: float, active: int, pending: int, delta_code: int, target: int
) -> None:
    info = decision_info(utilization, active, pending)
    legacy, delta, desired = threshold(None), threshold(DELTA), threshold(DESIRED_1_10)
    assert legacy.act(np.zeros(1), info) == delta.act(np.zeros(1), info) == delta_code
    code = desired.act(np.zeros(1), info)
    assert DESIRED_1_10.target_for(code, active + pending) == target
    assert desired.last_decision is not None and legacy.last_decision is not None
    assert desired.last_decision.reason == legacy.last_decision.reason
    assert desired.last_decision.target_replicas == target
    assert legacy.last_decision.target_replicas is None


def test_threshold_is_never_proportional_under_desired() -> None:
    controller = threshold(DESIRED_1_10)
    code = controller.act(np.zeros(1), decision_info(1.0, 2))  # saturated: HPA would jump
    assert DESIRED_1_10.target_for(code, 2) == 3


def test_threshold_cooldown_is_unchanged_under_desired() -> None:
    delta, desired = threshold(DELTA, cooldown=2), threshold(DESIRED_1_10, cooldown=2)
    reasons = {}
    for name, controller in (("delta", delta), ("desired", desired)):
        steps = [
            decision_info(0.9, 2),
            decision_info(0.9, 2, 1, tick=0, applied_replica_change=1),
            decision_info(0.9, 2, 1, tick=1, applied_replica_change=0),
            decision_info(0.9, 3, 0, tick=2, applied_replica_change=0),
        ]
        for info in steps:
            controller.act(np.zeros(1), info)
        reasons[name] = controller.last_decision
    assert reasons["delta"] is not None and reasons["desired"] is not None
    assert reasons["delta"].reason == reasons["desired"].reason == "above_high"


def predictive(contract: ActionContract | None) -> PredictiveController:
    return PredictiveController.from_config(SimulatorConfig(), action_contract=contract)


def feed(
    controller: PredictiveController,
    rates: list[float],
    *,
    active: int,
    pending: int,
    queued: float,
) -> int:
    code = HOLD
    for tick, rate in enumerate(rates):
        code = controller.act(
            np.zeros(1),
            {
                "tick": tick,
                "request_rate": rate,
                "queued_requests": queued,
                "active_replicas": active,
                "pending_replicas": pending,
            },
        )
    return code


def test_predictive_identifiers_are_unchanged() -> None:
    assert (FORECAST_METHOD, CAPACITY_POLICY) == ("linear-trend", "forecast-plus-backlog-v1")


def test_predictive_same_forecast_and_desired_encoded_as_target() -> None:
    rates = [100.0, 150.0, 200.0, 250.0]
    legacy, delta, desired = predictive(None), predictive(DELTA), predictive(DESIRED_1_10)
    codes = [feed(c, rates, active=2, pending=0, queued=0.0) for c in (legacy, delta, desired)]
    decisions = [c.last_decision for c in (legacy, delta, desired)]
    assert all(d is not None for d in decisions)
    forecasts = {d.forecast_rps for d in decisions if d}
    wanted = {d.desired_replicas for d in decisions if d}
    assert len(forecasts) == 1 and len(wanted) == 1
    assert legacy.forecasts == desired.forecasts
    (want,) = wanted
    assert want is not None and want > 3
    assert codes[0] == codes[1] == SCALE_UP  # one step under delta-v1
    assert DESIRED_1_10.target_for(codes[2], 2) == want  # the whole target at once


def test_predictive_backlog_hold_holds_committed_under_desired() -> None:
    controller = predictive(DESIRED_1_10)
    code = feed(controller, [10.0, 10.0], active=6, pending=0, queued=5.0)
    assert controller.last_decision is not None
    assert controller.last_decision.reason == "backlog_hold"
    assert DESIRED_1_10.target_for(code, 6) == 6


def test_predictive_scale_down_goes_straight_to_desired() -> None:
    controller = predictive(DESIRED_1_10)
    code = feed(controller, [10.0, 10.0], active=8, pending=0, queued=0.0)
    assert controller.last_decision is not None and controller.last_decision.reason == "scale_down"
    assert DESIRED_1_10.target_for(code, 8) == controller.last_decision.desired_replicas == 1


def test_static_requests_its_fixed_target_directly() -> None:
    config = desired_config(initial_replicas=1)
    controller = StaticController(
        5, config.replicas, action_contract=ActionContract.from_config(config)
    )
    infos = run_episode(AutoscalingEnv(config, idle(config)), controller, seed=0)
    assert infos[0]["applied_replica_change"] == 4
    assert all(info["applied_replica_change"] == 0 for info in infos[1:])


def test_random_desired_draws_any_target_code_and_is_seeded() -> None:
    config = desired_config()
    contract = ActionContract.from_config(config)
    first = [
        RandomController(seed=1, action_contract=contract).act(np.zeros(1), {}) for _ in range(1)
    ]
    controller = RandomController(seed=1, action_contract=contract)
    codes = [controller.act(np.zeros(1), {}) for _ in range(200)]
    assert set(codes) == set(range(10)) and codes[0] == first[0]


# --- reward and action-magnitude diagnostics ----------------------------------------------------


@pytest.mark.parametrize("config", [SimulatorConfig(), desired_config()])
def test_churn_penalty_counts_scaling_events_not_magnitude(config: SimulatorConfig) -> None:
    env = env_of(config)
    code = SCALE_UP if config.action.semantics == DELTA_V1 else 6  # +1 vs +6 replicas
    info = env.step(code)[4]
    assert abs(info["applied_replica_change"]) == (1 if config.action.semantics == DELTA_V1 else 6)
    assert info["reward_components"]["churn_penalty"] == RewardWeights().churn * 1.0


def test_reward_weights_are_unchanged() -> None:
    assert RewardWeights().model_dump() == {
        "latency": 1.0,
        "cost": 1.0,
        "sla": 1.0,
        "queue": 1.0,
        "churn": 0.1,
    }


def test_action_magnitude_distinguishes_one_from_six_replicas() -> None:
    one = env_of(SimulatorConfig())
    six = env_of(desired_config())
    infos_one = [one.step(SCALE_UP)[4], one.step(HOLD)[4]]
    infos_six = [six.step(6)[4], six.step(6)[4]]
    for infos in (infos_one, infos_six):
        assert summarize_episode(infos, SimulatorConfig()).scaling_actions == 1
    small, large = summarize_action_magnitude(infos_one), summarize_action_magnitude(infos_six)
    assert (small.scaling_events, small.total_absolute_replica_change) == (1, 1)
    assert (large.scaling_events, large.total_absolute_replica_change) == (1, 6)
    assert large.mean_absolute_replica_change_when_scaling == 6.0
    assert large.max_absolute_replica_change_in_one_tick == 6
    assert large.max_pending_replicas == 6 and small.max_pending_replicas == 1


def test_action_magnitude_counts_both_directions_and_idle_episodes() -> None:
    env = env_of(desired_config(initial_replicas=2))
    infos = [env.step(7)[4], env.step(2)[4], env.step(2)[4]]
    metrics = summarize_action_magnitude(infos)
    assert (metrics.scaling_events, metrics.total_absolute_replica_change) == (2, 11)
    assert metrics.as_metrics()["action.max_absolute_replica_change_in_one_tick"] == 6.0
    quiet = summarize_action_magnitude([env.step(2)[4]])
    assert quiet.mean_absolute_replica_change_when_scaling == 0.0
    with pytest.raises(ValueError):
        summarize_action_magnitude([])


# --- model compatibility ----------------------------------------------------------------------


def compatibility(config: SimulatorConfig) -> EnvironmentCompatibility:
    return EnvironmentCompatibility.from_config(config, "v1")


def test_compatibility_records_action_semantics() -> None:
    assert compatibility(SimulatorConfig()).action_semantics_version == DELTA_V1
    desired = compatibility(desired_config())
    assert desired.action_semantics_version == DESIRED_REPLICAS_V1 and desired.action_count == 10


def test_same_action_count_different_semantics_is_incompatible() -> None:
    delta = compatibility(SimulatorConfig())
    three_levels = compatibility(desired_config(min_replicas=8, initial_replicas=8))
    assert delta.action_count == three_levels.action_count == 3
    assert delta.mismatches(three_levels) == {
        "action_semantics_version": (DELTA_V1, DESIRED_REPLICAS_V1)
    }
    for check in (delta.require_compatible, delta.require_compatible_for_robustness):
        with pytest.raises(ValueError, match="action_semantics_version"):
            check(three_levels)
    assert "action_semantics_version" not in ROBUSTNESS_PERTURBABLE_FIELDS


def test_pre_79_compatibility_loads_as_delta_v1() -> None:
    payload = compatibility(SimulatorConfig()).model_dump(mode="json")
    del payload["action_semantics_version"]
    old = EnvironmentCompatibility.model_validate_json(json.dumps(payload))
    assert old.action_semantics_version == DELTA_V1
    old.require_compatible(compatibility(SimulatorConfig()))


def _bundle(algorithm: str, config: SimulatorConfig, directory: Path) -> Path:
    env = AutoscalingEnv(config, benchmark_trace("syn-train-spike"))
    if algorithm == "dqn":
        hp: Any = DQNHyperparameters(
            buffer_size=500, learning_starts=16, batch_size=16, net_arch=(16, 16)
        )
        model: Any = build_dqn(Monitor(env), hp, seed=0)
    else:
        hp = PPOHyperparameters(n_steps=64, batch_size=32, n_epochs=2, net_arch=(16, 16))
        model = build_ppo(Monitor(env), hp, seed=0)
    model.learn(total_timesteps=64)
    return save_model_bundle(
        model,
        directory,
        metadata=ModelMetadata(
            algorithm=algorithm,  # type: ignore[arg-type]
            config_version=f"{algorithm}-v1",
            benchmark_version="v1",
            training_workload_id="syn-train-spike",
            training_workload_split="train",
            seed=0,
            total_timesteps=64,
            hyperparameters=hp.as_params(),
            scalerl_version="test",
        ),
        compatibility=EnvironmentCompatibility.from_env(env, "v1"),
    )


@pytest.fixture(scope="module")
def bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], Path]:
    configs = {DELTA_V1: SimulatorConfig(), DESIRED_REPLICAS_V1: desired_config()}
    return {
        (algorithm, semantics): _bundle(
            algorithm, config, tmp_path_factory.mktemp(f"{algorithm}-{semantics}")
        )
        for algorithm in ("dqn", "ppo")
        for semantics, config in configs.items()
    }


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_desired_models_train_and_infer_discretely(
    algorithm: str, bundles: dict[tuple[str, str], Path]
) -> None:
    config = desired_config()
    env = AutoscalingEnv(config, benchmark_trace("syn-val-bursty"))
    controller = load_sb3_controller(bundles[(algorithm, DESIRED_REPLICAS_V1)], env)
    assert isinstance(controller.model, DQN if algorithm == "dqn" else PPO)
    assert controller.model.action_space == spaces.Discrete(10)
    infos = run_episode(env, controller, seed=0)
    assert all(0 <= info["requested_action"] < 10 for info in infos)
    assert all(isinstance(info["requested_replica_target"], int) for info in infos)
    _, recorded = read_model_bundle(bundles[(algorithm, DESIRED_REPLICAS_V1)])
    assert recorded.action_semantics_version == DESIRED_REPLICAS_V1


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_bundles_load_only_under_their_own_contract(
    algorithm: str, bundles: dict[tuple[str, str], Path]
) -> None:
    envs = {
        DELTA_V1: AutoscalingEnv(SimulatorConfig(), benchmark_trace("syn-val-bursty")),
        DESIRED_REPLICAS_V1: AutoscalingEnv(desired_config(), benchmark_trace("syn-val-bursty")),
    }
    for trained, bundle_semantics in (
        (DELTA_V1, DELTA_V1),
        (DESIRED_REPLICAS_V1, DESIRED_REPLICAS_V1),
    ):
        load_sb3_controller(bundles[(algorithm, bundle_semantics)], envs[trained])
    for model_semantics, env_semantics in (
        (DELTA_V1, DESIRED_REPLICAS_V1),
        (DESIRED_REPLICAS_V1, DELTA_V1),
    ):
        for robustness in (False, True):
            with pytest.raises(ValueError, match="incompatible environment"):
                load_sb3_controller(
                    bundles[(algorithm, model_semantics)],
                    envs[env_semantics],
                    robustness_evaluation=robustness,
                )


def test_same_count_desired_model_never_loads_in_delta_env(tmp_path: Path) -> None:
    three = desired_config(min_replicas=8, initial_replicas=8)
    bundle = _bundle("dqn", three, tmp_path / "three")
    delta_env = AutoscalingEnv(SimulatorConfig(), benchmark_trace("syn-val-bursty"))
    assert delta_env.action_space == spaces.Discrete(3)
    with pytest.raises(ValueError, match="action_semantics_version"):
        load_sb3_controller(bundle, delta_env)
    delayed = SimulatorConfig(dynamics=DynamicsConfig(telemetry_delay_ticks=1))
    with pytest.raises(ValueError, match="action_semantics_version"):
        load_sb3_controller(
            bundle,
            AutoscalingEnv(delayed, benchmark_trace("syn-val-bursty")),
            robustness_evaluation=True,
        )


def test_pre_79_bundle_without_the_field_loads_as_delta(
    bundles: dict[tuple[str, str], Path], tmp_path: Path
) -> None:
    source = bundles[("dqn", DELTA_V1)]
    old = tmp_path / "old"
    old.mkdir()
    for name in ("model.zip", "metadata.json"):
        (old / name).write_bytes((source / name).read_bytes())
    payload = json.loads((source / "compatibility.json").read_text())
    del payload["action_semantics_version"]
    (old / "compatibility.json").write_text(json.dumps(payload))
    load_sb3_controller(old, AutoscalingEnv(SimulatorConfig(), benchmark_trace("syn-val-bursty")))
    with pytest.raises(ValueError, match="action_semantics_version"):
        load_sb3_controller(
            old, AutoscalingEnv(desired_config(), benchmark_trace("syn-val-bursty"))
        )
