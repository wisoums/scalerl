"""Tests for seeded stochastic startup delay (#81) and the startup-robustness-v1 extension.

Synthetic fixtures and validation workloads only; tiny models; no outputs/.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from stable_baselines3.common.monitor import Monitor

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.environment import (
    DESIRED_REPLICAS_V1,
    FIXED_V1,
    TRI_POINT_EXPECTED_MULTIPLIER,
    TRI_POINT_MULTIPLICATIVE_V1,
    TRI_POINT_MULTIPLIERS,
    TRI_POINT_PROBABILITIES,
    ActionConfig,
    AutoscalingEnv,
    DynamicsConfig,
    ReplicaConfig,
    ReplicaPool,
    SimulatorConfig,
)
from scalerl.environment.gym_env import PHYSICAL_ONLY_KEYS
from scalerl.environment.startup import draw_tri_point_multiplier
from scalerl.evaluation import startup_robustness as sr
from scalerl.evaluation.robustness import (
    CAPACITY_JITTER,
    COMBINED_ROBUSTNESS,
    DELAYED_TELEMETRY,
    NOMINAL,
    ROBUSTNESS_SCENARIO_VERSION,
    ROBUSTNESS_SCENARIOS,
)
from scalerl.mlops import EnvironmentCompatibility
from scalerl.mlops.spec import ROBUSTNESS_PERTURBABLE_FIELDS
from scalerl.workloads import steady_workload

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "benchmarks" / "v1" / "startup-robustness-v1.json"
DECISION = REPO / "benchmarks" / "v1" / "action-contract-v2.json"
SPEC_ID = "bec43981bfc8"
DESIRED = ActionConfig(semantics=DESIRED_REPLICAS_V1)


def config(
    *, model: str = TRI_POINT_MULTIPLICATIVE_V1, seed: int = 0, initial: int = 2, **dynamics: Any
) -> SimulatorConfig:
    return SimulatorConfig(
        action=DESIRED,
        replicas=ReplicaConfig(initial_replicas=initial),
        dynamics=DynamicsConfig(startup_delay_model=model, startup_delay_seed=seed, **dynamics),  # type: ignore[arg-type]
    )


def env_of(cfg: SimulatorConfig, rate: float = 10.0) -> AutoscalingEnv:
    trace = steady_workload(duration_seconds=3600, control_interval_seconds=30, rate=rate)
    env = AutoscalingEnv(cfg, trace)
    env.reset(seed=0)
    return env


def started(env: AutoscalingEnv, codes: list[int]) -> list[list[float]]:
    return [env.step(code)[4].get("startup_realized_delays_seconds", []) for code in codes]


# --- model definition --------------------------------------------------------------------------


def test_tri_point_model_is_the_predeclared_one() -> None:
    assert TRI_POINT_MULTIPLIERS == (0.5, 1.0, 1.5)
    assert TRI_POINT_PROBABILITIES == (0.25, 0.50, 0.25)
    assert TRI_POINT_EXPECTED_MULTIPLIER == 1.0 and sum(TRI_POINT_PROBABILITIES) == 1.0
    rng = np.random.default_rng(0)
    draws = [draw_tri_point_multiplier(rng) for _ in range(500)]
    assert set(draws) <= set(TRI_POINT_MULTIPLIERS) and len(set(draws)) == 3
    again = np.random.default_rng(0)
    assert [draw_tri_point_multiplier(again) for _ in range(500)] == draws


def test_canonical_realized_delays_are_30_60_or_90_seconds() -> None:
    env = env_of(config(seed=3, initial=1))
    delays = [d for batch in started(env, [9, 0, 9, 0, 9]) for d in batch]
    assert delays and set(delays) <= {30.0, 60.0, 90.0}


# --- fixed-v1 default regression ---------------------------------------------------------------


def test_default_dynamics_mean_fixed_startup_and_serialize_unchanged() -> None:
    dynamics = DynamicsConfig()
    assert (dynamics.startup_delay_model, dynamics.startup_delay_seed) == (FIXED_V1, 0)
    dumped = SimulatorConfig().model_dump(mode="json")["dynamics"]
    assert dumped == {
        "capacity_jitter_fraction": 0.0,
        "dynamics_seed": 0,
        "telemetry_delay_ticks": 0,
    }
    assert DynamicsConfig.model_validate(dumped) == dynamics
    stochastic = DynamicsConfig(
        startup_delay_model=TRI_POINT_MULTIPLICATIVE_V1, startup_delay_seed=2
    )
    assert stochastic.model_dump()["startup_delay_model"] == TRI_POINT_MULTIPLICATIVE_V1
    assert DynamicsConfig.model_validate_json(stochastic.model_dump_json()) == stochastic
    with pytest.raises(ValidationError):
        DynamicsConfig(startup_delay_model="lognormal-v1")  # type: ignore[arg-type]


def test_robustness_v1_scenarios_serialize_as_before() -> None:
    # The #19 plan hash depends on these dumps; no startup field may appear.
    for scenario in ROBUSTNESS_SCENARIOS.values():
        dumped = scenario.dynamics(3).model_dump(mode="json")
        assert set(dumped) == {"capacity_jitter_fraction", "dynamics_seed", "telemetry_delay_ticks"}
        assert scenario.dynamics(3).startup_delay_model == FIXED_V1


def test_fixed_startup_never_draws_from_the_startup_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    env = env_of(config(model=FIXED_V1))
    monkeypatch.setattr(env, "_startup_rng", None)  # any draw would fail
    for code in (9, 0, 9, 4):
        info = env.step(code)[4]
        assert not PHYSICAL_ONLY_KEYS & set(info)  # no new keys in the nominal info


def test_zero_nominal_delay_never_draws() -> None:
    calls: list[int] = []
    pool = ReplicaPool(
        ReplicaConfig(initial_replicas=2, startup_delay_seconds=0.0),
        startup_multiplier=lambda: calls.append(1) or 1.5,  # type: ignore[func-returns-value]
    )
    assert pool.scale_up(3) == 3 and pool.active_count == 5 and not calls


def test_fixed_pool_matches_the_pre_81_schedule() -> None:
    pool = ReplicaPool(ReplicaConfig(initial_replicas=2, startup_delay_seconds=60.0))
    pool.scale_up(3)
    assert pool.pending_by_ticks_until_active(30) == (0, 3)
    assert pool.physical_pending_by_ticks_until_active(30) == (0, 3)
    pool.advance(30)
    pool.scale_up(1)
    assert pool.pending_by_ticks_until_active(30) == (3, 1)
    pool.advance(30)
    assert (pool.active_count, pool.pending_count) == (5, 1)


# --- seeding and RNG separation -----------------------------------------------------------------


REQUESTS = [7, 2, 9, 0, 6, 9, 3, 9]


def test_same_seed_and_requests_give_the_same_realization() -> None:
    a = started(env_of(config(seed=5)), REQUESTS)
    b = started(env_of(config(seed=5)), REQUESTS)
    assert a == b and any(len(set(batch)) > 1 for batch in a)
    assert a != started(env_of(config(seed=6)), REQUESTS)


def test_reset_restarts_the_realization_and_order_is_irrelevant() -> None:
    env = env_of(config(seed=4))
    first = started(env, REQUESTS)
    env.reset(seed=0)
    assert started(env, REQUESTS) == first
    other = env_of(config(seed=9))  # interleave another environment's draws
    x = env_of(config(seed=4))
    interleaved = []
    for code in REQUESTS:
        other.step(code)
        interleaved.append(x.step(code)[4]["startup_realized_delays_seconds"])
    assert interleaved == first


def test_capacity_and_startup_streams_are_independent() -> None:
    jitter = {"capacity_jitter_fraction": 0.1, "dynamics_seed": 2}

    def multipliers(cfg: SimulatorConfig, codes: list[int]) -> list[float]:
        env = env_of(cfg)
        return [env.step(code)[4]["capacity_multiplier"] for code in codes]

    base = multipliers(config(model=FIXED_V1, **jitter), [1] * 8)
    assert multipliers(config(seed=0, **jitter), REQUESTS) == base  # startup draws don't shift it
    assert multipliers(config(seed=7, **jitter), [9, 0] * 4) == base
    # ...and capacity jitter does not shift the startup realization.
    assert started(env_of(config(seed=5, **jitter)), REQUESTS) == started(
        env_of(config(seed=5)), REQUESTS
    )


# --- multi-replica requests and cancellation ------------------------------------------------------


def test_multi_replica_request_gets_independent_readiness() -> None:
    for seed in range(20):
        env = env_of(config(seed=seed))
        info = env.step(6)[4]  # target 7 from 2 committed: five replicas in one decision
        delays = info["startup_realized_delays_seconds"]
        if len(set(delays)) > 1:
            break
    assert info["applied_replica_change"] == 5 and len(delays) == 5 and len(set(delays)) > 1
    assert info["active_replicas"] == 2  # none serves the tick it was requested
    active = [2]
    for _ in range(3):
        active.append(env.step(6)[4]["active_replicas"])
    # replicas become active tick by tick in realized-delay order, never early
    for ticks, expected in ((1, 30.0), (2, 60.0), (3, 90.0)):
        ready = sum(1 for d in delays if d <= expected)
        assert active[ticks] == 2 + ready


def test_cancellation_is_newest_first_and_cancelled_never_activate() -> None:
    draws = iter([1.5, 0.5, 1.0, 0.5])
    pool = ReplicaPool(
        ReplicaConfig(initial_replicas=1, startup_delay_seconds=60.0),
        startup_multiplier=lambda: next(draws),
    )
    pool.scale_up(3)  # realized 90, 30, 60
    assert [s.realized_delay_seconds for s in pool.last_started] == [90.0, 30.0, 60.0]
    assert pool.scale_down(2) == 2  # cancels the 60 s and the 30 s (newest first)
    assert pool.pending_count == 1
    pool.advance(30)
    pool.advance(30)
    assert pool.active_count == 1  # the cancelled 30 s replica never activated
    pool.advance(30)
    assert (pool.active_count, pool.pending_count) == (2, 0)  # the 90 s one did
    pool.scale_up(1)
    assert pool.last_started[0].realized_delay_seconds == 30.0  # stream continues


# --- no future readiness leak ---------------------------------------------------------------------


def _seeds_with_first_draws(first: float, second: float) -> tuple[int, int]:
    found: dict[float, int] = {}
    for seed in range(200):
        value = draw_tri_point_multiplier(np.random.default_rng(seed))
        found.setdefault(value, seed)
    return found[first], found[second]


@pytest.mark.parametrize("delay", [0, 1])
def test_controller_view_hides_the_sampled_realization(delay: int) -> None:
    nominal_seed, late_seed = _seeds_with_first_draws(1.0, 1.5)
    views, observations, raw = [], [], []
    for seed in (nominal_seed, late_seed):
        env = env_of(config(seed=seed, initial=1, telemetry_delay_ticks=delay))
        info = env.step(1)[4]  # target 2: one new replica
        raw.append(info["startup_realized_delays_seconds"])
        view = env.decision_info(info)
        assert not PHYSICAL_ONLY_KEYS & set(view)
        assert not any("realized" in key or "ready_tick" in key for key in view)
        views.append(view)
        observations.append(env._observation())
        for snapshot in env.visible_telemetry():
            assert not PHYSICAL_ONLY_KEYS & set(snapshot.measurements)
    assert raw == [[60.0], [90.0]]  # physical truth differs...
    assert views[0] == views[1]  # ...but nothing a controller sees does
    assert np.array_equal(observations[0], observations[1])
    assert observations[0].shape == (12,)


def test_late_replica_is_counted_as_due_now_with_a_fixed_shape() -> None:
    env = env_of(config(seed=_seeds_with_first_draws(1.5, 1.5)[0], initial=1))
    env.step(1)
    env.step(1)
    assert env.replica_counts["pending_replicas"] == 1  # 90 s realized: still starting
    assert env._pending_buckets() == (1, 0)  # nominally due now; the realization stays hidden
    assert env.observation_space.shape == (12,)


# --- compatibility ----------------------------------------------------------------------------


def test_compatibility_records_the_model_not_the_seed() -> None:
    fixed = EnvironmentCompatibility.from_config(config(model=FIXED_V1), "v1")
    tri_a = EnvironmentCompatibility.from_config(config(seed=1), "v1")
    tri_b = EnvironmentCompatibility.from_config(config(seed=2), "v1")
    assert fixed.startup_delay_model == FIXED_V1
    assert tri_a == tri_b and tri_a.startup_delay_model == TRI_POINT_MULTIPLICATIVE_V1
    with pytest.raises(ValueError, match="startup_delay_model"):
        fixed.require_compatible(tri_a)
    assert fixed.require_compatible_for_robustness(tri_a) == ("startup_delay_model",)
    assert "startup_delay_model" in ROBUSTNESS_PERTURBABLE_FIELDS
    assert "action_semantics_version" not in ROBUSTNESS_PERTURBABLE_FIELDS


def test_robustness_path_still_rejects_other_mismatches() -> None:
    fixed = EnvironmentCompatibility.from_config(config(model=FIXED_V1), "v1")
    delta = EnvironmentCompatibility.from_config(
        SimulatorConfig(dynamics=DynamicsConfig(startup_delay_model=TRI_POINT_MULTIPLICATIVE_V1)),
        "v1",
    )
    with pytest.raises(ValueError, match="action_semantics_version"):
        fixed.require_compatible_for_robustness(delta)
    other = config().model_copy(update={"replicas": ReplicaConfig(startup_delay_seconds=90.0)})
    with pytest.raises(ValueError, match="observation_shape"):
        fixed.require_compatible_for_robustness(EnvironmentCompatibility.from_config(other, "v1"))
    bigger = config().model_copy(update={"replicas": ReplicaConfig(service_capacity_rps=60.0)})
    with pytest.raises(ValueError, match="service_capacity_rps"):
        fixed.require_compatible_for_robustness(EnvironmentCompatibility.from_config(bigger, "v1"))


def test_old_compatibility_json_loads_as_fixed_startup() -> None:
    payload = EnvironmentCompatibility.from_config(config(model=FIXED_V1), "v1").model_dump(
        mode="json"
    )
    del payload["startup_delay_model"]
    old = EnvironmentCompatibility.model_validate_json(json.dumps(payload))
    assert old.startup_delay_model == FIXED_V1


# --- robustness-v1 frozen ---------------------------------------------------------------------


def test_robustness_v1_is_unchanged() -> None:
    assert ROBUSTNESS_SCENARIO_VERSION == "robustness-v1"
    assert list(ROBUSTNESS_SCENARIOS) == [
        "nominal",
        "capacity-jitter",
        "delayed-telemetry",
        "combined-robustness",
    ]
    assert [(s.capacity_jitter_fraction, s.telemetry_delay_ticks) for s in (
        NOMINAL, CAPACITY_JITTER, DELAYED_TELEMETRY, COMBINED_ROBUSTNESS
    )] == [(0.0, 0), (0.1, 0), (0.0, 1), (0.1, 1)]  # fmt: skip
    assert not hasattr(NOMINAL, "startup_delay_model")
    assert not set(sr.STARTUP_ROBUSTNESS_SCENARIOS) & set(ROBUSTNESS_SCENARIOS)


def test_startup_robustness_scenarios_and_seed_plan() -> None:
    assert sr.STARTUP_ROBUSTNESS_VERSION == "startup-robustness-v1"
    assert list(sr.STARTUP_ROBUSTNESS_SCENARIOS) == [
        "startup-delay-jitter",
        "combined-startup-robustness",
    ]
    jitter, combined = sr.STARTUP_DELAY_JITTER, sr.COMBINED_STARTUP_ROBUSTNESS
    assert (jitter.capacity_jitter_fraction, jitter.telemetry_delay_ticks) == (0.0, 0)
    assert (combined.capacity_jitter_fraction, combined.telemetry_delay_ticks) == (0.1, 1)
    assert jitter.seed_pairs() == tuple((0, s) for s in range(5))
    assert combined.seed_pairs() == tuple((s, s) for s in range(5))
    dynamics = combined.dynamics(startup_delay_seed=3, dynamics_seed=3)
    assert dynamics.startup_delay_model == TRI_POINT_MULTIPLICATIVE_V1


# --- diagnostics ------------------------------------------------------------------------------


def test_startup_diagnostics_and_unavailable_values() -> None:
    cfg = config(seed=5)
    env = env_of(cfg)
    infos = [env.step(code)[4] for code in REQUESTS]
    diag = sr.summarize_startup(infos, cfg)
    delays = [d for info in infos for d in info["startup_realized_delays_seconds"]]
    assert diag.replicas_requested == len(delays)
    assert diag.mean_realized_delay_seconds == pytest.approx(sum(delays) / len(delays))
    assert {diag.min_realized_delay_seconds, diag.max_realized_delay_seconds} <= {30.0, 60.0, 90.0}
    assert diag.multi_replica_batches >= 1
    quiet = sr.summarize_startup([env.step(9)[4]], cfg)  # at max: nothing started
    assert quiet.replicas_requested == 0 and quiet.mean_realized_delay_seconds is None
    assert "startup.mean_realized_delay_seconds" not in quiet.as_metrics()
    fixed_cfg = config(model=FIXED_V1)
    fixed_env = env_of(fixed_cfg)
    fixed = sr.summarize_startup([fixed_env.step(6)[4]], fixed_cfg)
    assert (
        fixed.replicas_requested,
        fixed.mean_multiplier,
        fixed.max_batch_readiness_spread_seconds,
    ) == (
        5,
        1.0,
        0.0,
    )


# --- experiment spec and guards ---------------------------------------------------------------


def test_committed_spec_is_the_frozen_plan() -> None:
    spec = sr.load_frozen_spec(SPEC, DECISION)
    assert spec.experiment_id == SPEC_ID
    assert spec.startup_model["id"] == TRI_POINT_MULTIPLICATIVE_V1
    assert spec.startup_model["multipliers"] == [0.5, 1.0, 1.5]
    assert spec.startup_model["probabilities"] == [0.25, 0.5, 0.25]
    assert spec.startup_model["expected_multiplier"] == 1.0
    assert spec.action_semantics == DESIRED_REPLICAS_V1
    assert spec.validation_workload_ids == sr.VALIDATION_WORKLOADS and spec.test_workload_ids == ()
    assert spec.startup_seeds == spec.combined_dynamics_seeds == (0, 1, 2, 3, 4)
    assert len(spec.robustness_v1_scenarios) == 4
    assert [m["selected_candidate_id"] for m in spec.ppo_models] == ["ppo-c08"] * 5
    assert "no canonical desired-replicas-v1 DQN" in spec.dqn
    assert spec.reward_changed is False and spec.held_out_data_used is False
    assert spec.simulator_config["action"] == {"semantics": DESIRED_REPLICAS_V1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_workload_ids", ["syn-test-spike-hard"]),
        ("validation_workload_ids", ["syn-val-bursty", "syn-test-spike-hard"]),
        ("validation_workload_ids", ["syn-train-bursty"]),
        ("startup_seeds", [0, 1]),
        ("action_semantics", "delta-v1"),
        ("reward_changed", True),
        ("held_out_data_used", True),
        ("simulator_config", SimulatorConfig().model_dump(mode="json")),
    ],
)
def test_spec_rejects_departures(field: str, value: Any) -> None:
    payload = json.loads(SPEC.read_text())
    payload[field] = value
    with pytest.raises(ValidationError):
        sr.ExperimentSpec.model_validate_json(json.dumps(payload))


def test_edited_startup_model_is_refused(tmp_path: Path) -> None:
    payload = json.loads(SPEC.read_text())
    payload["startup_model"]["probabilities"] = [0.2, 0.6, 0.2]
    with pytest.raises(ValidationError, match="predeclared"):
        sr.ExperimentSpec.model_validate_json(json.dumps(payload))


def test_cli_has_no_test_escape(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        sr.main(["run", "--include-test"])
    with pytest.raises(SystemExit):
        sr.main(["run", "--workload", "syn-test-spike-hard"])
    assert sr.main(["check", "--spec", str(SPEC), "--action-decision", str(DECISION)]) == 0
    printed = capsys.readouterr().out
    for text in (SPEC_ID, "desired-replicas-v1", "tri-point-multiplicative-v1",
                 "[0.5, 1.0, 1.5]", "[0.25, 0.5, 0.25]", "robustness-v1 scenarios    4",
                 "test workload count        0", "reward changed             false"):  # fmt: skip
        assert text in printed


def test_cases_cover_the_plan_with_distinct_seeds() -> None:
    spec = sr.load_frozen_spec(SPEC, DECISION)
    all_cases = sr.cases(spec)
    assert len(all_cases) == 3 * 8 * 11
    assert all(c.workload_id in sr.VALIDATION_WORKLOADS for c in all_cases)
    nominal = [c for c in all_cases if c.scenario_name == "nominal"]
    assert all(
        c.startup_delay_seed is None and c.scenario_version == "robustness-v1" for c in nominal
    )
    for case in all_cases:
        cfg = sr.case_config(case)
        assert cfg.action.semantics == DESIRED_REPLICAS_V1
        if case.startup_delay_seed is None:
            assert cfg.dynamics == NOMINAL.dynamics(0)
        else:
            assert cfg.dynamics.startup_delay_seed == case.startup_delay_seed
            assert cfg.dynamics.dynamics_seed == case.dynamics_seed
    ppo = [c for c in all_cases if c.training_run_id]
    assert {c.training_seed for c in ppo} == {0, 1, 2, 3, 4}


# --- end to end (rule controllers + a tiny PPO stand-in) ----------------------------------------


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture(scope="module")
def tiny_ppo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from scalerl.rl import ModelMetadata, save_model_bundle
    from scalerl.training.ppo import PPOHyperparameters, build_ppo

    cfg = SimulatorConfig(action=DESIRED)
    env = AutoscalingEnv(cfg, build_workload(load_benchmark_manifest().get("syn-train-spike")))
    hp = PPOHyperparameters(n_steps=64, batch_size=32, n_epochs=1, net_arch=(8, 8))
    model = build_ppo(Monitor(env), hp, seed=0)
    model.learn(total_timesteps=64)
    return save_model_bundle(
        model,
        tmp_path_factory.mktemp("ppo"),
        metadata=ModelMetadata(
            algorithm="ppo",
            config_version="ppo-v1",
            benchmark_version="v1",
            training_workload_id="syn-train-spike",
            training_workload_split="train",
            seed=0,
            total_timesteps=64,
            hyperparameters=hp.as_params(),
            scalerl_version="test",
        ),  # fmt: skip
        compatibility=EnvironmentCompatibility.from_env(env, "v1"),
    )


def test_run_is_resumable_and_logs_distinct_seed_metadata(
    tmp_path: Path, tracking_uri: str, tiny_ppo: Path
) -> None:
    from mlflow import MlflowClient

    spec = sr.load_frozen_spec(SPEC, DECISION)
    variants = ("threshold-v1", "ppo-desired-replicas-v1-seed2")
    out = tmp_path / "out"
    rows = sr.run_experiment(
        spec, out=out, tracking_uri=tracking_uri, progress=lambda _: None,
        locate=lambda _: tiny_ppo, only_variants=variants,
    )  # fmt: skip
    assert len(rows) == 3 * 2 * 11
    ppo_rows = [r for r in rows if r["controller"] == "ppo"]
    assert {r["perturbed_compatibility"] for r in ppo_rows if r["scenario"] == "nominal"} == {""}
    assert {
        r["perturbed_compatibility"]
        for r in ppo_rows
        if r["scenario"] == "combined-startup-robustness"
    } == {"startup_delay_model,telemetry_delay_ticks"}
    stochastic = [r for r in rows if r["startup_delay_seed"] is not None]
    assert all(r["startup.replicas_requested"] == sum(len(b) for b in r["startup_realizations"])
               for r in stochastic)  # fmt: skip
    client = MlflowClient(tracking_uri)
    tags = client.get_run(stochastic[-1]["mlflow_run_id"]).data.tags
    assert tags["scalerl.startup_delay_model"] == TRI_POINT_MULTIPLICATIVE_V1
    assert tags["scalerl.startup_robustness_version"] == "startup-robustness-v1"
    assert tags["scalerl.startup_delay_seed"] == str(stochastic[-1]["startup_delay_seed"])
    assert tags["scalerl.dynamics_seed"] == str(stochastic[-1]["dynamics_seed"])
    assert tags["scalerl.action_semantics"] == DESIRED_REPLICAS_V1
    nominal_run = next(r for r in rows if r["scenario"] == "nominal")
    assert (
        client.get_run(nominal_run["mlflow_run_id"]).data.tags["scalerl.startup_delay_seed"]
        == "none"
    )

    experiment = client.get_experiment_by_name(sr.EXPERIMENT_NAME)
    assert experiment is not None
    count = len(client.search_runs([experiment.experiment_id], max_results=1000))
    raw = out / "raw-results.jsonl"
    raw.write_text("".join(line + "\n" for line in raw.read_text().splitlines()[:-2]))
    again = sr.run_experiment(
        spec, out=out, tracking_uri=tracking_uri, progress=lambda _: None,
        locate=lambda _: tiny_ppo, only_variants=variants,
    )  # fmt: skip
    assert len(client.search_runs([experiment.experiment_id], max_results=1000)) == count
    assert [r["startup_realizations"] for r in again] == [r["startup_realizations"] for r in rows]

    summary = sr.summarize(again)
    levels = {item["level"] for item in summary}
    assert levels == {"variant", "ppo-training-seeds"}
    jitter = next(
        item for item in summary
        if item["level"] == "variant" and item["scenario"] == "startup-delay-jitter"
        and item["metric"] == "sla_violation_rate"
    )  # fmt: skip
    assert jitter["n"] == 5 and jitter["replicate_unit"] == "startup_seed"
    freeze = sr.build_freeze(spec, again)
    assert freeze.held_out_data_used is False and freeze.reward_changed is False
    assert len(freeze.mlflow_run_ids) == len(again)


def test_committed_freeze_artifact() -> None:
    path = REPO / "benchmarks" / "v1" / "startup-robustness-freeze-v1.json"
    freeze = sr.StartupRobustnessFreeze.model_validate_json(path.read_text())
    assert freeze.freeze_id == "5907ade36ebd"
    assert freeze.experiment_spec_id == SPEC_ID
    assert freeze.startup_robustness_version == "startup-robustness-v1"
    assert freeze.startup_model["id"] == TRI_POINT_MULTIPLICATIVE_V1
    assert list(freeze.scenarios) == ["startup-delay-jitter", "combined-startup-robustness"]
    assert freeze.startup_seeds == (0, 1, 2, 3, 4)
    assert freeze.action_semantics == DESIRED_REPLICAS_V1
    assert len(freeze.mlflow_run_ids) == 3 * 8 * 11
    assert all(case.split("|")[1] in sr.VALIDATION_WORKLOADS for case in freeze.mlflow_run_ids)
    assert freeze.held_out_data_used is False and freeze.reward_changed is False
    assert freeze.declares_overall_winner is False
    assert "-test-" not in path.read_text()
