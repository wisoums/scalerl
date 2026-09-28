"""Held-out Azure TEST evaluation ``heldout-evaluation-v1`` (#46)."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import StaticController, ThresholdController, run_episode
from scalerl.controllers.proactive_predictive import (
    HistoricalDemandProfile,
    ProactivePredictiveController,
)
from scalerl.environment import (
    DESIRED_REPLICAS_V1,
    AutoscalingEnv,
    SimulatorConfig,
    SourceWindow,
)
from scalerl.environment.observation import Measurement, ObservationConstants, build_observation
from scalerl.evaluation import heldout as ho
from scalerl.evaluation import sim_to_real as s2r
from scalerl.evaluation.multiseed import ControllerManifest as MultiseedManifest
from scalerl.evaluation.multiseed import make_plan, reference_variants
from scalerl.evaluation.predictive_baseline import final_contract_config
from scalerl.evaluation.robustness import ROBUSTNESS_SCENARIOS
from scalerl.evaluation.startup_robustness import _scenarios as startup_scenarios
from scalerl.mlops import EnvironmentCompatibility
from scalerl.workloads import WorkloadTrace

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "v1"
SPEC_ID = "b32b2f3d3dd6"
DQN_RUN = "920b0ddb58df4d2b9f550431d2f8ceeb"
PPO_RUN = "9bac13a9447b4c64a44d9732568c949e"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return ho.load_frozen_spec(BENCH / "heldout-evaluation-v1.json", BENCH)


@pytest.fixture(scope="module")
def cases(spec: dict[str, Any]) -> list[ho.Case]:
    return ho.generate_cases(spec)


# --- the predeclared spec ------------------------------------------------------------------------


def test_committed_spec_is_the_frozen_one(spec: dict[str, Any]) -> None:
    assert ho.spec_id(spec) == SPEC_ID
    assert spec == ho.build_spec(BENCH)
    assert spec["experiment_version"] == "heldout-evaluation-v1"
    assert spec["source_issue"] == "#46" and spec["benchmark_version"] == "v1"
    assert spec["test_data_used_for_design"] is False
    assert spec["test_outcomes_used_for_design"] is False
    assert spec["model_selection_on_test"] is False
    assert spec["q_learning_included"] is False
    assert spec["declares_winner"] is False
    assert spec["mlflow"]["experiment"] == "scalerl-heldout-v1"
    assert spec["mlflow"]["tracking_uri"] == "sqlite:///outputs/mlflow.db"


def test_edited_spec_is_refused(tmp_path: Path) -> None:
    payload = json.loads((BENCH / "heldout-evaluation-v1.json").read_text())
    payload["workloads"]["primary_azure_test"] = ["azure-test-993600"]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        ho.load_frozen_spec(path, BENCH)


def test_exactly_the_two_azure_test_workloads(spec: dict[str, Any]) -> None:
    workloads = spec["workloads"]["primary_azure_test"]
    assert workloads == ["azure-test-993600", "azure-test-1166400"]
    manifest = load_benchmark_manifest()
    assert all(manifest.get(w).split == "test" and manifest.get(w).source == "azure"
               for w in workloads)  # fmt: skip
    assert sorted(
        w.id for w in manifest.workloads if w.split == "test" and w.source == "azure"
    ) == sorted(workloads)
    assert set(spec["workloads"]["excluded_from_primary"]) == {
        "syn-test-seasonal-shifted", "syn-test-spike-hard"
    }  # fmt: skip


def test_cases_never_use_train_validation_or_synthetic_test(cases: list[ho.Case]) -> None:
    manifest = load_benchmark_manifest()
    for case in cases:
        entry = manifest.get(case.workload_id)
        assert entry.split == "test" and entry.source == "azure"
    assert {c.workload_id for c in cases} == {"azure-test-993600", "azure-test-1166400"}


def test_exact_controller_set_and_learned_artifacts(spec: dict[str, Any]) -> None:
    variants = {v["variant_id"]: v for v in spec["controllers"]}
    assert list(variants) == [
        "static-v1", "random-v1", "threshold-v1", "predictive-v1",
        "predictive-seasonal-v1", "dqn-c14-seed0", "ppo-c08-seed4",
    ]  # fmt: skip
    assert variants["static-v1"]["params"] == {"target_replicas": 5}
    assert variants["random-v1"]["evaluation_seeds"] == [0, 1, 2, 3, 4]
    assert variants["threshold-v1"]["params"] == {
        "low_threshold": 0.2, "high_threshold": 0.6, "cooldown_ticks": 3
    }  # fmt: skip
    assert variants["predictive-v1"]["params"]["history_window_ticks"] == 4
    assert variants["predictive-seasonal-v1"]["params"]["profile_id"] == "d050d3b8ca0f"
    dqn, ppo = variants["dqn-c14-seed0"], variants["ppo-c08-seed4"]
    assert (dqn["candidate_id"], dqn["training_seed"], dqn["training_run_id"]) == (
        "dqn-c14", 0, DQN_RUN
    )  # fmt: skip
    assert dqn["model_artifact_uri"] == f"runs:/{DQN_RUN}/model"
    assert (ppo["candidate_id"], ppo["training_seed"], ppo["training_run_id"]) == (
        "ppo-c08", 4, PPO_RUN
    )  # fmt: skip
    assert ppo["model_artifact_uri"] == f"runs:/{PPO_RUN}/model"
    assert all(v["evaluation_seeds"] == [0] for k, v in variants.items() if k != "random-v1")
    assert "q-learning" in spec["excluded_controllers"]
    assert {v["controller"] for v in spec["controllers"]} == {
        "static", "random", "threshold", "predictive", "predictive-seasonal", "dqn", "ppo"
    }  # fmt: skip


def test_no_model_selection_on_test(cases: list[ho.Case]) -> None:
    learned = {
        (c.variant.controller, c.variant.training_run_id) for c in cases if c.variant.learned
    }
    assert learned == {("dqn", DQN_RUN), ("ppo", PPO_RUN)}  # one artifact per family, period
    controllers = s2r.load(BENCH / "controller-deployment-manifest-v1.json")["controllers"]
    for family, run in (("dqn", DQN_RUN), ("ppo", PPO_RUN)):
        assert controllers[family]["canonical_selection"]["canonical_training_run_id"] == run


def test_contracts_and_upstream_ids(spec: dict[str, Any]) -> None:
    assert spec["reward"] == {
        "contract": "reward-contract-v1",
        "contract_id": "37158c261364",
        "variant": "full-cost-low-v1",
        "weights": {"latency": 1.0, "cost": 0.5, "sla": 1.0, "queue": 1.0, "churn": 0.1},
        "role": "secondary metric (episode_reward); system metrics are primary",
    }
    assert spec["action_semantics"] == DESIRED_REPLICAS_V1
    assert SimulatorConfig.model_validate(spec["simulator_config"]) == final_contract_config()
    up = spec["upstream"]
    assert (up["sim_to_real_protocol_id"], up["live_replay_manifest_id"],
            up["controller_deployment_manifest_id"]) == (
        "b7bf45c109f1", "7803b71fdfe8", "553ddf3e512b"
    )  # fmt: skip
    assert up["selection_spec_id"] == "418876d6c8e9"
    assert (up["action_decision_id"], up["action_experiment_spec_id"]) == (
        "0eb5562b01e6", "899dfbb64217"
    )  # fmt: skip
    assert up["predictive_baseline_artifact_id"] == "0d315559c680"
    assert (up["startup_robustness_spec_id"], up["startup_robustness_freeze_id"]) == (
        "60f1f14972a7", "25d65bbba36f"
    )  # fmt: skip
    assert (up["reward_ablation_spec_id"], up["reward_contract_id"]) == (
        "f5aff8f1e8c2", "37158c261364"
    )  # fmt: skip


def test_robustness_and_startup_definitions_are_the_frozen_ones(spec: dict[str, Any]) -> None:
    scenarios = spec["primary_matrix"]["scenarios"]
    assert set(scenarios) == set(ROBUSTNESS_SCENARIOS)
    for name, frozen in ROBUSTNESS_SCENARIOS.items():
        s = scenarios[name]
        assert s["version"] == "robustness-v1"
        assert s["capacity_jitter_fraction"] == frozen.capacity_jitter_fraction
        assert s["telemetry_delay_ticks"] == frozen.telemetry_delay_ticks
    assert scenarios["nominal"]["dynamics_seeds"] == [0]
    assert scenarios["delayed-telemetry"]["dynamics_seeds"] == [0]
    assert scenarios["capacity-jitter"]["dynamics_seeds"] == [0, 1, 2, 3, 4]
    assert scenarios["combined-robustness"]["dynamics_seeds"] == [0, 1, 2, 3, 4]
    assert scenarios["capacity-jitter"]["capacity_jitter_fraction"] == 0.1
    startup = spec["startup_matrix"]["scenarios"]
    assert startup == startup_scenarios()
    assert startup["startup-delay-jitter"]["seed_pairs_dynamics_startup"] == [
        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]
    ]  # fmt: skip
    assert startup["combined-startup-robustness"]["seed_pairs_dynamics_startup"] == [
        [s, s] for s in range(5)
    ]
    assert startup["combined-startup-robustness"]["telemetry_delay_ticks"] == 1
    model = spec["startup_matrix"]["startup_model"]
    assert (model["id"], model["multipliers"], model["probabilities"]) == (
        "tri-point-multiplicative-v1", [0.5, 1.0, 1.5], [0.25, 0.5, 0.25]
    )  # fmt: skip


# --- the matrix ----------------------------------------------------------------------------------


def test_expected_case_counts(spec: dict[str, Any], cases: list[ho.Case]) -> None:
    ho.require_expected_counts(spec, cases)
    assert ho.case_counts(cases) == {"primary": 264, "startup": 220, "main": 484, "replay": 15}
    per = Counter((c.kind, c.workload_id, c.scenario) for c in cases if c.kind != "replay")
    for workload in ("azure-test-993600", "azure-test-1166400"):
        assert per["primary", workload, "nominal"] == 11
        assert per["primary", workload, "capacity-jitter"] == 55
        assert per["primary", workload, "delayed-telemetry"] == 11
        assert per["primary", workload, "combined-robustness"] == 55
        assert per["startup", workload, "startup-delay-jitter"] == 55
        assert per["startup", workload, "combined-startup-robustness"] == 55
    assert len({c.case_id for c in cases}) == len(cases)
    assert [c.index for c in cases] == list(range(len(cases)))


def test_count_guard_rejects_a_changed_matrix(spec: dict[str, Any], cases: list[ho.Case]) -> None:
    with pytest.raises(ValueError, match="case counts"):
        ho.require_expected_counts(spec, cases[:-1])
    edited = json.loads(json.dumps(spec))
    edited["primary_matrix"]["scenarios"]["nominal"]["dynamics_seeds"] = [0, 1]
    with pytest.raises(ValueError, match="case counts"):
        ho.require_expected_counts(edited, ho.generate_cases(edited))


def test_no_pseudo_replication(cases: list[ho.Case]) -> None:
    realizations = Counter(
        (c.kind, c.variant.variant_id, c.workload_id, c.slice["slice_id"] if c.slice else None,
         c.scenario, c.dynamics_seed, c.startup_delay_seed, c.evaluation_seed)
        for c in cases
    )  # fmt: skip
    assert set(realizations.values()) == {1}
    for case in cases:
        if case.capacity_jitter_fraction == 0 and case.startup_delay_seed is None:
            assert case.dynamics_seed == 0
        if case.variant.variant_id != "random-v1":
            assert case.evaluation_seed == 0


def test_matched_seeds_across_controllers(cases: list[ho.Case]) -> None:
    by_condition: dict[tuple[Any, ...], set[str]] = {}
    for case in cases:
        if case.kind == "replay":
            continue
        key = (case.kind, case.workload_id, case.scenario, case.dynamics_seed,
               case.startup_delay_seed)  # fmt: skip
        by_condition.setdefault(key, set()).add(case.variant.variant_id)
    assert all(len(v) == 7 for v in by_condition.values())


def test_case_configs_and_loading_paths(spec: dict[str, Any], cases: list[ho.Case]) -> None:
    for case in cases:
        cfg = case.config(spec)
        assert cfg.action.semantics == DESIRED_REPLICAS_V1
        assert cfg.dynamics.capacity_jitter_fraction == case.capacity_jitter_fraction
        assert cfg.dynamics.telemetry_delay_ticks == case.telemetry_delay_ticks
        assert cfg.dynamics.dynamics_seed == case.dynamics_seed
        assert cfg.dynamics.startup_delay_model == case.startup_delay_model
        assert case.robustness_path == (
            case.telemetry_delay_ticks > 0 or case.startup_delay_seed is not None
        )
        assert cfg.model_copy(update={"dynamics": final_contract_config().dynamics}) == (
            final_contract_config()
        )


def test_development_multiseed_guard_still_rejects_test_workloads() -> None:
    manifest = MultiseedManifest(benchmark_version="v1", variants=tuple(reference_variants()))
    with pytest.raises(ValueError, match="held-out test workload"):
        make_plan(manifest, workload_ids=("azure-test-993600",))
    with pytest.raises(ValueError, match="held-out test workload"):
        make_plan(manifest, workload_ids=("azure-test-1166400",))


# --- replay references ---------------------------------------------------------------------------


def test_replay_slices_are_the_frozen_72_windows(
    spec: dict[str, Any], cases: list[ho.Case]
) -> None:
    replay = spec["replay_references"]
    frozen = s2r.load(BENCH / "live-replay-manifest-v1.json")
    assert replay["manifest_id"] == "7803b71fdfe8" == s2r.content_id(frozen)
    assert replay["source_workload_id"] == "azure-test-1166400"
    assert [(s["slice_id"], s["start_offset_seconds"], s["start_bin"], s["end_bin_exclusive"])
            for s in replay["slices"]] == [
        ("azure-test-1166400-w0-s600-d600", 600.0, 20, 40),
        ("azure-test-1166400-w1-s1500-d600", 1500.0, 50, 70),
        ("azure-test-1166400-w2-s2400-d600", 2400.0, 80, 100),
    ]  # fmt: skip
    for entry, original in zip(replay["slices"], frozen["slices"], strict=True):
        assert entry["slice_checksum"] == original["slice_checksum"]
        assert entry["source_trace_fingerprint"] == original["source_trace_fingerprint"]
        assert entry["schedule_ids"] == [s["schedule_id"] for s in original["schedules"]]
        assert entry["load_seeds"] == [0, 1, 2]
    refs = [c for c in cases if c.kind == "replay"]
    assert len(refs) == 15 and {c.scenario for c in refs} == {"nominal"}
    assert {c.variant.variant_id for c in refs} == set(ho.REPLAY_CONTROLLERS)
    for case in refs:  # one simulator run, mapped to all three load schedules
        assert case.slice is not None and len(case.slice["schedule_ids"]) == 3
        assert case.dynamics_seed == 0 and case.evaluation_seed == 0


def synthetic_source(ticks: int = 120) -> WorkloadTrace:
    rng = np.random.default_rng(7)
    return WorkloadTrace([float(round(x * 30) / 30) for x in rng.uniform(0, 400, ticks)], 30.0)


def rebind_replay(spec: dict[str, Any], source: WorkloadTrace) -> dict[str, Any]:
    """A copy of the spec whose replay slices point at ``source`` (for offline tests)."""
    edited = json.loads(json.dumps(spec))
    fingerprint = ho.trace_fingerprint(source)
    edited["replay_references"]["trace_fingerprint"] = fingerprint
    for entry in edited["replay_references"]["slices"]:
        start, end = entry["start_bin"], entry["end_bin_exclusive"]
        counts = s2r.slice_counts(source, start, end - start)
        entry["source_trace_fingerprint"] = fingerprint
        entry["slice_checksum"] = s2r._sha256({"counts_per_bin": counts, "interval": 30.0})
    return edited


def test_slice_trace_is_exact_and_refuses_other_sources(spec: dict[str, Any]) -> None:
    source = synthetic_source()
    edited = rebind_replay(spec, source)
    entry = edited["replay_references"]["slices"][1]
    sliced = ho.slice_trace(source, entry)
    assert sliced.request_rates == source.request_rates[50:70]
    with pytest.raises(ValueError, match="not the frozen #72 source"):
        ho.slice_trace(source, spec["replay_references"]["slices"][1])
    tampered = dict(entry, slice_checksum="0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        ho.slice_trace(source, tampered)


# --- observation and seasonal-profile offsets ------------------------------------------------


def test_source_window_offsets_episode_progress_only() -> None:
    cfg = final_contract_config()
    source = synthetic_source()
    full = AutoscalingEnv(cfg, source)
    sliced = AutoscalingEnv(
        cfg, WorkloadTrace(source.request_rates[50:70], 30.0), source_window=SourceWindow(50, 120)
    )
    assert sliced.episode_ticks == 20 and full.episode_ticks == 120
    obs, _ = sliced.reset(seed=0)
    progress = sliced.observation_features.index("episode_progress")
    assert obs[progress] == pytest.approx(50 / 120)
    history = [i for i, n in enumerate(sliced.observation_features) if n.startswith("demand")]
    assert not obs[history].any()  # traffic history starts empty at the slice boundary
    constants = ObservationConstants.from_config(cfg)
    for k in range(1, 21):
        obs, _, _, truncated, info = sliced.step(0)
        assert obs[progress] == pytest.approx((50 + k) / 120)
        rates = [s.request_rate for s in sliced.visible_telemetry()]
        expected = build_observation(
            constants,
            request_rates_newest_first=rates,
            latest=Measurement(info["utilization"], info["queued_requests"],
                               info["p95_latency_seconds"], info["infrastructure_cost"]),
            active_replicas=sliced.replica_counts["active_replicas"],
            pending_by_ticks=sliced._pending_buckets(),
            completed_ticks=50 + k,
            episode_ticks=120,
        )  # fmt: skip
        assert np.array_equal(obs, expected)
        assert truncated == (k == 20)
    assert info["tick"] == 19  # the info tick numbering is unchanged


def test_full_episode_behavior_is_unchanged_without_a_window() -> None:
    cfg = final_contract_config()
    source = synthetic_source()
    a = run_episode(AutoscalingEnv(cfg, source), _threshold(cfg), seed=0)
    env = AutoscalingEnv(cfg, source, source_window=None)
    b = run_episode(env, _threshold(cfg), seed=0)
    assert a == b
    obs, _ = env.reset(seed=0)
    assert obs[env.observation_features.index("episode_progress")] == 0.0


def test_source_window_validation() -> None:
    cfg = final_contract_config()
    source = synthetic_source()
    with pytest.raises(ValueError, match="past the end"):
        AutoscalingEnv(cfg, WorkloadTrace(source.request_rates[:30], 30.0),
                       source_window=SourceWindow(100, 120))  # fmt: skip
    with pytest.raises(ValueError, match="source episode"):
        AutoscalingEnv(cfg, WorkloadTrace(source.request_rates[:20], 30.0),
                       source_window=SourceWindow(0, 60))  # fmt: skip
    with pytest.raises(ValueError, match="start_tick"):
        SourceWindow(-1, 120)
    with pytest.raises(ValueError, match="needs"):
        AutoscalingEnv(cfg, WorkloadTrace(source.request_rates[:20], 30.0))


def _threshold(cfg: SimulatorConfig) -> ThresholdController:
    from scalerl.environment import ActionContract

    return ThresholdController(
        low_threshold=0.2, high_threshold=0.6, cooldown_ticks=3, min_replicas=1,
        max_replicas=10, action_contract=ActionContract.from_config(cfg),
    )  # fmt: skip


def profile() -> HistoricalDemandProfile:
    rng = np.random.default_rng(3)
    return HistoricalDemandProfile.from_traces(
        {f"train-{i}": WorkloadTrace(list(rng.uniform(0, 300, 120)), 30.0) for i in range(3)}
    )


def test_offset_profile_reads_source_positions() -> None:
    p = profile()
    for start in (20, 50, 80):
        shifted = ho.offset_profile(p, start)
        for t in range(-3, 130):
            assert shifted.rate_at(t) == (p.rate_at(start + t) if t >= 0 else None)
    assert ho.offset_profile(p, 0) == p
    with pytest.raises(ValueError):
        ho.offset_profile(p, 120)


def test_seasonal_controller_on_a_slice_uses_profile_ticks_from_the_start_bin() -> None:
    cfg = final_contract_config()
    p = profile()
    source = synthetic_source()
    start = 50
    controller = ProactivePredictiveController.from_config(cfg, profile=ho.offset_profile(p, start))
    env = AutoscalingEnv(
        cfg, WorkloadTrace(source.request_rates[start:start + 20], 30.0),
        source_window=SourceWindow(start, 120),
    )  # fmt: skip
    run_episode(env, controller, seed=0)
    used = [r for r in controller.forecasts if r.historical_profile_forecast_rps is not None]
    assert used
    for record in used:
        expected = p.rate_at(start + record.target_tick)
        assert expected is not None and record.profile_level_factor is not None
        assert record.historical_profile_forecast_rps == pytest.approx(
            expected * record.profile_level_factor
        )


# --- fairness ------------------------------------------------------------------------------------


def test_same_realization_for_every_controller_and_order_independent(
    spec: dict[str, Any], cases: list[ho.Case]
) -> None:
    traces = {w: synthetic_source() for w in spec["workloads"]["primary_azure_test"]}
    chosen = [
        c for c in cases
        if c.kind == "startup" and c.scenario == "combined-startup-robustness"
        and c.workload_id == "azure-test-993600" and c.dynamics_seed == 2
        and c.variant.variant_id in ("static-v1", "threshold-v1", "random-v1")
    ]  # fmt: skip

    def evaluate(order: list[ho.Case]) -> dict[str, Any]:
        out = {}
        for case in order:
            metrics, infos, _ = ho.evaluate_case(
                case, spec, traces[case.workload_id], profile=None, learned=_no_learned
            )
            out[case.case_id] = (metrics, [i["capacity_multiplier"] for i in infos])
        return out

    forward = evaluate(chosen)
    backward = evaluate(list(reversed(chosen)))
    assert forward == backward
    multipliers = {tuple(m) for _, m in forward.values()}
    assert len(multipliers) == 1  # same exogenous capacity sequence for every controller


def _no_learned(*_: Any) -> Any:
    raise AssertionError("no learned policy in this test")


# --- end to end: provenance, resume, recovery ----------------------------------------------------


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture(scope="module")
def tiny_ppo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from scalerl.rl import ModelMetadata, save_model_bundle
    from scalerl.training.ppo import PPOHyperparameters, build_ppo

    cfg = final_contract_config()
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
            seed=4,
            total_timesteps=64,
            hyperparameters=hp.as_params(),
            scalerl_version="test",
        ),  # fmt: skip
        compatibility=EnvironmentCompatibility.from_env(env, "v1"),
    )


def offline_inputs(spec: dict[str, Any], bundle: Path) -> tuple[dict[str, Any], ho.Inputs]:
    from scalerl.rl import load_sb3_controller

    source = synthetic_source()
    traces = {"azure-test-993600": synthetic_source(), "azure-test-1166400": source}
    edited = rebind_replay(spec, source)

    def learned(variant: ho.Variant, env: AutoscalingEnv, robustness: bool) -> Any:
        return load_sb3_controller(bundle, env, robustness_evaluation=robustness)

    return edited, ho.Inputs(traces, profile(), learned)


SELECTED = ("threshold-v1", "predictive-seasonal-v1", "ppo-c08-seed4")


def selected(case: ho.Case) -> bool:
    if case.variant.variant_id not in SELECTED:
        return False
    if case.kind == "replay":
        return True
    return case.workload_id == "azure-test-1166400" and case.dynamics_seed in (0, 1)


def test_run_records_provenance_and_is_resumable(
    spec: dict[str, Any], tmp_path: Path, tracking_uri: str, tiny_ppo: Path
) -> None:
    from mlflow import MlflowClient

    edited, inputs = offline_inputs(spec, tiny_ppo)
    out = tmp_path / "out"
    rows = ho.run(edited, inputs, out=out, tracking_uri=tracking_uri, require_clean=False,
                  select=selected, progress=lambda _: None)  # fmt: skip
    expected = [c for c in ho.generate_cases(edited) if selected(c)]
    assert [r["case_id"] for r in rows] == [c.case_id for c in expected]
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("scalerl-heldout-v1")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    assert len(runs) == len(rows)

    identity = ho.spec_id(edited)
    for row in rows:
        tags = client.get_run(row["mlflow_run_id"]).data.tags
        assert set(ho.REQUIRED_TAGS) <= set(tags)
        assert tags["scalerl.heldout_spec_id"] == identity == row["heldout_spec_id"]
        assert tags["scalerl.workload_split"] == "test"
        assert tags["scalerl.action_semantics"] == DESIRED_REPLICAS_V1
        assert tags["scalerl.reward_contract_id"] == "37158c261364"
        assert tags["scalerl.protocol_id"] == "b7bf45c109f1"
        assert tags["scalerl.replay_manifest_id"] == "7803b71fdfe8"
        assert tags["scalerl.controller_manifest_id"] == "553ddf3e512b"
        assert tags["scalerl.robustness_scenario"] == row["scenario"]
        assert tags["scalerl.dynamics_seed"] == str(row["dynamics_seed"])
        assert tags["scalerl.evaluation_seed"] == str(row["evaluation_seed"])
        assert tags["scalerl.step_infos_sha256"] == row["step_infos_sha256"]
        if row["controller"] == "ppo":
            assert tags["scalerl.model_source_run_id"] == PPO_RUN
            assert tags["scalerl.training_seed"] == "4"
            assert tags["scalerl.candidate_id"] == "ppo-c08"
        if row["case_kind"] == "replay":
            assert tags["scalerl.replay_schedule_ids"].count(",") == 2
            assert row["episode_ticks"] == 20.0
        else:
            assert row["episode_ticks"] == 120.0
        if row["startup_delay_seed"] is not None:
            assert tags["scalerl.startup_delay_seed"] == str(row["startup_delay_seed"])
            assert tags["scalerl.startup_delay_model"] == "tri-point-multiplicative-v1"
    ppo = [r for r in rows if r["controller"] == "ppo"]
    assert {r["perturbed_compatibility"] for r in ppo if r["scenario"] == "nominal"} == {""}
    assert {r["perturbed_compatibility"] for r in ppo if r["scenario"] == "delayed-telemetry"} == {
        "telemetry_delay_ticks"
    }
    assert {
        r["perturbed_compatibility"] for r in ppo if r["scenario"] == "combined-startup-robustness"
    } == {"startup_delay_model,telemetry_delay_ticks"}
    for row in rows:
        assert row["replica_seconds"] == pytest.approx(row["infrastructure_cost"] / 0.1 * 3600)
        assert row["sla_violation_count"] == pytest.approx(
            row["sla_violation_rate"] * row["episode_ticks"]
        )

    # resume: complete rows are never rerun
    again = ho.run(edited, inputs, out=out, tracking_uri=tracking_uri, require_clean=False,
                   select=selected, progress=lambda _: None)  # fmt: skip
    assert again == rows
    # a lost/torn local row is recovered from its FINISHED MLflow run, not rerun
    raw = out / "raw-results.jsonl"
    lines = raw.read_text().splitlines()
    raw.write_text("".join(line + "\n" for line in lines[:-3]) + lines[-3][:40])
    recovered = ho.run(edited, inputs, out=out, tracking_uri=tracking_uri, require_clean=False,
                       select=selected, progress=lambda _: None)  # fmt: skip
    assert recovered == rows
    assert len(client.search_runs([experiment.experiment_id], max_results=1000)) == len(rows)

    # a duplicated row is an error, never silently double-counted
    raw.write_text(raw.read_text() + raw.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        ho.run(edited, inputs, out=out, tracking_uri=tracking_uri, require_clean=False,
               select=selected, progress=lambda _: None)  # fmt: skip


def test_run_refuses_a_dirty_tree(
    spec: dict[str, Any], tmp_path: Path, tracking_uri: str, monkeypatch: pytest.MonkeyPatch,
    tiny_ppo: Path,
) -> None:  # fmt: skip
    edited, inputs = offline_inputs(spec, tiny_ppo)
    monkeypatch.setattr(ho, "git_state", lambda: ("abc", True))
    with pytest.raises(RuntimeError, match="clean git tree"):
        ho.run(edited, inputs, out=tmp_path, tracking_uri=tracking_uri, select=selected)


def test_learned_factory_refuses_a_different_bundle(
    spec: dict[str, Any], tmp_path: Path, tiny_ppo: Path
) -> None:
    import shutil

    variant = ho.Variant.from_json(
        next(v for v in spec["controllers"] if v["variant_id"] == "ppo-c08-seed4")
    )
    cache = tmp_path / "models"
    shutil.copytree(tiny_ppo, cache / PPO_RUN / "model")
    load = ho.mlflow_learned_factory(cache, None)
    env = AutoscalingEnv(final_contract_config(), synthetic_source())
    with pytest.raises(ValueError, match="not the frozen #72 artifact"):
        load(variant, env, False)  # the stand-in bundle has no matching training run ID


# --- aggregation and the result artifact ---------------------------------------------------------


def fake_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    identity = ho.spec_id(spec)
    rng = random.Random(0)
    rows = []
    for case in ho.generate_cases(spec):
        metrics = {k: rng.random() for k in ho.METRIC_KEYS}
        rows.append(
            ho.result_row(
                identity,
                case,
                run_id=f"run{case.index:04d}",
                git_sha="a" * 40,
                git_dirty=False,
                metrics=metrics,
                perturbed=(),
                infos_sha256="0" * 64,
                spec=spec,
            )  # fmt: skip
        )
    return rows


def test_aggregation_is_deterministic_and_never_pools(spec: dict[str, Any]) -> None:
    rows = [r for r in fake_rows(spec) if r["case_kind"] != "replay"]
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    assert ho.summarize(rows) == ho.summarize(shuffled)
    summary = ho.summarize(rows)
    groups = {(s["case_kind"], s["workload_or_slice"], s["scenario"]) for s in summary}
    assert len(groups) == 2 * (4 + 2)
    nominal = next(s for s in summary if s["scenario"] == "nominal"
                   and s["controller_variant_id"] == "dqn-c14-seed0")  # fmt: skip
    assert nominal["n"] == 1 and nominal["std"] is None  # no fabricated variance
    jitter = next(s for s in summary if s["scenario"] == "capacity-jitter"
                  and s["controller_variant_id"] == "random-v1"
                  and s["replicate_unit"] == "raw run")  # fmt: skip
    assert (jitter["n"], jitter["n_evaluation_seeds"], jitter["n_dynamics_seeds"]) == (25, 5, 5)
    per_seed = next(s for s in summary if s["scenario"] == "capacity-jitter"
                    and s["replicate_unit"] == "evaluation-seed mean")  # fmt: skip
    assert per_seed["n"] == 5 and per_seed["controller"] == "random"
    startup = next(s for s in summary if s["scenario"] == "startup-delay-jitter"
                   and s["controller_variant_id"] == "threshold-v1")  # fmt: skip
    assert (startup["n"], startup["n_startup_seeds"]) == (5, 5)
    deltas = ho.paired_deltas(rows)
    assert all(d["reference_case_id"].split("|")[1] == "threshold-v1" for d in deltas)
    assert len(deltas) == len(rows) - sum(1 for r in rows if r["controller"] == "threshold")


def test_result_artifact_is_traceable(spec: dict[str, Any]) -> None:
    rows = fake_rows(spec)
    results = ho.build_results(spec, rows, pre_run_sha="b" * 40, spec_at_pre_run=spec)
    assert results["results_id"] == ho.results_id(results)
    assert results["expected_counts"] == results["actual_counts"] == {
        "primary": 264, "startup": 220, "main": 484, "replay": 15
    }  # fmt: skip
    assert results["runs"] == {r["case_id"]: r["mlflow_run_id"] for r in rows}
    assert results["test_data_used_for_training"] is False
    assert results["test_data_used_for_model_selection"] is False
    assert results["post_hoc_protocol_change"] is False
    assert results["declares_winner"] is False
    ids = {i for g in results["groups"].values() for i in g["mlflow_run_ids"]}
    assert ids == {r["mlflow_run_id"] for r in rows if r["case_kind"] != "replay"}
    assert {s["group"] for s in results["summaries"]} == set(results["groups"])
    refs = results["replay_references"]
    assert len(refs) == 15 and all(len(r["maps_to_schedule_ids"]) == 3 for r in refs)
    assert all(r["independent_simulator_replicates"] == 1 for r in refs)
    with pytest.raises(ValueError, match="dirty"):
        ho.build_results(spec, [dict(r, git_dirty=True) for r in rows], pre_run_sha="b",
                         spec_at_pre_run=spec)  # fmt: skip
    with pytest.raises(ValueError, match="one row per predeclared case"):
        ho.build_results(spec, rows[:-1], pre_run_sha="b", spec_at_pre_run=spec)
    other = json.loads(json.dumps(spec))
    other["purpose"] = "changed after results"
    with pytest.raises(ValueError, match="pre-run SHA"):
        ho.build_results(spec, rows, pre_run_sha="b", spec_at_pre_run=other)


def test_static_reference_under_desired_replicas() -> None:
    cfg = final_contract_config()
    env = AutoscalingEnv(cfg, synthetic_source())
    infos = run_episode(env, StaticController(5, cfg.replicas, action_contract=env.action_contract))
    assert max(i["active_replicas"] for i in infos) == 5


def test_pre_run_checklist(spec: dict[str, Any], tmp_path: Path) -> None:
    text = ho.describe_plan(spec, tracking_uri="sqlite:///outputs/mlflow.db", out=tmp_path)
    for expected in (
        SPEC_ID, "264 (expected 264)", "220 (expected 220)", "484 (expected 484)",
        "15 (expected 15)", "azure-test-993600, azure-test-1166400", DQN_RUN, PPO_RUN,
        "scalerl-heldout-v1", "git_dirty=",
    ):  # fmt: skip
        assert expected in text


def test_committed_results_artifact(spec: dict[str, Any]) -> None:
    results = json.loads((BENCH / "heldout-results-v1.json").read_text())
    assert results["results_id"] == ho.results_id(results) == "d9d3fb985f2d"
    assert results["heldout_spec_id"] == SPEC_ID
    pre_run = "ebccda67c72f677c731cb0d6c6c67cfcd21bfc05"
    assert results["pre_run_sha"] == pre_run and results["execution_shas"] == [pre_run]
    assert results["expected_counts"] == results["actual_counts"] == {
        "primary": 264, "startup": 220, "main": 484, "replay": 15
    }  # fmt: skip
    assert sorted(results["runs"]) == sorted(c.case_id for c in ho.generate_cases(spec))
    assert len(set(results["runs"].values())) == 499
    grouped = [i for g in results["groups"].values() for i in g["mlflow_run_ids"]]
    assert sorted(grouped) == sorted(
        run for case, run in results["runs"].items() if not case.startswith("replay|")
    )
    assert results["controllers"] == spec["controllers"]
    assert results["upstream"] == spec["upstream"] and results["reward"] == spec["reward"]
    assert results["superseded_run_ids"] == []
    for key in ("test_data_used_for_training", "test_data_used_for_model_selection",
                "post_hoc_protocol_change", "declares_winner"):  # fmt: skip
        assert results[key] is False
    refs = results["replay_references"]
    assert len(refs) == 15 and all(r["independent_simulator_replicates"] == 1 for r in refs)
    slices = {s["slice_id"]: s["schedule_ids"] for s in spec["replay_references"]["slices"]}
    assert all(r["maps_to_schedule_ids"] == slices[r["slice_id"]] for r in refs)
    assert all(results["runs"][r["case_id"]] == r["mlflow_run_id"] for r in refs)
