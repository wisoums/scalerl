"""Tests for multi-seed evaluation (#19). Train/validation workloads and tiny models only."""

import csv
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import PredictiveController, ThresholdController
from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig
from scalerl.evaluation import multiseed
from scalerl.evaluation.multiseed import (
    ControllerManifest,
    ControllerVariant,
    EvaluationPlan,
    build_manifest,
    describe,
    dynamics_seeds_for,
    generate_cases,
    make_controller,
    make_plan,
    mlflow_model_locator,
    paired_deltas,
    read_rows,
    reference_variants,
    run_plan,
    summarize,
)
from scalerl.evaluation.robustness import (
    CAPACITY_JITTER,
    COMBINED_ROBUSTNESS,
    DELAYED_TELEMETRY,
    NOMINAL,
    apply_scenario,
    evaluate_robustness,
)
from scalerl.training.dqn import DQNHyperparameters, train_dqn
from scalerl.training.ppo import PPOHyperparameters, train_ppo
from scalerl.tuning.threshold import ThresholdTuningResult

WORKLOAD = "syn-val-bursty"
TINY_DQN = DQNHyperparameters(buffer_size=200, learning_starts=16, batch_size=16, net_arch=(8,))
TINY_PPO = PPOHyperparameters(n_steps=32, batch_size=16, n_epochs=1, net_arch=(8,))


# --- fixtures: a tiny but real preparation directory -----------------------------------------


@pytest.fixture(scope="module")
def prep(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("prep")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    ThresholdTuningResult(
        study_name="threshold-v1",
        objective_name="threshold-sla-first",
        objective_version="v1",
        search_space_version="threshold-grid-v1",
        benchmark_version="v1",
        optuna_storage="sqlite:///threshold.db",
        tuning_workload_ids=("syn-train-spike", "syn-val-bursty"),
        validation_workload_ids=("syn-val-bursty",),
        trial_count=18,
        selected_trial_number=14,
        selected_params={"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3},
        selected_validation_metrics={"validation_sla_violation_rate": 0.25},
        selected_mlflow_run_ids=("run-a", "run-b"),
    ).save(root / "threshold-v1.json")
    for algorithm, train, hp, steps in (
        ("dqn", train_dqn, TINY_DQN, 32),
        ("ppo", train_ppo, TINY_PPO, 32),
    ):
        (root / f"{algorithm}-tuning-v1.json").write_text(
            json.dumps(
                {
                    "study_name": f"{algorithm}-v1",
                    "objective_name": f"{algorithm}-sla-first",
                    "objective_version": "v1",
                    "search_space_version": f"{algorithm}-search-v1",
                    "sampler": "tpe",
                    "sampler_seed": 42,
                    "optuna_storage": "sqlite:///x.db",
                    "selected_trial_number": 3,
                    "selected_training_run_id": "tune-run",
                    "validation_workload_ids": [WORKLOAD],
                    "timesteps": steps,
                    "selected_validation_metrics": {"validation_sla_violation_rate": 0.3},
                    "selected_hyperparameters": hp.model_dump(mode="json"),
                }
            )
        )
        for seed in multiseed.TRAINING_SEEDS:
            options: dict[str, Any] = {
                "workload_id": "syn-train-spike",
                "validation_workload_ids": (WORKLOAD,),
                "hyperparameters": hp,
                "hyperparameter_source": f"optuna:{algorithm}-v1#trial3",
                "timesteps": steps,
                "seed": seed,
                "tracking_uri": uri,
                "log_interval": 32,
            }
            if algorithm == "ppo":
                options["checkpoint_interval"] = None
            train(**options).save(root / f"{algorithm}-seed{seed}.json")
    return root, uri


@pytest.fixture(scope="module")
def manifest(prep: tuple[Path, str]) -> ControllerManifest:
    return build_manifest(prep[0])


@pytest.fixture
def tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prep: tuple[Path, str]
) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield prep[1]  # the store that holds the tiny training runs and their models


def small_plan(manifest: ControllerManifest, **kwargs: Any) -> EvaluationPlan:
    options: dict[str, Any] = {
        "workload_ids": (WORKLOAD,),
        "dynamics_seeds": (0, 1),
        "variant_ids": (
            "threshold-v1",
            "predictive-v1",
            "static-v1",
            "random-v1",
            "dqn-seed0",
            "ppo-seed1",
        ),
    }
    return make_plan(manifest, **{**options, **kwargs})


# --- manifest ---------------------------------------------------------------------------------


def test_manifest_keeps_every_training_seed_with_lineage(manifest: ControllerManifest) -> None:
    ids = [variant.variant_id for variant in manifest.variants]

    assert ids == [
        "static-v1",
        "random-v1",
        "threshold-v1",
        "predictive-v1",
        *(f"dqn-seed{seed}" for seed in range(5)),
        *(f"ppo-seed{seed}" for seed in range(5)),
    ]
    dqn = [v for v in manifest.variants if v.controller == "dqn"]
    assert [v.training_seed for v in dqn] == [0, 1, 2, 3, 4]
    assert len({v.training_run_id for v in dqn}) == 5
    for variant in dqn:
        assert variant.model_artifact_uri == f"runs:/{variant.training_run_id}/model"
        assert variant.hyperparameter_source == "optuna:dqn-v1#trial3"
        assert variant.training_workload_id == "syn-train-spike"
        assert variant.compatibility is not None
        assert variant.lineage["tuning"]["study_name"] == "dqn-v1"  # type: ignore[index]
        assert variant.params == TINY_DQN.as_params()
    threshold = manifest.get("threshold-v1")
    assert threshold.params == {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3}
    assert threshold.lineage["selected_trial_number"] == 14
    assert manifest.get("predictive-v1").params["capacity_policy"] == "forecast-plus-backlog-v1"
    assert manifest.get("static-v1").params == {"target_replicas": 5}
    assert manifest.get("random-v1").evaluation_seeds == (0, 1, 2, 3, 4)
    assert "No training seed is selected" in manifest.notes


def test_manifest_round_trips(manifest: ControllerManifest, tmp_path: Path) -> None:
    assert ControllerManifest.load(manifest.save(tmp_path / "m.json")) == manifest


@pytest.mark.parametrize(
    "fields",
    [
        {"controller": "dqn"},  # learned without lineage
        {"training_seed": 0},  # rule controller with model lineage
        {"evaluation_seeds": ()},
        {"evaluation_seeds": (1, 1)},
        {"evaluation_seeds": (True,)},
        {"variant_id": "Bad ID"},
        {"lineage": {"validation_workload_ids": ["syn-test-spike-hard"]}},
    ],
)
def test_invalid_variants_are_rejected(fields: dict[str, Any]) -> None:
    base: dict[str, Any] = {"variant_id": "threshold-x", "controller": "threshold", "version": "v"}
    with pytest.raises(ValidationError):
        ControllerVariant(**{**base, **fields})


def test_learned_variants_must_come_from_a_train_workload(manifest: ControllerManifest) -> None:
    variant = manifest.get("dqn-seed0")
    with pytest.raises(ValidationError, match="TRAIN workload"):
        ControllerVariant(**{**dict(variant), "training_workload_id": "syn-val-bursty"})


def test_duplicate_variant_ids_are_rejected(manifest: ControllerManifest) -> None:
    with pytest.raises(ValidationError, match="unique"):
        ControllerManifest(benchmark_version="v1", variants=(manifest.variants[0],) * 2)


# --- plan -------------------------------------------------------------------------------------


def test_canonical_plan_is_versioned_and_complete(manifest: ControllerManifest) -> None:
    plan = make_plan(manifest)

    assert plan.plan_version == "multiseed-v1" and plan.summary_method == "descriptive-v1"
    assert plan.workload_ids == ("syn-val-steady-high", "syn-val-ramp-down", "syn-val-bursty")
    assert plan.scenario_names == (
        "nominal",
        "capacity-jitter",
        "delayed-telemetry",
        "combined-robustness",
    )
    assert plan.dynamics_seeds == (0, 1, 2, 3, 4)
    assert plan.robustness_version == "robustness-v1"
    assert len(plan.variants) == 14
    assert len(plan.plan_id) == 12 and plan.plan_id == make_plan(manifest).plan_id


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"workload_ids": ("syn-test-spike-hard",)}, "held-out test"),
        ({"workload_ids": (WORKLOAD, "azure-test-993600")}, "held-out test"),
        ({"workload_ids": (WORKLOAD, WORKLOAD)}, "unique"),
        ({"scenario_names": ("nominal", "nominal")}, "unique"),
        ({"scenario_names": ("chaos",)}, "unknown robustness scenario"),
        ({"dynamics_seeds": (0, 0)}, "unique"),
        ({"dynamics_seeds": (-1,)}, "non-negative"),
        ({"dynamics_seeds": (True,)}, "valid integer"),
        ({"dynamics_seeds": (1.0,)}, "valid integer"),
        ({"variant_ids": ("dqn-seed9",)}, "unknown controller variant"),
        ({"calibration_workload_ids": ("syn-test-spike-hard",)}, "held-out"),
    ],
)
def test_invalid_plans_are_rejected_before_anything_runs(
    manifest: ControllerManifest, kwargs: dict[str, Any], error: str
) -> None:
    with pytest.raises((ValueError, ValidationError), match=error):
        make_plan(manifest, **kwargs)


def test_base_config_must_be_nominal(manifest: ControllerManifest) -> None:
    jittered = apply_scenario(SimulatorConfig(), CAPACITY_JITTER)
    with pytest.raises(ValidationError, match="must be nominal"):
        make_plan(manifest, simulator_config=jittered, simulator_config_source="predeclared")


def test_cli_rejects_test_workloads_before_evaluating(
    manifest: ControllerManifest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(multiseed, "run_plan", lambda *a, **k: pytest.fail("ran"))
    path = manifest.save(tmp_path / "m.json")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        multiseed.main(
            ["run", "--manifest", str(path), "--workload", "syn-test-seasonal-shifted"]
            + ["--output-dir", str(out)]
        )
    assert "held-out test workload" in capsys.readouterr().err
    assert not out.exists()


# --- cases ------------------------------------------------------------------------------------


def test_cases_are_the_declared_product_without_pseudo_replication(
    manifest: ControllerManifest,
) -> None:
    plan = make_plan(manifest)
    cases = generate_cases(plan)

    # Per workload: nominal and delayed get dynamics seed 0 only; jitter scenarios get 5.
    assert dynamics_seeds_for(NOMINAL, (0, 1, 2, 3, 4)) == (0,)
    assert dynamics_seeds_for(DELAYED_TELEMETRY, (0, 1, 2, 3, 4)) == (0,)
    assert dynamics_seeds_for(COMBINED_ROBUSTNESS, (0, 1, 2, 3, 4)) == (0, 1, 2, 3, 4)
    realizations = 1 + 5 + 1 + 5
    runs_per_realization = 13 + 5  # 13 single-seed variants + Random's 5 evaluation seeds
    assert len(cases) == 3 * realizations * runs_per_realization
    assert len({case.case_id for case in cases}) == len(cases)
    nominal = [c for c in cases if c.scenario.name in ("nominal", "delayed-telemetry")]
    assert {c.dynamics_seed for c in nominal} == {0}
    assert [c.index for c in cases] == list(range(len(cases)))
    assert [c.case_id for c in generate_cases(plan)] == [c.case_id for c in cases]
    assert cases[0].case_id == "static-v1|syn-val-steady-high|nominal|dyn0|eval0"


# --- controllers and #65 integration ----------------------------------------------------------


def test_factories_give_fresh_controllers(
    manifest: ControllerManifest, prep: tuple[Path, str]
) -> None:
    env = AutoscalingEnv(SimulatorConfig(), build_workload(load_benchmark_manifest().get(WORKLOAD)))
    for variant_id in ("threshold-v1", "predictive-v1", "static-v1", "random-v1"):
        first = make_controller(manifest.get(variant_id), env)
        second = make_controller(manifest.get(variant_id), env)
        assert first is not second
    threshold = make_controller(manifest.get("threshold-v1"), env)
    assert isinstance(threshold, ThresholdController) and threshold.cooldown_ticks == 3
    predictive = make_controller(manifest.get("predictive-v1"), env)
    assert isinstance(predictive, PredictiveController) and predictive.request_history == ()


def test_evaluation_seed_reaches_the_controller_but_not_the_dynamics(
    manifest: ControllerManifest,
) -> None:
    trace = build_workload(load_benchmark_manifest().get(WORKLOAD))
    env = AutoscalingEnv(SimulatorConfig(), trace)

    def run(evaluation_seed: int) -> Any:
        controller = make_controller(manifest.get("random-v1"), env)
        return evaluate_robustness(
            controller,
            trace,
            scenario=CAPACITY_JITTER,
            dynamics_seed=2,
            evaluation_seed=evaluation_seed,
        )

    a, a_again, b = run(0), run(0), run(1)

    actions = [[i["requested_action"] for i in r.infos] for r in (a, a_again, b)]
    assert actions[0] == actions[1] != actions[2]  # Random is reproducible per evaluation seed
    multipliers = [[i["capacity_multiplier"] for i in r.infos] for r in (a, b)]
    assert multipliers[0] == multipliers[1]  # the evaluation seed never moves the weather
    assert (a.evaluation_seed, b.evaluation_seed) == (0, 1)


def test_learned_bundles_evaluate_strictly_or_via_the_robustness_path(
    manifest: ControllerManifest, tracking: str, tmp_path: Path
) -> None:
    locate = mlflow_model_locator(tmp_path / "models", tracking)
    trace = build_workload(load_benchmark_manifest().get(WORKLOAD))
    for variant_id in ("dqn-seed2", "ppo-seed4"):
        variant = manifest.get(variant_id)
        nominal = make_controller(
            variant, AutoscalingEnv(SimulatorConfig(), trace), model_dir=locate
        )
        assert nominal.perturbed_compatibility == ()  # type: ignore[attr-defined]
        delayed_env = AutoscalingEnv(apply_scenario(SimulatorConfig(), DELAYED_TELEMETRY), trace)
        delayed = make_controller(variant, delayed_env, model_dir=locate)
        assert delayed.perturbed_compatibility == ("telemetry_delay_ticks",)  # type: ignore[attr-defined]
        wrong = AutoscalingEnv(
            apply_scenario(
                SimulatorConfig(replicas=ReplicaConfig(max_replicas=12)), DELAYED_TELEMETRY
            ),
            trace,
        )
        with pytest.raises(ValueError, match="max_replicas"):
            make_controller(variant, wrong, model_dir=locate)


# --- running, raw results, MLflow -------------------------------------------------------------


def test_run_plan_writes_every_case_once_with_full_lineage(
    manifest: ControllerManifest, tracking: str, tmp_path: Path
) -> None:
    plan = small_plan(manifest)
    out = tmp_path / "multiseed"

    rows = run_plan(plan, output_dir=out, tracking_uri=tracking)

    cases = generate_cases(plan)
    assert len(rows) == len(cases) == (5 + 5) * (1 + 2 + 1 + 2)
    assert [row["case_id"] for row in rows] == [case.case_id for case in cases]
    assert EvaluationPlan.load(out / "evaluation-plan.json") == plan
    assert ControllerManifest.load(out / "controller-manifest.json").variants == plan.variants
    row = next(
        r for r in rows if r["case_id"] == f"dqn-seed0|{WORKLOAD}|combined-robustness|dyn1|eval0"
    )
    variant = manifest.get("dqn-seed0")
    assert row["training_seed"] == 0 and row["training_run_id"] == variant.training_run_id
    assert row["model_artifact_uri"] == variant.model_artifact_uri
    assert (row["capacity_jitter_fraction"], row["telemetry_delay_ticks"]) == (0.1, 1)
    assert row["perturbed_compatibility"] == "telemetry_delay_ticks"
    assert row["workload_split"] == "validation"
    assert {f"{k}" for k in multiseed.METRIC_KEYS} <= set(row)
    assert 0.9 <= row["min_capacity_multiplier"] <= row["max_capacity_multiplier"] <= 1.1

    run = MlflowClient(tracking).get_run(row["mlflow_run_id"])
    tags, metrics = run.data.tags, run.data.metrics
    assert tags["scalerl.controller"] == "dqn"  # the controller name stays the algorithm
    assert tags["scalerl.controller_variant_id"] == "dqn-seed0"
    assert tags["scalerl.training_seed"] == "0" and tags["scalerl.evaluation_seed"] == "0"
    assert tags["scalerl.model_source_run_id"] == variant.training_run_id
    assert tags["scalerl.evaluation_plan_id"] == plan.plan_id
    assert tags["scalerl.evaluation_plan_version"] == "multiseed-v1"
    assert tags["scalerl.robustness_scenario"] == "combined-robustness"
    assert tags["scalerl.dynamics_seed"] == "1"
    assert tags["scalerl.robustness.perturbed_compatibility"] == "telemetry_delay_ticks"
    assert metrics["sla_violation_rate"] == row["sla_violation_rate"]
    assert metrics["dynamics.mean_capacity_multiplier"] == row["mean_capacity_multiplier"]

    # Matched dynamics: every controller saw the same weather for a realization.
    for scenario, seed in (("capacity-jitter", 1), ("combined-robustness", 0)):
        realization = [
            r for r in rows if (r["robustness_scenario"], r["dynamics_seed"]) == (scenario, seed)
        ]
        assert len({r["mean_capacity_multiplier"] for r in realization}) == 1
        assert len({r["controller_variant_id"] for r in realization}) == 6

    # Deterministic raw CSV, summaries, and deltas exist.
    with (out / "raw-results.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [r["case_id"] for r in csv_rows] == [r["case_id"] for r in rows]
    assert float(csv_rows[5]["sla_violation_rate"]) == rows[5]["sla_violation_rate"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["summary_method"] == "descriptive-v1"
    assert (out / "paired-deltas.csv").exists()


def test_resume_skips_completed_cases_and_never_duplicates(
    manifest: ControllerManifest, tracking: str, tmp_path: Path
) -> None:
    plan = small_plan(manifest, variant_ids=("threshold-v1", "static-v1"))
    out = tmp_path / "resume"
    full = run_plan(plan, output_dir=out, tracking_uri=tracking)
    raw = out / "raw-results.jsonl"
    lines = raw.read_text().splitlines()
    # Simulate an interruption: two rows lost, the last one half-written.
    raw.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][: len(lines[-2]) // 2])

    with pytest.raises(FileExistsError, match="resume"):
        run_plan(plan, output_dir=out, tracking_uri=tracking)
    client = MlflowClient(tracking)
    experiment = client.get_experiment_by_name("scalerl-multiseed")
    assert experiment is not None
    before = len(client.search_runs([experiment.experiment_id], max_results=10_000))
    resumed = run_plan(plan, output_dir=out, tracking_uri=tracking, resume=True)
    after = len(client.search_runs([experiment.experiment_id], max_results=10_000))

    assert [r["case_id"] for r in resumed] == [r["case_id"] for r in full]
    assert len({r["case_id"] for r in read_rows(raw)}) == len(read_rows(raw)) == len(full)
    # The two lost cases were recovered from their finished MLflow runs, not re-run.
    assert after == before
    assert [r["mlflow_run_id"] for r in resumed] == [r["mlflow_run_id"] for r in full]
    with pytest.raises(ValueError, match="differs"):
        run_plan(small_plan(manifest, variant_ids=("static-v1",)), output_dir=out, resume=True)


# --- statistics -------------------------------------------------------------------------------


def test_describe_matches_hand_computed_values() -> None:
    summary = describe([1.0, 2.0, 3.0, 4.0, 10.0])

    assert (summary.n, summary.mean, summary.median) == (5, 4.0, 3.0)
    assert summary.std == pytest.approx(math.sqrt(12.5))  # sample SD
    assert (summary.minimum, summary.maximum) == (1.0, 10.0)
    assert (summary.q1, summary.q3, summary.iqr) == (2.0, 4.0, 2.0)
    assert (summary.q1, summary.q3) == tuple(np.percentile([1, 2, 3, 4, 10], [25, 75]))


def test_single_and_constant_values_are_reported_honestly() -> None:
    single = describe([0.25])
    constant = describe([0.5, 0.5, 0.5])

    assert single.std is None and (single.median, single.iqr) == (0.25, 0.0)
    assert constant.std == 0.0 and constant.iqr == 0.0
    with pytest.raises(ValueError, match="empty"):
        describe([])


def row(
    variant: str, controller: str, seed: int | None, scenario: str, dyn: int, sla: float
) -> dict[str, Any]:
    base = {key: 0.0 for key in multiseed.METRIC_KEYS}
    return {
        **base,
        "sla_violation_rate": sla,
        "controller": controller,
        "controller_variant_id": variant,
        "training_seed": seed,
        "evaluation_seed": 0,
        "workload_id": WORKLOAD,
        "robustness_scenario": scenario,
        "dynamics_seed": dyn,
        "mlflow_run_id": f"{variant}-{scenario}-{dyn}",
        "case_index": 0,
    }


def test_learned_training_seeds_stay_distinguishable_in_summaries() -> None:
    rows = [
        row(f"dqn-seed{seed}", "dqn", seed, "capacity-jitter", dyn, sla)
        for seed, slas in ((0, (0.1, 0.3)), (1, (0.5, 0.7)))
        for dyn, sla in zip((0, 1), slas, strict=True)
    ] + [row("threshold-v1", "threshold", None, "nominal", 0, 0.2)]

    summaries = [s for s in summarize(rows) if s["metric"] == "sla_violation_rate"]
    by_key = {
        (s["level"], s["controller_variant_id"], s["robustness_scenario"]): s for s in summaries
    }

    seed0 = by_key[("variant", "dqn-seed0", "capacity-jitter")]
    assert (seed0["n"], seed0["mean"], seed0["n_dynamics_seeds"]) == (2, pytest.approx(0.2), 2)
    family = by_key[("controller", "*", "capacity-jitter")]
    # Across training seeds: one value per seed (its dynamics mean), not four pooled rows.
    assert (family["n"], family["n_raw_runs"], family["n_training_seeds"]) == (2, 4, 2)
    assert family["mean"] == pytest.approx(0.4) and family["std"] == pytest.approx(math.sqrt(0.08))
    nominal = by_key[("variant", "threshold-v1", "nominal")]
    assert (nominal["n"], nominal["std"], nominal["n_dynamics_seeds"]) == (1, None, 1)
    assert family["mlflow_run_ids"].count("dqn-seed") == 4


def test_paired_deltas_only_pair_matching_realizations() -> None:
    rows = [
        row("threshold-v1", "threshold", None, "capacity-jitter", 0, 0.2),
        row("threshold-v1", "threshold", None, "capacity-jitter", 1, 0.4),
        row("dqn-seed0", "dqn", 0, "capacity-jitter", 0, 0.5),
        row("dqn-seed0", "dqn", 0, "capacity-jitter", 1, 0.1),
        row("dqn-seed0", "dqn", 0, "capacity-jitter", 2, 0.9),  # no threshold run for dyn 2
        row("dqn-seed0", "dqn", 0, "delayed-telemetry", 0, 0.9),  # no threshold in this scenario
    ]

    deltas = paired_deltas(rows)

    assert [(d["dynamics_seed"], d["robustness_scenario"]) for d in deltas] == [
        (0, "capacity-jitter"),
        (1, "capacity-jitter"),
    ]
    assert [d["delta_sla_violation_rate"] for d in deltas] == [
        pytest.approx(0.3),
        pytest.approx(-0.3),
    ]
    assert all(d["training_seed"] == 0 for d in deltas)


def test_reference_variants_are_fixed_not_tuned() -> None:
    variants = {v.variant_id: v for v in reference_variants()}

    assert variants["static-v1"].params == {"target_replicas": 5}
    assert variants["predictive-v1"].params["history_window_ticks"] == 4
    assert variants["predictive-v1"].params["target_utilization"] == 0.8
