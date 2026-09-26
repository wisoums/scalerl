"""Multi-seed controller evaluation and descriptive statistics (#19).

Answers: are controller results stable across the randomness that matters? It
evaluates fixed controllers, including every training seed of each learned
policy, across workloads × ``robustness-v1`` scenarios (#65) × seeds, keeps one
raw row per evaluation case, and summarizes distributions. It does not pick a
winner or a "best seed".

Three kinds of seeds stay separate:

* **training seed**: which learned policy (DQN/PPO seed 0-4 are five models);
* **evaluation seed**: the episode/controller reset seed (matters for Random;
  learned policies are deterministic at evaluation);
* **dynamics seed**: the #65 physical-world realization (capacity jitter). Every
  controller gets the same multiplier sequence for the same workload,
  scenario, and dynamics seed ("same weather"). Scenarios without jitter use
  dynamics seed 0 only: repeating a deterministic run would be
  pseudo-replication, not new evidence.

Development mode is TRAIN/VALIDATION only: held-out test workloads are rejected
when the plan is built, before anything runs. #46 reuses this library through
its own final-evaluation workflow.

Outputs (deterministic order) in the output directory::

    evaluation-plan.json      the resolved, versioned plan (saved before running)
    controller-manifest.json  the controllers/models evaluated
    raw-results.jsonl         one row per case, appended as each case finishes
    raw-results.csv           the same rows, in plan order
    summary.json / .csv       descriptive statistics (variant and controller level)
    paired-deltas.csv         matched differences against Threshold

Command line::

    python -m scalerl.evaluation.multiseed manifest --prep-dir outputs/multiseed-v1/prep \\
        --output outputs/multiseed-v1/controller-manifest.json
    python -m scalerl.evaluation.multiseed run \\
        --manifest outputs/multiseed-v1/controller-manifest.json \\
        --tracking-uri sqlite:///outputs/mlflow.db --output-dir outputs/multiseed-v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers import (
    Controller,
    PredictiveController,
    RandomController,
    StaticController,
    ThresholdController,
)
from scalerl.controllers.predictive import (
    BACKLOG_RECOVERY_TICKS,
    CAPACITY_POLICY,
    FORECAST_METHOD,
)
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation.metrics import EpisodeMetrics
from scalerl.evaluation.robustness import (
    ROBUSTNESS_SCENARIO_VERSION,
    ROBUSTNESS_SCENARIOS,
    RobustnessScenario,
    apply_scenario,
    evaluate_robustness_tracked,
    get_scenario,
)
from scalerl.mlops import SimulatorConfigSource
from scalerl.workloads import WorkloadTrace

MULTISEED_PLAN_VERSION: Final = "multiseed-v1"
MANIFEST_VERSION: Final = "controllers-multiseed-v1"
SUMMARY_METHOD: Final = "descriptive-v1"
STATIC_REFERENCE_TARGET = 5  # static-v1: a fixed half-fleet reference, predeclared, not tuned
TRAINING_SEEDS = (0, 1, 2, 3, 4)
DYNAMICS_SEEDS = (0, 1, 2, 3, 4)
RANDOM_EVALUATION_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_WORKLOADS = ("syn-val-steady-high", "syn-val-ramp-down", "syn-val-bursty")
METRIC_KEYS = tuple(EpisodeMetrics.__dataclass_fields__)
DYNAMICS_KEYS = (
    "mean_capacity_multiplier",
    "min_capacity_multiplier",
    "max_capacity_multiplier",
)
DELTA_KEYS = (
    "sla_violation_rate",
    "normalized_cost",
    "queue_pressure",
    "mean_p95_latency_seconds",
    "churn_rate",
)
ControllerKind = Literal["static", "random", "threshold", "predictive", "dqn", "ppo"]
LEARNED: tuple[ControllerKind, ...] = ("dqn", "ppo")


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def _unique(name: str, values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


def _non_negative_seeds(name: str, seeds: Sequence[int]) -> None:
    _unique(name, seeds)
    if any(seed < 0 for seed in seeds):
        raise ValueError(f"{name} must be non-negative integers")


# --- controller manifest -------------------------------------------------------------------------


class ControllerVariant(_Strict):
    """One exact controller configuration or learned model to evaluate."""

    variant_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9.-]*$")
    controller: ControllerKind
    version: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    evaluation_seeds: tuple[int, ...] = (0,)
    # Learned models only.
    training_seed: int | None = None
    training_workload_id: str | None = None
    training_run_id: str | None = None
    model_artifact_uri: str | None = None
    hyperparameter_source: str | None = None
    compatibility: dict[str, JsonValue] | None = None
    # Development evidence and provenance (tuning study/trial, validation metrics).
    lineage: dict[str, JsonValue] = Field(default_factory=dict)
    validation_metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.evaluation_seeds:
            raise ValueError("evaluation_seeds must not be empty")
        _non_negative_seeds("evaluation_seeds", self.evaluation_seeds)
        learned = (
            self.training_seed,
            self.training_workload_id,
            self.training_run_id,
            self.model_artifact_uri,
        )
        if self.controller in LEARNED:
            if any(value is None for value in learned):
                raise ValueError(
                    f"{self.variant_id}: learned variants need training_seed, "
                    "training_workload_id, training_run_id, and model_artifact_uri"
                )
            assert self.training_seed is not None and self.training_workload_id is not None
            if self.training_seed < 0:
                raise ValueError("training_seed must be non-negative")
            split = load_benchmark_manifest().get(self.training_workload_id).split
            if split != "train":
                raise ValueError(
                    f"{self.variant_id}: model trained on a {split} workload; learned "
                    "variants must come from a TRAIN workload"
                )
        elif any(value is not None for value in learned):
            raise ValueError(f"{self.variant_id}: only dqn/ppo variants carry model lineage")
        for workload_id in _as_strings(self.lineage.get("validation_workload_ids")):
            if load_benchmark_manifest().get(workload_id).split == "test":
                raise ValueError(
                    f"{self.variant_id}: selected on held-out test workload {workload_id!r}"
                )
        return self


def _as_strings(value: JsonValue) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


class ControllerManifest(_Strict):
    """Every controller/model a multi-seed plan may evaluate (evidence for #72)."""

    manifest_version: Literal["controllers-multiseed-v1"] = MANIFEST_VERSION
    benchmark_version: str
    variants: tuple[ControllerVariant, ...] = Field(min_length=1)
    notes: str = (
        "No training seed is selected here: every seed is evaluated. #72 freezes a "
        "deployment artifact later under a rule declared before held-out/live evaluation."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        _unique("variant_id", [variant.variant_id for variant in self.variants])
        return self

    def get(self, variant_id: str) -> ControllerVariant:
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        raise ValueError(f"unknown controller variant {variant_id!r}")

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ControllerManifest:
        return cls.model_validate_json(Path(path).read_text())


def reference_variants() -> list[ControllerVariant]:
    """Static and Random references and the fixed #63 Predictive policy."""
    return [
        ControllerVariant(
            variant_id="static-v1",
            controller="static",
            version="static-v1",
            params={"target_replicas": STATIC_REFERENCE_TARGET},
            lineage={"note": "fixed half-fleet cost reference; predeclared, not tuned"},
        ),
        ControllerVariant(
            variant_id="random-v1",
            controller="random",
            version="random-v1",
            evaluation_seeds=RANDOM_EVALUATION_SEEDS,
            lineage={"note": "uniformly random actions; a sanity check, not a target"},
        ),
        ControllerVariant(
            variant_id="predictive-v1",
            controller="predictive",
            version=CAPACITY_POLICY,
            params={
                "history_window_ticks": 4,
                "target_utilization": 0.8,
                "forecast_method": FORECAST_METHOD,
                "capacity_policy": CAPACITY_POLICY,
                "backlog_recovery_ticks": BACKLOG_RECOVERY_TICKS,
            },
            lineage={"note": "fixed #14/#63 policy; not tuned"},
        ),
    ]


def build_manifest(prep_dir: str | Path) -> ControllerManifest:
    """Manifest from ``scripts/multiseed_prepare.sh`` results (no selection happens here)."""
    from scalerl.training.dqn import DQNTrainingResult
    from scalerl.training.ppo import PPOTrainingResult
    from scalerl.tuning.threshold import ThresholdTuningResult

    prep = Path(prep_dir)
    threshold = ThresholdTuningResult.load(prep / "threshold-v1.json")
    variants = reference_variants()
    variants.insert(
        2,
        ControllerVariant(
            variant_id="threshold-v1",
            controller="threshold",
            version=f"{threshold.objective_name}-{threshold.objective_version}",
            params=dict(threshold.selected_params),
            lineage={
                "study_name": threshold.study_name,
                "search_space_version": threshold.search_space_version,
                "selected_trial_number": threshold.selected_trial_number,
                "optuna_storage": threshold.optuna_storage,
                "tuning_workload_ids": list(threshold.tuning_workload_ids),
                "validation_workload_ids": list(threshold.validation_workload_ids),
                "selected_mlflow_run_ids": list(threshold.selected_mlflow_run_ids),
            },
            validation_metrics=dict(threshold.selected_validation_metrics),
        ),
    )
    for algorithm, result_type in (("dqn", DQNTrainingResult), ("ppo", PPOTrainingResult)):
        tuning = json.loads((prep / f"{algorithm}-tuning-v1.json").read_text())
        tuning_lineage: dict[str, JsonValue] = {
            key: tuning[key]
            for key in (
                "study_name",
                "objective_name",
                "objective_version",
                "search_space_version",
                "sampler",
                "sampler_seed",
                "optuna_storage",
                "selected_trial_number",
                "selected_training_run_id",
                "validation_workload_ids",
                "timesteps",
            )
        }
        tuning_lineage["selected_validation_metrics"] = tuning["selected_validation_metrics"]
        for seed in TRAINING_SEEDS:
            result: Any = result_type.load(prep / f"{algorithm}-seed{seed}.json")
            variants.append(
                ControllerVariant(
                    variant_id=f"{algorithm}-seed{seed}",
                    controller=algorithm,  # type: ignore[arg-type]
                    version=getattr(result, f"{algorithm}_config_version"),
                    params=result.hyperparameters.as_params(),
                    training_seed=result.seed,
                    training_workload_id=result.training_workload_id,
                    training_run_id=result.training_run_id,
                    model_artifact_uri=result.model_artifact_uri,
                    hyperparameter_source=result.hyperparameter_source,
                    compatibility=result.compatibility.model_dump(mode="json"),
                    lineage={
                        "tuning": tuning_lineage,
                        "timesteps": result.timesteps,
                        "validation_workload_ids": list(result.validation_workload_ids),
                        "validation_run_ids": list(result.validation_run_ids),
                    },
                    validation_metrics=dict(result.validation_metrics),
                )
            )
    return ControllerManifest(
        benchmark_version=load_benchmark_manifest().version, variants=tuple(variants)
    )


# --- evaluation plan ---------------------------------------------------------------------------


class EvaluationPlan(_Strict):
    """Everything that defines a multi-seed evaluation; saved before it runs."""

    plan_version: Literal["multiseed-v1"] = MULTISEED_PLAN_VERSION
    benchmark_version: str
    workload_ids: tuple[str, ...] = Field(min_length=1)
    scenario_names: tuple[str, ...] = Field(min_length=1)
    robustness_version: Literal["robustness-v1"] = ROBUSTNESS_SCENARIO_VERSION
    dynamics_seeds: tuple[int, ...] = Field(min_length=1)
    variants: tuple[ControllerVariant, ...] = Field(min_length=1)
    simulator_config: SimulatorConfig = Field(default_factory=SimulatorConfig)
    simulator_config_source: SimulatorConfigSource = "default"
    calibration_workload_ids: tuple[str, ...] = ()
    calibration_note: str | None = None
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    summary_method: Literal["descriptive-v1"] = SUMMARY_METHOD

    @model_validator(mode="after")
    def _check(self) -> Self:
        _unique("workload_ids", self.workload_ids)
        _unique("scenario_names", self.scenario_names)
        _non_negative_seeds("dynamics_seeds", self.dynamics_seeds)
        _unique("variants", [variant.variant_id for variant in self.variants])
        manifest = load_benchmark_manifest()
        if self.benchmark_version != manifest.version:
            raise ValueError(f"plan benchmark {self.benchmark_version!r} is not {manifest.version}")
        for workload_id in self.workload_ids:
            split = manifest.get(workload_id).split
            if split == "test":
                raise ValueError(
                    f"multi-seed development evaluation uses train/validation workloads only; "
                    f"{workload_id!r} is a held-out test workload"
                )
        for name in self.scenario_names:
            scenario = get_scenario(name)
            if scenario.version != self.robustness_version:
                raise ValueError(f"scenario {name!r} is not {self.robustness_version}")
        for workload_id in self.calibration_workload_ids:
            if manifest.get(workload_id).split == "test":
                raise ValueError(f"calibration may not cite held-out workload {workload_id!r}")
        if self.simulator_config.dynamics != SimulatorConfig().dynamics:
            raise ValueError("the base simulator config must be nominal; scenarios set dynamics")
        return self

    @property
    def plan_id(self) -> str:
        """Stable short hash of the plan's content."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> EvaluationPlan:
        return cls.model_validate_json(Path(path).read_text())


def make_plan(
    manifest: ControllerManifest,
    *,
    workload_ids: Sequence[str] = DEFAULT_WORKLOADS,
    scenario_names: Sequence[str] | None = None,
    dynamics_seeds: Sequence[int] = DYNAMICS_SEEDS,
    variant_ids: Sequence[str] | None = None,
    simulator_config: SimulatorConfig | None = None,
    simulator_config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
) -> EvaluationPlan:
    """The canonical v1 plan unless narrowed; validated before anything runs."""
    variants = (
        manifest.variants
        if variant_ids is None
        else tuple(manifest.get(variant_id) for variant_id in variant_ids)
    )
    return EvaluationPlan(
        benchmark_version=manifest.benchmark_version,
        workload_ids=tuple(workload_ids),
        scenario_names=tuple(scenario_names or ROBUSTNESS_SCENARIOS),
        dynamics_seeds=tuple(dynamics_seeds),
        variants=variants,
        simulator_config=simulator_config or SimulatorConfig(),
        simulator_config_source=simulator_config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
    )


# --- cases ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationCase:
    """One episode: a controller variant on a workload, scenario, and seed pair."""

    index: int
    variant: ControllerVariant
    workload_id: str
    scenario: RobustnessScenario
    dynamics_seed: int
    evaluation_seed: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.variant.variant_id}|{self.workload_id}|{self.scenario.name}"
            f"|dyn{self.dynamics_seed}|eval{self.evaluation_seed}"
        )


def dynamics_seeds_for(scenario: RobustnessScenario, seeds: Sequence[int]) -> tuple[int, ...]:
    """Declared dynamics seeds for stochastic scenarios; seed 0 only otherwise.

    Without capacity jitter the dynamics seed changes nothing, so extra seeds
    would only duplicate one deterministic episode (pseudo-replication).
    """
    if scenario.capacity_jitter_fraction == 0:
        return (0,)
    return tuple(seeds)


def generate_cases(plan: EvaluationPlan) -> list[EvaluationCase]:
    """Every case of ``plan`` in a deterministic order."""
    cases: list[EvaluationCase] = []
    for workload_id in plan.workload_ids:
        for name in plan.scenario_names:
            scenario = get_scenario(name)
            for dynamics_seed in dynamics_seeds_for(scenario, plan.dynamics_seeds):
                for variant in plan.variants:
                    for evaluation_seed in variant.evaluation_seeds:
                        cases.append(
                            EvaluationCase(
                                len(cases),
                                variant,
                                workload_id,
                                scenario,
                                dynamics_seed,
                                evaluation_seed,
                            )
                        )
    return cases


# --- controllers ---------------------------------------------------------------------------------


ModelLocator = Callable[[ControllerVariant], Path]


def make_controller(
    variant: ControllerVariant,
    env: AutoscalingEnv,
    *,
    model_dir: ModelLocator | None = None,
) -> Controller:
    """A fresh controller for one case (no state carried between cases).

    Learned bundles are loaded against ``env``: strictly, or, under a delayed
    telemetry scenario, through the explicit robustness-only path (#65).
    """
    config = env.config
    params = variant.params
    if variant.controller == "static":
        return StaticController(int(_number(params["target_replicas"])), config.replicas)
    if variant.controller == "random":
        return RandomController()  # seeded by the runner's reset(evaluation_seed)
    if variant.controller == "threshold":
        return ThresholdController(
            low_threshold=_number(params["low_threshold"]),
            high_threshold=_number(params["high_threshold"]),
            cooldown_ticks=int(_number(params["cooldown_ticks"])),
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
        )
    if variant.controller == "predictive":
        return PredictiveController.from_config(
            config,
            history_window_ticks=int(_number(params["history_window_ticks"])),
            target_utilization=_number(params["target_utilization"]),
        )
    from scalerl.rl import load_sb3_controller

    if model_dir is None:
        raise ValueError(f"{variant.variant_id}: learned variants need a model locator")
    return load_sb3_controller(
        model_dir(variant),
        env,
        robustness_evaluation=config.dynamics.telemetry_delay_ticks > 0,
    )


def _number(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def mlflow_model_locator(cache_dir: Path, tracking_uri: str | None) -> ModelLocator:
    """Download each learned model bundle once (``runs:/<id>/model``) into ``cache_dir``."""

    def locate(variant: ControllerVariant) -> Path:
        target = cache_dir / variant.variant_id
        bundle = target / "model"
        if not (bundle / "model.zip").is_file():
            from mlflow import MlflowClient

            assert variant.training_run_id is not None
            target.mkdir(parents=True, exist_ok=True)
            MlflowClient(tracking_uri).download_artifacts(
                variant.training_run_id, "model", str(target)
            )
        return bundle

    return locate


# --- running -------------------------------------------------------------------------------------


RAW_FILE = "raw-results.jsonl"


def result_row(
    plan: EvaluationPlan,
    case: EvaluationCase,
    *,
    split: str,
    metrics: Mapping[str, float],
    dynamics: Mapping[str, float],
    mlflow_run_id: str,
    perturbed: Sequence[str],
) -> dict[str, Any]:
    variant = case.variant
    return {
        "plan_version": plan.plan_version,
        "plan_id": plan.plan_id,
        "benchmark_version": plan.benchmark_version,
        "case_index": case.index,
        "case_id": case.case_id,
        "controller": variant.controller,
        "controller_variant_id": variant.variant_id,
        "controller_version": variant.version,
        "training_seed": variant.training_seed,
        "training_run_id": variant.training_run_id,
        "model_artifact_uri": variant.model_artifact_uri,
        "hyperparameter_source": variant.hyperparameter_source,
        "workload_id": case.workload_id,
        "workload_split": split,
        "robustness_scenario": case.scenario.name,
        "robustness_version": case.scenario.version,
        "capacity_jitter_fraction": case.scenario.capacity_jitter_fraction,
        "telemetry_delay_ticks": case.scenario.telemetry_delay_ticks,
        "evaluation_seed": case.evaluation_seed,
        "dynamics_seed": case.dynamics_seed,
        "mlflow_run_id": mlflow_run_id,
        "perturbed_compatibility": ",".join(perturbed),
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        **{key: float(dynamics[key]) for key in DYNAMICS_KEYS},
    }


def case_tags(plan: EvaluationPlan, case: EvaluationCase) -> dict[str, str]:
    tags = {
        "scalerl.evaluation_plan_version": plan.plan_version,
        "scalerl.evaluation_plan_id": plan.plan_id,
        "scalerl.evaluation_case_id": case.case_id,
        "scalerl.controller_variant_id": case.variant.variant_id,
        "scalerl.controller_version": case.variant.version,
    }
    if case.variant.training_seed is not None:
        tags["scalerl.training_seed"] = str(case.variant.training_seed)
    if case.variant.model_artifact_uri is not None:
        tags["scalerl.model_artifact_uri"] = case.variant.model_artifact_uri
    return tags


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Completed rows; a torn last line from an interrupted write is ignored."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _repair_torn_tail(path: Path) -> None:
    """Drop a partial last line left by an interrupted write, so appends stay line-aligned."""
    if not path.exists():
        return
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        path.write_bytes(content[: content.rfind(b"\n") + 1])


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover_from_mlflow(
    plan: EvaluationPlan,
    case: EvaluationCase,
    *,
    split: str,
    tracking_uri: str | None,
    experiment_name: str,
) -> dict[str, Any] | None:
    """A finished MLflow run of this case whose local row was lost, if any."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`scalerl.evaluation_plan_id` = '{plan.plan_id}' and "
            f"tags.`scalerl.evaluation_case_id` = '{case.case_id}' and "
            "attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    metrics = run.data.metrics
    return result_row(
        plan,
        case,
        split=split,
        metrics=metrics,
        dynamics={key: metrics[f"dynamics.{key}"] for key in DYNAMICS_KEYS},
        mlflow_run_id=run.info.run_id,
        perturbed=_as_list(run.data.tags.get("scalerl.robustness.perturbed_compatibility")),
    )


def _as_list(value: str | None) -> list[str]:
    return [item for item in (value or "").split(",") if item]


def run_plan(
    plan: EvaluationPlan,
    *,
    output_dir: str | Path,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-multiseed",
    azure_csv_path: str | Path | None = None,
    resume: bool = False,
    model_dir: ModelLocator | None = None,
    traces: Mapping[str, WorkloadTrace] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every case once (one MLflow ``evaluate`` run each) and write all outputs.

    The plan is saved before anything runs. Rows are appended as each case
    finishes; with ``resume=True`` completed cases (by case ID, locally or as a
    finished MLflow run of this plan) are skipped, so no case is counted twice.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_path, raw_path = out / "evaluation-plan.json", out / RAW_FILE
    if plan_path.exists():
        if not resume:
            raise FileExistsError(f"{plan_path} exists; use resume to continue that plan")
        if EvaluationPlan.load(plan_path) != plan:
            raise ValueError("the existing evaluation plan differs; use a new output directory")
    elif raw_path.exists():
        raise FileExistsError(f"{raw_path} exists without a plan")
    plan.save(plan_path)
    ControllerManifest(benchmark_version=plan.benchmark_version, variants=plan.variants).save(
        out / "controller-manifest.json"
    )

    manifest = load_benchmark_manifest()
    entries = {workload_id: manifest.get(workload_id) for workload_id in plan.workload_ids}
    if traces is None:
        traces = build_workloads(list(entries.values()), azure_csv_path=azure_csv_path)
    locate = model_dir or mlflow_model_locator(out / "models", tracking_uri)
    _repair_torn_tail(raw_path)
    completed = {row["case_id"] for row in read_rows(raw_path) if row["plan_id"] == plan.plan_id}

    cases = generate_cases(plan)
    for case in cases:
        if case.case_id in completed:
            continue
        entry = entries[case.workload_id]
        recovered = (
            _recover_from_mlflow(
                plan,
                case,
                split=entry.split,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
            )
            if resume
            else None
        )
        if recovered is not None:
            _append_row(raw_path, recovered)
            completed.add(case.case_id)
            continue
        scenario_config = apply_scenario(
            plan.simulator_config, case.scenario, dynamics_seed=case.dynamics_seed
        )
        env = AutoscalingEnv(scenario_config, traces[case.workload_id], plan.reward_weights)
        controller = make_controller(case.variant, env, model_dir=locate)
        result, run_id = evaluate_robustness_tracked(
            controller,
            traces[case.workload_id],
            controller_name=case.variant.controller,
            entry=entry,
            scenario=case.scenario,
            dynamics_seed=case.dynamics_seed,
            evaluation_seed=case.evaluation_seed,
            config=plan.simulator_config,
            base_config_source=plan.simulator_config_source,
            calibration_workload_ids=plan.calibration_workload_ids,
            calibration_note=plan.calibration_note,
            reward_weights=plan.reward_weights,
            hyperparameters=case.variant.params,
            model_source_run_id=case.variant.training_run_id,
            extra_tags=case_tags(plan, case),
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"multiseed-{case.case_id.replace('|', '-')}",
        )
        row = result_row(
            plan,
            case,
            split=entry.split,
            metrics=result.metrics.as_metrics(),
            dynamics={key: getattr(result.dynamics, key) for key in DYNAMICS_KEYS},
            mlflow_run_id=run_id,
            perturbed=getattr(controller, "perturbed_compatibility", ()),
        )
        _append_row(raw_path, row)
        completed.add(case.case_id)
        if progress is not None:
            progress(f"[{len(completed)}/{len(cases)}] {case.case_id}")

    rows = [row for row in read_rows(raw_path) if row["plan_id"] == plan.plan_id]
    write_outputs(rows, out)
    return sorted(rows, key=lambda row: row["case_index"])


# --- statistics ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSummary:
    """Descriptive statistics of one metric over a group (``descriptive-v1``)."""

    n: int
    mean: float
    median: float
    std: float | None  # sample standard deviation; None with a single value
    minimum: float
    maximum: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def describe(values: Sequence[float]) -> MetricSummary:
    """Mean, median, sample SD (n ≥ 2), min/max, and quartiles (inclusive, linear).

    Quartiles use linear interpolation between order statistics
    (``statistics.quantiles(method="inclusive")``, NumPy's default); a single
    value is its own median and quartiles, with no standard deviation.
    """
    if not values:
        raise ValueError("cannot describe an empty group")
    data = [float(value) for value in values]
    if len(data) == 1:
        q1 = q3 = data[0]
    else:
        q1, _, q3 = statistics.quantiles(data, n=4, method="inclusive")
    return MetricSummary(
        n=len(data),
        mean=statistics.fmean(data),
        median=statistics.median(data),
        std=statistics.stdev(data) if len(data) > 1 else None,
        minimum=min(data),
        maximum=max(data),
        q1=q1,
        q3=q3,
    )


SUMMARY_FIELDS = (
    "level",
    "controller",
    "controller_variant_id",
    "workload_id",
    "robustness_scenario",
    "metric",
    "replicate_unit",
    "n",
    "n_raw_runs",
    "n_training_seeds",
    "n_evaluation_seeds",
    "n_dynamics_seeds",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "q1",
    "q3",
    "iqr",
    "mlflow_run_ids",
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Two-level descriptive summaries per workload and scenario (never pooled across them).

    * ``variant`` level: one controller configuration or learned model, over
      its raw runs (dynamics seeds × evaluation seeds). A deterministic
      controller in a scenario without jitter has one run and zero spread.
    * ``controller`` level: across the controller's variants (for DQN/PPO its
      five training seeds), each represented by its variant-level mean. This is
      the primary spread for learned policies; a single-variant rule controller
      has n = 1 here (its dynamics spread is in the variant row).

    Every row names the raw MLflow runs it aggregates.
    """
    summaries: list[dict[str, Any]] = []
    by_variant: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row["controller"],
            row["controller_variant_id"],
            row["workload_id"],
            row["robustness_scenario"],
        )
        by_variant.setdefault(key, []).append(row)

    variant_means: dict[tuple[str, str, str], list[tuple[str, dict[str, float]]]] = {}
    for (controller, variant_id, workload, scenario), group in sorted(by_variant.items()):
        means = {}
        for metric in METRIC_KEYS:
            summary = describe([row[metric] for row in group])
            means[metric] = summary.mean
            summaries.append(
                _summary_row(
                    "variant",
                    controller,
                    variant_id,
                    workload,
                    scenario,
                    metric,
                    "raw run",
                    summary,
                    group,
                )
            )
        variant_means.setdefault((controller, workload, scenario), []).append((variant_id, means))

    for (controller, workload, scenario), members in sorted(variant_means.items()):
        group = [
            row
            for row in rows
            if (row["controller"], row["workload_id"], row["robustness_scenario"])
            == (controller, workload, scenario)
        ]
        for metric in METRIC_KEYS:
            summary = describe([means[metric] for _, means in members])
            summaries.append(
                _summary_row(
                    "controller",
                    controller,
                    "*",
                    workload,
                    scenario,
                    metric,
                    "variant mean",
                    summary,
                    group,
                )
            )
    return summaries


def _summary_row(
    level: str,
    controller: str,
    variant_id: str,
    workload: str,
    scenario: str,
    metric: str,
    unit: str,
    summary: MetricSummary,
    group: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "level": level,
        "controller": controller,
        "controller_variant_id": variant_id,
        "workload_id": workload,
        "robustness_scenario": scenario,
        "metric": metric,
        "replicate_unit": unit,
        "n": summary.n,
        "n_raw_runs": len(group),
        "n_training_seeds": len({r["training_seed"] for r in group} - {None}),
        "n_evaluation_seeds": len({r["evaluation_seed"] for r in group}),
        "n_dynamics_seeds": len({r["dynamics_seed"] for r in group}),
        "mean": summary.mean,
        "median": summary.median,
        "std": summary.std,
        "min": summary.minimum,
        "max": summary.maximum,
        "q1": summary.q1,
        "q3": summary.q3,
        "iqr": summary.iqr,
        "mlflow_run_ids": " ".join(sorted(r["mlflow_run_id"] for r in group)),
    }


def paired_deltas(
    rows: Sequence[Mapping[str, Any]], reference: str = "threshold-v1"
) -> list[dict[str, Any]]:
    """``metric − reference metric`` for rows sharing workload, scenario, and dynamics seed.

    Descriptive only (no ranking or significance). Rows whose dynamics
    realization has no reference run are never paired.
    """
    anchors = {
        (row["workload_id"], row["robustness_scenario"], row["dynamics_seed"]): row
        for row in rows
        if row["controller_variant_id"] == reference
    }
    deltas = []
    for row in sorted(rows, key=lambda r: r["case_index"]):
        if row["controller_variant_id"] == reference:
            continue
        anchor = anchors.get((row["workload_id"], row["robustness_scenario"], row["dynamics_seed"]))
        if anchor is None:
            continue
        deltas.append(
            {
                "controller": row["controller"],
                "controller_variant_id": row["controller_variant_id"],
                "training_seed": row["training_seed"],
                "evaluation_seed": row["evaluation_seed"],
                "workload_id": row["workload_id"],
                "robustness_scenario": row["robustness_scenario"],
                "dynamics_seed": row["dynamics_seed"],
                "reference_variant_id": reference,
                "mlflow_run_id": row["mlflow_run_id"],
                "reference_mlflow_run_id": anchor["mlflow_run_id"],
                **{f"delta_{key}": row[key] - anchor[key] for key in DELTA_KEYS},
            }
        )
    return deltas


# --- outputs ----------------------------------------------------------------------


def write_outputs(rows: Sequence[Mapping[str, Any]], out: Path) -> None:
    """Raw CSV (plan order), summaries, and paired deltas; deterministic content."""
    ordered = sorted(rows, key=lambda row: row["case_index"])
    if not ordered:
        return
    _write_csv(out / "raw-results.csv", ordered, list(ordered[0]))
    summaries = summarize(ordered)
    _write_csv(out / "summary.csv", summaries, list(SUMMARY_FIELDS))
    _write_text(
        out / "summary.json",
        json.dumps({"summary_method": SUMMARY_METHOD, "rows": summaries}, indent=2, allow_nan=False)
        + "\n",
    )
    deltas = paired_deltas(ordered)
    if deltas:
        _write_csv(out / "paired-deltas.csv", deltas, list(deltas[0]))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    os.chmod(handle.name, 0o644)  # temp files are private by default
    os.replace(handle.name, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    return "" if value is None else value


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as handle:
        handle.write(content)
    os.chmod(handle.name, 0o644)  # temp files are private by default
    os.replace(handle.name, path)
    return path


# --- command line ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed controller evaluation (#19).")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest_cmd = commands.add_parser("manifest", help="build the controller manifest")
    manifest_cmd.add_argument("--prep-dir", type=Path, default=Path("outputs/multiseed-v1/prep"))
    manifest_cmd.add_argument(
        "--output", type=Path, default=Path("outputs/multiseed-v1/controller-manifest.json")
    )

    run_cmd = commands.add_parser("run", help="evaluate a plan (train/validation only)")
    run_cmd.add_argument("--manifest", type=Path, required=True)
    run_cmd.add_argument("--workload", action="append", dest="workloads")
    run_cmd.add_argument("--scenario", action="append", dest="scenarios")
    run_cmd.add_argument("--dynamics-seed", action="append", type=int, dest="dynamics_seeds")
    run_cmd.add_argument("--variant", action="append", dest="variants")
    run_cmd.add_argument("--azure-csv", type=Path)
    run_cmd.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    run_cmd.add_argument("--experiment-name", default="scalerl-multiseed")
    run_cmd.add_argument("--output-dir", type=Path, default=Path("outputs/multiseed-v1"))
    run_cmd.add_argument("--resume", action="store_true")

    summarize_cmd = commands.add_parser("summarize", help="recompute summaries from raw rows")
    summarize_cmd.add_argument("--output-dir", type=Path, default=Path("outputs/multiseed-v1"))

    args = parser.parse_args(argv)
    if args.command == "manifest":
        path = build_manifest(args.prep_dir).save(args.output)
        print(f"controller manifest written to {path}")
        return 0
    if args.command == "summarize":
        plan = EvaluationPlan.load(args.output_dir / "evaluation-plan.json")
        rows = [r for r in read_rows(args.output_dir / RAW_FILE) if r["plan_id"] == plan.plan_id]
        write_outputs(rows, args.output_dir)
        print(f"{len(rows)} rows summarized in {args.output_dir}")
        return 0

    try:
        plan = make_plan(
            ControllerManifest.load(args.manifest),
            workload_ids=args.workloads or DEFAULT_WORKLOADS,
            scenario_names=args.scenarios,
            dynamics_seeds=args.dynamics_seeds or DYNAMICS_SEEDS,
            variant_ids=args.variants,
        )
    except ValueError as error:  # includes pydantic validation errors
        parser.error(f"invalid evaluation plan: {error}")
    rows = run_plan(
        plan,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        azure_csv_path=args.azure_csv,
        resume=args.resume,
        progress=print,
    )
    print(f"plan {plan.plan_id}: {len(rows)} cases; outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
