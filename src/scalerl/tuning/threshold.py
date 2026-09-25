"""Tune the threshold autoscaler's thresholds and cooldown (#13).

Uses the shared study infrastructure (``run_study``, #54): Optuna's grid
sampler runs the exact 18-point grid, each workload evaluation is one #17
MLflow run through ``TrialContext.track``, and tuning workloads are limited to
train/validation. Only ``low_threshold``, ``high_threshold``, and
``cooldown_ticks`` are tuned; the simulator configuration is given and never
changed.

Selection rule ``threshold-sla-first`` v1, over VALIDATION workloads only:

1. lowest mean SLA violation rate (also the Optuna objective);
2. lowest mean normalized infrastructure cost;
3. lowest mean normalized queue pressure;
4. lowest mean churn rate;
5. smallest ``(high_threshold, low_threshold, cooldown_ticks)`` as an exact tie-break.

Reward is logged as a secondary metric and never used for selection.

Run a synthetic train/validation study locally::

    python -m scalerl.tuning.threshold \\
        --storage sqlite:///outputs/threshold-optuna.db \\
        --tracking-uri sqlite:///outputs/mlflow.db \\
        --output outputs/threshold-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from scalerl.benchmarks import (
    AzureWorkload,
    BenchmarkManifest,
    WorkloadEntry,
    build_workloads,
    load_benchmark_manifest,
)
from scalerl.controllers import ThresholdController
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import RunSpec, SimulatorConfigSource
from scalerl.tuning.spec import (
    GridValue,
    StudySpec,
    require_tuning_workloads,
    safe_storage_label,
)
from scalerl.tuning.study import RUN_IDS_ATTR, TrialContext, run_study
from scalerl.workloads import WorkloadTrace

THRESHOLD_GRID: dict[str, list[GridValue]] = {
    "high_threshold": [0.6, 0.7, 0.8],
    "low_threshold": [0.2, 0.3],
    "cooldown_ticks": [3, 5, 10],
}
OBJECTIVE_NAME = "threshold-sla-first"
OBJECTIVE_VERSION = "v1"
SEARCH_SPACE_VERSION = "threshold-grid-v1"
SELECTED_TRIAL_ATTR = "selected_trial_number"
SELECTION_KEYS = (
    "validation_sla_violation_rate",
    "validation_normalized_cost",
    "validation_queue_pressure",
    "validation_churn_rate",
)
_PARAM_ORDER = ("high_threshold", "low_threshold", "cooldown_ticks")


# --- workload metrics --------------------------------------------------------------------


def evaluate_episode(env: AutoscalingEnv, controller: ThresholdController) -> EpisodeMetrics:
    """Run one full episode and summarize it with the shared evaluation metrics."""
    return evaluate_controller_episode(env, controller).metrics


# --- selection -------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialAggregate:
    """Validation aggregates and parameters of one completed trial."""

    number: int
    params: Mapping[str, float | int]
    validation_sla_violation_rate: float
    validation_normalized_cost: float
    validation_queue_pressure: float
    validation_churn_rate: float


def select_trial(candidates: Sequence[TrialAggregate]) -> TrialAggregate:
    """Apply ``threshold-sla-first`` v1: SLA, then cost, queue, churn, then parameters."""
    if not candidates:
        raise ValueError("no completed trials to select from")
    return min(
        candidates,
        key=lambda c: (
            c.validation_sla_violation_rate,
            c.validation_normalized_cost,
            c.validation_queue_pressure,
            c.validation_churn_rate,
            tuple(c.params[name] for name in _PARAM_ORDER),
        ),
    )


class ThresholdTuningResult(BaseModel):
    """Reproducible record of a threshold study and its selected configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_name: str
    objective_name: str
    objective_version: str
    search_space_version: str
    benchmark_version: str
    optuna_storage: str
    tuning_workload_ids: tuple[str, ...]
    validation_workload_ids: tuple[str, ...]
    trial_count: int
    selected_trial_number: int
    selected_params: dict[str, float | int]
    selected_validation_metrics: dict[str, float]
    selected_mlflow_run_ids: tuple[str, ...]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> ThresholdTuningResult:
        return cls.model_validate_json(Path(path).read_text())


# --- study -------------------------------------------------------------------------------


def default_tuning_workload_ids(manifest: BenchmarkManifest | None = None) -> tuple[str, ...]:
    """The benchmark's synthetic train and validation workloads."""
    manifest = manifest or load_benchmark_manifest()
    return tuple(entry.id for entry in manifest.tuning if not isinstance(entry, AzureWorkload))


def threshold_study_spec(
    workload_ids: Sequence[str],
    *,
    name: str = "threshold-v1",
    storage: str | None = None,
    sampler_seed: int = 0,
    identity_context: Mapping[str, Any] | None = None,
) -> StudySpec:
    """The exact 18-point grid study over ``workload_ids``.

    ``identity_context`` (simulator config, provenance, workload fingerprint)
    is stored with the study, so it cannot be resumed with different inputs.
    """
    return StudySpec(
        name=name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        direction="minimize",
        tuning_workload_ids=tuple(workload_ids),
        sampler="grid",
        sampler_seed=sampler_seed,
        grid=THRESHOLD_GRID,
        storage=storage,
        identity_context=dict(identity_context or {}),
    )


def run_threshold_study(
    *,
    config: SimulatorConfig | None = None,
    config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    workload_ids: Sequence[str] | None = None,
    azure_csv_path: str | Path | None = None,
    study_name: str = "threshold-v1",
    storage: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-threshold-tuning",
) -> ThresholdTuningResult:
    """Evaluate the exact grid on train/validation workloads and select a configuration.

    Workload traces are built once (Azure entries in one pass over
    ``azure_csv_path``); each trial gets fresh environments. Resuming a
    persisted study runs only grid points not yet evaluated; resuming it with a
    different simulator config, provenance, or workload data is refused.
    """
    config = config or SimulatorConfig()
    ids = tuple(workload_ids) if workload_ids is not None else default_tuning_workload_ids()
    entries = require_tuning_workloads(ids)
    validation = tuple(entry for entry in entries if entry.split == "validation")
    if not validation:
        raise ValueError("threshold tuning needs at least one validation workload for selection")
    # Validate simulator-config provenance before any study state is created.
    _run_spec(entries[0], {}, config, config_source, calibration_workload_ids, calibration_note)
    traces = build_workloads(entries, azure_csv_path=azure_csv_path)
    spec = threshold_study_spec(
        ids,
        name=study_name,
        storage=storage,
        identity_context={
            "simulator_config": config.model_dump(mode="json"),
            "simulator_config_source": config_source,
            "calibration_workload_ids": list(calibration_workload_ids),
            "reward_weights": RewardWeights().model_dump(mode="json"),
            "workload_fingerprint": workload_fingerprint(traces),
        },
    )

    def objective(context: TrialContext) -> float:
        params = {
            name: context.trial.suggest_categorical(name, values)
            for name, values in THRESHOLD_GRID.items()
        }
        results = {
            entry.id: _evaluate_workload(
                context,
                entry,
                traces[entry.id],
                params,
                config,
                config_source,
                calibration_workload_ids,
                calibration_note,
                tracking_uri,
                experiment_name,
            )
            for entry in entries
        }
        for split in ("train", "validation"):
            split_results = [results[e.id] for e in entries if e.split == split]
            if split_results:
                for key, value in _aggregate(split_results).items():
                    context.trial.set_user_attr(f"{split}_{key}", value)
        return float(context.trial.user_attrs["validation_sla_violation_rate"])

    study = run_study(spec, objective)
    return _select_and_record(study, spec, entries, validation)


def _evaluate_workload(
    context: TrialContext,
    entry: WorkloadEntry,
    trace: WorkloadTrace,
    params: Mapping[str, Any],
    config: SimulatorConfig,
    config_source: SimulatorConfigSource,
    calibration_workload_ids: Sequence[str],
    calibration_note: str | None,
    tracking_uri: str | None,
    experiment_name: str,
) -> EpisodeMetrics:
    run_spec = _run_spec(
        entry, params, config, config_source, calibration_workload_ids, calibration_note
    )
    controller = ThresholdController(
        low_threshold=params["low_threshold"],
        high_threshold=params["high_threshold"],
        cooldown_ticks=params["cooldown_ticks"],
        min_replicas=config.replicas.min_replicas,
        max_replicas=config.replicas.max_replicas,
    )
    with context.track(run_spec, tracking_uri=tracking_uri, experiment_name=experiment_name) as run:
        metrics = evaluate_episode(AutoscalingEnv(config, trace), controller)
        run.log_metrics(metrics.as_metrics())
    return metrics


def _run_spec(
    entry: WorkloadEntry,
    params: Mapping[str, Any],
    config: SimulatorConfig,
    config_source: SimulatorConfigSource,
    calibration_workload_ids: Sequence[str],
    calibration_note: str | None,
) -> RunSpec:
    return RunSpec(
        run_kind="tune",
        controller="threshold",
        workload_id=entry.id,
        workload_split=entry.split,
        simulator_config=config,
        simulator_config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=RewardWeights(),
        seed=0,
        hyperparameters=dict(params),
    )


def workload_fingerprint(traces: Mapping[str, WorkloadTrace]) -> str:
    """SHA-256 of the exact traces a study evaluates (catches changed source data)."""
    payload = {
        workload_id: [trace.control_interval_seconds, list(trace.request_rates)]
        for workload_id, trace in traces.items()
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _aggregate(results: Sequence[EpisodeMetrics]) -> dict[str, float]:
    return {
        "sla_violation_rate": statistics.fmean(r.sla_violation_rate for r in results),
        "normalized_cost": statistics.fmean(r.normalized_cost for r in results),
        "queue_pressure": statistics.fmean(r.queue_pressure for r in results),
        "churn_rate": statistics.fmean(r.churn_rate for r in results),
        "mean_p95_latency_seconds": statistics.fmean(r.mean_p95_latency_seconds for r in results),
        "episode_reward": statistics.fmean(r.episode_reward for r in results),
    }


def _select_and_record(
    study: Any,
    spec: StudySpec,
    entries: Sequence[WorkloadEntry],
    validation: Sequence[WorkloadEntry],
) -> ThresholdTuningResult:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    aggregates = [
        TrialAggregate(
            number=trial.number,
            params=trial.params,
            **{key: float(trial.user_attrs[key]) for key in SELECTION_KEYS},
        )
        for trial in completed
    ]
    selected = select_trial(aggregates)
    trial = next(t for t in completed if t.number == selected.number)
    study.set_user_attr(SELECTED_TRIAL_ATTR, selected.number)
    study.set_user_attr("selected_params", dict(selected.params))
    study.set_user_attr("selection_rule", f"{OBJECTIVE_NAME}-{OBJECTIVE_VERSION}")
    return ThresholdTuningResult(
        study_name=spec.name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        optuna_storage=safe_storage_label(spec.storage),
        tuning_workload_ids=tuple(entry.id for entry in entries),
        validation_workload_ids=tuple(entry.id for entry in validation),
        trial_count=len(completed),
        selected_trial_number=selected.number,
        selected_params=dict(selected.params),
        selected_validation_metrics={
            key: float(value)
            for key, value in trial.user_attrs.items()
            if key.startswith("validation_")
        },
        selected_mlflow_run_ids=tuple(trial.user_attrs[RUN_IDS_ATTR]),
    )


# --- command line ------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the threshold autoscaler (#13).")
    parser.add_argument("--storage", default="sqlite:///outputs/threshold-optuna.db")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-threshold-tuning")
    parser.add_argument("--study-name", default="threshold-v1")
    parser.add_argument("--output", type=Path, default=Path("outputs/threshold-v1.json"))
    parser.add_argument(
        "--workload",
        action="append",
        dest="workloads",
        help="train/validation workload ID (repeatable); defaults to synthetic train/validation",
    )
    parser.add_argument("--azure-csv", type=Path, help="local Azure trace for Azure workloads")
    parser.add_argument("--simulator-config", type=Path, help="SimulatorConfig JSON file")
    parser.add_argument(
        "--config-source",
        choices=["default", "predeclared", "calibrated_train_validation"],
        default="default",
    )
    parser.add_argument("--calibration-workload", action="append", default=[])
    parser.add_argument("--calibration-note")
    args = parser.parse_args(argv)

    source: Literal["default", "predeclared", "calibrated_train_validation"] = args.config_source
    config = (
        SimulatorConfig.model_validate_json(args.simulator_config.read_text())
        if args.simulator_config
        else SimulatorConfig()
    )
    _prepare_sqlite_directory(args.storage)
    result = run_threshold_study(
        config=config,
        config_source=source,
        calibration_workload_ids=args.calibration_workload,
        calibration_note=args.calibration_note,
        workload_ids=args.workloads,
        azure_csv_path=args.azure_csv,
        study_name=args.study_name,
        storage=args.storage,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    path = result.save(args.output)
    print(f"selected trial {result.selected_trial_number}: {result.selected_params}")
    print(f"validation metrics: {result.selected_validation_metrics}")
    print(f"result written to {path}")
    return 0


def _prepare_sqlite_directory(storage: str) -> None:
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        Path(storage.removeprefix(prefix)).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sys.exit(main())
