"""Tracked evaluation of the predictive baseline on train/validation workloads (#14).

Each workload is one #17 MLflow run (``run_kind="evaluate"``,
``controller="predictive"``) logging the predeclared controller settings, the
forecast horizon, the capacity policy (forecast plus backlog recovery, #63),
forecast accuracy (scored after the episode), and the shared system metrics.
This development path accepts train/validation workloads only; held-out
evaluation is a separate, later step. Settings are not tuned here.

Run locally::

    python -m scalerl.evaluation.predictive --tracking-uri sqlite:///outputs/mlflow.db
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scalerl.benchmarks import AzureWorkload, build_workloads, load_benchmark_manifest
from scalerl.controllers import PredictiveController
from scalerl.controllers.predictive import (
    BACKLOG_RECOVERY_TICKS,
    CAPACITY_POLICY,
    FORECAST_METHOD,
)
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.evaluation.forecast import ForecastAccuracy, score_forecasts
from scalerl.evaluation.metrics import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import RunSpec, SimulatorConfigSource, start_tracked_run


@dataclass(frozen=True)
class PredictiveEvaluation:
    """One workload's tracked predictive evaluation."""

    workload_id: str
    split: str
    mlflow_run_id: str
    forecast_horizon_ticks: int
    metrics: EpisodeMetrics
    forecast: ForecastAccuracy


def development_workload_ids() -> tuple[str, ...]:
    """The benchmark's synthetic train and validation workloads."""
    manifest = load_benchmark_manifest()
    return tuple(entry.id for entry in manifest.tuning if not isinstance(entry, AzureWorkload))


def evaluate_predictive(
    *,
    config: SimulatorConfig | None = None,
    config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    workload_ids: Sequence[str] | None = None,
    azure_csv_path: str | Path | None = None,
    history_window_ticks: int = 4,
    target_utilization: float = 0.8,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-predictive",
) -> list[PredictiveEvaluation]:
    """Evaluate the predictive controller once per workload, one MLflow run each."""
    config = config or SimulatorConfig()
    manifest = load_benchmark_manifest()
    ids = tuple(workload_ids) if workload_ids is not None else development_workload_ids()
    entries = []
    for workload_id in ids:
        entry = manifest.get(workload_id)
        if entry.split not in ("train", "validation"):
            raise ValueError(
                f"predictive development evaluation uses train/validation workloads only; "
                f"{workload_id!r} is a held-out {entry.split} workload"
            )
        entries.append(entry)
    traces = build_workloads(entries, azure_csv_path=azure_csv_path)

    results = []
    for entry in entries:
        controller = PredictiveController.from_config(
            config,
            history_window_ticks=history_window_ticks,
            target_utilization=target_utilization,
        )
        spec = RunSpec(
            run_kind="evaluate",
            controller="predictive",
            workload_id=entry.id,
            workload_split=entry.split,
            simulator_config=config,
            simulator_config_source=config_source,
            calibration_workload_ids=tuple(calibration_workload_ids),
            calibration_note=calibration_note,
            seed=0,
            evaluation_seeds=(0,),
            hyperparameters={
                "history_window_ticks": history_window_ticks,
                "target_utilization": target_utilization,
                "forecast_method": FORECAST_METHOD,
                "forecast_horizon_ticks": controller.forecast_horizon_ticks,
                "capacity_policy": CAPACITY_POLICY,
                "backlog_recovery_ticks": BACKLOG_RECOVERY_TICKS,
            },
        )
        with start_tracked_run(
            spec, tracking_uri=tracking_uri, experiment_name=experiment_name
        ) as run:
            episode = evaluate_controller_episode(
                AutoscalingEnv(config, traces[entry.id]), controller
            )
            forecast = score_forecasts(controller.forecasts, episode.infos)
            run.log_metrics({**episode.metrics.as_metrics(), **forecast.as_metrics()})
        results.append(
            PredictiveEvaluation(
                entry.id,
                entry.split,
                run.run_id,
                controller.forecast_horizon_ticks,
                episode.metrics,
                forecast,
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the predictive baseline (#14).")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-predictive")
    parser.add_argument("--workload", action="append", dest="workloads")
    parser.add_argument("--azure-csv", type=Path)
    parser.add_argument("--simulator-config", type=Path, help="SimulatorConfig JSON file")
    parser.add_argument(
        "--config-source",
        choices=["default", "predeclared", "calibrated_train_validation"],
        default="default",
    )
    parser.add_argument("--calibration-workload", action="append", default=[])
    parser.add_argument("--calibration-note")
    parser.add_argument("--output", type=Path, help="optional JSON summary")
    args = parser.parse_args(argv)

    config = (
        SimulatorConfig.model_validate_json(args.simulator_config.read_text())
        if args.simulator_config
        else SimulatorConfig()
    )
    results = evaluate_predictive(
        config=config,
        config_source=args.config_source,
        calibration_workload_ids=args.calibration_workload,
        calibration_note=args.calibration_note,
        workload_ids=args.workloads,
        azure_csv_path=args.azure_csv,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    summary: list[dict[str, Any]] = []
    for result in results:
        row = asdict(result)
        summary.append(row)
        print(
            f"{result.workload_id:<26} {result.split:<10} run={result.mlflow_run_id} "
            f"SLA={result.metrics.sla_violation_rate:.3f} "
            f"cost={result.metrics.normalized_cost:.3f} "
            f"MAE={result.forecast.forecast_mae_rps}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
