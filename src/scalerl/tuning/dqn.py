"""Tune DQN hyperparameters on train/validation workloads with Optuna (#15, #54).

Each trial trains one DQN (with the trial's suggested settings) on the declared
TRAIN workload and evaluates the final policy deterministically on the
declared VALIDATION workloads. All of a trial's runs are ``run_kind="tune"``
and go through ``TrialContext.track``, so the study, trial, sampler, seed,
objective/search-space versions, and parameters are on every MLflow run, and
every run ID is on the trial. Held-out test workloads are rejected before any
training starts. Every trial's model is logged (small for v1); retrain the
selected configuration with ``python -m scalerl.training.dqn --hyperparameters
<this result>`` to get the ``train`` run of record.

Search space ``dqn-search-v1`` (predeclared from standard SB3/DQN ranges,
never from held-out results): learning rate, gamma, batch size, target update
interval, exploration fraction, final exploration epsilon, and network
architecture. Replay buffer, learning starts, training frequency, gradient
steps, and initial epsilon stay at the ``dqn-v1`` defaults. Sampler: seeded
TPE (trial-seeded, so resumes are exact), sequential, no pruning.

Selection rule ``dqn-sla-first`` v1, over VALIDATION workloads only:

1. lowest mean SLA violation rate (also the Optuna objective);
2. lowest mean normalized infrastructure cost;
3. lowest mean normalized queue pressure;
4. lowest mean churn rate;
5. lowest trial number (deterministic tie-break).

Episode reward is recorded but never used for selection.

Run a development study locally::

    python -m scalerl.tuning.dqn --train-workload syn-train-bursty \\
        --validation-workload syn-val-bursty --n-trials 20 --timesteps 200000 \\
        --storage sqlite:///outputs/dqn-optuna.db \\
        --tracking-uri sqlite:///outputs/mlflow.db --output outputs/dqn-tuning-v1.json
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.environment import SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.mlops import SimulatorConfigSource
from scalerl.training.dqn import (
    DEFAULT_LOG_INTERVAL,
    DEFAULT_TIMESTEPS,
    DQN_CONFIG_VERSION,
    DQNHyperparameters,
    DQNRunSettings,
    dqn_run_spec,
    require_training_workload,
    require_validation_workloads,
    run_params,
    train_and_validate,
)
from scalerl.tuning.spec import StudySpec, safe_storage_label
from scalerl.tuning.study import RUN_IDS_ATTR, TrialContext, run_study
from scalerl.tuning.threshold import workload_fingerprint

if TYPE_CHECKING:
    import optuna

OBJECTIVE_NAME = "dqn-sla-first"
OBJECTIVE_VERSION = "v1"
SEARCH_SPACE_VERSION = "dqn-search-v1"
DEFAULT_SAMPLER_SEED = 42
DEFAULT_N_TRIALS = 20
SELECTED_TRIAL_ATTR = "selected_trial_number"
TRAINING_RUN_ATTR = "training_run_id"
SELECTION_KEYS = (
    "validation_sla_violation_rate",
    "validation_normalized_cost",
    "validation_queue_pressure",
    "validation_churn_rate",
)
NET_ARCHS: dict[str, tuple[int, ...]] = {
    "64x64": (64, 64),
    "128x128": (128, 128),
    "256x256": (256, 256),
}
GAMMAS = [0.95, 0.98, 0.99, 0.995]
BATCH_SIZES = [32, 64, 128]
TARGET_UPDATE_INTERVALS = [250, 1_000, 5_000]


def suggest_hyperparameters(trial: optuna.Trial) -> DQNHyperparameters:
    """Sample ``dqn-search-v1``; unsearched settings keep their ``dqn-v1`` defaults."""
    net_arch = trial.suggest_categorical("net_arch", list(NET_ARCHS))
    return DQNHyperparameters(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_categorical("gamma", GAMMAS),
        batch_size=trial.suggest_categorical("batch_size", BATCH_SIZES),
        target_update_interval=trial.suggest_categorical(
            "target_update_interval", TARGET_UPDATE_INTERVALS
        ),
        exploration_fraction=trial.suggest_float("exploration_fraction", 0.05, 0.5),
        exploration_final_eps=trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        net_arch=NET_ARCHS[str(net_arch)],
    )


# --- selection -------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialAggregate:
    """Validation aggregates of one completed trial."""

    number: int
    validation_sla_violation_rate: float
    validation_normalized_cost: float
    validation_queue_pressure: float
    validation_churn_rate: float


def select_trial(candidates: Sequence[TrialAggregate]) -> TrialAggregate:
    """Apply ``dqn-sla-first`` v1: SLA, then cost, queue pressure, churn, then trial number."""
    if not candidates:
        raise ValueError("no completed trials to select from")
    return min(
        candidates,
        key=lambda c: (
            c.validation_sla_violation_rate,
            c.validation_normalized_cost,
            c.validation_queue_pressure,
            c.validation_churn_rate,
            c.number,
        ),
    )


class DQNTuningResult(BaseModel):
    """Reproducible record of a DQN study and its selected configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_name: str
    objective_name: str
    objective_version: str
    search_space_version: str
    dqn_config_version: str
    benchmark_version: str
    optuna_storage: str
    sampler: Literal["tpe"]
    sampler_seed: int
    training_workload_id: str
    validation_workload_ids: tuple[str, ...]
    timesteps: int
    seed: int
    trial_count: int
    selected_trial_number: int
    selected_params: dict[str, Any]
    selected_hyperparameters: DQNHyperparameters
    selected_validation_metrics: dict[str, float]
    selected_training_run_id: str
    selected_mlflow_run_ids: tuple[str, ...]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> DQNTuningResult:
        return cls.model_validate_json(Path(path).read_text())


# --- study -------------------------------------------------------------------------------


def run_dqn_study(
    *,
    training_workload_id: str,
    validation_workload_ids: Sequence[str],
    n_trials: int = DEFAULT_N_TRIALS,
    timesteps: int = DEFAULT_TIMESTEPS,
    seed: int = 0,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
    config: SimulatorConfig | None = None,
    config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    azure_csv_path: str | Path | None = None,
    study_name: str = "dqn-v1",
    storage: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-dqn-tuning",
    log_interval: int = DEFAULT_LOG_INTERVAL,
    suggest: Callable[[optuna.Trial], DQNHyperparameters] = suggest_hyperparameters,
) -> DQNTuningResult:
    """Run (or resume) the seeded TPE study and select a configuration.

    Workload splits and simulator-config provenance are checked before the
    study is created or any model is trained. Every trial trains with the same
    ``seed``, so trials differ only in their hyperparameters.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    training_entry = require_training_workload(training_workload_id)
    validation_entries = require_validation_workloads(validation_workload_ids)
    settings = DQNRunSettings(
        timesteps=timesteps,
        seed=seed,
        config=config or SimulatorConfig(),
        config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=RewardWeights(),
        log_interval=log_interval,
    )
    for entry in (training_entry, *validation_entries):  # provenance checks, before training
        dqn_run_spec("tune", entry, settings, run_params(DQNHyperparameters()))
    entries = (training_entry, *validation_entries)
    traces = build_workloads(entries, azure_csv_path=azure_csv_path)
    spec = StudySpec(
        name=study_name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        direction="minimize",
        tuning_workload_ids=tuple(entry.id for entry in entries),
        sampler="tpe",
        sampler_seed=sampler_seed,
        pruner="none",
        storage=storage,
        n_trials=n_trials,
        identity_context={
            "training_workload_id": training_entry.id,
            "timesteps": timesteps,
            "seed": seed,
            "dqn_config_version": DQN_CONFIG_VERSION,
            "simulator_config": settings.config.model_dump(mode="json"),
            "simulator_config_source": config_source,
            "calibration_workload_ids": list(calibration_workload_ids),
            "reward_weights": settings.reward_weights.model_dump(mode="json"),
            "workload_fingerprint": workload_fingerprint(traces),
        },
    )

    def objective(context: TrialContext) -> float:
        hyperparameters = suggest(context.trial)

        def track(run_spec: Any) -> Any:
            return context.track(
                run_spec, tracking_uri=tracking_uri, experiment_name=experiment_name
            )

        outcome = train_and_validate(
            training_entry=training_entry,
            validation_entries=validation_entries,
            traces=traces,
            hyperparameters=hyperparameters,
            settings=settings,
            track=track,
            training_run_kind="tune",
            validation_run_kind="tune",
            extra_params={"hyperparameter_source": f"optuna:{spec.name}#trial{context.number}"},
        )
        context.trial.set_user_attr(TRAINING_RUN_ATTR, outcome.training_run_id)
        context.trial.set_user_attr("hyperparameters", hyperparameters.as_params())
        for key, value in outcome.aggregate.items():
            context.trial.set_user_attr(f"validation_{key}", value)
        return float(outcome.aggregate["sla_violation_rate"])

    study = run_study(spec, objective)
    return _select_and_record(study, spec, settings, training_entry.id, validation_entries)


def _select_and_record(
    study: Any,
    spec: StudySpec,
    settings: DQNRunSettings,
    training_workload_id: str,
    validation_entries: Sequence[Any],
) -> DQNTuningResult:
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    aggregates = [
        TrialAggregate(
            number=trial.number,
            **{key: float(trial.user_attrs[key]) for key in SELECTION_KEYS},
        )
        for trial in completed
    ]
    selected = select_trial(aggregates)
    trial = next(t for t in completed if t.number == selected.number)
    study.set_user_attr(SELECTED_TRIAL_ATTR, selected.number)
    study.set_user_attr("selection_rule", f"{OBJECTIVE_NAME}-{OBJECTIVE_VERSION}")
    return DQNTuningResult(
        study_name=spec.name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        dqn_config_version=DQN_CONFIG_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        optuna_storage=safe_storage_label(spec.storage),
        sampler="tpe",
        sampler_seed=spec.sampler_seed,
        training_workload_id=training_workload_id,
        validation_workload_ids=tuple(entry.id for entry in validation_entries),
        timesteps=settings.timesteps,
        seed=settings.seed,
        trial_count=len(completed),
        selected_trial_number=selected.number,
        selected_params=dict(trial.params),
        selected_hyperparameters=DQNHyperparameters.model_validate(
            trial.user_attrs["hyperparameters"], strict=False
        ),
        selected_validation_metrics={
            key: float(value)
            for key, value in trial.user_attrs.items()
            if key.startswith("validation_")
        },
        selected_training_run_id=str(trial.user_attrs[TRAINING_RUN_ATTR]),
        selected_mlflow_run_ids=tuple(trial.user_attrs[RUN_IDS_ATTR]),
    )


# --- command line ------------------------------------------------------------------------


LOCAL_STORAGE = "sqlite:///outputs/dqn-optuna.db"


def default_storage() -> str:
    """``OPTUNA_STORAGE_URI`` (set in the Docker Compose trainer), else local SQLite."""
    return os.environ.get("OPTUNA_STORAGE_URI") or LOCAL_STORAGE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune DQN on train/validation workloads (#15).")
    parser.add_argument("--train-workload", required=True, help="TRAIN workload ID")
    parser.add_argument(
        "--validation-workload",
        action="append",
        dest="validation_workloads",
        required=True,
        help="VALIDATION workload ID (repeatable)",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0, help="training seed of every trial")
    parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED)
    parser.add_argument(
        "--storage",
        default=default_storage(),
        help=f"Optuna storage URI; defaults to $OPTUNA_STORAGE_URI, else {LOCAL_STORAGE}",
    )
    parser.add_argument("--study-name", default="dqn-v1")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-dqn-tuning")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--azure-csv", type=Path, help="local Azure trace for Azure workloads")
    parser.add_argument("--simulator-config", type=Path, help="SimulatorConfig JSON file")
    parser.add_argument(
        "--config-source",
        choices=["default", "predeclared", "calibrated_train_validation"],
        default="default",
    )
    parser.add_argument("--calibration-workload", action="append", default=[])
    parser.add_argument("--calibration-note")
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn-tuning-v1.json"))
    args = parser.parse_args(argv)

    source: SimulatorConfigSource = args.config_source
    _prepare_sqlite_directory(args.storage)
    result = run_dqn_study(
        training_workload_id=args.train_workload,
        validation_workload_ids=args.validation_workloads,
        n_trials=args.n_trials,
        timesteps=args.timesteps,
        seed=args.seed,
        sampler_seed=args.sampler_seed,
        config=(
            SimulatorConfig.model_validate_json(args.simulator_config.read_text())
            if args.simulator_config
            else None
        ),
        config_source=source,
        calibration_workload_ids=args.calibration_workload,
        calibration_note=args.calibration_note,
        azure_csv_path=args.azure_csv,
        study_name=args.study_name,
        storage=args.storage,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        log_interval=args.log_interval,
    )
    path = result.save(args.output)
    print(f"selected trial {result.selected_trial_number}: {result.selected_params}")
    print(f"validation metrics: {result.selected_validation_metrics}")
    print(f"retrain with: python -m scalerl.training.dqn --hyperparameters {path} ...")
    print(f"result written to {path}")
    return 0


def _prepare_sqlite_directory(storage: str) -> None:
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        Path(storage.removeprefix(prefix)).parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    sys.exit(main())
