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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from scalerl.benchmarks import load_benchmark_manifest
from scalerl.environment import SimulatorConfig
from scalerl.mlops import SimulatorConfigSource
from scalerl.training.dqn import (
    DEFAULT_LOG_INTERVAL,
    DEFAULT_TIMESTEPS,
    DQN_ALGORITHM,
    DQN_CONFIG_VERSION,
    DQNHyperparameters,
)
from scalerl.tuning.sb3 import (
    DEFAULT_N_TRIALS,
    DEFAULT_SAMPLER_SEED,
    SELECTED_TRIAL_ATTR,
    SELECTION_KEYS,
    TRAINING_RUN_ATTR,
    StudyDefinition,
    TrialAggregate,
    prepare_sqlite_directory,
    run_sb3_study,
    select_trial,
)
from scalerl.tuning.spec import safe_storage_label

if TYPE_CHECKING:
    import optuna

__all__ = [
    "DEFAULT_N_TRIALS",
    "DEFAULT_SAMPLER_SEED",
    "OBJECTIVE_NAME",
    "OBJECTIVE_VERSION",
    "SEARCH_SPACE_VERSION",
    "SELECTED_TRIAL_ATTR",
    "SELECTION_KEYS",
    "TRAINING_RUN_ATTR",
    "DQNTuningResult",
    "TrialAggregate",
    "main",
    "run_dqn_study",
    "select_trial",
    "suggest_hyperparameters",
]

OBJECTIVE_NAME = "dqn-sla-first"
OBJECTIVE_VERSION = "v1"
SEARCH_SPACE_VERSION = "dqn-search-v1"
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


# --- result ----------------------------------------------------------------------------


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
    """Run (or resume) the seeded TPE study (``tuning.sb3``) and select a configuration.

    Budget, workload splits, and simulator-config provenance are checked before
    the study is created or any model is trained. Every trial trains with the
    same ``seed``, so trials differ only in their hyperparameters.
    """
    outcome = run_sb3_study(
        StudyDefinition(
            algorithm=DQN_ALGORITHM,
            objective_name=OBJECTIVE_NAME,
            objective_version=OBJECTIVE_VERSION,
            search_space_version=SEARCH_SPACE_VERSION,
            suggest=suggest,
            # train_freq is not searched, so the dqn-v1 value applies to every trial.
            rollout_sizes=(DQNHyperparameters().train_freq,),
        ),
        training_workload_id=training_workload_id,
        validation_workload_ids=validation_workload_ids,
        n_trials=n_trials,
        timesteps=timesteps,
        seed=seed,
        sampler_seed=sampler_seed,
        config=config,
        config_source=config_source,
        calibration_workload_ids=calibration_workload_ids,
        calibration_note=calibration_note,
        azure_csv_path=azure_csv_path,
        study_name=study_name,
        storage=storage,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        log_interval=log_interval,
    )
    return DQNTuningResult(
        study_name=outcome.spec.name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        dqn_config_version=DQN_CONFIG_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        optuna_storage=safe_storage_label(outcome.spec.storage),
        sampler="tpe",
        sampler_seed=outcome.spec.sampler_seed,
        training_workload_id=outcome.training_workload_id,
        validation_workload_ids=outcome.validation_workload_ids,
        timesteps=outcome.settings.timesteps,
        seed=outcome.settings.seed,
        trial_count=outcome.trial_count,
        selected_trial_number=outcome.selected_trial_number,
        selected_params=outcome.selected_params,
        selected_hyperparameters=DQNHyperparameters.model_validate(
            outcome.selected_hyperparameters, strict=False
        ),
        selected_validation_metrics=outcome.selected_validation_metrics,
        selected_training_run_id=outcome.selected_training_run_id,
        selected_mlflow_run_ids=outcome.selected_mlflow_run_ids,
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
    prepare_sqlite_directory(args.storage)
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


if __name__ == "__main__":
    sys.exit(main())
