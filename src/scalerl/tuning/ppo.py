"""Tune PPO hyperparameters on train/validation workloads with Optuna (#16, #54).

Uses the same study runner as DQN (``scalerl.tuning.sb3``): each trial trains
one PPO on the declared TRAIN workload and validates the final policy on the
declared VALIDATION workloads, all as ``run_kind="tune"`` runs tracked through
``TrialContext.track``. Seeded TPE, sequential trials, no pruning; held-out
test workloads are rejected before any training starts.

Search space ``ppo-search-v1`` (predeclared from standard SB3/PPO ranges,
never from held-out results):

==========================  ==========================================
learning_rate               log-uniform 1e-5 – 1e-3
gamma                       {0.95, 0.98, 0.99, 0.995}
gae_lambda                  {0.9, 0.95, 0.98}
clip_range                  {0.1, 0.2, 0.3}
ent_coef                    {0.0, 0.001, 0.01}
n_steps (rollout size)      {512, 1024, 2048}
batch_size                  {64, 128, 256}
n_epochs                    {5, 10, 20}
net_arch (actor = critic)   {64×64, 128×128, 256×256}
==========================  ==========================================

Every ``batch_size`` divides every ``n_steps``, so no combination truncates a
minibatch, and the trial budget must be a multiple of 2048 (the least common
multiple of the rollout sizes), so every trial trains exactly the same number
of timesteps. ``vf_coef``, ``max_grad_norm``, ``normalize_advantage``, and the
normalization policy stay at their ``ppo-v1`` values.

Selection rule ``ppo-sla-first`` v1 (validation only): SLA violation rate,
then normalized cost, queue pressure, churn, then trial number; reward is
recorded but never selects. Retrain the selection as the ``train`` run of
record with ``python -m scalerl.training.ppo --hyperparameters <result>``.

Run a development study locally::

    python -m scalerl.tuning.ppo --train-workload syn-train-bursty \\
        --validation-workload syn-val-bursty --n-trials 20 --timesteps 204800 \\
        --storage sqlite:///outputs/ppo-optuna.db \\
        --tracking-uri sqlite:///outputs/mlflow.db --output outputs/ppo-tuning-v1.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from scalerl.benchmarks import load_benchmark_manifest
from scalerl.environment import SimulatorConfig
from scalerl.mlops import SimulatorConfigSource
from scalerl.training.common import DEFAULT_LOG_INTERVAL
from scalerl.training.ppo import (
    DEFAULT_TIMESTEPS,
    PPO_ALGORITHM,
    PPO_CONFIG_VERSION,
    PPOHyperparameters,
)
from scalerl.tuning.sb3 import (
    DEFAULT_N_TRIALS,
    DEFAULT_SAMPLER_SEED,
    StudyDefinition,
    prepare_sqlite_directory,
    run_sb3_study,
)
from scalerl.tuning.spec import safe_storage_label

if TYPE_CHECKING:
    import optuna

OBJECTIVE_NAME = "ppo-sla-first"
OBJECTIVE_VERSION = "v1"
SEARCH_SPACE_VERSION = "ppo-search-v1"
NET_ARCHS: dict[str, tuple[int, ...]] = {
    "64x64": (64, 64),
    "128x128": (128, 128),
    "256x256": (256, 256),
}
GAMMAS = [0.95, 0.98, 0.99, 0.995]
GAE_LAMBDAS = [0.9, 0.95, 0.98]
CLIP_RANGES = [0.1, 0.2, 0.3]
ENT_COEFS = [0.0, 0.001, 0.01]
N_STEPS = [512, 1_024, 2_048]
BATCH_SIZES = [64, 128, 256]
N_EPOCHS = [5, 10, 20]
BUDGET_MULTIPLE = math.lcm(*N_STEPS)  # 2048: every trial trains exactly `timesteps`


def suggest_hyperparameters(trial: optuna.Trial) -> PPOHyperparameters:
    """Sample ``ppo-search-v1``; unsearched settings keep their ``ppo-v1`` defaults."""
    net_arch = trial.suggest_categorical("net_arch", list(NET_ARCHS))
    return PPOHyperparameters(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gamma=trial.suggest_categorical("gamma", GAMMAS),
        gae_lambda=trial.suggest_categorical("gae_lambda", GAE_LAMBDAS),
        clip_range=trial.suggest_categorical("clip_range", CLIP_RANGES),
        ent_coef=trial.suggest_categorical("ent_coef", ENT_COEFS),
        n_steps=trial.suggest_categorical("n_steps", N_STEPS),
        batch_size=trial.suggest_categorical("batch_size", BATCH_SIZES),
        n_epochs=trial.suggest_categorical("n_epochs", N_EPOCHS),
        net_arch=NET_ARCHS[str(net_arch)],
    )


class PPOTuningResult(BaseModel):
    """Reproducible record of a PPO study and its selected configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_name: str
    objective_name: str
    objective_version: str
    search_space_version: str
    ppo_config_version: str
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
    selected_hyperparameters: PPOHyperparameters
    selected_validation_metrics: dict[str, float]
    selected_training_run_id: str
    selected_mlflow_run_ids: tuple[str, ...]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> PPOTuningResult:
        return cls.model_validate_json(Path(path).read_text())


def run_ppo_study(
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
    study_name: str = "ppo-v1",
    storage: str | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-ppo-tuning",
    log_interval: int = DEFAULT_LOG_INTERVAL,
    suggest: Callable[[optuna.Trial], PPOHyperparameters] = suggest_hyperparameters,
    rollout_sizes: Sequence[int] = tuple(N_STEPS),
) -> PPOTuningResult:
    """Run (or resume) the seeded TPE study (``tuning.sb3``) and select a configuration.

    ``timesteps`` must be a whole number of rollouts for every rollout size the
    search can propose (``rollout_sizes``). Trials do not write checkpoints.
    """
    outcome = run_sb3_study(
        StudyDefinition(
            algorithm=PPO_ALGORITHM,
            objective_name=OBJECTIVE_NAME,
            objective_version=OBJECTIVE_VERSION,
            search_space_version=SEARCH_SPACE_VERSION,
            suggest=suggest,
            rollout_sizes=tuple(rollout_sizes),
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
    return PPOTuningResult(
        study_name=outcome.spec.name,
        objective_name=OBJECTIVE_NAME,
        objective_version=OBJECTIVE_VERSION,
        search_space_version=SEARCH_SPACE_VERSION,
        ppo_config_version=PPO_CONFIG_VERSION,
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
        selected_hyperparameters=PPOHyperparameters.model_validate(
            outcome.selected_hyperparameters, strict=False
        ),
        selected_validation_metrics=outcome.selected_validation_metrics,
        selected_training_run_id=outcome.selected_training_run_id,
        selected_mlflow_run_ids=outcome.selected_mlflow_run_ids,
    )


# --- command line ------------------------------------------------------------------------


LOCAL_STORAGE = "sqlite:///outputs/ppo-optuna.db"


def default_storage() -> str:
    """``OPTUNA_STORAGE_URI`` (set in the Docker Compose trainer), else local SQLite."""
    return os.environ.get("OPTUNA_STORAGE_URI") or LOCAL_STORAGE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune PPO on train/validation workloads (#16).")
    parser.add_argument("--train-workload", required=True, help="TRAIN workload ID")
    parser.add_argument(
        "--validation-workload",
        action="append",
        dest="validation_workloads",
        required=True,
        help="VALIDATION workload ID (repeatable)",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"per trial; a multiple of {BUDGET_MULTIPLE} so every n_steps fits exactly",
    )
    parser.add_argument("--seed", type=int, default=0, help="training seed of every trial")
    parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED)
    parser.add_argument(
        "--storage",
        default=default_storage(),
        help=f"Optuna storage URI; defaults to $OPTUNA_STORAGE_URI, else {LOCAL_STORAGE}",
    )
    parser.add_argument("--study-name", default="ppo-v1")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-ppo-tuning")
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
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo-tuning-v1.json"))
    args = parser.parse_args(argv)

    source: SimulatorConfigSource = args.config_source
    prepare_sqlite_directory(args.storage)
    result = run_ppo_study(
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
    print(f"retrain with: python -m scalerl.training.ppo --hyperparameters {path} ...")
    print(f"result written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
