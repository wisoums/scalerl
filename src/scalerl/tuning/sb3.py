"""Shared Optuna study for Stable-Baselines3 algorithms (DQN #15, PPO #16).

One trial trains one model with the trial's suggested hyperparameters on the
declared TRAIN workload, then validates the final policy on the declared
VALIDATION workloads, all as ``run_kind="tune"`` runs tracked through
``TrialContext.track`` (#54). Samplers are seeded TPE, trials run
sequentially, and there is no pruning. Held-out test workloads and budgets SB3
cannot train exactly are rejected before the study is created.

Selection is SLA-first over VALIDATION metrics only, identical for every
algorithm:

1. lowest mean SLA violation rate (also the Optuna objective);
2. lowest mean normalized infrastructure cost;
3. lowest mean normalized queue pressure;
4. lowest mean churn rate;
5. lowest trial number (deterministic tie-break).

Episode reward is recorded but never used for selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from scalerl.benchmarks import build_workloads
from scalerl.environment import SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.mlops import RunSpec, SimulatorConfigSource
from scalerl.training.common import (
    DEFAULT_LOG_INTERVAL,
    AlgorithmSpec,
    TrainingSettings,
    build_run_spec,
    require_checkpoint_interval,
    require_training_workload,
    require_validation_workloads,
    require_whole_rollouts,
    train_and_validate,
)
from scalerl.tuning.spec import StudySpec
from scalerl.tuning.study import RUN_IDS_ATTR, TrialContext, run_study
from scalerl.tuning.threshold import workload_fingerprint

if TYPE_CHECKING:
    import optuna

DEFAULT_SAMPLER_SEED = 42
DEFAULT_N_TRIALS = 20
SELECTED_TRIAL_ATTR = "selected_trial_number"
TRAINING_RUN_ATTR = "training_run_id"
HYPERPARAMETERS_ATTR = "hyperparameters"
SELECTION_KEYS = (
    "validation_sla_violation_rate",
    "validation_normalized_cost",
    "validation_queue_pressure",
    "validation_churn_rate",
)


@dataclass(frozen=True)
class TrialAggregate:
    """Validation aggregates of one completed trial."""

    number: int
    validation_sla_violation_rate: float
    validation_normalized_cost: float
    validation_queue_pressure: float
    validation_churn_rate: float


def select_trial(candidates: Sequence[TrialAggregate]) -> TrialAggregate:
    """SLA-first selection: SLA, then cost, queue pressure, churn, then trial number."""
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


@dataclass(frozen=True)
class StudyDefinition:
    """An algorithm's search: objective identity, search space, and its rollout sizes."""

    algorithm: AlgorithmSpec
    objective_name: str
    objective_version: str
    search_space_version: str
    suggest: Callable[[optuna.Trial], Any]
    # Every rollout size the search can propose: the budget must fit each one exactly,
    # so no trial secretly trains longer than another.
    rollout_sizes: tuple[int, ...]


@dataclass(frozen=True)
class StudyOutcome:
    """The selected trial of a finished study, with everything needed for a result file."""

    spec: StudySpec
    settings: TrainingSettings
    training_workload_id: str
    validation_workload_ids: tuple[str, ...]
    trial_count: int
    selected_trial_number: int
    selected_params: dict[str, Any]
    selected_hyperparameters: dict[str, JsonValue]
    selected_validation_metrics: dict[str, float]
    selected_training_run_id: str
    selected_mlflow_run_ids: tuple[str, ...]


def run_sb3_study(
    definition: StudyDefinition,
    *,
    training_workload_id: str,
    validation_workload_ids: Sequence[str],
    n_trials: int,
    timesteps: int,
    seed: int,
    sampler_seed: int,
    config: SimulatorConfig | None,
    config_source: SimulatorConfigSource,
    calibration_workload_ids: Sequence[str],
    calibration_note: str | None,
    azure_csv_path: str | Path | None,
    study_name: str,
    storage: str | None,
    tracking_uri: str | None,
    experiment_name: str,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    checkpoint_interval: int | None = None,
) -> StudyOutcome:
    """Run (or resume) the seeded TPE study and select a configuration.

    Budget, workload splits, and simulator-config provenance are checked
    before the study is created or any model is trained. Every trial trains
    with the same ``seed``, so trials differ only in their hyperparameters.
    """
    algorithm = definition.algorithm
    for size in definition.rollout_sizes:
        require_whole_rollouts(
            timesteps, size, unit=algorithm.rollout_unit, label=f"SB3 {algorithm.label}"
        )
        require_checkpoint_interval(checkpoint_interval, size, unit=algorithm.rollout_unit)
    training_entry = require_training_workload(training_workload_id, algorithm.label)
    validation_entries = require_validation_workloads(validation_workload_ids, algorithm.label)
    settings = TrainingSettings(
        timesteps=timesteps,
        seed=seed,
        config=config or SimulatorConfig(),
        config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=RewardWeights(),
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
    )
    for entry in (training_entry, *validation_entries):  # provenance checks, before training
        build_run_spec("tune", algorithm.name, entry, settings, {})
    entries = (training_entry, *validation_entries)
    traces = build_workloads(entries, azure_csv_path=azure_csv_path)
    spec = StudySpec(
        name=study_name,
        objective_name=definition.objective_name,
        objective_version=definition.objective_version,
        search_space_version=definition.search_space_version,
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
            f"{algorithm.name}_config_version": algorithm.config_version,
            "simulator_config": settings.config.model_dump(mode="json"),
            "simulator_config_source": config_source,
            "calibration_workload_ids": list(calibration_workload_ids),
            "reward_weights": settings.reward_weights.model_dump(mode="json"),
            "workload_fingerprint": workload_fingerprint(traces),
        },
    )

    def objective(context: TrialContext) -> float:
        hyperparameters = definition.suggest(context.trial)

        def track(run_spec: RunSpec) -> Any:
            return context.track(
                run_spec, tracking_uri=tracking_uri, experiment_name=experiment_name
            )

        outcome = train_and_validate(
            algorithm,
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
        context.trial.set_user_attr(HYPERPARAMETERS_ATTR, hyperparameters.as_params())
        for key, value in outcome.aggregate.items():
            context.trial.set_user_attr(f"validation_{key}", value)
        return float(outcome.aggregate["sla_violation_rate"])

    study = run_study(spec, objective)
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    selected = select_trial(
        [
            TrialAggregate(
                number=trial.number,
                **{key: float(trial.user_attrs[key]) for key in SELECTION_KEYS},
            )
            for trial in completed
        ]
    )
    trial = next(t for t in completed if t.number == selected.number)
    study.set_user_attr(SELECTED_TRIAL_ATTR, selected.number)
    study.set_user_attr(
        "selection_rule", f"{definition.objective_name}-{definition.objective_version}"
    )
    return StudyOutcome(
        spec=spec,
        settings=settings,
        training_workload_id=training_entry.id,
        validation_workload_ids=tuple(entry.id for entry in validation_entries),
        trial_count=len(completed),
        selected_trial_number=selected.number,
        selected_params=dict(trial.params),
        selected_hyperparameters=dict(trial.user_attrs[HYPERPARAMETERS_ATTR]),
        selected_validation_metrics={
            key: float(value)
            for key, value in trial.user_attrs.items()
            if key.startswith("validation_")
        },
        selected_training_run_id=str(trial.user_attrs[TRAINING_RUN_ATTR]),
        selected_mlflow_run_ids=tuple(trial.user_attrs[RUN_IDS_ATTR]),
    )


def prepare_sqlite_directory(storage: str) -> None:
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        Path(storage.removeprefix(prefix)).parent.mkdir(parents=True, exist_ok=True)
