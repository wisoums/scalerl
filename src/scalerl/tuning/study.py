"""Run Optuna studies whose trials log through ScaleRL's MLflow contract.

Requires the optional ``tuning`` extra (and ``mlops`` for tracked runs).
Optuna is imported only when a study runs; ``StudySpec`` works without it.

Responsibilities stay separate: Optuna proposes parameters and records trial
values; every workload evaluation is its own ``start_tracked_run`` MLflow run
(one workload per run, #17). A trial may therefore own several run IDs,
stored on the trial as the ``mlflow_run_ids`` user attribute and linked back
through ``hp.optuna.*`` params and ``scalerl.optuna.*`` tags on each run. When
the trial ends, every one of its runs is tagged with the final Optuna state
(``scalerl.optuna.trial_state`` = ``COMPLETE``, ``PRUNED``, or ``FAIL``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from scalerl.mlops import RunSpec
from scalerl.tuning.spec import StudySpec

if TYPE_CHECKING:
    import optuna

    from scalerl.mlops import TrackedRun

INSTALL_HINT = 'Optuna tuning requires the tuning extra: pip install -e ".[tuning]"'
RUN_IDS_ATTR = "mlflow_run_ids"
IDENTITY_ATTR = "scalerl_study_identity"
TRIAL_STATE_TAG = "scalerl.optuna.trial_state"
_TUNING_RUN_KINDS = ("train", "tune")


class TrialContext:
    """One Optuna trial: parameter suggestions plus tracked MLflow runs."""

    def __init__(self, spec: StudySpec, trial: optuna.Trial) -> None:
        self._spec = spec
        self._trial = trial
        self._runs: list[tuple[str, str | None]] = []  # (run ID, tracking URI)

    @property
    def trial(self) -> optuna.Trial:
        """The Optuna trial, for ``suggest_float``/``suggest_categorical``/..."""
        return self._trial

    @property
    def number(self) -> int:
        return self._trial.number

    @property
    def study_name(self) -> str:
        return self._spec.name

    @property
    def params(self) -> dict[str, Any]:
        """Parameters suggested so far in this trial."""
        return dict(self._trial.params)

    @property
    def mlflow_run_ids(self) -> tuple[str, ...]:
        """MLflow runs started by this trial, in order."""
        return tuple(run_id for run_id, _ in self._runs)

    @contextmanager
    def track(self, run_spec: RunSpec, **tracking: Any) -> Iterator[TrackedRun]:
        """Start one MLflow run (one workload) that belongs to this trial.

        ``run_spec`` must be a ``train`` or ``tune`` run on one of the study's
        tuning workloads. The trial's Optuna metadata and suggested parameters
        are added to its hyperparameters; ``tracking`` is passed to
        ``start_tracked_run``. Pruning inside the block ends the run ``FINISHED``
        (other errors mark it ``FAILED``); the trial's final state is tagged on
        all its runs when the trial ends.
        """
        import optuna

        from scalerl.mlops import start_tracked_run

        if run_spec.run_kind not in _TUNING_RUN_KINDS:
            raise ValueError(
                f"tuning trials may only track train/tune runs, not {run_spec.run_kind}"
            )
        if run_spec.workload_id not in self._spec.tuning_workload_ids:
            raise ValueError(
                f"workload {run_spec.workload_id!r} is not one of study "
                f"{self._spec.name!r}'s tuning workloads"
            )
        spec = RunSpec(
            **{
                **dict(run_spec),
                "hyperparameters": {
                    **run_spec.hyperparameters,
                    "optuna": {
                        **self._spec.trial_metadata(self.number),
                        "params": self.params,
                    },
                },
            }
        )

        pruned: optuna.TrialPruned | None = None
        with start_tracked_run(spec, **tracking) as run:
            self._runs.append((run.run_id, tracking.get("tracking_uri")))
            self._trial.set_user_attr(RUN_IDS_ATTR, list(self.mlflow_run_ids))
            run.set_tag("scalerl.optuna.study", self._spec.name)
            run.set_tag("scalerl.optuna.trial", str(self.number))
            try:
                yield run
            except optuna.TrialPruned as error:
                pruned = error
        if pruned is not None:
            raise pruned

    def report(self, value: float, step: int) -> None:
        """Report an intermediate train/validation value for pruning."""
        self._trial.report(value, step)

    def should_prune(self) -> bool:
        return self._trial.should_prune()

    def prune(self) -> None:
        """Stop this trial as pruned (raises ``optuna.TrialPruned``)."""
        import optuna

        raise optuna.TrialPruned()

    def _tag_trial_state(self, state: str) -> None:
        """Tag every run this trial opened with the trial's final state."""
        if not self._runs:
            return
        from mlflow import MlflowClient

        for run_id, tracking_uri in self._runs:
            MlflowClient(tracking_uri=tracking_uri).set_tag(run_id, TRIAL_STATE_TAG, state)


Objective = Callable[[TrialContext], float]


def run_study(
    spec: StudySpec,
    objective: Objective,
    *,
    catch: tuple[type[Exception], ...] = (),
) -> optuna.Study:
    """Create or resume ``spec``'s study and run its remaining trial budget.

    Trials run sequentially unless ``spec.n_jobs > 1``. An exception from the
    objective marks the trial ``FAILED`` and is re-raised, unless its type is in
    ``catch``, in which case the study continues.
    """
    _require_optuna()
    import optuna

    study = optuna.create_study(
        study_name=spec.name,
        storage=spec.storage,
        direction=spec.direction,
        sampler=_sampler(spec),
        pruner=_pruner(spec),
        load_if_exists=spec.load_if_exists,
    )
    _check_identity(study, spec)

    # A grid study's default budget is its grid: re-running a finished grid
    # must not make GridSampler re-evaluate an already visited point.
    budget = spec.n_trials if spec.n_trials is not None else spec.grid_size
    finished = sum(1 for trial in study.trials if trial.state.is_finished())
    remaining = None if budget is None else budget - finished
    if remaining is not None and remaining <= 0:
        return study

    contexts: dict[int, TrialContext] = {}

    def run_trial(trial: optuna.Trial) -> float:
        trial.set_user_attr(RUN_IDS_ATTR, [])
        context = contexts[trial.number] = TrialContext(spec, trial)
        try:
            return objective(context)
        except optuna.TrialPruned:
            raise  # Optuna records PRUNED and calls tag_runs
        except Exception as error:
            if not isinstance(error, catch):
                # Uncaught errors end optimize() before callbacks run.
                failed = contexts.pop(trial.number)
                failed._tag_trial_state(optuna.trial.TrialState.FAIL.name)
            raise

    def tag_runs(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        context = contexts.pop(trial.number, None)
        if context is not None:
            context._tag_trial_state(trial.state.name)

    study.optimize(
        run_trial,
        n_trials=remaining,
        timeout=spec.timeout_seconds,
        n_jobs=spec.n_jobs,
        catch=catch,
        callbacks=[tag_runs],
    )
    return study


def _require_optuna() -> None:
    try:
        import optuna
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(spec: StudySpec) -> optuna.samplers.BaseSampler:
    """Seeded sampler that behaves the same whether or not the study was resumed.

    GridSampler reads visited combinations from storage, so resuming never
    repeats a grid point. Random and TPE samplers keep an in-memory RNG that
    Optuna does not persist, so they are seeded per trial instead.
    """
    import optuna

    from scalerl.tuning._sampler import TrialSeededSampler

    if spec.sampler == "grid":
        assert spec.grid is not None
        return optuna.samplers.GridSampler(
            {name: list(values) for name, values in spec.grid.items()}, seed=spec.sampler_seed
        )
    if spec.sampler == "tpe":
        return TrialSeededSampler(
            lambda seed: optuna.samplers.TPESampler(seed=seed), spec.sampler_seed
        )
    return TrialSeededSampler(
        lambda seed: optuna.samplers.RandomSampler(seed=seed), spec.sampler_seed
    )


def _pruner(spec: StudySpec) -> optuna.pruners.BasePruner:
    import optuna

    if spec.pruner == "median":
        return optuna.pruners.MedianPruner()
    return optuna.pruners.NopPruner()


def _check_identity(study: optuna.Study, spec: StudySpec) -> None:
    """Refuse to resume a study created with a different definition."""
    identity = spec.identity()
    existing = study.user_attrs.get(IDENTITY_ATTR)
    if existing is None:
        study.set_user_attr(IDENTITY_ATTR, identity)
    elif existing != identity:
        changed = _changed_fields(existing, identity)
        raise ValueError(
            f"study {spec.name!r} already exists with a different definition "
            f"({', '.join(changed)}); use a new study name"
        )


def _changed_fields(existing: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """Name the identity fields that differ, down to ``identity_context`` entries."""
    changed = []
    for key in identity:
        if existing.get(key) == identity[key]:
            continue
        if key == "identity_context":
            old, new = existing.get(key) or {}, identity[key]
            changed += [
                f"{key}.{name}" for name in set(old) | set(new) if old.get(name) != new.get(name)
            ]
        else:
            changed.append(key)
    return sorted(changed)
