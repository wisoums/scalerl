"""Algorithm-neutral Stable-Baselines3 training pipeline shared by DQN (#15) and PPO (#16).

An :class:`AlgorithmSpec` describes what differs between algorithms (how the
model is built, its hyperparameters' ``hp.*`` params, its rollout size, and
which SB3 logger values form its learning curve). Everything else is shared,
so DQN and PPO follow one contract:

* one TRAIN workload per ``train`` run, VALIDATION workloads only for
  validation, held-out test refused before any trace is built or model trained;
* budgets that are an exact number of SB3 rollouts, so the recorded
  ``training_steps`` is what the model was trained for;
* the final model saved as a compatibility-checked bundle under ``model/``,
  reloaded through :func:`~scalerl.rl.load_sb3_controller`, and validated with
  the shared ``evaluate_controller_episode`` in one run per workload tagged
  ``scalerl.model_source_run_id``;
* a learning curve from SB3's public callback hooks, and optional periodic
  checkpoints under ``checkpoints/``.

The policy only ever receives ``AutoscalingEnv`` observations and rewards
through the Gymnasium API; no wrapper normalizes either.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from gymnasium import Env
from pydantic import BaseModel, JsonValue
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import scalerl
from scalerl.benchmarks import WorkloadEntry, build_workloads, load_benchmark_manifest
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import EnvironmentCompatibility, RunKind, RunSpec, SimulatorConfigSource
from scalerl.mlops.tracking import TrackedRun
from scalerl.rl import ModelMetadata, load_sb3_controller, save_model_bundle
from scalerl.rl.sb3 import Algorithm
from scalerl.workloads import WorkloadTrace

DEFAULT_LOG_INTERVAL = 1_000
MODEL_ARTIFACT_PATH = "model"
CHECKPOINT_ARTIFACT_PATH = "checkpoints"
MODEL_SOURCE_TAG = "scalerl.model_source_run_id"
VALIDATION_KEYS = (
    "sla_violation_rate",
    "normalized_cost",
    "queue_pressure",
    "churn_rate",
    "mean_p95_latency_seconds",
    "episode_reward",
)

TrackFactory = Callable[[RunSpec], AbstractContextManager[TrackedRun]]


@dataclass(frozen=True)
class AlgorithmSpec:
    """What differs between SB3 algorithms; everything else is shared."""

    name: Algorithm  # MLflow ``controller`` and bundle ``algorithm``
    label: str  # for messages, e.g. "DQN"
    config_version: str
    build: Callable[[Env[Any, Any], Any, int], BaseAlgorithm]
    run_params: Callable[[Any], dict[str, JsonValue]]
    rollout_size: Callable[[Any], int]
    rollout_unit: str  # the hyperparameter that sets the rollout size
    recorded_metrics: tuple[tuple[str, str], ...]  # (SB3 logger key, MLflow metric)


# --- guardrails ---------------------------------------------------------------------------


def require_training_workload(workload_id: str, label: str) -> WorkloadEntry:
    """The manifest entry of a TRAIN workload; validation and held-out test are refused."""
    entry = load_benchmark_manifest().get(workload_id)
    if entry.split != "train":
        kind = "a held-out test" if entry.split == "test" else f"a {entry.split}"
        raise ValueError(
            f"{label} trains on train workloads only; {workload_id!r} is {kind} workload"
        )
    return entry


def require_validation_workloads(
    workload_ids: Sequence[str], label: str
) -> tuple[WorkloadEntry, ...]:
    """Manifest entries of VALIDATION workloads; test (and train) workloads are refused."""
    if not workload_ids:
        raise ValueError(f"{label} development needs at least one validation workload")
    if len(set(workload_ids)) != len(workload_ids):
        raise ValueError("validation workloads must be unique")
    manifest = load_benchmark_manifest()
    entries = []
    for workload_id in workload_ids:
        entry = manifest.get(workload_id)
        if entry.split != "validation":
            kind = "a held-out test" if entry.split == "test" else f"a {entry.split}"
            raise ValueError(
                f"{label} validation uses validation workloads only; "
                f"{workload_id!r} is {kind} workload"
            )
        entries.append(entry)
    return tuple(entries)


def require_whole_rollouts(
    timesteps: int, rollout_size: int, *, unit: str, label: str = "SB3"
) -> None:
    """Refuse budgets SB3 cannot train exactly.

    SB3 checks the budget only between rollouts (``train_freq`` steps for
    DQN, ``n_steps`` for PPO with one environment), so any other budget
    silently trains longer. The requested count is recorded as
    ``training_steps`` before training starts, so it must be exact.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    remainder = timesteps % rollout_size
    if remainder:
        exact = timesteps + rollout_size - remainder
        raise ValueError(
            f"timesteps ({timesteps}) must be a multiple of {unit} ({rollout_size}); "
            f"{label} collects whole rollouts and would train for {exact} timesteps instead"
        )


def require_checkpoint_interval(interval: int | None, rollout_size: int, *, unit: str) -> None:
    """Checkpoints fall on rollout boundaries, i.e. right after a completed update."""
    if interval is None:
        return
    if interval < 1 or interval % rollout_size:
        raise ValueError(
            f"checkpoint interval ({interval}) must be a positive multiple of "
            f"{unit} ({rollout_size})"
        )


def default_validation_workload_ids() -> tuple[str, ...]:
    """The benchmark's synthetic validation workloads."""
    return tuple(
        entry.id for entry in load_benchmark_manifest().validation if entry.source == "synthetic"
    )


# --- learning curve and checkpoints ---------------------------------------------------------


class MLflowTrainingCallback(BaseCallback):
    """Logs the learning curve to a tracked run every ``log_interval`` timesteps.

    Uses only SB3's public hooks: episode stats from the ``Monitor`` wrapper's
    ``info["episode"]``, and the values SB3's logger recorded (``recorded``
    maps logger keys to metric names), snapshotted at the start of each
    rollout and at the end of training, i.e. right after an update and before
    SB3 may clear them. A value SB3 has not produced yet (e.g. a loss before
    the first update) is omitted, never logged as zero.
    """

    def __init__(
        self,
        run: TrackedRun,
        *,
        recorded: Sequence[tuple[str, str]],
        log_interval: int = DEFAULT_LOG_INTERVAL,
    ) -> None:
        super().__init__()
        if log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        self._run = run
        self._interval = log_interval
        self._keys = tuple(recorded)
        self._window: list[float] = []
        self._returns: list[float] = []
        self._last_logged = -1
        self._recorded: dict[str, float] = {}
        self._logged_recorded: dict[str, float] = {}

    @property
    def episode_count(self) -> int:
        return len(self._returns)

    def _snapshot(self) -> None:
        values = self.model.logger.name_to_value
        for source, name in self._keys:
            if source in values:
                self._recorded[name] = float(values[source])

    def _on_rollout_start(self) -> None:
        self._snapshot()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            episode = info.get("episode")
            if episode is not None:
                self._window.append(float(episode["r"]))
                self._returns.append(float(episode["r"]))
        if self.num_timesteps % self._interval == 0:
            self._log()
        return True

    def _on_training_end(self) -> None:
        # The final update runs after the last rollout, so its values (loss, update
        # count, ...) are only visible now; log them at the final step if they changed.
        self._snapshot()
        if self._last_logged != self.num_timesteps:
            self._log()
        elif self._recorded != self._logged_recorded:
            self._run.log_metrics(self._recorded, step=self.num_timesteps)
            self._logged_recorded = dict(self._recorded)

    def _log(self) -> None:
        metrics: dict[str, float] = {"train/episodes": float(self.episode_count)}
        if self._window:
            metrics["train/episode_reward_mean"] = statistics.fmean(self._window)
        if self._returns:
            metrics["train/episode_reward_mean_100"] = statistics.fmean(self._returns[-100:])
        metrics |= self._recorded
        self._run.log_metrics(metrics, step=self.num_timesteps)
        self._logged_recorded = dict(self._recorded)
        self._window.clear()
        self._last_logged = self.num_timesteps


class CheckpointCallback(BaseCallback):
    """Saves the policy every ``interval`` timesteps and logs it under ``checkpoints/``.

    Saves happen at rollout starts (and at the end of training), when every
    collected step has already been used for an update, so ``step-N.zip`` is
    exactly the model after N timesteps of learning. Checkpoints are
    intermediate artifacts: the model of record is the final ``model/`` bundle.
    """

    def __init__(self, run: TrackedRun, directory: Path, *, interval: int) -> None:
        super().__init__()
        self._run = run
        self._directory = directory
        self._interval = interval
        self.saved: list[dict[str, Any]] = []

    def _save(self) -> None:
        step = self.num_timesteps
        if step == 0 or step % self._interval or any(c["step"] == step for c in self.saved):
            return
        name = f"step-{step:07d}.zip"
        path = self._directory / name
        self.model.save(path)
        self._run.log_artifact(path, artifact_path=CHECKPOINT_ARTIFACT_PATH)
        self.saved.append({"step": step, "file": f"{CHECKPOINT_ARTIFACT_PATH}/{name}"})

    def _on_rollout_start(self) -> None:
        self._save()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self._save()


# --- run specs and settings ------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSettings:
    """Everything, besides hyperparameters, that defines one training run."""

    timesteps: int
    seed: int
    config: SimulatorConfig
    config_source: SimulatorConfigSource
    calibration_workload_ids: tuple[str, ...]
    calibration_note: str | None
    reward_weights: RewardWeights
    log_interval: int = DEFAULT_LOG_INTERVAL
    checkpoint_interval: int | None = None


@dataclass(frozen=True)
class RunOutcome:
    training_run_id: str
    validation_run_ids: tuple[str, ...]
    validation_metrics: dict[str, EpisodeMetrics]
    aggregate: dict[str, float]
    compatibility: EnvironmentCompatibility
    training_episodes: int
    checkpoints: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def build_run_spec(
    run_kind: RunKind,
    controller: str,
    entry: WorkloadEntry,
    settings: TrainingSettings,
    hyperparameters: Mapping[str, JsonValue],
) -> RunSpec:
    return RunSpec(
        run_kind=run_kind,
        controller=controller,
        workload_id=entry.id,
        workload_split=entry.split,
        simulator_config=settings.config,
        simulator_config_source=settings.config_source,
        calibration_workload_ids=settings.calibration_workload_ids,
        calibration_note=settings.calibration_note,
        reward_weights=settings.reward_weights,
        seed=settings.seed,
        # Validation episodes are evaluated with reset seed 0; training runs evaluate nothing.
        evaluation_seeds=(0,) if entry.split == "validation" else (),
        hyperparameters=dict(hyperparameters),
        training_steps=settings.timesteps,
    )


# --- train + validate (shared by the CLIs and Optuna trials) ---------------------------------


def require_trainable(
    algorithm: AlgorithmSpec, hyperparameters: Any, settings: TrainingSettings
) -> None:
    rollout = algorithm.rollout_size(hyperparameters)
    require_whole_rollouts(
        settings.timesteps, rollout, unit=algorithm.rollout_unit, label=f"SB3 {algorithm.label}"
    )
    require_checkpoint_interval(settings.checkpoint_interval, rollout, unit=algorithm.rollout_unit)


def train_and_validate(
    algorithm: AlgorithmSpec,
    *,
    training_entry: WorkloadEntry,
    validation_entries: Sequence[WorkloadEntry],
    traces: Mapping[str, WorkloadTrace],
    hyperparameters: Any,
    settings: TrainingSettings,
    track: TrackFactory,
    training_run_kind: RunKind,
    validation_run_kind: RunKind,
    extra_params: Mapping[str, JsonValue] | None = None,
) -> RunOutcome:
    """Train on one workload (one tracked run), then validate the final policy per workload.

    The model is saved as a bundle and logged under ``model/`` on the training
    run. Validation loads that bundle back through :func:`load_sb3_controller`
    (so the compatibility check is exercised) and evaluates it with the shared
    ``evaluate_controller_episode``, one tracked run per validation workload.
    """
    require_trainable(algorithm, hyperparameters, settings)
    params = {**algorithm.run_params(hyperparameters), **dict(extra_params or {})}
    config, weights = settings.config, settings.reward_weights
    spec = build_run_spec(training_run_kind, algorithm.name, training_entry, settings, params)
    with track(spec) as run, tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        env = AutoscalingEnv(config, traces[training_entry.id], weights)
        model = algorithm.build(Monitor(env), hyperparameters, settings.seed)
        curve = MLflowTrainingCallback(
            run, recorded=algorithm.recorded_metrics, log_interval=settings.log_interval
        )
        callbacks: list[BaseCallback] = [curve]
        checkpoints: CheckpointCallback | None = None
        if settings.checkpoint_interval is not None:
            (workdir / "checkpoints").mkdir()
            checkpoints = CheckpointCallback(
                run, workdir / "checkpoints", interval=settings.checkpoint_interval
            )
            callbacks.append(checkpoints)
        started = time.perf_counter()
        model.learn(total_timesteps=settings.timesteps, callback=CallbackList(callbacks))
        if model.num_timesteps != settings.timesteps:
            raise RuntimeError(
                f"SB3 trained {model.num_timesteps} timesteps, not the recorded "
                f"{settings.timesteps}"
            )
        run.log_metrics(
            {
                "training_timesteps": float(model.num_timesteps),
                "training_episodes": float(curve.episode_count),
                "training_seconds": time.perf_counter() - started,
            }
        )
        saved_checkpoints = tuple(checkpoints.saved) if checkpoints is not None else ()
        if checkpoints is not None:
            run.log_artifact_dict(
                "manifest.json",
                {
                    "interval_timesteps": settings.checkpoint_interval,
                    "checkpoints": list(saved_checkpoints),
                    "model_of_record": f"{MODEL_ARTIFACT_PATH}/",
                    "note": "intermediate policies; the final model/ bundle is the model of record",
                },
                artifact_path=CHECKPOINT_ARTIFACT_PATH,
            )

        version = load_benchmark_manifest().version
        compatibility = EnvironmentCompatibility.from_env(env, version)
        metadata = ModelMetadata(
            algorithm=algorithm.name,
            config_version=algorithm.config_version,
            benchmark_version=version,
            training_workload_id=training_entry.id,
            training_workload_split="train",
            seed=settings.seed,
            total_timesteps=settings.timesteps,
            hyperparameters=hyperparameters.as_params(),
            scalerl_version=scalerl.__version__,
            training_run_id=run.run_id,
        )
        bundle = save_model_bundle(
            model, workdir / "bundle", metadata=metadata, compatibility=compatibility
        )
        for path in sorted(bundle.iterdir()):
            run.log_artifact(path, artifact_path=MODEL_ARTIFACT_PATH)

        results: dict[str, EpisodeMetrics] = {}
        validation_run_ids = []
        for entry in validation_entries:
            validation_env = AutoscalingEnv(config, traces[entry.id], weights)
            controller = load_sb3_controller(bundle, validation_env)
            validation_spec = build_run_spec(
                validation_run_kind, algorithm.name, entry, settings, params
            )
            with track(validation_spec) as validation_run:
                validation_run.set_tag(MODEL_SOURCE_TAG, run.run_id)
                validation_run.set_tag("scalerl.model_artifact_path", MODEL_ARTIFACT_PATH)
                evaluation = evaluate_controller_episode(validation_env, controller, seed=0)
                validation_run.log_metrics(evaluation.metrics.as_metrics())
            results[entry.id] = evaluation.metrics
            validation_run_ids.append(validation_run.run_id)

        aggregate = aggregate_validation(list(results.values()))
        run.log_metrics({f"validation.{key}": value for key, value in aggregate.items()})
        run.log_artifact_dict(
            "validation_summary.json",
            {
                "validation_run_ids": validation_run_ids,
                "aggregate": aggregate,
                "per_workload": {key: value.as_metrics() for key, value in results.items()},
            },
            artifact_path="scalerl",
        )
    return RunOutcome(
        training_run_id=run.run_id,
        validation_run_ids=tuple(validation_run_ids),
        validation_metrics=results,
        aggregate=aggregate,
        compatibility=compatibility,
        training_episodes=curve.episode_count,
        checkpoints=saved_checkpoints,
    )


def aggregate_validation(results: Sequence[EpisodeMetrics]) -> dict[str, float]:
    """Mean system metrics over validation workloads (reward stays secondary)."""
    if not results:
        raise ValueError("no validation results to aggregate")
    return {key: statistics.fmean(getattr(r, key) for r in results) for key in VALIDATION_KEYS}


@dataclass(frozen=True)
class TrainedRun:
    """A completed tracked training run with the workloads it used."""

    outcome: RunOutcome
    training_entry: WorkloadEntry
    validation_entries: tuple[WorkloadEntry, ...]


def train_sb3(
    algorithm: AlgorithmSpec,
    *,
    hyperparameters: Any,
    hyperparameter_source: str,
    workload_id: str,
    validation_workload_ids: Sequence[str] | None,
    settings: TrainingSettings,
    azure_csv_path: str | Path | None,
    tracking_uri: str | None,
    experiment_name: str,
) -> TrainedRun:
    """Guardrails first (budget, splits, provenance), then one tracked training run.

    Nothing is built or trained until every check has passed.
    """
    require_trainable(algorithm, hyperparameters, settings)
    training_entry = require_training_workload(workload_id, algorithm.label)
    validation_ids = (
        tuple(validation_workload_ids)
        if validation_workload_ids is not None
        else default_validation_workload_ids()
    )
    validation_entries = require_validation_workloads(validation_ids, algorithm.label)
    extra: dict[str, JsonValue] = {"hyperparameter_source": hyperparameter_source}
    params = {**algorithm.run_params(hyperparameters), **extra}
    for entry in (training_entry, *validation_entries):  # provenance checks, before training
        kind: RunKind = "train" if entry is training_entry else "evaluate"
        build_run_spec(kind, algorithm.name, entry, settings, params)
    traces = build_workloads([training_entry, *validation_entries], azure_csv_path=azure_csv_path)

    from scalerl.mlops import start_tracked_run

    def track(spec: RunSpec) -> AbstractContextManager[TrackedRun]:
        return start_tracked_run(spec, tracking_uri=tracking_uri, experiment_name=experiment_name)

    outcome = train_and_validate(
        algorithm,
        training_entry=training_entry,
        validation_entries=validation_entries,
        traces=traces,
        hyperparameters=hyperparameters,
        settings=settings,
        track=track,
        training_run_kind="train",
        validation_run_kind="evaluate",
        extra_params=extra,
    )
    return TrainedRun(outcome, training_entry, validation_entries)


# --- command-line helpers ------------------------------------------------------------------------


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_hyperparameters(path: str | Path, model: type[ModelT]) -> tuple[ModelT, str]:
    """Read a hyperparameter JSON, or a tuning result's selected configuration."""
    payload = json.loads(Path(path).read_text())
    if "selected_hyperparameters" in payload:
        source = f"optuna:{payload['study_name']}#trial{payload['selected_trial_number']}"
        return model.model_validate(payload["selected_hyperparameters"], strict=False), source
    return model.model_validate(payload, strict=False), f"file:{Path(path).name}"


def apply_overrides(hyperparameters: ModelT, overrides: Sequence[str]) -> ModelT:
    """Apply ``name=value`` overrides (JSON values; ``net_arch=128,128`` also accepted)."""
    if not overrides:
        return hyperparameters
    model = type(hyperparameters)
    values = hyperparameters.model_dump()
    for item in overrides:
        name, sep, raw = item.partition("=")
        if not sep or name not in model.model_fields:
            raise ValueError(f"invalid override {item!r}; use <hyperparameter>=<value>")
        if name == "net_arch":
            values[name] = tuple(int(width) for width in raw.split(","))
        else:
            values[name] = json.loads(raw)
    return model.model_validate(values, strict=False)
