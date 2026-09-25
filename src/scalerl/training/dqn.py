"""Train a Stable-Baselines3 DQN autoscaler on one TRAIN workload (#15).

DQN fits ScaleRL directly: the action space is ``Discrete(3)`` (0 scale down,
1 hold, 2 scale up) and the observation is the frozen v1 normalized vector
(#58: four recent completed-traffic samples plus system state). The policy
sees only that observation and the environment's reward; it never reads the
trace, ``info``, or anything about the future.

Pipeline (one workload per MLflow run, #17)::

    TRAIN workload ─→ AutoscalingEnv ─→ SB3 DQN.learn ─→ model bundle ─→ MLflow `train` run
                                                             │  (model/ artifacts)
    each VALIDATION workload ─→ load bundle (compatibility-checked) ─→ SB3Controller
        ─→ evaluate_controller_episode ─→ its own MLflow `evaluate` run,
           tagged scalerl.model_source_run_id = <train run>

Held-out test workloads are rejected before any training starts. The v1
hyperparameters are predeclared, not tuned; tuning lives in
``scalerl.tuning.dqn`` and its selection is retrained here with
``--hyperparameters``.

Run locally::

    python -m scalerl.training.dqn --workload syn-train-bursty \\
        --validation-workload syn-val-bursty --timesteps 200000 --seed 0 \\
        --tracking-uri sqlite:///outputs/mlflow.db --output outputs/dqn-v1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gymnasium import Env
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import scalerl
from scalerl.benchmarks import WorkloadEntry, build_workloads, load_benchmark_manifest
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import EnvironmentCompatibility, RunKind, RunSpec, SimulatorConfigSource
from scalerl.mlops.tracking import TrackedRun
from scalerl.rl import ModelMetadata, load_sb3_controller, save_model_bundle
from scalerl.workloads import WorkloadTrace

DQN_CONFIG_VERSION = "dqn-v1"
DEFAULT_TIMESTEPS = 200_000
DEFAULT_LOG_INTERVAL = 1_000
MODEL_ARTIFACT_PATH = "model"
MODEL_SOURCE_TAG = "scalerl.model_source_run_id"
_SB3_RECORDED = (
    ("train/loss", "train/loss"),
    ("train/n_updates", "train/n_updates"),
    ("rollout/exploration_rate", "train/exploration_rate"),
)
VALIDATION_KEYS = (
    "sla_violation_rate",
    "normalized_cost",
    "queue_pressure",
    "churn_rate",
    "mean_p95_latency_seconds",
    "episode_reward",
)


class DQNHyperparameters(BaseModel):
    """The DQN settings ScaleRL exposes; defaults are the predeclared ``dqn-v1`` config.

    Starting points are SB3's DQN defaults, adapted to a 200k-step budget on
    120-tick episodes (smaller replay buffer, earlier learning, more frequent
    target updates, longer exploration). They are documented, not tuned, and
    not claimed to be optimal.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    learning_rate: float = Field(default=5e-4, gt=0, le=1)
    buffer_size: int = Field(default=100_000, ge=1)
    learning_starts: int = Field(default=1_000, ge=0)
    batch_size: int = Field(default=64, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    train_freq: int = Field(default=4, ge=1)
    gradient_steps: int = Field(default=1, ge=1)
    target_update_interval: int = Field(default=1_000, ge=1)
    exploration_fraction: float = Field(default=0.2, gt=0, le=1)
    exploration_initial_eps: float = Field(default=1.0, ge=0, le=1)
    exploration_final_eps: float = Field(default=0.05, ge=0, le=1)
    net_arch: tuple[int, ...] = Field(default=(64, 64), min_length=1)

    @model_validator(mode="after")
    def _check(self) -> DQNHyperparameters:
        if any(width < 1 for width in self.net_arch):
            raise ValueError("net_arch layer widths must be positive")
        if self.exploration_final_eps > self.exploration_initial_eps:
            raise ValueError("exploration_final_eps must not exceed exploration_initial_eps")
        return self

    def sb3_kwargs(self) -> dict[str, Any]:
        values = self.model_dump()
        net_arch = list(values.pop("net_arch"))
        return {**values, "policy_kwargs": {"net_arch": net_arch}}

    def as_params(self) -> dict[str, JsonValue]:
        params: dict[str, JsonValue] = self.model_dump(mode="json")
        return params


# --- guardrails ---------------------------------------------------------------------------


def require_training_workload(workload_id: str) -> WorkloadEntry:
    """The manifest entry of a TRAIN workload; validation and held-out test are refused."""
    entry = load_benchmark_manifest().get(workload_id)
    if entry.split != "train":
        kind = "a held-out test" if entry.split == "test" else f"a {entry.split}"
        raise ValueError(f"DQN trains on train workloads only; {workload_id!r} is {kind} workload")
    return entry


def require_validation_workloads(workload_ids: Sequence[str]) -> tuple[WorkloadEntry, ...]:
    """Manifest entries of VALIDATION workloads; test (and train) workloads are refused."""
    if not workload_ids:
        raise ValueError("DQN development needs at least one validation workload")
    if len(set(workload_ids)) != len(workload_ids):
        raise ValueError("validation workloads must be unique")
    manifest = load_benchmark_manifest()
    entries = []
    for workload_id in workload_ids:
        entry = manifest.get(workload_id)
        if entry.split != "validation":
            kind = "a held-out test" if entry.split == "test" else f"a {entry.split}"
            raise ValueError(
                f"DQN validation uses validation workloads only; {workload_id!r} is {kind} workload"
            )
        entries.append(entry)
    return tuple(entries)


def require_exact_timesteps(timesteps: int, hyperparameters: DQNHyperparameters) -> None:
    """Refuse budgets SB3 DQN cannot train exactly.

    SB3's off-policy ``learn`` checks the budget only between rollouts of
    ``train_freq`` steps, so a budget that is not a multiple of ``train_freq``
    silently trains longer (e.g. 1 → 4, 97 → 100 with ``train_freq=4``). The
    requested count is recorded as ``training_steps`` before training starts,
    so it must be exactly what the model is trained for.
    """
    if timesteps < 1:
        raise ValueError("timesteps must be at least 1")
    remainder = timesteps % hyperparameters.train_freq
    if remainder:
        exact = timesteps + hyperparameters.train_freq - remainder
        raise ValueError(
            f"timesteps ({timesteps}) must be a multiple of train_freq "
            f"({hyperparameters.train_freq}); SB3 DQN collects whole rollouts and would "
            f"train for {exact} timesteps instead"
        )


def default_validation_workload_ids() -> tuple[str, ...]:
    """The benchmark's synthetic validation workloads."""
    return tuple(
        entry.id for entry in load_benchmark_manifest().validation if entry.source == "synthetic"
    )


# --- model construction and learning curve ---------------------------------------------------


def build_dqn(env: Env[Any, Any], hyperparameters: DQNHyperparameters, *, seed: int) -> DQN:
    """A CPU SB3 DQN on ``env``, seeded for reproducible setup (network init, exploration)."""
    return DQN("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **hyperparameters.sb3_kwargs())


class MLflowTrainingCallback(BaseCallback):
    """Logs the learning curve to a tracked run every ``log_interval`` timesteps.

    Uses only SB3's public hooks: episode stats from the ``Monitor`` wrapper's
    ``info["episode"]``, and the values SB3's logger recorded (loss, update
    count, exploration rate), snapshotted at the start of each rollout, i.e.
    right after the previous training step and before SB3 may clear them. A
    value SB3 has not produced yet (e.g. loss before ``learning_starts``) is
    omitted, never logged as zero.
    """

    def __init__(self, run: TrackedRun, *, log_interval: int = DEFAULT_LOG_INTERVAL) -> None:
        super().__init__()
        if log_interval < 1:
            raise ValueError("log_interval must be at least 1")
        self._run = run
        self._interval = log_interval
        self._window: list[float] = []
        self._returns: list[float] = []
        self._last_logged = -1
        self._recorded: dict[str, float] = {}

    @property
    def episode_count(self) -> int:
        return len(self._returns)

    def _on_rollout_start(self) -> None:
        recorded = self.model.logger.name_to_value
        for source, name in _SB3_RECORDED:
            if source in recorded:
                self._recorded[name] = float(recorded[source])

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
        if self._last_logged != self.num_timesteps:
            self._log()

    def _log(self) -> None:
        metrics: dict[str, float] = {"train/episodes": float(self.episode_count)}
        if self._window:
            metrics["train/episode_reward_mean"] = statistics.fmean(self._window)
        if self._returns:
            metrics["train/episode_reward_mean_100"] = statistics.fmean(self._returns[-100:])
        metrics |= self._recorded
        self._run.log_metrics(metrics, step=self.num_timesteps)
        self._window.clear()
        self._last_logged = self.num_timesteps


# --- train + validate (shared by the CLI and Optuna trials) ---------------------------------


TrackFactory = Callable[[RunSpec], AbstractContextManager[TrackedRun]]


@dataclass(frozen=True)
class DQNRunSettings:
    """Everything, besides hyperparameters, that defines one DQN training run."""

    timesteps: int
    seed: int
    config: SimulatorConfig
    config_source: SimulatorConfigSource
    calibration_workload_ids: tuple[str, ...]
    calibration_note: str | None
    reward_weights: RewardWeights
    log_interval: int = DEFAULT_LOG_INTERVAL


@dataclass(frozen=True)
class DQNRunOutcome:
    training_run_id: str
    validation_run_ids: tuple[str, ...]
    validation_metrics: dict[str, EpisodeMetrics]
    aggregate: dict[str, float]
    compatibility: EnvironmentCompatibility
    training_episodes: int


def dqn_run_spec(
    run_kind: RunKind,
    entry: WorkloadEntry,
    settings: DQNRunSettings,
    hyperparameters: Mapping[str, JsonValue],
) -> RunSpec:
    return RunSpec(
        run_kind=run_kind,
        controller="dqn",
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


def run_params(
    hyperparameters: DQNHyperparameters, extra: Mapping[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """``hp.*`` params of a DQN run: algorithm, config version, and every setting."""
    return {
        "algorithm": "sb3-dqn",
        "dqn_config_version": DQN_CONFIG_VERSION,
        **hyperparameters.as_params(),
        **dict(extra or {}),
    }


def train_and_validate(
    *,
    training_entry: WorkloadEntry,
    validation_entries: Sequence[WorkloadEntry],
    traces: Mapping[str, WorkloadTrace],
    hyperparameters: DQNHyperparameters,
    settings: DQNRunSettings,
    track: TrackFactory,
    training_run_kind: RunKind,
    validation_run_kind: RunKind,
    extra_params: Mapping[str, JsonValue] | None = None,
) -> DQNRunOutcome:
    """Train on one workload (one tracked run), then validate the final policy per workload.

    The model is saved as a bundle and logged under ``model/`` on the training
    run. Validation loads that bundle back through :func:`load_sb3_controller`
    (so the compatibility check is exercised) and evaluates it with the shared
    ``evaluate_controller_episode``, one tracked run per validation workload.
    """
    require_exact_timesteps(settings.timesteps, hyperparameters)
    params = run_params(hyperparameters, extra_params)
    config, weights = settings.config, settings.reward_weights
    with track(dqn_run_spec(training_run_kind, training_entry, settings, params)) as run:
        env = AutoscalingEnv(config, traces[training_entry.id], weights)
        model = build_dqn(Monitor(env), hyperparameters, seed=settings.seed)
        callback = MLflowTrainingCallback(run, log_interval=settings.log_interval)
        started = time.perf_counter()
        model.learn(total_timesteps=settings.timesteps, callback=callback)
        if model.num_timesteps != settings.timesteps:
            raise RuntimeError(
                f"SB3 trained {model.num_timesteps} timesteps, not the recorded "
                f"{settings.timesteps}"
            )
        run.log_metrics(
            {
                "training_timesteps": float(model.num_timesteps),
                "training_episodes": float(callback.episode_count),
                "training_seconds": time.perf_counter() - started,
            }
        )
        compatibility = EnvironmentCompatibility.from_env(env, load_benchmark_manifest().version)
        metadata = ModelMetadata(
            algorithm="dqn",
            config_version=DQN_CONFIG_VERSION,
            benchmark_version=load_benchmark_manifest().version,
            training_workload_id=training_entry.id,
            training_workload_split="train",
            seed=settings.seed,
            total_timesteps=settings.timesteps,
            hyperparameters=hyperparameters.as_params(),
            scalerl_version=scalerl.__version__,
            training_run_id=run.run_id,
        )

        with tempfile.TemporaryDirectory() as directory:
            bundle = save_model_bundle(
                model, Path(directory) / "bundle", metadata=metadata, compatibility=compatibility
            )
            for path in sorted(bundle.iterdir()):
                run.log_artifact(path, artifact_path=MODEL_ARTIFACT_PATH)

            results: dict[str, EpisodeMetrics] = {}
            validation_run_ids = []
            for entry in validation_entries:
                validation_env = AutoscalingEnv(config, traces[entry.id], weights)
                controller = load_sb3_controller(bundle, validation_env)
                spec = dqn_run_spec(validation_run_kind, entry, settings, params)
                with track(spec) as validation_run:
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
    return DQNRunOutcome(
        training_run_id=run.run_id,
        validation_run_ids=tuple(validation_run_ids),
        validation_metrics=results,
        aggregate=aggregate,
        compatibility=compatibility,
        training_episodes=callback.episode_count,
    )


def aggregate_validation(results: Sequence[EpisodeMetrics]) -> dict[str, float]:
    """Mean system metrics over validation workloads (reward stays secondary)."""
    if not results:
        raise ValueError("no validation results to aggregate")
    return {key: statistics.fmean(getattr(r, key) for r in results) for key in VALIDATION_KEYS}


# --- training entry point -----------------------------------------------------------------------


class DQNTrainingResult(BaseModel):
    """Reproducible record of one DQN training run and its validation (no model bytes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["dqn"] = "dqn"
    dqn_config_version: str
    benchmark_version: str
    hyperparameter_source: str
    training_run_id: str
    training_workload_id: str
    validation_workload_ids: tuple[str, ...]
    validation_run_ids: tuple[str, ...]
    seed: int
    timesteps: int
    training_episodes: int
    hyperparameters: DQNHyperparameters
    simulator_config_source: SimulatorConfigSource
    model_artifact_path: str
    model_artifact_uri: str
    compatibility: EnvironmentCompatibility
    validation_metrics: dict[str, float]
    validation_metrics_by_workload: dict[str, dict[str, float]]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> DQNTrainingResult:
        return cls.model_validate_json(Path(path).read_text())


def train_dqn(
    *,
    workload_id: str,
    validation_workload_ids: Sequence[str] | None = None,
    hyperparameters: DQNHyperparameters | None = None,
    hyperparameter_source: str = DQN_CONFIG_VERSION,
    timesteps: int = DEFAULT_TIMESTEPS,
    seed: int = 0,
    config: SimulatorConfig | None = None,
    config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    reward_weights: RewardWeights | None = None,
    azure_csv_path: str | Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-dqn",
    log_interval: int = DEFAULT_LOG_INTERVAL,
) -> DQNTrainingResult:
    """Train DQN on one TRAIN workload and validate it; one tracked run per workload.

    Every guardrail (train split, validation split, simulator-config
    provenance) runs before any trace is built or any model is trained.
    """
    hyperparameters = hyperparameters or DQNHyperparameters()
    require_exact_timesteps(timesteps, hyperparameters)
    training_entry = require_training_workload(workload_id)
    validation_ids = (
        tuple(validation_workload_ids)
        if validation_workload_ids is not None
        else default_validation_workload_ids()
    )
    validation_entries = require_validation_workloads(validation_ids)
    settings = DQNRunSettings(
        timesteps=timesteps,
        seed=seed,
        config=config or SimulatorConfig(),
        config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=reward_weights or RewardWeights(),
        log_interval=log_interval,
    )
    params = run_params(hyperparameters, {"hyperparameter_source": hyperparameter_source})
    for entry in (training_entry, *validation_entries):  # provenance checks, before training
        dqn_run_spec("train" if entry is training_entry else "evaluate", entry, settings, params)
    traces = build_workloads([training_entry, *validation_entries], azure_csv_path=azure_csv_path)

    from scalerl.mlops import start_tracked_run

    def track(spec: RunSpec) -> AbstractContextManager[TrackedRun]:
        return start_tracked_run(spec, tracking_uri=tracking_uri, experiment_name=experiment_name)

    outcome = train_and_validate(
        training_entry=training_entry,
        validation_entries=validation_entries,
        traces=traces,
        hyperparameters=hyperparameters,
        settings=settings,
        track=track,
        training_run_kind="train",
        validation_run_kind="evaluate",
        extra_params={"hyperparameter_source": hyperparameter_source},
    )
    return DQNTrainingResult(
        dqn_config_version=DQN_CONFIG_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        hyperparameter_source=hyperparameter_source,
        training_run_id=outcome.training_run_id,
        training_workload_id=training_entry.id,
        validation_workload_ids=tuple(entry.id for entry in validation_entries),
        validation_run_ids=outcome.validation_run_ids,
        seed=seed,
        timesteps=timesteps,
        training_episodes=outcome.training_episodes,
        hyperparameters=hyperparameters,
        simulator_config_source=config_source,
        model_artifact_path=MODEL_ARTIFACT_PATH,
        model_artifact_uri=f"runs:/{outcome.training_run_id}/{MODEL_ARTIFACT_PATH}",
        compatibility=outcome.compatibility,
        validation_metrics=outcome.aggregate,
        validation_metrics_by_workload={
            key: value.as_metrics() for key, value in outcome.validation_metrics.items()
        },
    )


# --- command line ------------------------------------------------------------------------


def load_hyperparameters(path: str | Path) -> tuple[DQNHyperparameters, str]:
    """Read a DQNHyperparameters JSON, or a DQN tuning result's selected configuration."""
    payload = json.loads(Path(path).read_text())
    if "selected_hyperparameters" in payload:
        source = f"optuna:{payload['study_name']}#trial{payload['selected_trial_number']}"
        selected = payload["selected_hyperparameters"]
        return DQNHyperparameters.model_validate(selected, strict=False), source
    return DQNHyperparameters.model_validate(payload, strict=False), f"file:{Path(path).name}"


def apply_overrides(
    hyperparameters: DQNHyperparameters, overrides: Sequence[str]
) -> DQNHyperparameters:
    """Apply ``name=value`` overrides (JSON values; ``net_arch=128,128`` also accepted)."""
    if not overrides:
        return hyperparameters
    values = hyperparameters.model_dump()
    for item in overrides:
        name, sep, raw = item.partition("=")
        if not sep or name not in DQNHyperparameters.model_fields:
            raise ValueError(f"invalid override {item!r}; use <hyperparameter>=<value>")
        if name == "net_arch":
            values[name] = tuple(int(width) for width in raw.split(","))
        else:
            values[name] = json.loads(raw)
    return DQNHyperparameters.model_validate(values, strict=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SB3 DQN autoscaler (#15).")
    parser.add_argument("--workload", required=True, help="TRAIN workload ID")
    parser.add_argument(
        "--validation-workload",
        action="append",
        dest="validation_workloads",
        help="VALIDATION workload ID (repeatable); defaults to the synthetic validation set",
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        help="DQNHyperparameters JSON or a DQN tuning result (uses its selected config)",
    )
    parser.add_argument(
        "--hp", action="append", default=[], metavar="NAME=VALUE", help="override one setting"
    )
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
    parser.add_argument("--reward-weights", type=Path, help="RewardWeights JSON file")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-dqn")
    parser.add_argument("--output", type=Path, default=Path("outputs/dqn-v1.json"))
    args = parser.parse_args(argv)

    if args.hyperparameters is not None:
        hyperparameters, source = load_hyperparameters(args.hyperparameters)
    else:
        hyperparameters, source = DQNHyperparameters(), DQN_CONFIG_VERSION
    if args.hp:
        hyperparameters = apply_overrides(hyperparameters, args.hp)
        source = f"{source}+overrides"
    source_value: SimulatorConfigSource = args.config_source
    result = train_dqn(
        workload_id=args.workload,
        validation_workload_ids=args.validation_workloads,
        hyperparameters=hyperparameters,
        hyperparameter_source=source,
        timesteps=args.timesteps,
        seed=args.seed,
        config=(
            SimulatorConfig.model_validate_json(args.simulator_config.read_text())
            if args.simulator_config
            else None
        ),
        config_source=source_value,
        calibration_workload_ids=args.calibration_workload,
        calibration_note=args.calibration_note,
        reward_weights=(
            RewardWeights.model_validate_json(args.reward_weights.read_text())
            if args.reward_weights
            else None
        ),
        azure_csv_path=args.azure_csv,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        log_interval=args.log_interval,
    )
    path = result.save(args.output)
    print(f"training run {result.training_run_id} ({result.training_workload_id})")
    runs = dict(zip(result.validation_workload_ids, result.validation_run_ids, strict=True))
    print(f"validation runs {runs}")
    print(f"validation metrics (mean): {result.validation_metrics}")
    print(f"model: {result.model_artifact_uri}; result written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
