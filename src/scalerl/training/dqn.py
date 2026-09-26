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
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from gymnasium import Env
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from stable_baselines3 import DQN

from scalerl.benchmarks import WorkloadEntry, load_benchmark_manifest
from scalerl.environment import SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.mlops import EnvironmentCompatibility, RunKind, RunSpec, SimulatorConfigSource
from scalerl.training import common
from scalerl.training.common import (
    DEFAULT_LOG_INTERVAL,
    MODEL_ARTIFACT_PATH,
    MODEL_SOURCE_TAG,
    VALIDATION_KEYS,
    AlgorithmSpec,
    MLflowTrainingCallback,
    RunOutcome,
    TrackFactory,
    TrainingSettings,
    aggregate_validation,
    default_validation_workload_ids,
)
from scalerl.workloads import WorkloadTrace

__all__ = [
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_TIMESTEPS",
    "DQN",
    "DQN_ALGORITHM",
    "DQN_CONFIG_VERSION",
    "MODEL_ARTIFACT_PATH",
    "MODEL_SOURCE_TAG",
    "VALIDATION_KEYS",
    "DQNHyperparameters",
    "DQNRunSettings",
    "DQNTrainingResult",
    "MLflowTrainingCallback",
    "aggregate_validation",
    "apply_overrides",
    "build_dqn",
    "default_validation_workload_ids",
    "dqn_run_spec",
    "load_hyperparameters",
    "main",
    "require_exact_timesteps",
    "require_training_workload",
    "require_validation_workloads",
    "run_params",
    "train_and_validate",
    "train_dqn",
]

DQN_CONFIG_VERSION = "dqn-v1"
DEFAULT_TIMESTEPS = 200_000
# SB3 logger values that form DQN's learning curve (key → MLflow metric).
DQN_RECORDED_METRICS = (
    ("train/loss", "train/loss"),
    ("train/n_updates", "train/n_updates"),
    ("rollout/exploration_rate", "train/exploration_rate"),
)
DQNRunSettings = TrainingSettings  # backwards-compatible name
DQNRunOutcome = RunOutcome


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


def build_dqn(env: Env[Any, Any], hyperparameters: DQNHyperparameters, *, seed: int) -> DQN:
    """A CPU SB3 DQN on ``env``, seeded for reproducible setup (network init, exploration)."""
    return DQN("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **hyperparameters.sb3_kwargs())


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


DQN_ALGORITHM = AlgorithmSpec(
    name="dqn",
    label="DQN",
    config_version=DQN_CONFIG_VERSION,
    build=lambda env, hyperparameters, seed: build_dqn(env, hyperparameters, seed=seed),
    run_params=run_params,
    rollout_size=lambda hyperparameters: hyperparameters.train_freq,
    rollout_unit="train_freq",
    recorded_metrics=DQN_RECORDED_METRICS,
)


# --- guardrails (shared implementation, DQN wording) -----------------------------------------


def require_training_workload(workload_id: str) -> WorkloadEntry:
    """The manifest entry of a TRAIN workload; validation and held-out test are refused."""
    return common.require_training_workload(workload_id, "DQN")


def require_validation_workloads(workload_ids: Sequence[str]) -> tuple[WorkloadEntry, ...]:
    """Manifest entries of VALIDATION workloads; test (and train) workloads are refused."""
    return common.require_validation_workloads(workload_ids, "DQN")


def require_exact_timesteps(timesteps: int, hyperparameters: DQNHyperparameters) -> None:
    """Refuse budgets SB3 DQN cannot train exactly (see ``common.require_whole_rollouts``).

    SB3's off-policy loop checks the budget only between rollouts of
    ``train_freq`` steps (e.g. 1 → 4, 97 → 100 with ``train_freq=4``).
    """
    common.require_whole_rollouts(
        timesteps, hyperparameters.train_freq, unit="train_freq", label="SB3 DQN"
    )


def dqn_run_spec(
    run_kind: RunKind,
    entry: WorkloadEntry,
    settings: TrainingSettings,
    hyperparameters: Mapping[str, JsonValue],
) -> RunSpec:
    return common.build_run_spec(run_kind, "dqn", entry, settings, hyperparameters)


def train_and_validate(
    *,
    training_entry: WorkloadEntry,
    validation_entries: Sequence[WorkloadEntry],
    traces: Mapping[str, WorkloadTrace],
    hyperparameters: DQNHyperparameters,
    settings: TrainingSettings,
    track: TrackFactory,
    training_run_kind: RunKind,
    validation_run_kind: RunKind,
    extra_params: Mapping[str, JsonValue] | None = None,
) -> RunOutcome:
    """DQN through the shared train → bundle → reload → validate pipeline."""
    return common.train_and_validate(
        DQN_ALGORITHM,
        training_entry=training_entry,
        validation_entries=validation_entries,
        traces=traces,
        hyperparameters=hyperparameters,
        settings=settings,
        track=track,
        training_run_kind=training_run_kind,
        validation_run_kind=validation_run_kind,
        extra_params=extra_params,
    )


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

    Every guardrail (budget, train split, validation split, simulator-config
    provenance) runs before any trace is built or any model is trained.
    """
    hyperparameters = hyperparameters or DQNHyperparameters()
    settings = TrainingSettings(
        timesteps=timesteps,
        seed=seed,
        config=config or SimulatorConfig(),
        config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=reward_weights or RewardWeights(),
        log_interval=log_interval,
    )
    trained = common.train_sb3(
        DQN_ALGORITHM,
        hyperparameters=hyperparameters,
        hyperparameter_source=hyperparameter_source,
        workload_id=workload_id,
        validation_workload_ids=validation_workload_ids,
        settings=settings,
        azure_csv_path=azure_csv_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    outcome = trained.outcome
    return DQNTrainingResult(
        dqn_config_version=DQN_CONFIG_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        hyperparameter_source=hyperparameter_source,
        training_run_id=outcome.training_run_id,
        training_workload_id=trained.training_entry.id,
        validation_workload_ids=tuple(entry.id for entry in trained.validation_entries),
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
    return common.load_hyperparameters(path, DQNHyperparameters)


def apply_overrides(
    hyperparameters: DQNHyperparameters, overrides: Sequence[str]
) -> DQNHyperparameters:
    """Apply ``name=value`` overrides (JSON values; ``net_arch=128,128`` also accepted)."""
    return common.apply_overrides(hyperparameters, overrides)


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
