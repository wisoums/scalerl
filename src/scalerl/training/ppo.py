"""Train a Stable-Baselines3 PPO autoscaler on one TRAIN workload (#16).

PPO is ScaleRL's second deep-RL controller. Instead of DQN's Q-values it learns
an **actor** π(action | state), a probability over the three actions (0 scale
down, 1 hold, 2 scale up), and a **critic** V(state), the expected return
from a state, which provides the advantages for the actor's update. Training
alternates: collect a rollout of ``n_steps`` transitions with the current
policy → estimate advantages (GAE) → update actor and critic for ``n_epochs``
over minibatches → collect the next rollout. There is no replay buffer,
target network, or epsilon exploration (those are DQN's); PPO explores by
sampling its own stochastic policy. Evaluation is deterministic (most likely
action) through the same ``SB3Controller`` as DQN.

Same information and reward as DQN: the frozen v1 observation (#58) and
``AutoscalingEnv``'s reward, nothing else. Normalization policy (v1):

* observation: already normalized by ``AutoscalingEnv`` (``env-v1``); no
  SB3 ``VecNormalize``;
* reward: the ScaleRL reward as is; no ``VecNormalize`` reward scaling;
* advantages: PPO's own per-minibatch ``normalize_advantage=True`` (an
  optimizer detail, not a change to observations or rewards).

Shares the training pipeline with DQN (``scalerl.training.common``): one
workload per MLflow run, held-out test refused before training, exact rollout
budgets, compatibility-checked ``model/`` bundle reloaded for validation, and
periodic ``checkpoints/``.

Run locally::

    python -m scalerl.training.ppo --workload syn-train-bursty \\
        --validation-workload syn-val-bursty --timesteps 204800 --seed 0 \\
        --tracking-uri sqlite:///outputs/mlflow.db --output outputs/ppo-v1.json
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from gymnasium import Env
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from stable_baselines3 import PPO

from scalerl.benchmarks import load_benchmark_manifest
from scalerl.environment import SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.mlops import EnvironmentCompatibility, SimulatorConfigSource
from scalerl.training import common
from scalerl.training.common import (
    DEFAULT_LOG_INTERVAL,
    MODEL_ARTIFACT_PATH,
    AlgorithmSpec,
    TrainingSettings,
)

PPO_CONFIG_VERSION = "ppo-v1"
DEFAULT_TIMESTEPS = 204_800  # exactly 100 rollouts of n_steps=2048 (~200k, like DQN)
DEFAULT_CHECKPOINT_INTERVAL = 51_200  # every 25 rollouts → 4 checkpoints per default run
OBSERVATION_NORMALIZATION = "env-v1"  # AutoscalingEnv's own normalization; no VecNormalize
REWARD_NORMALIZATION = "none"  # the ScaleRL reward as is; no VecNormalize
# SB3 logger values that form PPO's learning curve (key → MLflow metric).
PPO_RECORDED_METRICS = (
    ("train/loss", "train/loss"),
    ("train/n_updates", "train/n_updates"),
    ("train/policy_gradient_loss", "train/policy_gradient_loss"),
    ("train/value_loss", "train/value_loss"),
    ("train/entropy_loss", "train/entropy_loss"),
    ("train/approx_kl", "train/approx_kl"),
    ("train/clip_fraction", "train/clip_fraction"),
    ("train/clip_range", "train/clip_range"),
    ("train/explained_variance", "train/explained_variance"),
    ("train/learning_rate", "train/learning_rate"),
)


class PPOHyperparameters(BaseModel):
    """The PPO settings ScaleRL exposes; defaults are the predeclared ``ppo-v1`` config.

    They are SB3's PPO defaults: one 2048-step rollout per update (with one
    environment), 64-sample minibatches (32 per epoch, dividing the rollout
    exactly, so SB3 never truncates a minibatch), 10 epochs, and separate
    64×64 tanh actor and critic networks. Documented starting values, not
    tuned, not claimed optimal.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    learning_rate: float = Field(default=3e-4, gt=0, le=1)
    n_steps: int = Field(default=2_048, ge=2)
    batch_size: int = Field(default=64, ge=2)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, le=1)
    normalize_advantage: bool = True
    ent_coef: float = Field(default=0.0, ge=0, le=1)
    vf_coef: float = Field(default=0.5, gt=0, le=10)
    max_grad_norm: float = Field(default=0.5, gt=0, le=100)
    net_arch: tuple[int, ...] = Field(default=(64, 64), min_length=1)

    @model_validator(mode="after")
    def _check(self) -> PPOHyperparameters:
        if any(width < 1 for width in self.net_arch):
            raise ValueError("net_arch layer widths must be positive")
        if self.batch_size > self.n_steps:
            raise ValueError("batch_size must not exceed the rollout size (n_steps)")
        if self.n_steps % self.batch_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must divide the rollout size "
                f"(n_steps={self.n_steps}) so no minibatch is truncated"
            )
        return self

    def sb3_kwargs(self) -> dict[str, Any]:
        values = self.model_dump()
        net_arch = list(values.pop("net_arch"))
        # Separate actor (pi) and critic (vf) networks of the same widths.
        return {**values, "policy_kwargs": {"net_arch": {"pi": net_arch, "vf": net_arch}}}

    def as_params(self) -> dict[str, JsonValue]:
        params: dict[str, JsonValue] = self.model_dump(mode="json")
        return params


def build_ppo(env: Env[Any, Any], hyperparameters: PPOHyperparameters, *, seed: int) -> PPO:
    """A CPU SB3 PPO on ``env`` (no VecNormalize), seeded for reproducible setup."""
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **hyperparameters.sb3_kwargs())


def run_params(
    hyperparameters: PPOHyperparameters, extra: Mapping[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """``hp.*`` params of a PPO run: algorithm, config version, normalization, every setting."""
    return {
        "algorithm": "sb3-ppo",
        "ppo_config_version": PPO_CONFIG_VERSION,
        "observation_normalization": OBSERVATION_NORMALIZATION,
        "reward_normalization": REWARD_NORMALIZATION,
        **hyperparameters.as_params(),
        **dict(extra or {}),
    }


PPO_ALGORITHM = AlgorithmSpec(
    name="ppo",
    label="PPO",
    config_version=PPO_CONFIG_VERSION,
    build=lambda env, hyperparameters, seed: build_ppo(env, hyperparameters, seed=seed),
    run_params=run_params,
    rollout_size=lambda hyperparameters: hyperparameters.n_steps,
    rollout_unit="n_steps",
    recorded_metrics=PPO_RECORDED_METRICS,
)


# --- training entry point -----------------------------------------------------------------------


class PPOTrainingResult(BaseModel):
    """Reproducible record of one PPO training run and its validation (no model bytes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    algorithm: Literal["ppo"] = "ppo"
    ppo_config_version: str
    benchmark_version: str
    hyperparameter_source: str
    training_run_id: str
    training_workload_id: str
    validation_workload_ids: tuple[str, ...]
    validation_run_ids: tuple[str, ...]
    seed: int
    timesteps: int
    rollouts: int
    training_episodes: int
    hyperparameters: PPOHyperparameters
    observation_normalization: str
    reward_normalization: str
    simulator_config_source: SimulatorConfigSource
    model_artifact_path: str
    model_artifact_uri: str
    checkpoint_interval: int | None
    checkpoints: tuple[dict[str, Any], ...]
    compatibility: EnvironmentCompatibility
    validation_metrics: dict[str, float]
    validation_metrics_by_workload: dict[str, dict[str, float]]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> PPOTrainingResult:
        return cls.model_validate_json(Path(path).read_text())


def train_ppo(
    *,
    workload_id: str,
    validation_workload_ids: Sequence[str] | None = None,
    hyperparameters: PPOHyperparameters | None = None,
    hyperparameter_source: str = PPO_CONFIG_VERSION,
    timesteps: int = DEFAULT_TIMESTEPS,
    seed: int = 0,
    config: SimulatorConfig | None = None,
    config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    reward_weights: RewardWeights | None = None,
    azure_csv_path: str | Path | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-ppo",
    log_interval: int = DEFAULT_LOG_INTERVAL,
    checkpoint_interval: int | None = DEFAULT_CHECKPOINT_INTERVAL,
) -> PPOTrainingResult:
    """Train PPO on one TRAIN workload and validate it; one tracked run per workload.

    Every guardrail (exact rollout budget, checkpoint interval, train split,
    validation split, simulator-config provenance) runs before any trace is
    built or any model is trained. ``checkpoint_interval=None`` disables
    checkpoints.
    """
    hyperparameters = hyperparameters or PPOHyperparameters()
    settings = TrainingSettings(
        timesteps=timesteps,
        seed=seed,
        config=config or SimulatorConfig(),
        config_source=config_source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        reward_weights=reward_weights or RewardWeights(),
        log_interval=log_interval,
        checkpoint_interval=checkpoint_interval,
    )
    trained = common.train_sb3(
        PPO_ALGORITHM,
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
    return PPOTrainingResult(
        ppo_config_version=PPO_CONFIG_VERSION,
        benchmark_version=load_benchmark_manifest().version,
        hyperparameter_source=hyperparameter_source,
        training_run_id=outcome.training_run_id,
        training_workload_id=trained.training_entry.id,
        validation_workload_ids=tuple(entry.id for entry in trained.validation_entries),
        validation_run_ids=outcome.validation_run_ids,
        seed=seed,
        timesteps=timesteps,
        rollouts=timesteps // hyperparameters.n_steps,
        training_episodes=outcome.training_episodes,
        hyperparameters=hyperparameters,
        observation_normalization=OBSERVATION_NORMALIZATION,
        reward_normalization=REWARD_NORMALIZATION,
        simulator_config_source=config_source,
        model_artifact_path=MODEL_ARTIFACT_PATH,
        model_artifact_uri=f"runs:/{outcome.training_run_id}/{MODEL_ARTIFACT_PATH}",
        checkpoint_interval=checkpoint_interval,
        checkpoints=outcome.checkpoints,
        compatibility=outcome.compatibility,
        validation_metrics=outcome.aggregate,
        validation_metrics_by_workload={
            key: value.as_metrics() for key, value in outcome.validation_metrics.items()
        },
    )


# --- command line ------------------------------------------------------------------------


def load_hyperparameters(path: str | Path) -> tuple[PPOHyperparameters, str]:
    """Read a PPOHyperparameters JSON, or a PPO tuning result's selected configuration."""
    return common.load_hyperparameters(path, PPOHyperparameters)


def apply_overrides(
    hyperparameters: PPOHyperparameters, overrides: Sequence[str]
) -> PPOHyperparameters:
    """Apply ``name=value`` overrides (JSON values; ``net_arch=128,128`` also accepted)."""
    return common.apply_overrides(hyperparameters, overrides)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SB3 PPO autoscaler (#16).")
    parser.add_argument("--workload", required=True, help="TRAIN workload ID")
    parser.add_argument(
        "--validation-workload",
        action="append",
        dest="validation_workloads",
        help="VALIDATION workload ID (repeatable); defaults to the synthetic validation set",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="must be a multiple of n_steps (default 204800 = 100 rollouts of 2048)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        help="PPOHyperparameters JSON or a PPO tuning result (uses its selected config)",
    )
    parser.add_argument(
        "--hp", action="append", default=[], metavar="NAME=VALUE", help="override one setting"
    )
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="timesteps between checkpoints (a multiple of n_steps); 0 disables them",
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
    parser.add_argument("--reward-weights", type=Path, help="RewardWeights JSON file")
    parser.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
    parser.add_argument("--experiment-name", default="scalerl-ppo")
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo-v1.json"))
    args = parser.parse_args(argv)

    if args.hyperparameters is not None:
        hyperparameters, source = load_hyperparameters(args.hyperparameters)
    else:
        hyperparameters, source = PPOHyperparameters(), PPO_CONFIG_VERSION
    if args.hp:
        hyperparameters = apply_overrides(hyperparameters, args.hp)
        source = f"{source}+overrides"
    source_value: SimulatorConfigSource = args.config_source
    result = train_ppo(
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
        checkpoint_interval=args.checkpoint_interval or None,
    )
    path = result.save(args.output)
    print(f"training run {result.training_run_id} ({result.training_workload_id})")
    runs = dict(zip(result.validation_workload_ids, result.validation_run_ids, strict=True))
    print(f"validation runs {runs}")
    print(f"validation metrics (mean): {result.validation_metrics}")
    print(f"checkpoints: {[c['file'] for c in result.checkpoints]}")
    print(f"model: {result.model_artifact_uri}; result written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
