"""Run contract, lineage, and compatibility metadata; importable without MLflow."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Self

from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

import scalerl
from scalerl.benchmarks import load_benchmark_manifest
from scalerl.benchmarks.manifest import Split
from scalerl.environment.config import SimulatorConfig
from scalerl.environment.gym_env import AutoscalingEnv
from scalerl.environment.reward import RewardWeights
from scalerl.workloads import steady_workload

RunKind = Literal["train", "tune", "evaluate"]
SimulatorConfigSource = Literal["default", "predeclared", "calibrated_train_validation"]

# Runs of these kinds must never see held-out workloads.
_NON_EVALUATION_KINDS = ("train", "tune")
_CALIBRATION_SPLITS: tuple[Split, ...] = ("train", "validation")
_TRACKED_PACKAGES = (
    "mlflow",
    "gymnasium",
    "numpy",
    "pandas",
    "pydantic",
    "stable-baselines3",
    "torch",
)


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class RunSpec(_Strict):
    """Everything needed to identify and reproduce one tracked run.

    ``workload_split`` must match the benchmark manifest. ``train`` and
    ``tune`` runs may not target ``test`` workloads, and simulator-config
    calibration may cite only ``train``/``validation`` workloads.
    """

    run_kind: RunKind
    controller: str = Field(min_length=1)
    benchmark_version: Literal["v1"] = "v1"
    workload_id: str = Field(min_length=1)
    workload_split: Split
    simulator_config: SimulatorConfig
    simulator_config_source: SimulatorConfigSource
    calibration_workload_ids: tuple[str, ...] = ()
    calibration_note: str | None = None
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)
    seed: int | None = None
    evaluation_seeds: tuple[int, ...] = ()
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)
    training_steps: int | None = Field(default=None, ge=0)
    training_episodes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_lineage(self) -> Self:
        manifest = load_benchmark_manifest()
        splits = {entry.id: entry.split for entry in manifest.workloads}

        actual = splits.get(self.workload_id)
        if actual is None:
            raise ValueError(
                f"workload {self.workload_id!r} is not in benchmark {manifest.version}"
            )
        if actual != self.workload_split:
            raise ValueError(
                f"workload {self.workload_id!r} is in split {actual!r}, not {self.workload_split!r}"
            )
        if self.run_kind in _NON_EVALUATION_KINDS and actual == "test":
            raise ValueError(
                f"{self.run_kind} runs may not use held-out test workload {self.workload_id!r}"
            )

        if self.simulator_config_source == "calibrated_train_validation":
            if not self.calibration_workload_ids:
                raise ValueError("calibrated simulator configs must list calibration_workload_ids")
        elif self.calibration_workload_ids:
            raise ValueError(
                "calibration_workload_ids require simulator_config_source="
                "'calibrated_train_validation'"
            )
        for workload_id in self.calibration_workload_ids:
            split = splits.get(workload_id)
            if split is None:
                raise ValueError(f"calibration workload {workload_id!r} is not in the benchmark")
            if split not in _CALIBRATION_SPLITS:
                raise ValueError(
                    f"calibration may use train/validation workloads only; "
                    f"{workload_id!r} is a held-out {split} workload"
                )

        if self.simulator_config_source == "default" and self.simulator_config != SimulatorConfig():
            raise ValueError(
                "simulator_config_source='default' requires the default SimulatorConfig; "
                "use 'predeclared' or 'calibrated_train_validation' for custom values"
            )
        return self


class EnvironmentCompatibility(_Strict):
    """Observation/action contract a learned policy depends on.

    Besides the shapes, it holds every ``SimulatorConfig`` field that
    ``AutoscalingEnv`` uses to build or normalize observations, so two
    environments are compatible only if each feature means the same thing.
    ``min_replicas`` and ``initial_replicas`` are excluded: they change dynamics
    and the start state, not what any feature measures.
    """

    observation_shape: tuple[int, ...]
    action_count: int
    startup_delay_seconds: float
    control_interval_seconds: float
    episode_duration_seconds: float
    max_replicas: int
    service_capacity_rps: float
    cost_per_hour: float
    latency_target_seconds: float
    benchmark_version: str | None = None

    @classmethod
    def from_env(cls, env: AutoscalingEnv, benchmark_version: str | None = None) -> Self:
        shape = env.observation_space.shape
        if shape is None:
            raise ValueError("environment observation space has no shape")
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("environment action space must be Discrete")
        replicas = env.config.replicas
        return cls(
            observation_shape=tuple(int(size) for size in shape),
            action_count=int(env.action_space.n),
            startup_delay_seconds=replicas.startup_delay_seconds,
            control_interval_seconds=env.config.timing.control_interval_seconds,
            episode_duration_seconds=env.config.timing.episode_duration_seconds,
            max_replicas=replicas.max_replicas,
            service_capacity_rps=replicas.service_capacity_rps,
            cost_per_hour=replicas.cost_per_hour,
            latency_target_seconds=env.config.sla.latency_target_seconds,
            benchmark_version=benchmark_version,
        )

    @classmethod
    def from_config(cls, config: SimulatorConfig, benchmark_version: str | None = None) -> Self:
        """Derive the contract from a real ``AutoscalingEnv`` built on an idle trace."""
        trace = steady_workload(
            duration_seconds=config.timing.episode_duration_seconds,
            control_interval_seconds=config.timing.control_interval_seconds,
            rate=0.0,
        )
        return cls.from_env(AutoscalingEnv(config, trace), benchmark_version)

    def mismatches(self, other: EnvironmentCompatibility) -> dict[str, tuple[Any, Any]]:
        """Return ``{field: (self, other)}`` for every differing field."""
        mine, theirs = self.model_dump(), other.model_dump()
        return {key: (mine[key], theirs[key]) for key in mine if mine[key] != theirs[key]}

    def require_compatible(self, other: EnvironmentCompatibility) -> None:
        """Raise ``ValueError`` if ``other`` differs in any recorded field."""
        differences = self.mismatches(other)
        if differences:
            details = ", ".join(f"{key}: {a!r} != {b!r}" for key, (a, b) in differences.items())
            raise ValueError(f"incompatible environment: {details}")


def software_metadata(git_sha: str | None = None) -> dict[str, Any]:
    """Return code and dependency versions; never fails outside a Git checkout.

    The Git SHA comes from ``git_sha``, then ``SCALERL_GIT_SHA`` or
    ``GITHUB_SHA``, then ``git rev-parse`` next to the package, else
    ``"unknown"``.
    """
    sha, dirty = _git_state(git_sha)
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "scalerl_version": scalerl.__version__,
        "python_version": platform.python_version(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
    }


def _git_state(override: str | None) -> tuple[str, bool | None]:
    for sha in (override, os.environ.get("SCALERL_GIT_SHA"), os.environ.get("GITHUB_SHA")):
        if sha:
            return sha, None
    package_dir = Path(scalerl.__file__).resolve().parent
    sha = _git(package_dir, "rev-parse", "HEAD")
    if sha is None:
        return "unknown", None
    status = _git(package_dir, "status", "--porcelain")
    return sha, None if status is None else bool(status)


def _git(directory: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"
