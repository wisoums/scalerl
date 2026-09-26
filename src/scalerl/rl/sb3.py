"""Stable-Baselines3 policies as ordinary ScaleRL controllers, and their model bundles.

Supported algorithms: ``dqn`` (#15) and ``ppo`` (#16); a bundle's
``metadata.json`` names its algorithm, so one loader serves both.

A learned policy may use **only the observation**: :class:`SB3Controller`
ignores ``info`` entirely (unlike the rule-based baselines, which read
documented raw values from it) and always predicts deterministically.

A model bundle is a directory that can be used outside the process that
trained it::

    <bundle>/model.zip           SB3's native model (the canonical policy artifact)
    <bundle>/compatibility.json  EnvironmentCompatibility of the training env
    <bundle>/metadata.json       ModelMetadata (algorithm, workload, seed, ...)

:func:`load_sb3_controller` refuses to run a policy in an environment whose
compatibility contract differs from the one it was trained on. Comparing the
observation shape is not enough: two environments can have the same shape
while their features mean different things.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from scalerl.benchmarks import load_benchmark_manifest
from scalerl.environment.gym_env import AutoscalingEnv, Observation
from scalerl.mlops.spec import EnvironmentCompatibility

BUNDLE_VERSION: Final = "scalerl-sb3-bundle-v1"
MODEL_FILE = "model.zip"
COMPATIBILITY_FILE = "compatibility.json"
METADATA_FILE = "metadata.json"
Algorithm = Literal["dqn", "ppo"]
_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"dqn": DQN, "ppo": PPO}


class SB3Controller:
    """Deterministic :class:`~scalerl.controllers.Controller` for a discrete SB3 policy.

    Uses only the observation; ``info`` is ignored by design so the policy can
    never read raw simulator values, the trace, or anything about the future.
    Feed-forward SB3 policies are stateless between ticks, so ``reset`` does
    nothing and never changes the model.
    """

    def __init__(
        self, model: BaseAlgorithm, *, perturbed_compatibility: tuple[str, ...] = ()
    ) -> None:
        if not isinstance(model.action_space, spaces.Discrete):
            raise ValueError("SB3Controller needs a policy with a Discrete action space")
        self._model = model
        self._perturbed = perturbed_compatibility

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    @property
    def perturbed_compatibility(self) -> tuple[str, ...]:
        """Compatibility fields a robustness evaluation deliberately changed (else empty)."""
        return self._perturbed

    def reset(self, seed: int | None = None) -> None:
        """No per-episode state; deterministic prediction needs no seed."""

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        """Greedy action of the trained policy for ``observation``; ``info`` is ignored."""
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)


class ModelMetadata(BaseModel):
    """What a bundled model is and how it was trained (no binary content)."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    bundle_version: Literal["scalerl-sb3-bundle-v1"] = BUNDLE_VERSION
    algorithm: Algorithm
    config_version: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    training_workload_id: str = Field(min_length=1)
    training_workload_split: Literal["train"]
    seed: int
    total_timesteps: int = Field(ge=1)
    hyperparameters: dict[str, JsonValue]
    scalerl_version: str
    training_run_id: str | None = None


def save_model_bundle(
    model: BaseAlgorithm,
    directory: str | Path,
    *,
    metadata: ModelMetadata,
    compatibility: EnvironmentCompatibility,
) -> Path:
    """Write ``model.zip``, ``compatibility.json``, and ``metadata.json`` into ``directory``."""
    bundle = Path(directory)
    bundle.mkdir(parents=True, exist_ok=True)
    model.save(bundle / MODEL_FILE)
    (bundle / COMPATIBILITY_FILE).write_text(compatibility.model_dump_json(indent=2) + "\n")
    (bundle / METADATA_FILE).write_text(metadata.model_dump_json(indent=2) + "\n")
    return bundle


def read_model_bundle(directory: str | Path) -> tuple[ModelMetadata, EnvironmentCompatibility]:
    """Read a bundle's metadata and compatibility contract (the model is not loaded)."""
    bundle = Path(directory)
    for name in (MODEL_FILE, COMPATIBILITY_FILE, METADATA_FILE):
        if not (bundle / name).is_file():
            raise FileNotFoundError(f"model bundle {bundle} is missing {name}")
    metadata = ModelMetadata.model_validate_json((bundle / METADATA_FILE).read_text())
    compatibility = EnvironmentCompatibility.model_validate_json(
        (bundle / COMPATIBILITY_FILE).read_text()
    )
    return metadata, compatibility


def load_sb3_controller(
    directory: str | Path,
    env: AutoscalingEnv,
    *,
    benchmark_version: str | None = None,
    robustness_evaluation: bool = False,
) -> SB3Controller:
    """Load a bundled policy for ``env`` after verifying the full compatibility contract.

    The current contract is derived from ``env`` itself (``benchmark_version``
    defaults to the installed benchmark manifest's version) and must equal the
    one saved with the model; otherwise ``ValueError`` names every mismatch
    and no model is loaded.

    ``robustness_evaluation=True`` is only for evaluating a fixed trained
    policy under a predeclared robustness scenario (#65): it additionally
    permits a telemetry-delay difference (see
    ``EnvironmentCompatibility.require_compatible_for_robustness``), still
    rejects every other mismatch, and records the perturbed fields on the
    returned controller. Never use it for training or normal inference.
    """
    bundle = Path(directory)
    metadata, trained = read_model_bundle(bundle)
    version = benchmark_version or load_benchmark_manifest().version
    current = EnvironmentCompatibility.from_env(env, version)
    perturbed: tuple[str, ...] = ()
    if robustness_evaluation:
        perturbed = trained.require_compatible_for_robustness(current)
    else:
        trained.require_compatible(current)
    model = _ALGORITHMS[metadata.algorithm].load(bundle / MODEL_FILE, device="cpu")
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError("model observation space does not match the environment")
    return SB3Controller(model, perturbed_compatibility=perturbed)
