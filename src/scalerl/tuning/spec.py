"""Study contract and tuning-workload guardrail; importable without Optuna."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scalerl.benchmarks import BenchmarkManifest, WorkloadEntry, load_benchmark_manifest

SamplerName = Literal["grid", "tpe", "random"]
PrunerName = Literal["none", "median"]
GridValue = float | int | str | bool


def require_tuning_workloads(
    workload_ids: Iterable[str], manifest: BenchmarkManifest | None = None
) -> tuple[WorkloadEntry, ...]:
    """Return manifest entries for ``workload_ids``, rejecting anything outside train/validation.

    Held-out test workloads never participate in tuning.
    """
    manifest = manifest or load_benchmark_manifest()
    tuning = {entry.id: entry for entry in manifest.tuning}
    splits = {entry.id: entry.split for entry in manifest.workloads}
    entries = []
    for workload_id in workload_ids:
        if workload_id not in splits:
            raise ValueError(f"workload {workload_id!r} is not in benchmark {manifest.version}")
        if workload_id not in tuning:
            raise ValueError(
                f"tuning may use train/validation workloads only; {workload_id!r} is a "
                f"held-out {splits[workload_id]} workload"
            )
        entries.append(tuning[workload_id])
    return tuple(entries)


class StudySpec(BaseModel):
    """Everything that defines an Optuna study, validated before Optuna is touched.

    ``n_trials`` is the study's total budget: running the same spec against an
    existing study only runs the remaining trials. ``identity_context`` holds
    any other inputs that define the experiment (simulator config, provenance,
    workload data fingerprints); it is part of the persisted identity, so a
    study cannot be resumed with different inputs. Single-objective only;
    multi-objective studies are deferred until a consumer needs them.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    objective_name: str = Field(min_length=1)
    objective_version: str = Field(min_length=1)
    search_space_version: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]
    tuning_workload_ids: tuple[str, ...] = Field(min_length=1)
    sampler: SamplerName
    sampler_seed: int
    grid: dict[str, list[GridValue]] | None = None
    pruner: PrunerName = "none"
    storage: str | None = None
    n_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    n_jobs: int = Field(default=1, ge=1)
    load_if_exists: bool = True
    identity_context: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        require_tuning_workloads(self.tuning_workload_ids)
        if len(set(self.tuning_workload_ids)) != len(self.tuning_workload_ids):
            raise ValueError("tuning_workload_ids must be unique")
        if self.sampler == "grid":
            if not self.grid or any(not values for values in self.grid.values()):
                raise ValueError("grid sampler needs a non-empty grid for every parameter")
        elif self.grid is not None:
            raise ValueError("grid is only used by the grid sampler")
        elif self.n_trials is None and self.timeout_seconds is None:
            raise ValueError(f"{self.sampler} sampler needs n_trials or timeout_seconds")
        return self

    @property
    def grid_size(self) -> int | None:
        """Number of grid combinations, or ``None`` for non-grid samplers."""
        if self.grid is None:
            return None
        size = 1
        for values in self.grid.values():
            size *= len(values)
        return size

    def identity(self) -> dict[str, Any]:
        """Fields that must match when an existing study is resumed."""
        return self.model_dump(
            mode="json",
            include={
                "objective_name",
                "objective_version",
                "search_space_version",
                "direction",
                "tuning_workload_ids",
                "sampler",
                "sampler_seed",
                "grid",
                "pruner",
                "identity_context",
            },
        )

    def trial_metadata(self, trial_number: int) -> dict[str, Any]:
        """Optuna lineage recorded on every MLflow run the trial creates."""
        return {
            "study": self.name,
            "trial": trial_number,
            "sampler": self.sampler,
            "sampler_seed": self.sampler_seed,
            "pruner": self.pruner,
            "objective_name": self.objective_name,
            "objective_version": self.objective_version,
            "search_space_version": self.search_space_version,
            "storage": safe_storage_label(self.storage),
        }


def safe_storage_label(storage: str | None) -> str:
    """Describe the storage without credentials, e.g. ``sqlite:///optuna.db``."""
    if storage is None:
        return "in-memory"
    parts = urlsplit(storage)
    if parts.username is None and parts.password is None:
        return storage
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"
