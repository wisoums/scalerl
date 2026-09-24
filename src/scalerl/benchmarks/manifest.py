"""Typed, validated benchmark workload manifest and workload construction.

A manifest assigns every workload to exactly one split:

- ``train``: policy learning;
- ``validation``: threshold/predictive/RL tuning and model selection;
- ``test``: final held-out reporting only, after all choices are frozen.

Split helpers never merge ``test`` into training or tuning selections. Each
id encodes its split (``syn-train-*``, ``azure-val-*``, ``syn-test-*``, ...),
so moving a workload between splits requires renaming it deliberately.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scalerl.workloads import (
    AZURE_FUNCTIONS_2021,
    AzureWindow,
    WorkloadTrace,
    bursty_workload,
    diurnal_workload,
    load_azure_trace,
    load_azure_traces,
    ramp_workload,
    spike_workload,
    steady_workload,
)

Split = Literal["train", "validation", "test"]

V1_DURATION_SECONDS = 3600.0
V1_CONTROL_INTERVAL_SECONDS = 30.0

# Shipped as package data, so it resolves in source checkouts and installed wheels.
V1_MANIFEST: Traversable = files("scalerl.benchmarks").joinpath("v1", "workloads.json")

_SPLIT_ID_TOKENS: dict[str, str] = {"train": "train", "validation": "val", "test": "test"}
_SOURCE_ID_PREFIXES: dict[str, str] = {"synthetic": "syn", "azure": "azure"}


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


# --- generator parameters ---------------------------------------------------


class SteadyParameters(_Strict):
    rate: float


class DiurnalParameters(_Strict):
    baseline: float
    amplitude: float
    period_seconds: float
    phase_seconds: float


class RampParameters(_Strict):
    start_rate: float
    end_rate: float


class SpikeParameters(_Strict):
    baseline: float
    spike_rate: float
    spike_start_seconds: float
    spike_duration_seconds: float


class BurstyParameters(_Strict):
    baseline: float
    burst_rate: float
    burst_probability: float
    burst_duration_seconds: float
    noise_std: float
    seed: int


class AzureParameters(_Strict):
    start_seconds: float = Field(ge=0)


# --- workload entries -------------------------------------------------------


class _WorkloadEntry(_Strict):
    id: str = Field(min_length=1)
    split: Split
    source: Literal["synthetic", "azure"]
    duration_seconds: float = Field(gt=0)
    control_interval_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _id_matches_source_and_split(self) -> Self:
        prefix = f"{_SOURCE_ID_PREFIXES[self.source]}-{_SPLIT_ID_TOKENS[self.split]}-"
        if not self.id.startswith(prefix):
            raise ValueError(f"workload id {self.id!r} must start with {prefix!r}")
        return self


class _SyntheticEntry(_WorkloadEntry):
    source: Literal["synthetic"]


class SteadyWorkload(_SyntheticEntry):
    generator: Literal["steady"]
    parameters: SteadyParameters


class DiurnalWorkload(_SyntheticEntry):
    generator: Literal["diurnal"]
    parameters: DiurnalParameters


class RampWorkload(_SyntheticEntry):
    generator: Literal["ramp"]
    parameters: RampParameters


class SpikeWorkload(_SyntheticEntry):
    generator: Literal["spike"]
    parameters: SpikeParameters


class BurstyWorkload(_SyntheticEntry):
    generator: Literal["bursty"]
    parameters: BurstyParameters


SyntheticWorkload = Annotated[
    SteadyWorkload | DiurnalWorkload | RampWorkload | SpikeWorkload | BurstyWorkload,
    Field(discriminator="generator"),
]


class AzureWorkload(_WorkloadEntry):
    source: Literal["azure"]
    dataset: Literal["Azure Functions Invocation Trace 2021"]
    parameters: AzureParameters


WorkloadEntry = Annotated[SyntheticWorkload | AzureWorkload, Field(discriminator="source")]


# --- manifest ---------------------------------------------------------------


class BenchmarkManifest(_Strict):
    """A frozen benchmark version: workload membership by split."""

    version: Literal["v1"]
    frozen_on: str
    workloads: tuple[WorkloadEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_membership(self) -> Self:
        ids = [entry.id for entry in self.workloads]
        duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
        if duplicates:
            raise ValueError(f"duplicate workload ids: {duplicates}")

        for entry in self.workloads:
            if (entry.duration_seconds, entry.control_interval_seconds) != (
                V1_DURATION_SECONDS,
                V1_CONTROL_INTERVAL_SECONDS,
            ):
                raise ValueError(
                    f"{entry.id}: v1 workloads must use duration_seconds={V1_DURATION_SECONDS} "
                    f"and control_interval_seconds={V1_CONTROL_INTERVAL_SECONDS}"
                )

        windows = sorted(
            (entry.parameters.start_seconds, entry.duration_seconds, entry.id)
            for entry in self.workloads
            if isinstance(entry, AzureWorkload)
        )
        for (start, duration, first), (next_start, _, second) in zip(
            windows, windows[1:], strict=False
        ):
            if start + duration > next_start:
                raise ValueError(f"Azure windows {first} and {second} overlap")
        return self

    @property
    def train(self) -> tuple[WorkloadEntry, ...]:
        """Workloads for policy learning."""
        return self._split("train")

    @property
    def validation(self) -> tuple[WorkloadEntry, ...]:
        """Workloads for tuning and model selection."""
        return self._split("validation")

    @property
    def tuning(self) -> tuple[WorkloadEntry, ...]:
        """Train plus validation workloads; never includes held-out test workloads."""
        return self.train + self.validation

    @property
    def test(self) -> tuple[WorkloadEntry, ...]:
        """Held-out workloads for final reporting only, after all choices are frozen."""
        return self._split("test")

    def get(self, workload_id: str) -> WorkloadEntry:
        """Return the workload with ``workload_id``."""
        for entry in self.workloads:
            if entry.id == workload_id:
                return entry
        raise KeyError(f"unknown workload id: {workload_id!r}")

    def _split(self, split: Split) -> tuple[WorkloadEntry, ...]:
        return tuple(entry for entry in self.workloads if entry.split == split)


def load_benchmark_manifest(path: str | Path | None = None) -> BenchmarkManifest:
    """Load and validate a benchmark manifest; the packaged v1 manifest by default."""
    if path is None:
        return BenchmarkManifest.model_validate_json(V1_MANIFEST.read_text())
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    return BenchmarkManifest.model_validate_json(manifest_path.read_text())


# --- workload construction --------------------------------------------------


def build_workload(
    entry: WorkloadEntry, *, azure_csv_path: str | Path | None = None
) -> WorkloadTrace:
    """Build the ``WorkloadTrace`` for ``entry`` using the existing generators/loader.

    Azure entries need ``azure_csv_path`` (the locally extracted trace);
    synthetic entries ignore it.
    """
    duration, interval = entry.duration_seconds, entry.control_interval_seconds
    if isinstance(entry, AzureWorkload):
        if azure_csv_path is None:
            raise ValueError(
                f"{entry.id} is an Azure workload; pass azure_csv_path to the extracted "
                f"{AZURE_FUNCTIONS_2021} CSV (see data/README.md)"
            )
        return load_azure_trace(
            azure_csv_path,
            start_seconds=entry.parameters.start_seconds,
            duration_seconds=duration,
            control_interval_seconds=interval,
        )

    timing = {"duration_seconds": duration, "control_interval_seconds": interval}
    parameters = entry.parameters.model_dump() | timing
    if isinstance(entry, SteadyWorkload):
        return steady_workload(**parameters)
    if isinstance(entry, DiurnalWorkload):
        return diurnal_workload(**parameters)
    if isinstance(entry, RampWorkload):
        return ramp_workload(**parameters)
    if isinstance(entry, SpikeWorkload):
        return spike_workload(**parameters)
    return bursty_workload(**parameters)


def build_workloads(
    entries: Iterable[WorkloadEntry], *, azure_csv_path: str | Path | None = None
) -> dict[str, WorkloadTrace]:
    """Build several workloads, keyed by id in ``entries`` order.

    All Azure entries are binned in a single pass over ``azure_csv_path``
    instead of re-reading the large trace once per window.
    """
    entries = tuple(entries)
    azure = [entry for entry in entries if isinstance(entry, AzureWorkload)]
    azure_traces: dict[str, WorkloadTrace] = {}
    if azure:
        if azure_csv_path is None:
            raise ValueError(
                f"{azure[0].id} is an Azure workload; pass azure_csv_path to the extracted "
                f"{AZURE_FUNCTIONS_2021} CSV (see data/README.md)"
            )
        windows = [
            AzureWindow(
                entry.parameters.start_seconds,
                entry.duration_seconds,
                entry.control_interval_seconds,
            )
            for entry in azure
        ]
        traces = load_azure_traces(azure_csv_path, windows)
        azure_traces = {entry.id: trace for entry, trace in zip(azure, traces, strict=True)}

    return {
        entry.id: azure_traces[entry.id] if entry.id in azure_traces else build_workload(entry)
        for entry in entries
    }
