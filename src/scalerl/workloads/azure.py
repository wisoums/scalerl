"""Load request-rate traces from the Azure Functions Invocation Trace 2021.

The source is event-level: one row per invocation with ``end_timestamp`` and
``duration``, both in **seconds**. Microsoft modified the timestamps, so they
are treated as trace-relative seconds, never as calendar datetimes.

Each invocation arrives at ``end_timestamp - duration``. Arrivals inside the
half-open window ``[start_seconds, start_seconds + duration_seconds)`` are
counted per control interval (a start exactly on an interval boundary belongs
to the later interval), and each count becomes ``count / control_interval``
requests per second. Empty intervals are zero. Demand is never normalized,
clipped, or rescaled.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scalerl
from scalerl.workloads.generators import _non_negative, _tick_count
from scalerl.workloads.trace import WorkloadTrace

AZURE_FUNCTIONS_2021 = "Azure Functions Invocation Trace 2021"
REQUIRED_COLUMNS = ("end_timestamp", "duration")
DEFAULT_CHUNK_SIZE = 1_000_000


def load_azure_trace(
    csv_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    control_interval_seconds: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> WorkloadTrace:
    """Bin invocation arrivals from an extracted Azure CSV into a ``WorkloadTrace``.

    The CSV is streamed ``chunk_size`` rows at a time, so memory scales with
    the chunk and the output, not the source file. The result does not depend
    on ``chunk_size`` or row order. Any malformed row raises ``ValueError``
    rather than being dropped.
    """
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    window_start = _non_negative("start_seconds", start_seconds)
    duration = float(duration_seconds)
    interval = float(control_interval_seconds)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Azure trace file not found: {path}")
    _check_columns(path)

    # Explicit edges keep boundary starts in the later interval without
    # floating-point division; the last edge is the exact window end.
    edges = window_start + interval * np.arange(ticks + 1)
    edges[-1] = window_start + duration

    counts = np.zeros(ticks, dtype=np.int64)
    for chunk in _read_chunks(path, chunk_size):
        end = chunk["end_timestamp"].to_numpy()
        elapsed = chunk["duration"].to_numpy()
        _check_values(chunk.index, end, "end_timestamp")
        _check_values(chunk.index, elapsed, "duration")
        negative = elapsed < 0
        if negative.any():
            row = int(chunk.index[np.argmax(negative)])
            raise ValueError(f"duration at line {row + 2} is negative: {elapsed[negative][0]}")

        starts = end - elapsed
        starts = starts[(starts >= edges[0]) & (starts < edges[-1])]
        bins = np.searchsorted(edges, starts, side="right") - 1
        counts += np.bincount(bins, minlength=ticks)

    return WorkloadTrace((counts / interval).tolist(), interval)


def azure_trace_metadata(
    csv_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    control_interval_seconds: float,
) -> dict[str, Any]:
    """Describe an Azure slice for a processed-trace sidecar."""
    return {
        "dataset": AZURE_FUNCTIONS_2021,
        "source_file": Path(csv_path).name,
        "timestamp_semantics": "trace-relative seconds; arrival = end_timestamp - duration",
        "start_seconds": float(start_seconds),
        "duration_seconds": float(duration_seconds),
        "control_interval_seconds": float(control_interval_seconds),
        "loader": "scalerl.workloads.load_azure_trace",
        "scalerl_version": scalerl.__version__,
    }


def _check_columns(path: Path) -> None:
    try:
        header = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Azure trace file is empty: {path}") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"Azure trace is missing required columns: {missing}")


def _read_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    # round_trip parsing is exact; the default parser can shift a value across
    # an interval boundary.
    reader = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype=float,
        float_precision="round_trip",
        chunksize=chunk_size,
    )
    try:
        yield from reader
    except ValueError as error:
        raise ValueError(f"Azure trace timestamps/durations must be numeric: {error}") from error


def _check_values(index: pd.Index, values: np.ndarray, column: str) -> None:
    invalid = ~np.isfinite(values)
    if invalid.any():
        row = int(index[np.argmax(invalid)])
        raise ValueError(
            f"{column} at line {row + 2} must be a finite number, got {values[invalid][0]}"
        )
