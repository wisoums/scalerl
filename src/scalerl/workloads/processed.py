"""Save and load processed ``WorkloadTrace`` slices as inspectable files.

A slice ``name.csv`` holds ``tick,request_rate`` rows; ``name.json`` beside it
holds provenance metadata plus ``control_interval_seconds`` and
``tick_count``. Rates are written with ``repr`` so reloading is exact.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scalerl.workloads.trace import WorkloadTrace

FORMAT_VERSION = 1


def save_processed_trace(
    trace: WorkloadTrace, csv_path: str | Path, metadata: Mapping[str, Any]
) -> Path:
    """Write ``trace`` to ``csv_path`` and its metadata sidecar; return the sidecar path."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tick", "request_rate"])
        for tick, rate in enumerate(trace.request_rates):
            writer.writerow([tick, repr(rate)])

    sidecar = path.with_suffix(".json")
    content = {
        **metadata,
        "format_version": FORMAT_VERSION,
        "control_interval_seconds": trace.control_interval_seconds,
        "tick_count": len(trace),
    }
    sidecar.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    return sidecar


def load_processed_trace(csv_path: str | Path) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Read a slice written by :func:`save_processed_trace` and its metadata."""
    path = Path(csv_path)
    metadata: dict[str, Any] = json.loads(path.with_suffix(".json").read_text())

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if [int(row["tick"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path} ticks must be consecutive from 0")
    if len(rows) != metadata["tick_count"]:
        raise ValueError(
            f"{path} has {len(rows)} ticks but metadata records {metadata['tick_count']}"
        )

    rates = [float(row["request_rate"]) for row in rows]
    return WorkloadTrace(rates, metadata["control_interval_seconds"]), metadata
