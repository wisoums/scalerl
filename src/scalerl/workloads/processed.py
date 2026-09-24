"""Save and load processed ``WorkloadTrace`` slices as inspectable files.

A slice ``name.csv`` holds ``tick,request_rate`` rows; ``name.json`` beside it
holds provenance metadata plus ``control_interval_seconds``, ``tick_count``,
and the CSV's SHA-256. Rates are written with ``repr`` so reloading is exact.

Saving serializes everything before touching disk and replaces each file
atomically from a temporary file. The checksum rejects any CSV/JSON pair that
does not belong together, e.g. after an interrupted save.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scalerl.workloads.trace import WorkloadTrace

FORMAT_VERSION = 1


def save_processed_trace(
    trace: WorkloadTrace, csv_path: str | Path, metadata: Mapping[str, Any]
) -> Path:
    """Write ``trace`` to ``csv_path`` and its metadata sidecar; return the sidecar path.

    Non-JSON-serializable or non-finite metadata raises before any file changes.
    """
    path = Path(csv_path)
    sidecar = path.with_suffix(".json")

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["tick", "request_rate"])
    for tick, rate in enumerate(trace.request_rates):
        writer.writerow([tick, repr(rate)])
    csv_bytes = buffer.getvalue().encode()

    content = {
        **metadata,
        "format_version": FORMAT_VERSION,
        "control_interval_seconds": trace.control_interval_seconds,
        "tick_count": len(trace),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    json_bytes = (json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, csv_bytes)
    _replace_atomically(sidecar, json_bytes)
    return sidecar


def load_processed_trace(csv_path: str | Path) -> tuple[WorkloadTrace, dict[str, Any]]:
    """Read a slice written by :func:`save_processed_trace` and its metadata."""
    path = Path(csv_path)
    metadata: dict[str, Any] = json.loads(path.with_suffix(".json").read_text())
    csv_bytes = path.read_bytes()

    if hashlib.sha256(csv_bytes).hexdigest() != metadata.get("csv_sha256"):
        raise ValueError(f"{path} does not match the checksum in its metadata sidecar")

    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    if [int(row["tick"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path} ticks must be consecutive from 0")
    if len(rows) != metadata["tick_count"]:
        raise ValueError(
            f"{path} has {len(rows)} ticks but metadata records {metadata['tick_count']}"
        )

    rates = [float(row["request_rate"]) for row in rows]
    return WorkloadTrace(rates, metadata["control_interval_seconds"]), metadata


def _replace_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to a temporary file beside ``path``, then rename it over ``path``."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
