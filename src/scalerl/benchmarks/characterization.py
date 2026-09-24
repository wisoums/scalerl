"""Aggregate characterization and validation of benchmark Azure windows."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from scalerl.benchmarks.manifest import AzureWorkload, BenchmarkManifest, build_workloads
from scalerl.workloads import WorkloadTrace


def characterize_trace(trace: WorkloadTrace) -> dict[str, float | int | None]:
    """Return deterministic aggregate statistics of a trace's request rates."""
    rates = trace.request_rates
    mean = statistics.fmean(rates)
    peak = max(rates)
    std = statistics.pstdev(rates)
    return {
        "ticks": len(rates),
        "mean_rps": mean,
        "max_rps": peak,
        "std_rps": std,
        "coefficient_of_variation": std / mean if mean > 0 else None,
        "peak_to_mean": peak / mean if mean > 0 else None,
    }


def validate_azure_workloads(
    manifest: BenchmarkManifest, azure_csv_path: str | Path
) -> list[dict[str, Any]]:
    """Load every Azure workload in one pass and report its statistics and any problems.

    A window is valid when it has the expected tick count and interval, only
    finite non-negative rates, and at least one arrival.
    """
    entries = [entry for entry in manifest.workloads if isinstance(entry, AzureWorkload)]
    traces = build_workloads(entries, azure_csv_path=azure_csv_path)

    reports = []
    for entry in entries:
        trace = traces[entry.id]
        expected_ticks = round(entry.duration_seconds / entry.control_interval_seconds)

        problems = []
        if len(trace) != expected_ticks:
            problems.append(f"expected {expected_ticks} ticks, got {len(trace)}")
        if trace.control_interval_seconds != entry.control_interval_seconds:
            problems.append("control interval does not match the manifest")
        if not all(math.isfinite(rate) and rate >= 0 for rate in trace.request_rates):
            problems.append("request rates must be finite and non-negative")
        if sum(trace.request_rates) == 0:
            problems.append("window contains no invocations")

        reports.append(
            {
                "id": entry.id,
                "split": entry.split,
                "start_seconds": entry.parameters.start_seconds,
                "duration_seconds": entry.duration_seconds,
                "control_interval_seconds": entry.control_interval_seconds,
                "valid": not problems,
                "problems": problems,
                "statistics": characterize_trace(trace),
            }
        )
    return reports
