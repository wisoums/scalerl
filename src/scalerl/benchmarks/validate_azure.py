"""Validate and characterize the manifest's Azure windows against the real trace.

Local-only: requires the extracted Azure Functions 2021 CSV, which CI never
downloads. Usage::

    python -m scalerl.benchmarks.validate_azure --azure-csv data/raw/<file>.csv \\
        [--manifest benchmarks/v1/workloads.json] [--summary-out summary.json]

The optional summary holds only aggregate statistics and provenance, never raw
events. These statistics describe the frozen windows; they must not be used to
change split membership or tune simulator/controller settings.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import scalerl
from scalerl.benchmarks.manifest import (
    V1_MANIFEST_PATH,
    AzureWorkload,
    BenchmarkManifest,
    build_workload,
    load_benchmark_manifest,
)
from scalerl.workloads import AZURE_FUNCTIONS_2021, WorkloadTrace


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
    """Load every Azure workload and report its statistics and any problems.

    A window is valid when it has the expected tick count and interval, only
    finite non-negative rates, and at least one arrival.
    """
    reports = []
    for entry in manifest.workloads:
        if not isinstance(entry, AzureWorkload):
            continue
        trace = build_workload(entry, azure_csv_path=azure_csv_path)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--azure-csv", required=True, type=Path)
    parser.add_argument("--manifest", default=V1_MANIFEST_PATH, type=Path)
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args(argv)

    manifest = load_benchmark_manifest(args.manifest)
    reports = validate_azure_workloads(manifest, args.azure_csv)

    for report in reports:
        stats = report["statistics"]
        status = "ok" if report["valid"] else "INVALID: " + "; ".join(report["problems"])
        print(
            f"{report['id']:<22} {report['split']:<10} ticks={stats['ticks']} "
            f"mean={stats['mean_rps']:.3f} max={stats['max_rps']:.3f} "
            f"std={stats['std_rps']:.3f} cv={_fmt(stats['coefficient_of_variation'])} "
            f"peak/mean={_fmt(stats['peak_to_mean'])} {status}"
        )

    if args.summary_out is not None:
        summary = {
            "benchmark_version": manifest.version,
            "dataset": AZURE_FUNCTIONS_2021,
            "source_file": args.azure_csv.name,
            "scalerl_version": scalerl.__version__,
            "workloads": reports,
        }
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return 0 if all(report["valid"] for report in reports) else 1


def _fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())
