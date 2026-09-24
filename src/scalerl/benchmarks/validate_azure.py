"""Validate and characterize the manifest's Azure windows against the real trace.

Local-only: requires the extracted Azure Functions 2021 CSV, which CI never
downloads. Usage::

    python -m scalerl.benchmarks.validate_azure --azure-csv data/raw/<file>.csv \\
        [--manifest path/to/workloads.json] [--summary-out summary.json]

The optional summary holds only aggregate statistics and provenance, never raw
events. These statistics describe the frozen windows; they must not be used to
change split membership or tune simulator/controller settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import scalerl
from scalerl.benchmarks.characterization import validate_azure_workloads
from scalerl.benchmarks.manifest import load_benchmark_manifest
from scalerl.workloads import AZURE_FUNCTIONS_2021


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--azure-csv", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="defaults to the packaged v1 manifest")
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
