"""Tests for the Azure Functions 2021 trace loader and processed-trace cache."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scalerl.controllers import ThresholdController, run_episode
from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.workloads import (
    AZURE_FUNCTIONS_2021,
    WorkloadTrace,
    azure_trace_metadata,
    load_azure_trace,
    load_processed_trace,
    save_processed_trace,
)

# Fixture rows as (end_timestamp, duration) -> derived arrival (start) second:
#   100.5 - 0.5   = 100.0   window start: included
#   103   - 1     = 102.0
#   112   - 5     = 107.0   ends in [110, 120) but arrives in [100, 110)
#   109.75 - 0.25 = 109.5
#   115   - 20    = 95.0    ends inside the window but arrives before it: excluded
#   100   - 0.001 = 99.999  just before the window: excluded
#   120   - 0     = 120.0   on an internal boundary: later interval
#   126   - 2     = 124.0
#   131   - 1     = 130.0   window end: excluded
#   140   - 3     = 137.0   after the window: excluded
#   50    - 10    = 40.0    before the window: excluded
FIXTURE = Path(__file__).parent.parent / "fixtures" / "azure_functions_2021_small.csv"
WINDOW = {"start_seconds": 100.0, "duration_seconds": 30.0}


def load(interval: float = 10.0, path: Path = FIXTURE, **overrides: Any) -> WorkloadTrace:
    return load_azure_trace(path, **{**WINDOW, "control_interval_seconds": interval, **overrides})


def write_csv(
    path: Path, rows: list[tuple[Any, Any]], header: str = "end_timestamp,duration"
) -> Path:
    lines = [header] + [f"{end},{duration}" for end, duration in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# --- binning ----------------------------------------------------------------


def test_arrivals_are_binned_by_derived_start_time() -> None:
    trace = load(
        interval=10.0
    )  # [100,110): 100, 102, 107, 109.5 | [110,120): - | [120,130): 120, 124

    assert trace.request_rates == (0.4, 0.0, 0.2)
    assert trace.control_interval_seconds == 10.0
    assert len(trace) == 3


@pytest.mark.parametrize(
    ("interval", "expected_counts"),
    [
        (5.0, [2, 2, 0, 0, 2, 0]),  # [100,105) [105,110) ... [125,130)
        (10.0, [4, 0, 2]),
        (15.0, [4, 2]),  # [100,115) [115,130)
        (30.0, [6]),
    ],
)
def test_counts_become_requests_per_second_for_each_interval(
    interval: float, expected_counts: list[int]
) -> None:
    trace = load(interval=interval)

    assert trace.request_rates == tuple(count / interval for count in expected_counts)
    assert len(trace) == 30.0 / interval


def test_hand_computed_rates(tmp_path: Path) -> None:
    rows = [(1000 + i * 0.25, 0.0) for i in range(20)]  # 20 starts in [1000, 1010)
    rows += [(1020.5 + i, 0.5) for i in range(5)]  # 5 starts in [1020, 1030)
    path = write_csv(tmp_path / "rates.csv", rows)

    trace = load_azure_trace(
        path, start_seconds=1000, duration_seconds=30, control_interval_seconds=10
    )

    assert trace.request_rates == (2.0, 0.0, 0.5)


def test_window_boundaries_are_half_open(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "edges.csv", [(0, 0), (10, 0), (19.999, 0), (20, 0)])

    trace = load_azure_trace(
        path, start_seconds=0, duration_seconds=20, control_interval_seconds=10
    )

    # 0 -> bin 0 (start inclusive); 10 -> bin 1 (internal boundary);
    # 19.999 -> bin 1; 20 -> excluded (end exclusive).
    assert trace.request_rates == (0.1, 0.2)


def test_boundary_with_non_representable_interval_goes_to_later_bin(tmp_path: Path) -> None:
    # 0.1 * 3 is not exactly 0.3 in floating point; explicit edges keep a start
    # equal to the computed boundary in the later bin.
    boundary = 0.1 * 3
    path = write_csv(tmp_path / "float.csv", [(repr(boundary), 0)])

    trace = load_azure_trace(
        path, start_seconds=0, duration_seconds=0.5, control_interval_seconds=0.1
    )

    assert [rate > 0 for rate in trace.request_rates] == [False, False, False, True, False]


def test_empty_intervals_are_explicit_zeros(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "sparse.csv", [(35, 0)])

    trace = load_azure_trace(
        path, start_seconds=0, duration_seconds=60, control_interval_seconds=10
    )

    assert trace.request_rates == (0.0, 0.0, 0.0, 0.1, 0.0, 0.0)


def test_window_with_no_arrivals_is_all_zero() -> None:
    trace = load(start_seconds=1000.0)

    assert trace.request_rates == (0.0, 0.0, 0.0)


def test_demand_is_not_normalized_or_clipped(tmp_path: Path) -> None:
    rows = [(i * 0.001, 0) for i in range(10_000)]  # 10,000 starts in [0, 10)
    path = write_csv(tmp_path / "dense.csv", rows)

    trace = load_azure_trace(
        path, start_seconds=0, duration_seconds=10, control_interval_seconds=10
    )

    assert trace.request_rates == (1000.0,)


def test_extra_source_columns_are_ignored() -> None:
    # The fixture also carries the documented app/func columns.
    assert load() == load()


# --- chunking and determinism -----------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 1_000])
def test_chunk_size_does_not_change_the_trace(chunk_size: int) -> None:
    assert load(chunk_size=chunk_size) == load(chunk_size=1_000_000)


def test_row_order_does_not_change_the_trace(tmp_path: Path) -> None:
    lines = FIXTURE.read_text().splitlines()
    reversed_path = tmp_path / "reversed.csv"
    reversed_path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")

    assert load(path=reversed_path, chunk_size=2) == load()


def test_repeated_loads_are_identical() -> None:
    assert load(interval=5.0) == load(interval=5.0)


# --- validation -------------------------------------------------------------


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Azure trace file not found"):
        load(path=tmp_path / "missing.csv")


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")

    with pytest.raises(ValueError, match="file is empty"):
        load(path=path)


@pytest.mark.parametrize("header", ["end_timestamp", "duration", "app,func,end,duration"])
def test_missing_required_columns_are_rejected(tmp_path: Path, header: str) -> None:
    path = tmp_path / "columns.csv"
    path.write_text(header + "\n1,2\n")

    with pytest.raises(ValueError, match="missing required columns"):
        load(path=path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("abc", 1), "must be numeric"),
        ((101, "slow"), "must be numeric"),
        (("", 1), "end_timestamp at line 3 must be a finite number"),
        ((101, "nan"), "duration at line 3 must be a finite number"),
        (("inf", 1), "end_timestamp at line 3 must be a finite number"),
        ((101, "-inf"), "duration at line 3 must be a finite number"),
        ((101, -0.5), "duration at line 3 is negative"),
    ],
)
def test_malformed_rows_are_rejected_not_dropped(
    tmp_path: Path, row: tuple[Any, Any], message: str
) -> None:
    path = write_csv(tmp_path / "bad.csv", [(101, 1), row, (102, 1)])

    with pytest.raises(ValueError, match=message):
        load(path=path)


def test_malformed_row_outside_the_window_is_still_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "bad.csv", [(101, 1), (99999, -1)])

    with pytest.raises(ValueError, match="is negative"):
        load(path=path, chunk_size=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_seconds": -1.0}, "start_seconds must be finite and non-negative"),
        ({"start_seconds": float("nan")}, "start_seconds must be finite and non-negative"),
        ({"duration_seconds": 0.0}, "duration_seconds must be finite and greater than zero"),
        ({"duration_seconds": float("inf")}, "duration_seconds must be finite"),
        ({"control_interval_seconds": 0.0}, "control_interval_seconds must be finite"),
        ({"control_interval_seconds": -10.0}, "control_interval_seconds must be finite"),
        ({"duration_seconds": 25.0}, "duration_seconds must be a whole number of control"),
        ({"chunk_size": 0}, "chunk_size must be a positive integer"),
        ({"chunk_size": True}, "chunk_size must be a positive integer"),
    ],
)
def test_invalid_arguments_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load(**overrides)


# --- processed cache --------------------------------------------------------


def test_processed_slice_round_trips_exactly(tmp_path: Path) -> None:
    trace = load(interval=15.0)  # 4/15 and 2/15 are not exact decimals
    metadata = azure_trace_metadata(FIXTURE, **WINDOW, control_interval_seconds=15.0)
    csv_path = tmp_path / "processed" / "azure-slice.csv"

    sidecar = save_processed_trace(trace, csv_path, metadata)
    restored, restored_metadata = load_processed_trace(csv_path)

    assert sidecar == csv_path.with_suffix(".json")
    assert restored == trace
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert restored_metadata == {
        **metadata,
        "format_version": 1,
        "control_interval_seconds": 15.0,
        "tick_count": 2,
        "csv_sha256": checksum,
    }


def test_processed_files_are_inspectable(tmp_path: Path) -> None:
    csv_path = tmp_path / "slice.csv"
    save_processed_trace(
        load(interval=10.0),
        csv_path,
        azure_trace_metadata(FIXTURE, **WINDOW, control_interval_seconds=10.0),
    )

    with csv_path.open(newline="") as file:
        rows = list(csv.reader(file))
    metadata = json.loads(csv_path.with_suffix(".json").read_text())

    assert rows == [["tick", "request_rate"], ["0", "0.4"], ["1", "0.0"], ["2", "0.2"]]
    assert metadata["dataset"] == AZURE_FUNCTIONS_2021
    assert metadata["source_file"] == "azure_functions_2021_small.csv"  # file name, not path
    assert (metadata["start_seconds"], metadata["duration_seconds"]) == (100.0, 30.0)
    assert metadata["tick_count"] == 3
    assert "scalerl_version" in metadata


def test_processed_slice_with_inconsistent_metadata_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "slice.csv"
    save_processed_trace(load(), csv_path, {"dataset": AZURE_FUNCTIONS_2021})
    metadata = json.loads(csv_path.with_suffix(".json").read_text())
    csv_path.with_suffix(".json").write_text(json.dumps({**metadata, "tick_count": 5}))

    with pytest.raises(ValueError, match="has 3 ticks but metadata records 5"):
        load_processed_trace(csv_path)


def _save_original(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    csv_path = tmp_path / "slice.csv"
    save_processed_trace(load(), csv_path, {"dataset": AZURE_FUNCTIONS_2021, "note": "original"})
    snapshot = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    return csv_path, snapshot


def _replacement() -> WorkloadTrace:
    return load(start_seconds=1000.0)  # same tick count, different demand


@pytest.mark.parametrize(
    ("metadata", "error"),
    [({"bad": object()}, TypeError), ({"bad": float("nan")}, ValueError)],
    ids=["not_serializable", "nan"],
)
def test_failed_metadata_serialization_leaves_existing_slice_untouched(
    tmp_path: Path, metadata: dict[str, Any], error: type[Exception]
) -> None:
    csv_path, before = _save_original(tmp_path)
    replacement = _replacement()

    with pytest.raises(error):
        save_processed_trace(replacement, csv_path, metadata)

    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == before
    assert load_processed_trace(csv_path)[1]["note"] == "original"


def test_interrupted_save_is_detected_instead_of_mixing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, _ = _save_original(tmp_path)
    replacement = _replacement()
    real_replace = os.replace
    calls = []

    def fail_on_sidecar(source: str, target: str | Path) -> None:
        calls.append(target)
        if Path(target).suffix == ".json":
            raise OSError("disk full")
        real_replace(source, target)

    monkeypatch.setattr("scalerl.workloads.processed.os.replace", fail_on_sidecar)
    with pytest.raises(OSError, match="disk full"):
        save_processed_trace(replacement, csv_path, {"note": "replacement"})
    monkeypatch.undo()

    # New CSV beside the old JSON: rejected, and no temporary files left behind.
    assert len(calls) == 2
    assert sorted(file.name for file in tmp_path.iterdir()) == ["slice.csv", "slice.json"]
    with pytest.raises(ValueError, match="does not match the checksum"):
        load_processed_trace(csv_path)


def test_swapped_csv_with_equal_tick_count_is_rejected(tmp_path: Path) -> None:
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    save_processed_trace(load(), first, {"note": "first"})
    save_processed_trace(_replacement(), second, {"note": "second"})

    first.write_bytes(second.read_bytes())

    with pytest.raises(ValueError, match="does not match the checksum"):
        load_processed_trace(first)


# --- integration ------------------------------------------------------------


def test_azure_trace_drives_the_environment_through_the_shared_runner() -> None:
    trace = load(interval=10.0)
    config = SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=10.0, episode_duration_seconds=30.0),
        replicas=ReplicaConfig(min_replicas=1, max_replicas=3, initial_replicas=1),
    )
    env = AutoscalingEnv(config, trace)
    controller = ThresholdController(
        low_threshold=0.2, high_threshold=0.8, min_replicas=1, max_replicas=3
    )

    infos = run_episode(env, controller, seed=0)

    assert [info["request_rate"] for info in infos] == list(trace.request_rates)
    assert [info["arrived_requests"] for info in infos] == pytest.approx([4.0, 0.0, 2.0])
    assert sum(info["processed_requests"] for info in infos) + infos[-1][
        "queued_requests"
    ] == pytest.approx(6.0)
