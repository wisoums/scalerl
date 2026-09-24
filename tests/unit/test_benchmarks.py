"""Tests for the frozen v1 benchmark manifest and workload construction."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scalerl.benchmarks import (
    V1_MANIFEST,
    AzureWorkload,
    BenchmarkManifest,
    build_workload,
    build_workloads,
    characterize_trace,
    load_benchmark_manifest,
    validate_azure_workloads,
)
from scalerl.benchmarks.validate_azure import main as validate_azure_main
from scalerl.controllers import StaticController, ThresholdController, run_episode
from scalerl.environment import AutoscalingEnv, SimulatorConfig, TimingConfig
from scalerl.workloads import WorkloadTrace, bursty_workload, load_azure_trace

AZURE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "azure_functions_2021_small.csv"
RAW_MANIFEST: dict[str, Any] = json.loads(V1_MANIFEST.read_text())

EXPECTED_SPLITS = {
    "train": (
        "syn-train-steady-moderate",
        "syn-train-seasonal",
        "syn-train-ramp-up",
        "syn-train-spike",
        "syn-train-bursty",
        "azure-train-129600",
        "azure-train-302400",
        "azure-train-475200",
    ),
    "validation": (
        "syn-val-steady-high",
        "syn-val-ramp-down",
        "syn-val-bursty",
        "azure-val-734400",
    ),
    "test": (
        "syn-test-seasonal-shifted",
        "syn-test-spike-hard",
        "azure-test-993600",
        "azure-test-1166400",
    ),
}

EXPECTED_SYNTHETIC: dict[str, tuple[str, dict[str, Any]]] = {
    "syn-train-steady-moderate": ("steady", {"rate": 100.0}),
    "syn-train-seasonal": (
        "diurnal",
        {"baseline": 150.0, "amplitude": 100.0, "period_seconds": 3600.0, "phase_seconds": 0.0},
    ),
    "syn-train-ramp-up": ("ramp", {"start_rate": 40.0, "end_rate": 360.0}),
    "syn-train-spike": (
        "spike",
        {
            "baseline": 80.0,
            "spike_rate": 420.0,
            "spike_start_seconds": 1500.0,
            "spike_duration_seconds": 300.0,
        },
    ),
    "syn-train-bursty": (
        "bursty",
        {
            "baseline": 90.0,
            "burst_rate": 250.0,
            "burst_probability": 0.08,
            "burst_duration_seconds": 90.0,
            "noise_std": 10.0,
            "seed": 101,
        },
    ),
    "syn-val-steady-high": ("steady", {"rate": 250.0}),
    "syn-val-ramp-down": ("ramp", {"start_rate": 350.0, "end_rate": 60.0}),
    "syn-val-bursty": (
        "bursty",
        {
            "baseline": 120.0,
            "burst_rate": 280.0,
            "burst_probability": 0.12,
            "burst_duration_seconds": 60.0,
            "noise_std": 15.0,
            "seed": 202,
        },
    ),
    "syn-test-seasonal-shifted": (
        "diurnal",
        {"baseline": 180.0, "amplitude": 140.0, "period_seconds": 3600.0, "phase_seconds": 900.0},
    ),
    "syn-test-spike-hard": (
        "spike",
        {
            "baseline": 60.0,
            "spike_rate": 480.0,
            "spike_start_seconds": 900.0,
            "spike_duration_seconds": 180.0,
        },
    ),
}

EXPECTED_AZURE_OFFSETS = {
    "azure-train-129600": 129600.0,
    "azure-train-302400": 302400.0,
    "azure-train-475200": 475200.0,
    "azure-val-734400": 734400.0,
    "azure-test-993600": 993600.0,
    "azure-test-1166400": 1166400.0,
}


@pytest.fixture(scope="module")
def manifest() -> BenchmarkManifest:
    return load_benchmark_manifest()


def validate(raw: dict[str, Any]) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(json.dumps(raw))


def edited(edit: Any) -> dict[str, Any]:
    raw = copy.deepcopy(RAW_MANIFEST)
    edit(raw)
    return raw


def entry(raw: dict[str, Any], workload_id: str) -> dict[str, Any]:
    return next(item for item in raw["workloads"] if item["id"] == workload_id)


def v1_config() -> SimulatorConfig:
    return SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=3600.0)
    )


# --- committed manifest -----------------------------------------------------


def test_committed_manifest_loads(manifest: BenchmarkManifest) -> None:
    assert manifest.version == "v1"
    assert manifest.frozen_on == "2026-09-24"
    assert len(manifest.workloads) == 16


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_split_membership_is_exactly_frozen(manifest: BenchmarkManifest, split: str) -> None:
    assert tuple(entry.id for entry in getattr(manifest, split)) == EXPECTED_SPLITS[split]
    assert all(entry.split == split for entry in getattr(manifest, split))


def test_synthetic_parameters_are_exactly_frozen(manifest: BenchmarkManifest) -> None:
    synthetic = {
        entry.id: (entry.generator, entry.parameters.model_dump())
        for entry in manifest.workloads
        if not isinstance(entry, AzureWorkload)
    }

    assert synthetic == EXPECTED_SYNTHETIC


def test_azure_offsets_are_exactly_frozen(manifest: BenchmarkManifest) -> None:
    azure = {
        entry.id: entry.parameters.start_seconds
        for entry in manifest.workloads
        if isinstance(entry, AzureWorkload)
    }

    assert azure == EXPECTED_AZURE_OFFSETS
    assert {entry.dataset for entry in manifest.workloads if isinstance(entry, AzureWorkload)} == {
        "Azure Functions Invocation Trace 2021"
    }


def test_committed_azure_windows_do_not_overlap(manifest: BenchmarkManifest) -> None:
    windows = sorted(
        (entry.parameters.start_seconds, entry.parameters.start_seconds + entry.duration_seconds)
        for entry in manifest.workloads
        if isinstance(entry, AzureWorkload)
    )

    assert all(
        end <= next_start for (_, end), (next_start, _) in zip(windows, windows[1:], strict=False)
    )


def test_manifest_contains_no_filesystem_paths() -> None:
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for key, item in value.items() for s in [key, *strings(item)]]
        if isinstance(value, list):
            return [s for item in value for s in strings(item)]
        return []

    assert not [s for s in strings(RAW_MANIFEST) if "/" in s or "\\" in s or s.endswith(".csv")]


def test_every_workload_uses_the_v1_episode_contract(manifest: BenchmarkManifest) -> None:
    for workload in manifest.workloads:
        trace = build_workload(workload, azure_csv_path=AZURE_FIXTURE)
        assert len(trace) == 120
        assert trace.control_interval_seconds == 30.0


def test_default_manifest_is_package_data() -> None:
    # Loaded through importlib.resources, so it also resolves from an installed wheel.
    assert Path(str(V1_MANIFEST)).parts[-4:] == ("scalerl", "benchmarks", "v1", "workloads.json")
    assert load_benchmark_manifest() == load_benchmark_manifest(Path(str(V1_MANIFEST)))


def test_missing_manifest_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="benchmark manifest not found"):
        load_benchmark_manifest(tmp_path / "missing.json")


# --- split guardrails -------------------------------------------------------


def test_training_and_tuning_helpers_never_include_test(manifest: BenchmarkManifest) -> None:
    test_ids = {entry.id for entry in manifest.test}

    for selection in (manifest.train, manifest.validation, manifest.tuning):
        assert not test_ids & {entry.id for entry in selection}
    assert manifest.tuning == manifest.train + manifest.validation


def test_splits_partition_the_manifest(manifest: BenchmarkManifest) -> None:
    ids = [
        entry.id
        for split in (manifest.train, manifest.validation, manifest.test)
        for entry in split
    ]

    assert sorted(ids) == sorted(entry.id for entry in manifest.workloads)
    assert len(ids) == len(set(ids))


def test_no_helper_merges_all_splits() -> None:
    public = {name for name in dir(BenchmarkManifest) if not name.startswith("_")}

    assert not {"all", "all_workloads", "all_tuning_workloads", "everything"} & public


def test_moving_a_test_workload_into_train_is_rejected() -> None:
    raw = edited(lambda raw: entry(raw, "syn-test-spike-hard").update(split="train"))

    with pytest.raises(ValidationError, match="must start with 'syn-train-'"):
        validate(raw)


def test_get_returns_workloads_by_id(manifest: BenchmarkManifest) -> None:
    assert manifest.get("azure-val-734400").split == "validation"
    with pytest.raises(KeyError, match="unknown workload id"):
        manifest.get("syn-test-missing")


# --- manifest validation ----------------------------------------------------


def _duplicate(raw: dict[str, Any]) -> None:
    raw["workloads"].append(copy.deepcopy(raw["workloads"][0]))


def _remove_parameter(raw: dict[str, Any]) -> None:
    del entry(raw, "syn-train-bursty")["parameters"]["seed"]


def _overlapping_azure(raw: dict[str, Any]) -> None:
    entry(raw, "azure-train-302400")["parameters"]["start_seconds"] = 131000.0


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_duplicate, "duplicate workload ids"),
        (lambda raw: raw.update(version="v2"), "version"),
        (lambda raw: raw.update(workloads=[]), "workloads"),
        (lambda raw: raw.update(owner="me"), "Extra inputs are not permitted"),
        (lambda raw: entry(raw, "syn-train-spike").update(split="holdout"), "split"),
        (lambda raw: entry(raw, "syn-train-spike").update(source="recorded"), "source"),
        (lambda raw: entry(raw, "syn-train-spike").update(generator="sawtooth"), "generator"),
        (_remove_parameter, "seed"),
        (
            lambda raw: entry(raw, "syn-train-steady-moderate")["parameters"].update(jitter=1.0),
            "Extra inputs are not permitted",
        ),
        (
            lambda raw: entry(raw, "syn-train-steady-moderate")["parameters"].update(rate="100"),
            "rate",
        ),
        (lambda raw: entry(raw, "syn-train-ramp-up").update(duration_seconds=1800.0), "v1"),
        (lambda raw: entry(raw, "syn-val-ramp-down").update(control_interval_seconds=60.0), "v1"),
        (lambda raw: entry(raw, "azure-val-734400").update(dataset="Azure 2019"), "dataset"),
        (
            lambda raw: entry(raw, "azure-val-734400")["parameters"].update(start_seconds=-1.0),
            "start_seconds",
        ),
        (
            lambda raw: entry(raw, "azure-val-734400")["parameters"].update(path="data/raw/x.csv"),
            "Extra inputs are not permitted",
        ),
        (_overlapping_azure, "overlap"),
    ],
    ids=[
        "duplicate_id",
        "wrong_version",
        "empty",
        "unknown_top_level_field",
        "invalid_split",
        "invalid_source",
        "invalid_generator",
        "missing_parameter",
        "unknown_parameter",
        "non_numeric_parameter",
        "wrong_duration",
        "wrong_interval",
        "wrong_dataset",
        "negative_azure_start",
        "azure_path_in_manifest",
        "overlapping_azure_windows",
    ],
)
def test_invalid_manifests_are_rejected(edit: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        validate(edited(edit))


# --- construction -----------------------------------------------------------


def test_synthetic_workloads_rebuild_identically(manifest: BenchmarkManifest) -> None:
    for workload in manifest.workloads:
        if not isinstance(workload, AzureWorkload):
            assert build_workload(workload) == build_workload(workload)


def test_bursty_workloads_reproduce_from_their_committed_seed(manifest: BenchmarkManifest) -> None:
    workload = manifest.get("syn-train-bursty")
    parameters = EXPECTED_SYNTHETIC["syn-train-bursty"][1]
    timing = {"duration_seconds": 3600.0, "control_interval_seconds": 30.0}

    assert build_workload(workload) == bursty_workload(**timing, **parameters)
    assert build_workload(workload) != bursty_workload(**timing, **{**parameters, "seed": 102})
    assert build_workload(workload) != build_workload(manifest.get("syn-val-bursty"))


def test_azure_workload_requires_a_runtime_source_path(manifest: BenchmarkManifest) -> None:
    with pytest.raises(ValueError, match="pass azure_csv_path"):
        build_workload(manifest.get("azure-test-993600"))


def test_azure_workload_uses_the_existing_loader(manifest: BenchmarkManifest) -> None:
    workload = manifest.get("azure-train-129600")

    trace = build_workload(workload, azure_csv_path=AZURE_FIXTURE)

    assert trace == load_azure_trace(
        AZURE_FIXTURE,
        start_seconds=129600.0,
        duration_seconds=3600.0,
        control_interval_seconds=30.0,
    )


def test_build_workloads_matches_individual_builds(manifest: BenchmarkManifest) -> None:
    traces = build_workloads(manifest.workloads, azure_csv_path=AZURE_FIXTURE)

    assert list(traces) == [entry.id for entry in manifest.workloads]
    for entry in manifest.workloads:
        assert traces[entry.id] == build_workload(entry, azure_csv_path=AZURE_FIXTURE)


def test_build_workloads_needs_azure_path_only_for_azure_entries(
    manifest: BenchmarkManifest,
) -> None:
    synthetic = [entry for entry in manifest.tuning if not isinstance(entry, AzureWorkload)]

    assert len(build_workloads(synthetic)) == len(synthetic)
    with pytest.raises(ValueError, match="pass azure_csv_path"):
        build_workloads(manifest.tuning)


def test_azure_validation_reads_the_csv_once(
    manifest: BenchmarkManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scalerl.workloads.azure as azure_module

    passes = []
    real_read_chunks = azure_module._read_chunks

    def counting_read_chunks(*args: Any, **kwargs: Any) -> Any:
        passes.append(args)
        return real_read_chunks(*args, **kwargs)

    monkeypatch.setattr(azure_module, "_read_chunks", counting_read_chunks)

    reports = validate_azure_workloads(manifest, AZURE_FIXTURE)

    assert len(reports) == 6
    assert len(passes) == 1


def fixture_backed_azure_entry() -> AzureWorkload:
    """An ad-hoc Azure entry whose window covers the tiny fixture's timestamps."""
    return AzureWorkload.model_validate(
        {
            "id": "azure-train-0",
            "split": "train",
            "source": "azure",
            "dataset": "Azure Functions Invocation Trace 2021",
            "duration_seconds": 3600.0,
            "control_interval_seconds": 30.0,
            "parameters": {"start_seconds": 0.0},
        }
    )


# --- integration ------------------------------------------------------------


@pytest.mark.parametrize("source", ["synthetic", "azure"])
def test_benchmark_workloads_run_through_the_shared_runner(
    manifest: BenchmarkManifest, source: str
) -> None:
    if source == "synthetic":
        trace = build_workload(manifest.get("syn-train-spike"))
    else:
        trace = build_workload(fixture_backed_azure_entry(), azure_csv_path=AZURE_FIXTURE)
        assert sum(trace.request_rates) > 0
    config = v1_config()
    env = AutoscalingEnv(config, trace)

    for controller in (
        StaticController(3, config.replicas),
        ThresholdController(low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=10),
    ):
        infos = run_episode(env, controller, seed=0)
        assert len(infos) == 120
        assert [info["request_rate"] for info in infos] == list(trace.request_rates)


# --- local Azure validation tool ---------------------------------------------


def test_characterize_trace_reports_exact_statistics() -> None:
    stats = characterize_trace(WorkloadTrace([1.0, 3.0], control_interval_seconds=30.0))

    assert stats == {
        "ticks": 2,
        "mean_rps": 2.0,
        "max_rps": 3.0,
        "std_rps": 1.0,
        "coefficient_of_variation": 0.5,
        "peak_to_mean": 1.5,
    }


def test_characterize_trace_leaves_ratios_undefined_for_zero_demand() -> None:
    stats = characterize_trace(WorkloadTrace([0.0, 0.0], control_interval_seconds=30.0))

    assert stats["coefficient_of_variation"] is None
    assert stats["peak_to_mean"] is None


def test_empty_frozen_windows_are_reported_invalid(manifest: BenchmarkManifest) -> None:
    # The tiny fixture has no invocations inside any frozen window.
    reports = validate_azure_workloads(manifest, AZURE_FIXTURE)

    assert [report["id"] for report in reports] == list(EXPECTED_AZURE_OFFSETS)
    assert all(report["statistics"]["ticks"] == 120 for report in reports)
    assert all(report["problems"] == ["window contains no invocations"] for report in reports)


def _csv_with_one_invocation_per_window(tmp_path: Path) -> Path:
    rows = [f"app,func,{start + 45.0},1.0" for start in EXPECTED_AZURE_OFFSETS.values()]
    path = tmp_path / "windows.csv"
    path.write_text("\n".join(["app,func,end_timestamp,duration", *rows]) + "\n")
    return path


def test_validation_tool_writes_aggregate_only_summary(tmp_path: Path) -> None:
    source = _csv_with_one_invocation_per_window(tmp_path)
    summary_path = tmp_path / "summary.json"

    exit_code = validate_azure_main(
        ["--azure-csv", str(source), "--summary-out", str(summary_path)]
    )

    summary = json.loads(summary_path.read_text())
    assert exit_code == 0
    assert summary["benchmark_version"] == "v1"
    assert summary["source_file"] == "windows.csv"  # name only, no local path
    assert [report["id"] for report in summary["workloads"]] == list(EXPECTED_AZURE_OFFSETS)
    for report in summary["workloads"]:
        assert report["valid"] is True
        assert report["statistics"]["ticks"] == 120
        assert report["statistics"]["max_rps"] == pytest.approx(1 / 30)
        assert set(report) == {
            "id",
            "split",
            "start_seconds",
            "duration_seconds",
            "control_interval_seconds",
            "valid",
            "problems",
            "statistics",
        }


def test_validation_tool_fails_when_a_window_is_empty(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = validate_azure_main(["--azure-csv", str(AZURE_FIXTURE)])

    assert exit_code == 1
    assert "INVALID: window contains no invocations" in capsys.readouterr().out
