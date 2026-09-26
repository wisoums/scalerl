"""Tests for the #80 predictive-baseline experiment guards and runner (no outputs/, no tests)."""

import json
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from scalerl.controllers import (
    PredictiveController,
    ProactivePredictiveController,
    ThresholdController,
)
from scalerl.environment import DELTA_V1, DESIRED_REPLICAS_V1, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import predictive_baseline as experiment
from scalerl.evaluation.predictive_baseline import (
    ExperimentSpec,
    build_experiment_spec,
    final_contract_config,
    load_frozen_spec,
    make_controller,
    require_final_contract,
)

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "benchmarks" / "v1" / "predictive-baseline-experiment-v1.json"
DECISION = REPO / "benchmarks" / "v1" / "action-contract-v2.json"
SPEC_ID = "096b04ef953a"


def test_committed_spec_is_the_frozen_plan() -> None:
    spec = load_frozen_spec(SPEC, DECISION)
    assert spec.experiment_id == SPEC_ID == build_experiment_spec().experiment_id
    assert spec.action_semantics == DESIRED_REPLICAS_V1
    assert spec.action_decision_version == "action-contract-v2"
    assert (spec.action_experiment_spec_id, spec.action_candidate_set_id) == (
        "899dfbb64217",
        "cbe3c1c0719b",
    )
    assert spec.selection_spec_id == "418876d6c8e9"
    assert spec.existing_baseline.model_dump() == {
        "variant": "predictive-v1",
        "forecast_method": "linear-trend",
        "capacity_policy": "forecast-plus-backlog-v1",
    }
    assert spec.new_baseline.model_dump() == {
        "variant": "predictive-seasonal-v1",
        "forecast_method": "historical-profile-plus-linear-v1",
        "capacity_policy": "proactive-scaleout-conservative-scalein-v1",
    }
    assert spec.synthetic_validation_workload_ids == (
        "syn-val-steady-high",
        "syn-val-ramp-down",
        "syn-val-bursty",
    )
    assert spec.azure_profile_train_workload_ids == (
        "azure-train-129600",
        "azure-train-302400",
        "azure-train-475200",
    )
    assert spec.azure_validation_workload_ids == ("azure-val-734400",)
    assert spec.test_workload_ids == ()
    assert spec.dynamics["scenario"] == "nominal" and spec.dynamics["dynamics_seed"] == 0
    assert spec.dynamics["capacity_jitter_fraction"] == 0.0
    assert spec.dynamics["telemetry_delay_ticks"] == 0
    assert spec.evaluation_seed == 0
    assert spec.reward_weights == RewardWeights().model_dump() and spec.reward_changed is False
    assert spec.fixed_policy["blend_weight"] is None
    assert spec.simulator_config["action"] == {"semantics": DESIRED_REPLICAS_V1}
    assert "not a model of record" in str(spec.learned_context["dqn"])


def test_spec_hash_is_deterministic() -> None:
    assert build_experiment_spec() == build_experiment_spec()
    assert build_experiment_spec().model_dump_json(indent=2) + "\n" == SPEC.read_text()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_workload_ids", ["syn-test-seasonal-shifted"], "no test workloads"),
        (
            "synthetic_validation_workload_ids",
            ["syn-val-bursty", "syn-test-spike-hard"],
            "held-out",
        ),
        ("synthetic_validation_workload_ids", ["syn-val-bursty"], "exactly"),
        ("azure_profile_train_workload_ids", ["azure-train-129600", "azure-val-734400"], "exactly"),
        ("azure_profile_train_workload_ids", ["azure-test-993600"], "held-out"),
        ("azure_validation_workload_ids", ["azure-test-1166400"], "held-out"),
        ("simulator_config", SimulatorConfig().model_dump(mode="json"), "desired-replicas-v1"),
        ("reward_weights", {**RewardWeights().model_dump(), "churn": 1.0}, "reward"),
        ("dynamics", {"scenario": "capacity-jitter"}, "nominal"),
        ("action_semantics", DELTA_V1, None),
        ("evaluation_seed", 1, None),
        ("reward_changed", True, None),
    ],
)
def test_spec_rejects_departures(field: str, value: Any, message: str | None) -> None:
    payload = json.loads(SPEC.read_text())
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        ExperimentSpec.model_validate_json(json.dumps(payload))


def test_edited_spec_is_refused(tmp_path: Path) -> None:
    payload = json.loads(SPEC.read_text())
    payload["azure_capacity_note"] = "changed after results"
    edited = tmp_path / "spec.json"
    edited.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        load_frozen_spec(edited, DECISION)


def test_action_decision_is_verified(tmp_path: Path) -> None:
    decision = experiment.verify_action_decision(DECISION)
    assert decision.final_action_semantics == DESIRED_REPLICAS_V1
    payload = json.loads(DECISION.read_text())
    payload["experiment_spec_id"] = "000000000000"
    other = tmp_path / "decision.json"
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="experiment_spec_id"):
        experiment.verify_action_decision(other)


def test_package_default_delta_cannot_leak_in() -> None:
    assert SimulatorConfig().action.semantics == DELTA_V1  # the package default is unchanged
    config = final_contract_config()
    assert config.action.semantics == DESIRED_REPLICAS_V1
    require_final_contract(config)
    with pytest.raises(ValueError, match="delta-v1"):
        require_final_contract(SimulatorConfig())
    with pytest.raises(ValueError, match="delta-v1"):
        make_controller("predictive-v1", SimulatorConfig(), None)
    other = config.model_copy(
        update={"replicas": config.replicas.model_copy(update={"max_replicas": 12})}
    )
    with pytest.raises(ValueError, match="canonical"):
        require_final_contract(other)


def test_every_controller_acts_under_desired_replicas() -> None:
    config = final_contract_config()
    built = {v: make_controller(v, config, None) for v in experiment.CONTROLLERS}
    assert isinstance(built["threshold-v1"], ThresholdController)
    assert isinstance(built["predictive-v1"], PredictiveController)
    assert isinstance(built["predictive-seasonal-v1"], ProactivePredictiveController)
    for controller in built.values():
        assert controller.action_contract is not None  # type: ignore[attr-defined]
        assert controller.action_contract.semantics == DESIRED_REPLICAS_V1  # type: ignore[attr-defined]
    threshold = built["threshold-v1"]
    assert isinstance(threshold, ThresholdController)
    assert (threshold.high_threshold, threshold.low_threshold, threshold.cooldown_ticks) == (
        0.6,
        0.2,
        3,
    )


def test_cases_are_nominal_validation_only_with_profile_only_on_azure() -> None:
    spec = build_experiment_spec()
    synthetic = experiment.cases(spec, azure=False, profile=None)
    assert {c.workload_id for c in synthetic} == set(spec.synthetic_validation_workload_ids)
    assert len(synthetic) == 9 and all(c.profile_id is None for c in synthetic)
    assert all(c.case_id.endswith("|nominal|dyn0|eval0") for c in synthetic)
    from scalerl.controllers import HistoricalDemandProfile
    from scalerl.workloads import WorkloadTrace

    profile = HistoricalDemandProfile.from_traces({"x": WorkloadTrace([1.0] * 120, 30.0)})
    with_azure = experiment.cases(spec, azure=True, profile=profile)
    profiled = [c for c in with_azure if c.profile_id is not None]
    assert [(c.controller_variant, c.workload_id) for c in profiled] == [
        ("predictive-seasonal-v1", "azure-val-734400")
    ]
    assert not any("test" in c.workload_id for c in with_azure)


def test_cli_has_no_workload_or_test_option() -> None:
    with pytest.raises(SystemExit):
        experiment.main(["run", "--include-test"])
    with pytest.raises(SystemExit):
        experiment.main(["run", "--workload", "syn-test-seasonal-shifted"])


def test_cli_check_prints_the_plan(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    args = ["check", "--spec", str(SPEC), "--action-decision", str(DECISION)]
    assert experiment.main([*args, "--azure-csv", str(tmp_path / "absent.csv")]) == 0
    printed = capsys.readouterr().out
    expected = (
        SPEC_ID,
        "desired-replicas-v1",
        "action-contract-v2",
        "899dfbb64217",
        "cbe3c1c0719b",
        "418876d6c8e9",
        "test workload count     0",
        "reward changed          false",
        "Azure CSV               absent",
    )
    for text in expected:
        assert text in printed


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_run_without_azure_is_explicit_resumable_and_tagged(
    tmp_path: Path, tracking_uri: str
) -> None:
    spec = build_experiment_spec()
    out = tmp_path / "out"
    rows, azure, profile = experiment.run_experiment(
        spec, out=out, azure_csv_path=tmp_path / "absent.csv", tracking_uri=tracking_uri,
        progress=lambda _: None,
    )  # fmt: skip
    assert azure == "unavailable_external_data" and profile is None
    assert len(rows) == 9 and {r["workload_split"] for r in rows} == {"validation"}
    assert {r["action_semantics"] for r in rows} == {DESIRED_REPLICAS_V1}
    seasonal_rows = [r for r in rows if r["controller_variant"] == "predictive-seasonal-v1"]
    assert all(r["forecast.profile_used_count"] == 0.0 for r in seasonal_rows)
    assert all(r["forecast.horizon_ticks"] == 3.0 for r in seasonal_rows)
    assert all("forecast.forecast_count" not in r for r in rows if r["controller"] == "threshold")
    assert all(r["action.max_absolute_replica_change_in_one_tick"] >= 0 for r in rows)
    client = MlflowClient(tracking_uri)
    ml_experiment = client.get_experiment_by_name(experiment.EXPERIMENT_NAME)
    assert ml_experiment is not None
    runs = client.search_runs([ml_experiment.experiment_id])
    assert len(runs) == 9
    tags = client.get_run(seasonal_rows[0]["mlflow_run_id"]).data.tags
    assert tags["scalerl.action_semantics"] == DESIRED_REPLICAS_V1
    assert tags["scalerl.forecast_method"] == "historical-profile-plus-linear-v1"
    assert tags["scalerl.capacity_policy"] == "proactive-scaleout-conservative-scalein-v1"
    assert tags["scalerl.action_decision_version"] == "action-contract-v2"
    assert tags["scalerl.profile_id"] == "none" and tags["scalerl.experiment_id"] == SPEC_ID
    records = experiment.read_rows(out / "forecast-records.jsonl")
    assert records and all("actual_target_rps" in r for r in records)  # joined after the run

    # Resume: a lost + torn row is recovered from its finished run, never re-run.
    raw = out / "raw-results.jsonl"
    lines = raw.read_text().splitlines()
    raw.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:15])
    again, _, _ = experiment.run_experiment(
        spec, out=out, azure_csv_path=None, tracking_uri=tracking_uri, progress=lambda _: None
    )
    assert [r["case_id"] for r in again] == [r["case_id"] for r in rows]
    assert len(client.search_runs([ml_experiment.experiment_id])) == 9

    report = experiment.write_reports(spec, out, again, "unavailable_external_data", None)
    assert report["azure_validation_status"] == "unavailable_external_data"
    artifact = experiment.build_artifact(spec, again, "unavailable_external_data", None)
    assert artifact.azure_validation_status == "unavailable_external_data"
    assert artifact.held_out_data_used is False and artifact.declares_overall_winner is False
    assert set(artifact.validation_results) == set(spec.synthetic_validation_workload_ids)


def test_committed_baseline_artifact() -> None:
    path = REPO / "benchmarks" / "v1" / "predictive-baseline-v1.json"
    artifact = experiment.PredictiveBaselineArtifact.model_validate_json(path.read_text())
    assert artifact.artifact_id == "162b6fb3e9c9"
    assert artifact.experiment_spec_id == SPEC_ID
    assert artifact.frozen_baseline.variant == "predictive-seasonal-v1"
    assert artifact.existing_baseline.variant == "predictive-v1"
    assert artifact.action_semantics == DESIRED_REPLICAS_V1
    assert artifact.action_decision_version == "action-contract-v2"
    assert artifact.azure_validation_status == "available"
    assert artifact.historical_profile["train_workload_ids"] == [
        "azure-train-129600",
        "azure-train-302400",
        "azure-train-475200",
    ]
    assert artifact.historical_profile["profile_id"] == "d050d3b8ca0f"
    assert set(artifact.validation_results) == {
        "syn-val-steady-high",
        "syn-val-ramp-down",
        "syn-val-bursty",
        "azure-val-734400",
    }
    assert artifact.held_out_data_used is False and artifact.reward_changed is False
    assert artifact.declares_overall_winner is False
    assert "-test-" not in path.read_text()
