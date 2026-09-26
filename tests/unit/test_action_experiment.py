"""Tests for the #79 action-contract experiment: frozen spec, matched candidates, #78 selection.

Tiny budgets and synthetic evidence only; nothing reads outputs/ or test workloads.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from scalerl.environment import DELTA_V1, DESIRED_REPLICAS_V1
from scalerl.evaluation import action_semantics as experiment
from scalerl.evaluation.action_semantics import (
    FAMILIES,
    ExperimentSpec,
    TrainingEvidence,
    WorkloadEvidence,
    build_experiment_spec,
    evaluation_cases,
    load_frozen_inputs,
    run_selection,
    screening_path,
)
from scalerl.evaluation.model_selection import VALIDATION_WORKLOADS, SelectionSpec
from scalerl.tuning.candidates import CandidateSet, generate_candidate_set

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "benchmarks" / "v1" / "action-semantics-experiment-v1.json"
CANDIDATES = REPO / "benchmarks" / "v1" / "action-semantics-candidates-v1.json"
SELECTION = REPO / "benchmarks" / "v1" / "selection-v2-cost-under-sla.json"
EXPERIMENT_ID = "899dfbb64217"
CANDIDATE_SET_ID = "cbe3c1c0719b"
THRESHOLDS = {
    "syn-val-steady-high": 0.275,
    "syn-val-ramp-down": 0.35,
    "syn-val-bursty": 0.20833333333333334,
}


@pytest.fixture(scope="module")
def small() -> CandidateSet:
    return generate_candidate_set(count=3)


@pytest.fixture(scope="module")
def selection() -> SelectionSpec:
    return SelectionSpec.load(SELECTION)


# --- frozen candidate set ----------------------------------------------------------------------


def test_committed_candidate_set_is_the_seeded_generation() -> None:
    committed = CandidateSet.load(CANDIDATES)
    assert committed.candidate_set_id == CANDIDATE_SET_ID
    assert committed == generate_candidate_set()
    assert committed.generation_seed == 42 and committed.independent_of_results is True
    assert (len(committed.dqn.candidates), len(committed.ppo.candidates)) == (20, 20)
    assert committed.dqn.search_space_version == "dqn-search-v1"
    assert committed.ppo.search_space_version == "ppo-search-v1"
    assert [c.candidate_id for c in committed.dqn.candidates][:3] == [
        "dqn-c00",
        "dqn-c01",
        "dqn-c02",
    ]


def test_candidate_order_depends_only_on_the_seed(small: CandidateSet) -> None:
    committed = CandidateSet.load(CANDIDATES)
    # Generation evaluates nothing, so a prefix of the sequence is the same sequence.
    assert small.dqn.candidates == committed.dqn.candidates[:3]
    assert small.ppo.candidates == committed.ppo.candidates[:3]
    assert generate_candidate_set(count=3, seed=7) != small


def test_candidates_are_valid_v1_configurations() -> None:
    committed = CandidateSet.load(CANDIDATES)
    for algorithm in ("dqn", "ppo"):
        for candidate in committed.for_algorithm(algorithm).candidates:
            hyperparameters = committed.hyperparameters(algorithm, candidate.candidate_id)
            if algorithm == "ppo":
                assert 204_800 % hyperparameters.n_steps == 0  # type: ignore[union-attr]
            else:
                assert 200_000 % hyperparameters.train_freq == 0  # type: ignore[union-attr]


def test_candidate_set_rejects_reordering(small: CandidateSet) -> None:
    payload = json.loads(small.model_dump_json())
    payload["dqn"]["candidates"].reverse()
    with pytest.raises(ValidationError, match="generation order"):
        CandidateSet.model_validate_json(json.dumps(payload))


# --- frozen experiment spec --------------------------------------------------------------------


def test_committed_spec_is_the_frozen_plan(selection: SelectionSpec) -> None:
    spec, candidates, loaded = load_frozen_inputs(SPEC, CANDIDATES, SELECTION)
    assert spec.experiment_id == EXPERIMENT_ID
    assert spec == build_experiment_spec(candidates)
    assert spec.candidate_set_id == CANDIDATE_SET_ID
    assert loaded.spec_id == spec.selection_spec_id == "418876d6c8e9"
    assert spec.train_workload_id == "syn-train-bursty"
    assert spec.validation_workload_ids == VALIDATION_WORKLOADS
    assert spec.test_workload_ids == ()
    assert spec.timesteps == {"dqn": 200_000, "ppo": 204_800}
    assert (spec.screening_training_seed, spec.retraining_seeds) == (0, (0, 1, 2, 3, 4))
    assert spec.evaluation_seed == 0 and spec.dynamics_condition["scenario"] == "nominal"
    assert spec.decision_principle.preferred_contract == DESIRED_REPLICAS_V1
    assert "which contract has better validation metrics" in spec.decision_principle.not_decided_by
    assert spec.reward_weights == {
        "latency": 1.0,
        "cost": 1.0,
        "sla": 1.0,
        "queue": 1.0,
        "churn": 0.1,
    }
    assert [c.semantics for c in spec.action_contracts] == [DELTA_V1, DESIRED_REPLICAS_V1]
    assert spec.families == FAMILIES
    assert spec.rule_controllers["predictive"] == {
        "variant": "predictive-v1",
        "forecast_method": "linear-trend",
        "capacity_policy": "forecast-plus-backlog-v1",
        "params": {"history_window_ticks": 4, "target_utilization": 0.8},
        "desired_replicas_v1_encoding": "its computed desired replicas as the target",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_workload_ids", ["syn-test-spike-hard"], "no test workloads"),
        ("validation_workload_ids", ["syn-val-bursty", "syn-test-spike-hard"], "held-out"),
        ("validation_workload_ids", ["syn-val-bursty"], "exactly"),
        ("train_workload_id", "azure-test-993600", None),
        ("timesteps", {"dqn": 100_000, "ppo": 204_800}, "budgets"),
        ("retraining_seeds", [0, 1, 2], "retraining seeds"),
        ("screening_training_seed", 1, None),
        ("evaluation_seed", 3, None),
        ("selection_spec_id", "000000000000", None),
        ("dynamics_condition", {"scenario": "capacity-jitter"}, "nominal"),
    ],
)
def test_spec_rejects_departures_from_the_plan(field: str, value: Any, message: str | None) -> None:
    payload = json.loads(SPEC.read_text())
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        ExperimentSpec.model_validate_json(json.dumps(payload))


def test_edited_spec_is_refused_even_if_valid(tmp_path: Path) -> None:
    payload = json.loads(SPEC.read_text())
    payload["reward_definition"] = "something else"
    edited = tmp_path / "spec.json"
    edited.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        load_frozen_inputs(edited, CANDIDATES, SELECTION)


def test_other_selection_spec_is_refused(tmp_path: Path) -> None:
    payload = json.loads(SELECTION.read_text())
    payload["provenance"]["evidence"] = "edited"
    other = tmp_path / "selection.json"
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="418876d6c8e9"):
        load_frozen_inputs(SPEC, CANDIDATES, other)


def test_cli_refuses_a_test_workload_before_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(SPEC.read_text())
    payload["validation_workload_ids"] = ["syn-test-seasonal-shifted", *VALIDATION_WORKLOADS[1:]]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(payload))
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        experiment.main(
            ["screen", "--family", "dqn-delta-v1", "--spec", str(spec_path),
             "--candidates", str(CANDIDATES), "--selection-spec", str(SELECTION),
             "--output-dir", str(out)]
        )  # fmt: skip
    assert not out.exists()
    assert "held-out" in capsys.readouterr().err


def test_cli_has_no_workload_or_test_option() -> None:
    with pytest.raises(SystemExit):
        experiment.main(["screen", "--family", "dqn-delta-v1", "--include-test"])
    with pytest.raises(SystemExit):
        experiment.main(["evaluate", "--workload", "syn-test-spike-hard"])


def test_check_prints_the_frozen_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert experiment.main(["check", "--spec", str(SPEC), "--candidates", str(CANDIDATES),
                            "--selection-spec", str(SELECTION)]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    for expected in (EXPERIMENT_ID, CANDIDATE_SET_ID, "418876d6c8e9", "test workloads     none"):
        assert expected in printed


# --- selection at the configuration level ---------------------------------------------------------


def evidence(
    spec: ExperimentSpec,
    family: str,
    index: int,
    *,
    sla: dict[str, float],
    cost: float,
    reward: float = -100.0,
) -> TrainingEvidence:
    algorithm, semantics = experiment.parse_family(family)
    return TrainingEvidence(
        experiment_id=spec.experiment_id,
        candidate_set_id=spec.candidate_set_id,
        phase="screening",
        family=family,
        algorithm=algorithm,
        action_semantics=semantics,
        candidate_id=f"{algorithm}-c{index:02d}",
        candidate_index=index,
        training_seed=0,
        hyperparameters={},
        training_run_id=f"run-{family}-{index}",
        model_artifact_uri=f"runs:/run-{family}-{index}/model",
        compatibility={"action_semantics_version": semantics, "action_count": 3},
        training_episodes=1,
        workloads=tuple(
            WorkloadEvidence(
                workload_id=w,
                validation_run_id=f"val-{family}-{index}-{w}",
                metrics={
                    "sla_violation_rate": sla[w],
                    "normalized_cost": cost,
                    "queue_pressure": 0.1,
                    "churn_rate": 0.1,
                    "episode_reward": reward,
                    "mean_p95_latency_seconds": 1.0,
                },
                action={
                    "total_absolute_replica_change": 3.0,
                    "max_absolute_replica_change_in_one_tick": 1.0,
                },
            )
            for w in VALIDATION_WORKLOADS
        ),
    )


def write_family(out: Path, spec: ExperimentSpec, family: str, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        item = evidence(spec, family, index, **row)
        item.save(screening_path(out, family, item.candidate_id))


FEASIBLE = dict(THRESHOLDS)
BURSTY_MISS = {**THRESHOLDS, "syn-val-bursty": 0.21}


def test_selection_uses_the_frozen_78_rule_per_family(
    small: CandidateSet, selection: SelectionSpec, tmp_path: Path
) -> None:
    spec = build_experiment_spec(small)
    # Cheapest candidate misses on one workload only: per-workload feasibility rejects it.
    rows = [
        {"sla": BURSTY_MISS, "cost": 0.3, "reward": 0.0},  # best reward, infeasible
        {"sla": FEASIBLE, "cost": 0.6, "reward": -500.0},  # feasible, worst reward
        {"sla": {w: 0.0 for w in VALIDATION_WORKLOADS}, "cost": 0.9},
    ]
    for family in FAMILIES:
        write_family(tmp_path, spec, family, rows)
    results = run_selection(spec, small, selection, out=tmp_path)
    for family in FAMILIES:
        result = results[family]
        algorithm = family.split("-")[0]
        assert result.spec_id == "418876d6c8e9"
        assert result.feasible_candidate_ids == (f"{algorithm}-c01", f"{algorithm}-c02")
        assert result.selected_candidate_id == f"{algorithm}-c01"
    selected = json.loads((tmp_path / "selected-configs.json").read_text())
    assert selected["selection_spec_id"] == "418876d6c8e9"
    assert selected["families"]["dqn-desired-replicas-v1"]["feasible_candidate_count"] == 2


def test_zero_feasible_family_selects_nothing_and_cannot_be_retrained(
    small: CandidateSet, selection: SelectionSpec, tmp_path: Path
) -> None:
    spec = build_experiment_spec(small)
    for family in FAMILIES:
        write_family(
            tmp_path, spec, family, [{"sla": BURSTY_MISS, "cost": c} for c in (0.2, 0.4, 0.6)]
        )
    results = run_selection(spec, small, selection, out=tmp_path)
    result = results["ppo-desired-replicas-v1"]
    assert result.feasible_candidate_count == 0 and result.selected_candidate_id is None
    assert result.diagnostic_fallback_candidate_id is not None
    with pytest.raises(ValueError, match="nothing to retrain"):
        experiment.run_retraining(
            spec, small, "ppo-desired-replicas-v1", out=tmp_path, tracking_uri=None
        )


def test_incomplete_screening_is_refused(
    small: CandidateSet, selection: SelectionSpec, tmp_path: Path
) -> None:
    spec = build_experiment_spec(small)
    write_family(tmp_path, spec, "dqn-delta-v1", [{"sla": FEASIBLE, "cost": 0.5}])
    with pytest.raises(ValueError, match="screening incomplete"):
        run_selection(spec, small, selection, out=tmp_path)


def test_evidence_from_another_experiment_is_refused(
    small: CandidateSet, selection: SelectionSpec, tmp_path: Path
) -> None:
    spec = build_experiment_spec(small)
    for family in FAMILIES:
        write_family(tmp_path, spec, family, [{"sla": FEASIBLE, "cost": 0.5}] * 3)
    other = generate_candidate_set(count=3, seed=7)
    with pytest.raises(ValueError, match="another experiment"):
        run_selection(build_experiment_spec(other), other, selection, out=tmp_path)


def test_evaluation_cases_are_nominal_seed_separated_and_keep_every_seed(
    small: CandidateSet,
) -> None:
    spec = build_experiment_spec(small)
    models = [
        evidence(spec, family, 1, sla=FEASIBLE, cost=0.5).model_copy(
            update={"phase": "retraining", "training_seed": seed}
        )
        for family in FAMILIES
        for seed in (0, 1, 2, 3, 4)
    ]
    cases = evaluation_cases(spec, models)
    assert len(cases) == 3 * (4 + 20)
    assert all(case.case_id.endswith("|nominal|dyn0|eval0") for case in cases)
    learned = [c for c in cases if c.training_seed is not None]
    assert {c.training_seed for c in learned} == {0, 1, 2, 3, 4}
    assert len({c.controller_variant_id for c in learned}) == 20  # no best-seed pick
    rules = {c.controller_variant_id for c in cases if c.training_seed is None}
    assert rules == {
        "threshold-delta-v1",
        "threshold-desired-replicas-v1",
        "predictive-delta-v1",
        "predictive-desired-replicas-v1",
    }


def test_rule_controllers_use_canonical_parameters_under_each_contract() -> None:
    for semantics in (DELTA_V1, DESIRED_REPLICAS_V1):
        config, _ = experiment.contract_config(semantics)
        case = experiment.EvaluationCase(
            case_id="x", controller="threshold", controller_variant_id="t",
            controller_version="threshold-sla-first-v1", action_semantics=semantics,
            workload_id="syn-val-bursty",
        )  # fmt: skip
        controller = experiment.make_rule_controller(case, config)
        assert isinstance(controller, experiment.ThresholdController)
        assert (controller.high_threshold, controller.low_threshold, controller.cooldown_ticks) == (
            0.6,
            0.2,
            3,
        )
        assert controller.action_contract is not None
        assert controller.action_contract.semantics == semantics


def test_contract_configs_differ_only_in_action_semantics() -> None:
    delta, delta_source = experiment.contract_config(DELTA_V1)
    desired, desired_source = experiment.contract_config(DESIRED_REPLICAS_V1)
    assert (delta_source, desired_source) == ("default", "predeclared")
    assert desired.model_copy(update={"action": delta.action}) == delta


def test_invariant_checks_all_pass() -> None:
    checks = experiment.check_action_invariants()
    assert len(checks) >= 10 and all(check["passed"] for check in checks)


# --- end to end with tiny budgets -------------------------------------------------------------


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _runs(uri: str) -> int:
    client = MlflowClient(uri)
    experiment_ = client.get_experiment_by_name(experiment.EXPERIMENT_NAME)
    assert experiment_ is not None
    return len(client.search_runs([experiment_.experiment_id], max_results=10_000))


def test_pipeline_end_to_end_is_resumable_and_deduplicated(
    tmp_path: Path, tracking_uri: str, selection: SelectionSpec
) -> None:
    candidates = generate_candidate_set(count=1)
    spec = build_experiment_spec(candidates)
    out = tmp_path / "out"
    budgets = {"dqn": 64, "ppo": 2048}
    quiet = lambda _: None  # noqa: E731
    for family in FAMILIES:
        algorithm, semantics = experiment.parse_family(family)
        (item,) = experiment.run_screening(
            spec, candidates, family, out=out, tracking_uri=tracking_uri,
            timesteps=budgets[algorithm], progress=quiet,
        )  # fmt: skip
        assert item.compatibility["action_semantics_version"] == semantics
        assert item.compatibility["action_count"] == (3 if semantics == DELTA_V1 else 10)
        assert [w.workload_id for w in item.workloads] == list(VALIDATION_WORKLOADS)
    screened = _runs(tracking_uri)
    assert screened == 4 * (1 + 3)
    experiment.run_screening(
        spec, candidates, "dqn-desired-replicas-v1", out=out, tracking_uri=tracking_uri,
        timesteps=64, progress=quiet,
    )  # fmt: skip
    assert _runs(tracking_uri) == screened  # resume trains nothing twice

    # Force one successful selection per family so retraining/evaluation can run.
    results = run_selection(spec, candidates, selection, out=out)
    for family, result in results.items():
        if not result.selection_succeeded:
            forced = result.model_copy(
                update={
                    "feasible_candidate_ids": result.candidate_ids,
                    "infeasible_candidate_ids": (),
                    "feasible_candidate_count": 1,
                    "selection_succeeded": True,
                    "selected_candidate_id": result.candidate_ids[0],
                    "selected_objective_values": {"normalized_cost": 0.5},
                    "diagnostic_fallback_candidate_id": None,
                    "diagnostic_fallback": None,
                }
            )
            forced.save(out / f"{family}-selection.json")
    retrain_spec = spec.model_copy(update={"retraining_seeds": (0, 1)})
    for family in ("dqn-delta-v1", "dqn-desired-replicas-v1"):
        models = experiment.run_retraining(
            retrain_spec, candidates, family, out=out, tracking_uri=tracking_uri,
            timesteps=64, progress=quiet,
        )  # fmt: skip
        assert [m.training_seed for m in models] == [0, 1]
        assert all(
            m.phase == "retraining" and m.selection_spec_id == "418876d6c8e9" for m in models
        )
    rows = experiment.run_evaluation(
        retrain_spec, out=out, tracking_uri=tracking_uri, progress=quiet
    )
    assert len(rows) == 3 * (4 + 4)
    assert {row["robustness_scenario"] for row in rows} == {"nominal"}
    assert all(row["workload_split"] == "validation" for row in rows)
    evaluated = _runs(tracking_uri)

    # A lost local row is recovered from its finished MLflow run, never re-run.
    raw = out / "raw-results.jsonl"
    lines = raw.read_text().splitlines()
    raw.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:10])  # drop + tear the tail
    again = experiment.run_evaluation(
        retrain_spec, out=out, tracking_uri=tracking_uri, progress=quiet
    )
    assert [r["case_id"] for r in again] == [r["case_id"] for r in rows]
    assert _runs(tracking_uri) == evaluated

    learned = [r for r in rows if r["training_seed"] is not None]
    report = experiment._evaluation_matches_retraining(
        learned, experiment.load_evidence(sorted((out / "retraining").glob("*/seed*.json")))
    )
    assert report == {"compared": 12, "mismatches": 0}  # deterministic reload + evaluation
    desired_rows = [r for r in rows if r["action_semantics"] == DESIRED_REPLICAS_V1]
    assert all(r["action_max_absolute_replica_change_in_one_tick"] >= 0 for r in desired_rows)
    client = MlflowClient(tracking_uri)
    tags = client.get_run(learned[0]["mlflow_run_id"]).data.tags
    assert tags["scalerl.experiment_id"] == retrain_spec.experiment_id
    assert tags["scalerl.selection_spec_id"] == "418876d6c8e9"
    assert tags["scalerl.action_semantics"] in (DELTA_V1, DESIRED_REPLICAS_V1)
    training_tags = client.get_run(learned[0]["training_run_id"]).data.tags
    assert training_tags["scalerl.experiment_phase"] == "retraining"
    assert training_tags["scalerl.candidate_set_id"] == spec.candidate_set_id


# --- the frozen decision ----------------------------------------------------------------------


def test_committed_decision_freezes_desired_replicas_from_this_experiment() -> None:
    path = REPO / "benchmarks" / "v1" / "action-contract-v2.json"
    decision = experiment.ActionContractDecision.model_validate_json(path.read_text())
    assert decision.decision_id == "0eb5562b01e6"
    assert decision.final_action_semantics == DESIRED_REPLICAS_V1
    assert decision.experiment_spec_id == EXPERIMENT_ID
    assert decision.candidate_set_id == CANDIDATE_SET_ID
    assert decision.selection_spec_id == "418876d6c8e9"
    assert all(decision.criteria.values()) and len(decision.criteria) == 3
    assert decision.held_out_data_used is False and decision.reward_changed is False
    # DQN under desired-replicas-v1 had no feasible configuration: nothing was selected for it.
    dqn_desired = decision.selected_configurations["dqn-desired-replicas-v1"]
    assert isinstance(dqn_desired, dict) and dqn_desired["selected_candidate_id"] is None
    families = {model["family"] for model in decision.retrained_models}
    assert families == {"dqn-delta-v1", "ppo-delta-v1", "ppo-desired-replicas-v1"}
    assert len(decision.retrained_models) == 15
    assert "syn-test" not in path.read_text() and "azure-test" not in path.read_text()
