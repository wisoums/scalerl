"""Tests for ``selection-v2-cost-under-sla`` (#78). Synthetic fixtures only; no outputs/."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scalerl.evaluation import model_selection
from scalerl.evaluation.model_selection import (
    FEASIBILITY_TOLERANCE,
    VALIDATION_WORKLOADS,
    Candidate,
    SelectionSpec,
    WorkloadMetrics,
    check_spec_matches_evidence,
    freeze_spec_from_multiseed,
    multiseed_candidates,
    nominal_validation_rows,
    select,
)
from scalerl.tuning.sb3 import SELECTION_KEYS, TrialAggregate, select_trial

REPO = Path(__file__).resolve().parents[2]
COMMITTED_SPEC = REPO / "benchmarks" / "v1" / "selection-v2-cost-under-sla.json"
COMMITTED_SPEC_ID = "418876d6c8e9"
THRESHOLDS = {
    "syn-val-steady-high": 0.275,
    "syn-val-ramp-down": 0.35,
    "syn-val-bursty": 0.20833333333333334,
}


def make_spec(**overrides: Any) -> SelectionSpec:
    data: dict[str, Any] = {
        "benchmark_version": "v1",
        "reference": {
            "controller_version": "threshold-sla-first-v1",
            "params": {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3},
            "tuning_lineage": {"study_name": "threshold-v1", "selected_trial_number": 14},
        },
        "validation_workload_ids": list(VALIDATION_WORKLOADS),
        "sla_thresholds": [
            {"workload_id": w, "sla_violation_rate": s, "evaluation_run_id": f"run-{w}"}
            for w, s in THRESHOLDS.items()
        ],
        "provenance": {"evidence": "unit-test"},
    }
    data.update(overrides)
    return SelectionSpec.model_validate_json(json.dumps(data))


def metrics(
    sla: float, cost: float = 0.5, queue: float = 0.1, churn: float = 0.1
) -> WorkloadMetrics:
    return WorkloadMetrics(
        sla_violation_rate=sla, normalized_cost=cost, queue_pressure=queue, churn_rate=churn
    )


def candidate(
    candidate_id: str,
    *,
    sla: dict[str, float] | float | None = None,
    cost: float = 0.5,
    queue: float = 0.1,
    churn: float = 0.1,
    family: str = "dqn",
    index: int | None = None,
) -> Candidate:
    """A candidate with the given per-workload SLA (default: 0.1 everywhere, feasible)."""
    per = (
        sla
        if isinstance(sla, dict)
        else dict.fromkeys(VALIDATION_WORKLOADS, 0.1 if sla is None else sla)
    )
    return Candidate(
        candidate_id=candidate_id,
        family=family,
        tie_break_index=index,
        metrics_by_workload={w: metrics(per[w], cost, queue, churn) for w in VALIDATION_WORKLOADS},
    )


# --- A. historical v1 selectors are preserved ------------------------------------------


def test_v1_selector_still_sla_first() -> None:
    cheap = TrialAggregate(1, 0.20, 0.50, 0.1, 0.1)
    full_fleet = TrialAggregate(2, 0.10, 0.97, 0.05, 0.1)
    assert select_trial([cheap, full_fleet]).number == 2
    assert SELECTION_KEYS == (
        "validation_sla_violation_rate",
        "validation_normalized_cost",
        "validation_queue_pressure",
        "validation_churn_rate",
    )


def test_v2_differs_from_v1_on_the_same_evidence() -> None:
    spec = make_spec()
    cheap = candidate("cheap", sla=0.2, cost=0.5)
    full_fleet = candidate("full", sla=0.1, cost=0.97)
    assert select(spec, "dqn", [cheap, full_fleet]).selected_candidate_id == "cheap"


# --- B. per-workload feasibility ------------------------------------------------------


def test_feasibility_is_per_workload_not_on_the_mean() -> None:
    spec = make_spec()
    # Mean SLA (0.1) is far below every threshold, but bursty fails by 0.01.
    hidden = candidate(
        "hidden",
        sla={"syn-val-steady-high": 0.0, "syn-val-ramp-down": 0.0, "syn-val-bursty": 0.21833333},
        cost=0.1,
    )
    ok = candidate("ok", cost=0.9)
    result = select(spec, "dqn", [hidden, ok])
    assert result.infeasible_candidate_ids == ("hidden",)
    assert result.selected_candidate_id == "ok"
    bursty = next(w for w in result.candidates[0].workloads if w.workload_id == "syn-val-bursty")
    assert not bursty.feasible and bursty.sla_excess == pytest.approx(0.01)


def test_equality_is_feasible_and_tolerance_is_only_an_epsilon() -> None:
    spec = make_spec()
    assert spec.feasibility.tolerance == FEASIBILITY_TOLERANCE == 1e-12
    equal = candidate("equal", sla=dict(THRESHOLDS))
    noise = candidate("noise", sla={w: s + 1e-13 for w, s in THRESHOLDS.items()})
    over = candidate("over", sla={w: s + 1e-6 for w, s in THRESHOLDS.items()})
    result = select(spec, "dqn", [equal, noise, over])
    assert set(result.feasible_candidate_ids) == {"equal", "noise"}
    assert result.infeasible_candidate_ids == ("over",)
    with pytest.raises(ValidationError):
        make_spec(feasibility={"tolerance": 0.01})


# --- C. cost wins among feasible candidates ---------------------------------------------


def test_lower_cost_feasible_beats_better_sla_higher_cost() -> None:
    spec = make_spec()
    result = select(
        spec,
        "ppo",
        [
            candidate("best-sla", sla=0.0, cost=0.97, family="ppo"),
            candidate("cheaper", sla=0.2, cost=0.6, family="ppo"),
        ],
    )
    assert result.selected_candidate_id == "cheaper"
    assert result.selected_objective_values is not None
    assert result.selected_objective_values["normalized_cost"] == pytest.approx(0.6)


# --- D. tie-break order ---------------------------------------------------------------


def test_tie_break_order_queue_churn_sla_key() -> None:
    spec = make_spec()
    assert (
        select(
            spec,
            "dqn",
            [candidate("a", queue=0.3, churn=0.0), candidate("b", queue=0.2, churn=0.9)],
        ).selected_candidate_id
        == "b"
    )
    assert (
        select(
            spec, "dqn", [candidate("a", churn=0.3, sla=0.0), candidate("b", churn=0.2, sla=0.2)]
        ).selected_candidate_id
        == "b"
    )
    assert (
        select(
            spec, "dqn", [candidate("a", sla=0.2), candidate("b", sla=0.1)]
        ).selected_candidate_id
        == "b"
    )
    assert (
        select(
            spec, "dqn", [candidate("b", index=0), candidate("a", index=1)]
        ).selected_candidate_id
        == "b"
    )
    assert select(spec, "dqn", [candidate("b"), candidate("a")]).selected_candidate_id == "a"
    # Numeric key, not string order: seed 2 precedes seed 10.
    assert (
        select(
            spec, "dqn", [candidate("dqn-seed10", index=10), candidate("dqn-seed2", index=2)]
        ).selected_candidate_id
        == "dqn-seed2"
    )


def test_spec_rejects_a_different_tie_break_order() -> None:
    with pytest.raises(ValidationError, match="tie-break"):
        make_spec(
            tie_breaks=["churn_rate", "queue_pressure", "sla_violation_rate", "candidate_key"]
        )
    with pytest.raises(ValidationError, match="fallback"):
        make_spec(
            diagnostic_fallback_order=[
                "normalized_cost",
                "sla_violation_rate",
                "queue_pressure",
                "churn_rate",
                "candidate_key",
            ]
        )


# --- E. no feasible candidate ------------------------------------------------------------


def test_zero_feasible_selects_nothing_and_reports_diagnostic() -> None:
    spec = make_spec()
    worse = candidate("worse", sla=0.5, cost=0.1)
    closer = candidate(
        "closer",
        sla={"syn-val-steady-high": 0.3, "syn-val-ramp-down": 0.3, "syn-val-bursty": 0.3},
        cost=0.9,
    )
    result = select(spec, "dqn", [worse, closer])
    assert result.feasible_candidate_count == 0
    assert not result.selection_succeeded
    assert result.selected_candidate_id is None and result.selected_objective_values is None
    fallback = result.diagnostic_fallback
    assert fallback is not None and fallback.candidate_id == "closer"
    assert result.diagnostic_fallback_candidate_id == "closer"
    assert fallback.label.startswith("DIAGNOSTIC ONLY")
    excess = {w.workload_id: w.sla_excess for w in fallback.workloads}
    assert excess["syn-val-steady-high"] == pytest.approx(0.025)
    assert excess["syn-val-ramp-down"] == pytest.approx(-0.05)
    assert fallback.max_positive_sla_excess == pytest.approx(0.3 - THRESHOLDS["syn-val-bursty"])
    assert fallback.mean_sla_excess == pytest.approx(0.3 - sum(THRESHOLDS.values()) / 3)
    assert {w.workload_id: w.threshold_sla for w in fallback.workloads} == THRESHOLDS


def test_result_invariants_forbid_selecting_infeasible() -> None:
    result = select(make_spec(), "dqn", [candidate("bad", sla=0.9)])
    data = result.model_dump()
    data.update(selected_candidate_id="bad", selected_objective_values={"normalized_cost": 0.5})
    with pytest.raises(ValidationError):
        type(result).model_validate(data)


# --- F. workload handling ------------------------------------------------------------


@pytest.mark.parametrize(
    ("workloads", "message"),
    [
        (["syn-val-steady-high", "syn-val-steady-high", "syn-val-bursty"], "unique"),
        (["syn-val-steady-high", "syn-val-ramp-down", "syn-test-spike-hard"], "held-out"),
        (["syn-val-steady-high", "syn-val-ramp-down", "azure-test-993600"], "held-out"),
        (["syn-val-steady-high", "syn-val-ramp-down", "syn-train-bursty"], "not a validation"),
        (["syn-val-steady-high", "syn-val-ramp-down", "no-such-workload"], "unknown"),
    ],
)
def test_spec_rejects_bad_workloads(workloads: list[str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_spec(validation_workload_ids=workloads)


def test_spec_requires_a_threshold_for_every_workload() -> None:
    thresholds = [
        {"workload_id": w, "sla_violation_rate": s, "evaluation_run_id": "r"}
        for w, s in THRESHOLDS.items()
    ]
    with pytest.raises(ValidationError, match="cover exactly"):
        make_spec(sla_thresholds=thresholds[:2])
    with pytest.raises(ValidationError, match="unique"):
        make_spec(sla_thresholds=[*thresholds, thresholds[0]])


def test_candidate_must_cover_exactly_the_declared_workloads() -> None:
    spec = make_spec()
    full = candidate("x")
    missing = full.model_copy(
        update={"metrics_by_workload": dict(list(full.metrics_by_workload.items())[:2])}
    )
    with pytest.raises(ValueError, match="missing"):
        select(spec, "dqn", [missing])
    extra = full.model_copy(
        update={
            "metrics_by_workload": {**full.metrics_by_workload, "syn-train-bursty": metrics(0.0)}
        }
    )
    with pytest.raises(ValueError, match="unexpected"):
        select(spec, "dqn", [extra])


def test_equal_weight_mean_over_workloads() -> None:
    spec = make_spec()
    c = Candidate(
        candidate_id="c",
        family="dqn",
        metrics_by_workload={
            "syn-val-steady-high": metrics(0.0, cost=0.3),
            "syn-val-ramp-down": metrics(0.0, cost=0.6),
            "syn-val-bursty": metrics(0.0, cost=0.9),
        },
    )
    assert select(spec, "dqn", [c]).candidates[0].mean_metrics["normalized_cost"] == pytest.approx(
        0.6
    )


# --- G. metric validation ------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.01, 1.5])
def test_metrics_reject_nan_inf_negative_and_out_of_range(bad: float) -> None:
    with pytest.raises(ValidationError):
        WorkloadMetrics(sla_violation_rate=0.1, normalized_cost=bad, queue_pressure=0, churn_rate=0)
    with pytest.raises(ValidationError):
        make_spec(
            sla_thresholds=[
                {
                    "workload_id": w,
                    "sla_violation_rate": bad if i == 0 else s,
                    "evaluation_run_id": "r",
                }
                for i, (w, s) in enumerate(THRESHOLDS.items())
            ]
        )


def test_spec_rejects_version_scenario_reward_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="benchmark"):
        make_spec(benchmark_version="v0")
    with pytest.raises(ValidationError):
        make_spec(selection_condition={"scenario": "capacity-jitter"})
    with pytest.raises(ValidationError):
        make_spec(selection_condition={"dynamics_seed": 1})
    with pytest.raises(ValidationError):
        make_spec(reward_used_for_selection=True)
    with pytest.raises(ValidationError):
        make_spec(extra_objective="episode_reward")
    with pytest.raises(ValidationError):
        make_spec(selection_version="selection-v3")


def test_family_mismatch_and_duplicates_are_rejected() -> None:
    spec = make_spec()
    with pytest.raises(ValueError, match="family"):
        select(spec, "dqn", [candidate("p", family="ppo")])
    with pytest.raises(ValueError, match="unique"):
        select(spec, "dqn", [candidate("a"), candidate("a")])
    with pytest.raises(ValueError, match="no candidates"):
        select(spec, "dqn", [])


# --- H. robustness isolation -------------------------------------------------------------


def raw_row(variant: str, workload: str, sla: float, **overrides: Any) -> dict[str, Any]:
    controller = variant.split("-")[0]
    seed = int(variant.removeprefix(f"{controller}-seed")) if "seed" in variant else None
    row = {
        "plan_id": "plan123",
        "controller": controller,
        "controller_variant_id": variant,
        "training_seed": seed,
        "workload_id": workload,
        "workload_split": "validation",
        "robustness_scenario": "nominal",
        "robustness_version": "robustness-v1",
        "dynamics_seed": 0,
        "evaluation_seed": 0,
        "mlflow_run_id": f"{variant}-{workload}-{overrides.get('robustness_scenario', 'nominal')}",
        "sla_violation_rate": sla,
        "normalized_cost": 0.5,
        "queue_pressure": 0.1,
        "churn_rate": 0.1,
        "episode_reward": 0.0,
    }
    row.update(overrides)
    return row


def learned_variant(variant: str, seed: int) -> dict[str, Any]:
    return {
        "variant_id": variant,
        "controller": variant.split("-")[0],
        "version": "dqn-v1",
        "training_seed": seed,
        "training_run_id": f"train-{variant}",
        "model_artifact_uri": f"runs:/train-{variant}/model",
        "hyperparameter_source": "optuna:dqn-v1#trial16",
        "lineage": {
            "tuning": {
                "study_name": "dqn-v1",
                "objective_name": "dqn-sla-first",
                "objective_version": "v1",
                "selected_trial_number": 16,
            }
        },
    }


FIXTURE_MANIFEST: dict[str, Any] = {
    "variants": [
        {
            "variant_id": "threshold-v1",
            "controller": "threshold",
            "version": "threshold-sla-first-v1",
            "params": {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3},
            "lineage": {
                "study_name": "threshold-v1",
                "search_space_version": "threshold-grid-v1",
                "selected_trial_number": 14,
                "tuning_workload_ids": ["syn-train-bursty", *VALIDATION_WORKLOADS],
                "validation_workload_ids": list(VALIDATION_WORKLOADS),
                "selected_mlflow_run_ids": ["t1"],
            },
        },
        learned_variant("dqn-seed0", 0),
        learned_variant("dqn-seed1", 1),
    ]
}


def fixture_rows(robustness_sla: float = 0.0) -> list[dict[str, Any]]:
    rows = []
    for workload, threshold in THRESHOLDS.items():
        rows.append(raw_row("threshold-v1", workload, threshold))
        rows.append(raw_row("dqn-seed0", workload, threshold, normalized_cost=0.97))
        rows.append(raw_row("dqn-seed1", workload, threshold + 0.05, normalized_cost=0.6))
        for variant in ("threshold-v1", "dqn-seed0", "dqn-seed1"):
            # Robustness evidence that would flip the outcome if it leaked into selection.
            rows.append(
                raw_row(
                    variant,
                    workload,
                    robustness_sla,
                    robustness_scenario="capacity-jitter",
                    dynamics_seed=3,
                )
            )
            rows.append(raw_row(variant, workload, robustness_sla, evaluation_seed=1))
    return rows


def write_raw(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_only_nominal_rows_enter_selection(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "raw.jsonl", fixture_rows(robustness_sla=0.0))
    rows = nominal_validation_rows(raw)
    assert len(rows) == 9
    assert {(r["robustness_scenario"], r["dynamics_seed"], r["evaluation_seed"]) for r in rows} == {
        ("nominal", 0, 0)
    }
    spec = freeze_spec_from_multiseed(FIXTURE_MANIFEST, rows)
    result = select(spec, "dqn", multiseed_candidates(FIXTURE_MANIFEST, rows, "dqn", spec))
    # dqn-seed1 is perfect under robustness, infeasible nominally: nominal decides.
    assert result.feasible_candidate_ids == ("dqn-seed0",)
    assert result.selected_candidate_id == "dqn-seed0"


def test_raw_results_with_test_rows_are_refused(tmp_path: Path) -> None:
    rows = fixture_rows()
    rows.append(raw_row("dqn-seed0", "syn-test-spike-hard", 0.0, workload_split="test"))
    with pytest.raises(ValueError, match="held-out"):
        nominal_validation_rows(write_raw(tmp_path / "raw.jsonl", rows))


def test_mixed_plans_are_refused(tmp_path: Path) -> None:
    rows = fixture_rows()
    rows[0]["plan_id"] = "other"
    with pytest.raises(ValueError, match="mix"):
        nominal_validation_rows(write_raw(tmp_path / "raw.jsonl", rows))


# --- I. provenance round-trip -----------------------------------------------------------


def test_spec_and_result_round_trip(tmp_path: Path) -> None:
    rows = nominal_validation_rows(write_raw(tmp_path / "raw.jsonl", fixture_rows()))
    spec = freeze_spec_from_multiseed(FIXTURE_MANIFEST, rows)
    loaded = SelectionSpec.load(spec.save(tmp_path / "spec.json"))
    assert loaded == spec and loaded.spec_id == spec.spec_id
    assert (
        loaded.threshold("syn-val-bursty").evaluation_run_id
        == "threshold-v1-syn-val-bursty-nominal"
    )
    assert loaded.provenance["evaluation_plan_id"] == "plan123"
    assert loaded.reference.tuning_lineage["selected_trial_number"] == 14
    result = select(spec, "dqn", multiseed_candidates(FIXTURE_MANIFEST, rows, "dqn", spec))
    path = result.save(tmp_path / "dqn.json")
    assert type(result).model_validate_json(path.read_text()) == result
    provenance = result.candidates[0].provenance
    assert provenance["training_run_id"] == "train-dqn-seed0"
    assert provenance["tuning_selected_trial"] == 16
    assert provenance["tuning_objective"] == "dqn-sla-first-v1"


def test_committed_spec_is_frozen() -> None:
    spec = SelectionSpec.load(COMMITTED_SPEC)
    assert spec.spec_id == COMMITTED_SPEC_ID
    assert spec.validation_workload_ids == VALIDATION_WORKLOADS
    assert {t.workload_id: t.sla_violation_rate for t in spec.sla_thresholds} == THRESHOLDS
    assert spec.reference.params == {
        "high_threshold": 0.6,
        "low_threshold": 0.2,
        "cooldown_ticks": 3,
    }
    assert spec.reward_used_for_selection is False
    assert spec.provenance["held_out_data_used"] is False


def test_spec_must_match_the_evidence(tmp_path: Path) -> None:
    rows = nominal_validation_rows(write_raw(tmp_path / "raw.jsonl", fixture_rows()))
    check_spec_matches_evidence(freeze_spec_from_multiseed(FIXTURE_MANIFEST, rows), rows)
    with pytest.raises(ValueError, match="does not match"):
        check_spec_matches_evidence(make_spec(), rows)


# --- J. determinism ------------------------------------------------------------------


def test_input_order_does_not_change_the_result() -> None:
    spec = make_spec()
    pool = [
        candidate("a", cost=0.6, index=3),
        candidate("b", cost=0.6, index=1),
        candidate("c", sla=0.9, cost=0.1, index=0),
        candidate("d", cost=0.7, index=2),
    ]
    first = select(spec, "dqn", pool)
    assert select(spec, "dqn", list(reversed(pool))) == first
    assert first.selected_candidate_id == "b"
    assert first.candidate_ids == ("c", "b", "d", "a")


def test_spec_hash_is_stable_and_content_sensitive() -> None:
    assert make_spec().spec_id == make_spec().spec_id
    assert make_spec().spec_id != make_spec(provenance={"evidence": "other"}).spec_id


# --- K. reward exclusion ------------------------------------------------------------


def test_reward_is_not_an_input() -> None:
    assert "episode_reward" not in WorkloadMetrics.model_fields
    with pytest.raises(ValidationError):
        WorkloadMetrics(
            sla_violation_rate=0.1,
            normalized_cost=0.1,
            queue_pressure=0,
            churn_rate=0,
            episode_reward=-1.0,  # type: ignore[call-arg]
        )


def test_reward_in_raw_rows_does_not_affect_selection(tmp_path: Path) -> None:
    rows_a = fixture_rows()
    rows_b = [
        dict(row, episode_reward=1e6 if row["controller_variant_id"] == "dqn-seed1" else -1e6)
        for row in rows_a
    ]
    results = []
    for name, rows in (("a", rows_a), ("b", rows_b)):
        nominal = nominal_validation_rows(write_raw(tmp_path / f"{name}.jsonl", rows))
        spec = freeze_spec_from_multiseed(FIXTURE_MANIFEST, nominal)
        results.append(
            select(spec, "dqn", multiseed_candidates(FIXTURE_MANIFEST, nominal, "dqn", spec))
        )
    assert results[0] == results[1]


# --- L. #19 adapter and CLI ---------------------------------------------------------


def test_diagnose_cli_writes_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = write_raw(tmp_path / "raw.jsonl", fixture_rows())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(FIXTURE_MANIFEST))
    spec_path = tmp_path / "spec.json"
    assert (
        model_selection.main(
            [
                "freeze",
                "--manifest",
                str(manifest),
                "--raw-results",
                str(raw),
                "--output",
                str(spec_path),
            ]
        )
        == 0
    )
    out = tmp_path / "out"
    assert (
        model_selection.main(
            [
                "diagnose",
                "--spec",
                str(spec_path),
                "--manifest",
                str(manifest),
                "--raw-results",
                str(raw),
                "--output-dir",
                str(out),
                "--family",
                "dqn",
            ]
        )
        == 0
    )
    assert "dqn: 1/2 feasible; selected dqn-seed0" in capsys.readouterr().out
    result = json.loads((out / "dqn-selection.json").read_text())
    assert result["selected_candidate_id"] == "dqn-seed0"
    assert result["reward_used_for_selection"] is False
    with (out / "candidate-table.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert [(r["candidate_id"], r["feasible"], r["selected"]) for r in table] == [
        ("dqn-seed0", "True", "True"),
        ("dqn-seed1", "False", "False"),
    ]


def test_adapter_requires_one_nominal_row_per_workload(tmp_path: Path) -> None:
    rows = nominal_validation_rows(write_raw(tmp_path / "raw.jsonl", fixture_rows()))
    spec = freeze_spec_from_multiseed(FIXTURE_MANIFEST, rows)
    incomplete = [
        r
        for r in rows
        if not (r["controller_variant_id"] == "dqn-seed1" and r["workload_id"] == "syn-val-bursty")
    ]
    with pytest.raises(ValueError, match="expected one nominal row"):
        multiseed_candidates(FIXTURE_MANIFEST, incomplete, "dqn", spec)
    with pytest.raises(ValueError, match="no ppo variants"):
        multiseed_candidates(FIXTURE_MANIFEST, rows, "ppo", spec)


def test_cli_rejects_test_evidence(tmp_path: Path) -> None:
    rows = [*fixture_rows(), raw_row("dqn-seed0", "azure-test-993600", 0.0, workload_split="test")]
    raw = write_raw(tmp_path / "raw.jsonl", rows)
    with pytest.raises(SystemExit):
        model_selection.main(
            [
                "freeze",
                "--manifest",
                "m.json",
                "--raw-results",
                str(raw),
                "--output",
                str(tmp_path / "s.json"),
            ]
        )
