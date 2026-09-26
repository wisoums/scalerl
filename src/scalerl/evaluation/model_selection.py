"""Constrained cost-aware model selection, ``selection-v2-cost-under-sla`` (#78).

#19 showed that the v1 learned-policy selectors (``dqn-sla-first`` /
``ppo-sla-first`` v1: SLA first, cost second) reward a trivial policy that opens
almost the whole fleet. Those v1 selectors and their results stay unchanged
(``scalerl.tuning.sb3.select_trial``); this module adds a separate, versioned
rule for future selection:

    Among candidates that meet an acceptable service level, which uses the
    least infrastructure?

1. **Feasibility** (per workload, all must hold): on every declared validation
   workload, ``candidate SLA <= Threshold SLA + 1e-12``. The reference is the
   tuned Threshold ``threshold-v1`` (high 0.6, low 0.2, cooldown 3) measured
   in #19's nominal validation runs. The tolerance is only a floating-point
   equality epsilon, not an SLA allowance. One easy workload cannot hide a
   failure on another.
2. **Among feasible candidates**, minimize (equal-weight arithmetic mean over
   the declared workloads, one vote per workload) normalized cost, then queue
   pressure, then churn rate, then SLA violation rate, then the deterministic
   candidate key (``tie_break_index`` if given, e.g. an Optuna trial number,
   then ``candidate_id``). Reward is never used.
3. **No feasible candidate**: nothing is selected (``selected_candidate_id =
   None``, ``selection_succeeded = False``); the SLA constraint is never
   weakened. A *diagnostic-only* fallback (lowest mean SLA, then cost, queue,
   churn, key) is reported with its per-workload SLA excess.

Selection uses **nominal validation conditions only** (``robustness-v1``
``nominal``, dynamics seed 0, evaluation seed 0); robustness scenarios remain
evaluation evidence, never a hidden objective. Selection happens within one
candidate family (e.g. DQN seeds, or PPO seeds); families are never ranked
against each other. Held-out test workloads are rejected everywhere.

Command line::

    python -m scalerl.evaluation.model_selection diagnose \\
        --spec benchmarks/v1/selection-v2-cost-under-sla.json \\
        --manifest outputs/multiseed-v1/controller-manifest.json \\
        --raw-results outputs/multiseed-v1/raw-results.jsonl --output-dir outputs/selection-v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scalerl.benchmarks import load_benchmark_manifest

SELECTION_VERSION: Final = "selection-v2-cost-under-sla"
FEASIBILITY_TOLERANCE: Final = 1e-12
REFERENCE_VARIANT_ID: Final = "threshold-v1"
REFERENCE_PARAMS: Final = {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3}
VALIDATION_WORKLOADS: Final = ("syn-val-steady-high", "syn-val-ramp-down", "syn-val-bursty")
METRICS: Final = ("sla_violation_rate", "normalized_cost", "queue_pressure", "churn_rate")
DIAGNOSTIC_LABEL: Final = (
    "DIAGNOSTIC ONLY: no candidate met the SLA constraint; nothing was selected"
)

TieBreak = Literal["queue_pressure", "churn_rate", "sla_violation_rate", "candidate_key"]
FallbackKey = Literal[
    "sla_violation_rate", "normalized_cost", "queue_pressure", "churn_rate", "candidate_key"
]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def _require_development_workloads(workload_ids: Iterable[str]) -> None:
    manifest = load_benchmark_manifest()
    for workload_id in workload_ids:
        try:
            split = manifest.get(workload_id).split
        except KeyError as error:
            raise ValueError(str(error)) from None
        if split == "test":
            raise ValueError(
                f"model selection uses train/validation evidence only; {workload_id!r} is a "
                "held-out test workload"
            )


def _unique(name: str, values: Sequence[Any]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


# --- frozen specification ------------------------------------------------------------------


class WorkloadThreshold(_Strict):
    """The reference controller's measured SLA on one validation workload."""

    workload_id: str = Field(min_length=1)
    sla_violation_rate: float = Field(ge=0, le=1)
    evaluation_run_id: str = Field(min_length=1)


class ReferenceController(_Strict):
    """The tuned Threshold baseline that defines acceptable service."""

    controller: Literal["threshold"] = "threshold"
    variant_id: Literal["threshold-v1"] = REFERENCE_VARIANT_ID
    controller_version: str = Field(min_length=1)
    params: dict[str, float | int]
    tuning_lineage: dict[str, JsonValue]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.params != REFERENCE_PARAMS:
            raise ValueError(f"reference params must be the canonical {REFERENCE_PARAMS}")
        return self


class FeasibilityRule(_Strict):
    metric: Literal["sla_violation_rate"] = "sla_violation_rate"
    mode: Literal["per_workload_all"] = "per_workload_all"
    comparator: Literal["<="] = "<="
    tolerance: float = Field(default=FEASIBILITY_TOLERANCE, ge=0, le=1e-9)
    tolerance_meaning: Literal["floating-point equality epsilon; not an SLA allowance"] = (
        "floating-point equality epsilon; not an SLA allowance"
    )


class SelectionCondition(_Strict):
    """Selection evidence is nominal validation only (robustness stays evaluation-only)."""

    scenario: Literal["nominal"] = "nominal"
    robustness_version: Literal["robustness-v1"] = "robustness-v1"
    dynamics_seed: Literal[0] = 0
    evaluation_seed: Literal[0] = 0
    split: Literal["validation"] = "validation"


class SelectionSpec(_Strict):
    """The frozen ``selection-v2-cost-under-sla`` rule and the evidence behind it."""

    selection_version: Literal["selection-v2-cost-under-sla"] = SELECTION_VERSION
    benchmark_version: str
    reference: ReferenceController
    validation_workload_ids: tuple[str, ...] = Field(min_length=1)
    sla_thresholds: tuple[WorkloadThreshold, ...]
    feasibility: FeasibilityRule = Field(default_factory=FeasibilityRule)
    primary_objective: Literal["normalized_cost"] = "normalized_cost"
    aggregation: Literal["equal_weight_mean_over_declared_workloads"] = (
        "equal_weight_mean_over_declared_workloads"
    )
    tie_breaks: tuple[TieBreak, TieBreak, TieBreak, TieBreak] = (
        "queue_pressure",
        "churn_rate",
        "sla_violation_rate",
        "candidate_key",
    )
    candidate_key: Literal["tie_break_index, then candidate_id"] = (
        "tie_break_index, then candidate_id"
    )
    selection_scope: Literal["within one candidate family; families are never ranked"] = (
        "within one candidate family; families are never ranked"
    )
    selection_condition: SelectionCondition = Field(default_factory=SelectionCondition)
    no_feasible_candidate: Literal[
        "select nothing; report a diagnostic-only fallback; never relax the constraint"
    ] = "select nothing; report a diagnostic-only fallback; never relax the constraint"
    diagnostic_fallback_order: tuple[
        FallbackKey, FallbackKey, FallbackKey, FallbackKey, FallbackKey
    ] = (
        "sla_violation_rate",
        "normalized_cost",
        "queue_pressure",
        "churn_rate",
        "candidate_key",
    )
    reward_used_for_selection: Literal[False] = False
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.benchmark_version != load_benchmark_manifest().version:
            raise ValueError(f"spec benchmark {self.benchmark_version!r} is not the installed one")
        _unique("validation_workload_ids", self.validation_workload_ids)
        _require_development_workloads(self.validation_workload_ids)
        manifest = load_benchmark_manifest()
        for workload_id in self.validation_workload_ids:
            if manifest.get(workload_id).split != "validation":
                raise ValueError(f"{workload_id!r} is not a validation workload")
        threshold_ids = [threshold.workload_id for threshold in self.sla_thresholds]
        _unique("sla_thresholds workloads", threshold_ids)
        if set(threshold_ids) != set(self.validation_workload_ids):
            raise ValueError("sla_thresholds must cover exactly the declared validation workloads")
        if self.tie_breaks != (
            "queue_pressure",
            "churn_rate",
            "sla_violation_rate",
            "candidate_key",
        ):
            raise ValueError(f"{SELECTION_VERSION} fixes its tie-break order")
        if self.diagnostic_fallback_order != (
            "sla_violation_rate",
            "normalized_cost",
            "queue_pressure",
            "churn_rate",
            "candidate_key",
        ):
            raise ValueError(f"{SELECTION_VERSION} fixes its diagnostic fallback order")
        return self

    @property
    def spec_id(self) -> str:
        """Stable short hash of the specification's content."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def threshold(self, workload_id: str) -> WorkloadThreshold:
        return next(t for t in self.sla_thresholds if t.workload_id == workload_id)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output

    @classmethod
    def load(cls, path: str | Path) -> SelectionSpec:
        return cls.model_validate_json(Path(path).read_text())


# --- candidates and selection --------------------------------------------------------------------


class WorkloadMetrics(_Strict):
    """A candidate's nominal validation metrics on one workload (reward is not needed)."""

    sla_violation_rate: float = Field(ge=0, le=1)
    normalized_cost: float = Field(ge=0, le=1)
    queue_pressure: float = Field(ge=0, le=1)
    churn_rate: float = Field(ge=0, le=1)


class Candidate(_Strict):
    """One selectable model/configuration and its evidence (generic across algorithms)."""

    candidate_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    tie_break_index: int | None = Field(default=None, ge=0)
    metrics_by_workload: dict[str, WorkloadMetrics]
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    def sort_key(self) -> tuple[int, str]:
        return (-1 if self.tie_break_index is None else self.tie_break_index, self.candidate_id)


class WorkloadAssessment(_Strict):
    workload_id: str
    threshold_sla: float
    candidate_sla: float
    sla_excess: float  # candidate − threshold (negative = better than the reference)
    feasible: bool


class CandidateAssessment(_Strict):
    candidate_id: str
    feasible: bool
    mean_metrics: dict[str, float]
    workloads: tuple[WorkloadAssessment, ...]
    provenance: dict[str, JsonValue]


class DiagnosticFallback(_Strict):
    """The closest infeasible candidate; it did NOT pass selection."""

    label: Literal["DIAGNOSTIC ONLY: no candidate met the SLA constraint; nothing was selected"] = (
        DIAGNOSTIC_LABEL
    )
    candidate_id: str
    workloads: tuple[WorkloadAssessment, ...]
    max_positive_sla_excess: float
    mean_sla_excess: float
    mean_metrics: dict[str, float]


class SelectionResult(_Strict):
    """Outcome of applying the frozen spec to one candidate family."""

    selection_version: Literal["selection-v2-cost-under-sla"] = SELECTION_VERSION
    spec_id: str
    family: str
    candidate_ids: tuple[str, ...]
    feasible_candidate_ids: tuple[str, ...]
    infeasible_candidate_ids: tuple[str, ...]
    feasible_candidate_count: int
    selection_succeeded: bool
    selected_candidate_id: str | None
    selected_objective_values: dict[str, float] | None
    diagnostic_fallback_candidate_id: str | None
    diagnostic_fallback: DiagnosticFallback | None
    candidates: tuple[CandidateAssessment, ...]
    reward_used_for_selection: Literal[False] = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.selection_succeeded:
            if self.selected_candidate_id is None or self.diagnostic_fallback is not None:
                raise ValueError("a successful selection has a selected candidate and no fallback")
            if self.selected_candidate_id not in self.feasible_candidate_ids:
                raise ValueError("only a feasible candidate can be selected")
        elif self.selected_candidate_id is not None or self.selected_objective_values is not None:
            raise ValueError("a failed selection selects nothing")
        return self

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n")
        return output


def select(spec: SelectionSpec, family: str, candidates: Sequence[Candidate]) -> SelectionResult:
    """Apply ``selection-v2-cost-under-sla`` to one family; independent of input order."""
    if not candidates:
        raise ValueError("no candidates to select from")
    _unique("candidate_id", [candidate.candidate_id for candidate in candidates])
    workloads = spec.validation_workload_ids
    tolerance = spec.feasibility.tolerance
    assessments = []
    for candidate in sorted(candidates, key=Candidate.sort_key):
        if candidate.family != family:
            raise ValueError(
                f"{candidate.candidate_id} is in family {candidate.family!r}, not {family!r}"
            )
        present = set(candidate.metrics_by_workload)
        if present != set(workloads):
            missing, extra = set(workloads) - present, present - set(workloads)
            raise ValueError(
                f"{candidate.candidate_id}: evidence must cover exactly the declared workloads "
                f"(missing {sorted(missing)}, unexpected {sorted(extra)})"
            )
        rows = []
        for workload_id in workloads:
            threshold = spec.threshold(workload_id).sla_violation_rate
            sla = candidate.metrics_by_workload[workload_id].sla_violation_rate
            rows.append(
                WorkloadAssessment(
                    workload_id=workload_id,
                    threshold_sla=threshold,
                    candidate_sla=sla,
                    sla_excess=sla - threshold,
                    feasible=sla <= threshold + tolerance,
                )
            )
        means = {
            metric: math.fsum(getattr(candidate.metrics_by_workload[w], metric) for w in workloads)
            / len(workloads)
            for metric in METRICS
        }
        assessments.append(
            (
                candidate,
                CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    feasible=all(row.feasible for row in rows),
                    mean_metrics=means,
                    workloads=tuple(rows),
                    provenance=candidate.provenance,
                ),
            )
        )

    feasible = [(c, a) for c, a in assessments if a.feasible]
    infeasible = [(c, a) for c, a in assessments if not a.feasible]
    selected: CandidateAssessment | None = None
    fallback: DiagnosticFallback | None = None
    if feasible:
        _, selected = min(
            feasible,
            key=lambda pair: (
                pair[1].mean_metrics["normalized_cost"],
                pair[1].mean_metrics["queue_pressure"],
                pair[1].mean_metrics["churn_rate"],
                pair[1].mean_metrics["sla_violation_rate"],
                pair[0].sort_key(),
            ),
        )
    else:
        _, closest = min(
            assessments,
            key=lambda pair: (
                pair[1].mean_metrics["sla_violation_rate"],
                pair[1].mean_metrics["normalized_cost"],
                pair[1].mean_metrics["queue_pressure"],
                pair[1].mean_metrics["churn_rate"],
                pair[0].sort_key(),
            ),
        )
        excesses = [row.sla_excess for row in closest.workloads]
        fallback = DiagnosticFallback(
            candidate_id=closest.candidate_id,
            workloads=closest.workloads,
            max_positive_sla_excess=max(0.0, *excesses),
            mean_sla_excess=math.fsum(excesses) / len(excesses),
            mean_metrics=closest.mean_metrics,
        )
    return SelectionResult(
        spec_id=spec.spec_id,
        family=family,
        candidate_ids=tuple(a.candidate_id for _, a in assessments),
        feasible_candidate_ids=tuple(a.candidate_id for _, a in feasible),
        infeasible_candidate_ids=tuple(a.candidate_id for _, a in infeasible),
        feasible_candidate_count=len(feasible),
        selection_succeeded=selected is not None,
        selected_candidate_id=selected.candidate_id if selected else None,
        selected_objective_values=dict(selected.mean_metrics) if selected else None,
        diagnostic_fallback_candidate_id=fallback.candidate_id if fallback else None,
        diagnostic_fallback=fallback,
        candidates=tuple(a for _, a in assessments),
    )


# --- #19 evidence adapter ------------------------------------------------------------------------


def nominal_validation_rows(
    raw_path: str | Path, condition: SelectionCondition | None = None
) -> list[dict[str, Any]]:
    """#19 raw rows that may enter selection: nominal, dynamics/evaluation seed 0.

    A raw file containing any held-out test row is refused outright (it is not
    development evidence); robustness rows are filtered out, never averaged in.
    """
    condition = condition or SelectionCondition()
    rows = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line.strip()]
    _require_development_workloads({row["workload_id"] for row in rows})
    if any(row["workload_split"] == "test" for row in rows):
        raise ValueError("raw results contain held-out test rows; refusing to read them")
    plans = {row["plan_id"] for row in rows}
    if len(plans) != 1:
        raise ValueError(f"raw results mix evaluation plans {sorted(plans)}")
    return [
        row
        for row in rows
        if row["robustness_scenario"] == condition.scenario
        and row["robustness_version"] == condition.robustness_version
        and row["dynamics_seed"] == condition.dynamics_seed
        and row["evaluation_seed"] == condition.evaluation_seed
        and row["workload_split"] == condition.split
    ]


def _one_row(
    rows: Sequence[Mapping[str, Any]], variant_id: str, workload_id: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["controller_variant_id"] == variant_id and row["workload_id"] == workload_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one nominal row for {variant_id} on {workload_id}, found {len(matches)}"
        )
    return matches[0]


def freeze_spec_from_multiseed(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    workload_ids: Sequence[str] = VALIDATION_WORKLOADS,
) -> SelectionSpec:
    """Build the v2 spec from #19's manifest and nominal raw rows (exact values, run IDs)."""
    reference = next(v for v in manifest["variants"] if v["variant_id"] == REFERENCE_VARIANT_ID)
    thresholds = []
    for workload_id in workload_ids:
        row = _one_row(rows, REFERENCE_VARIANT_ID, workload_id)
        thresholds.append(
            WorkloadThreshold(
                workload_id=workload_id,
                sla_violation_rate=float(row["sla_violation_rate"]),
                evaluation_run_id=str(row["mlflow_run_id"]),
            )
        )
    lineage = reference["lineage"]
    plan_id = {row["plan_id"] for row in rows}.pop()
    return SelectionSpec(
        benchmark_version=load_benchmark_manifest().version,
        reference=ReferenceController(
            controller_version=reference["version"],
            params=reference["params"],
            tuning_lineage={
                "study_name": lineage["study_name"],
                "objective": reference["version"],
                "search_space_version": lineage["search_space_version"],
                "selected_trial_number": lineage["selected_trial_number"],
                "tuning_workload_ids": lineage["tuning_workload_ids"],
                "validation_workload_ids": lineage["validation_workload_ids"],
                "selected_tuning_mlflow_run_ids": lineage["selected_mlflow_run_ids"],
            },
        ),
        validation_workload_ids=tuple(workload_ids),
        sla_thresholds=tuple(thresholds),
        provenance={
            "evidence": "#19 multi-seed validation evaluation (multiseed-v1)",
            "evaluation_plan_id": plan_id,
            "evaluation_experiment": "scalerl-multiseed",
            "threshold_measurement": "threshold-v1, nominal, dynamics seed 0, evaluation seed 0",
            "frozen_before": ["#79", "#20", "#72", "#46"],
            "held_out_data_used": False,
        },
    )


def multiseed_candidates(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], family: str, spec: SelectionSpec
) -> list[Candidate]:
    """The #19 learned variants of ``family`` (e.g. DQN seeds 0-4) as generic candidates."""
    variants = [v for v in manifest["variants"] if v["controller"] == family]
    if not variants:
        raise ValueError(f"no {family} variants in the manifest")
    candidates = []
    for variant in variants:
        metrics, run_ids = {}, {}
        for workload_id in spec.validation_workload_ids:
            row = _one_row(rows, variant["variant_id"], workload_id)
            metrics[workload_id] = WorkloadMetrics(
                **{metric: float(row[metric]) for metric in METRICS}
            )
            run_ids[workload_id] = row["mlflow_run_id"]
        candidates.append(
            Candidate(
                candidate_id=variant["variant_id"],
                family=family,
                tie_break_index=variant["training_seed"],
                metrics_by_workload=metrics,
                provenance={
                    "training_seed": variant.get("training_seed"),
                    "training_run_id": variant.get("training_run_id"),
                    "model_artifact_uri": variant.get("model_artifact_uri"),
                    "hyperparameter_source": variant.get("hyperparameter_source"),
                    "tuning_study": variant["lineage"]["tuning"]["study_name"],
                    "tuning_objective": variant["lineage"]["tuning"]["objective_name"]
                    + "-"
                    + variant["lineage"]["tuning"]["objective_version"],
                    "tuning_selected_trial": variant["lineage"]["tuning"]["selected_trial_number"],
                    "evaluation_run_ids": run_ids,
                    "level": "existing trained seed artifact (not a hyperparameter re-selection)",
                },
            )
        )
    return candidates


def check_spec_matches_evidence(spec: SelectionSpec, rows: Sequence[Mapping[str, Any]]) -> None:
    """The spec's thresholds must be exactly the Threshold rows in this evidence."""
    for threshold in spec.sla_thresholds:
        row = _one_row(rows, REFERENCE_VARIANT_ID, threshold.workload_id)
        if (
            row["mlflow_run_id"] != threshold.evaluation_run_id
            or float(row["sla_violation_rate"]) != threshold.sla_violation_rate
        ):
            raise ValueError(
                f"spec threshold for {threshold.workload_id} does not match the evidence"
            )


def candidate_table(results: Sequence[SelectionResult]) -> list[dict[str, Any]]:
    table = []
    for result in results:
        for assessment in result.candidates:
            table.append(
                {
                    "family": result.family,
                    "candidate_id": assessment.candidate_id,
                    "feasible": assessment.feasible,
                    "selected": assessment.candidate_id == result.selected_candidate_id,
                    "diagnostic_fallback": assessment.candidate_id
                    == result.diagnostic_fallback_candidate_id,
                    **{f"mean_{m}": assessment.mean_metrics[m] for m in METRICS},
                    **{
                        f"sla_excess_{row.workload_id}": row.sla_excess
                        for row in assessment.workloads
                    },
                }
            )
    return table


# --- command line ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{SELECTION_VERSION} model selection (#78).")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "diagnose"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest", type=Path, default=Path("outputs/multiseed-v1/controller-manifest.json")
        )
        command.add_argument(
            "--raw-results", type=Path, default=Path("outputs/multiseed-v1/raw-results.jsonl")
        )
        if name == "freeze":
            command.add_argument(
                "--output",
                type=Path,
                default=Path("benchmarks/v1/selection-v2-cost-under-sla.json"),
            )
        else:
            command.add_argument(
                "--spec", type=Path, default=Path("benchmarks/v1/selection-v2-cost-under-sla.json")
            )
            command.add_argument("--family", action="append", dest="families")
            command.add_argument("--output-dir", type=Path, default=Path("outputs/selection-v2"))
    args = parser.parse_args(argv)

    try:
        rows = nominal_validation_rows(args.raw_results)
    except ValueError as error:
        parser.error(str(error))
    manifest = json.loads(args.manifest.read_text())
    if args.command == "freeze":
        spec = freeze_spec_from_multiseed(manifest, rows)
        spec.save(args.output)
        print(f"{SELECTION_VERSION} spec {spec.spec_id} written to {args.output}")
        return 0

    spec = SelectionSpec.load(args.spec)
    check_spec_matches_evidence(spec, rows)
    results = []
    for family in args.families or ["dqn", "ppo"]:
        result = select(spec, family, multiseed_candidates(manifest, rows, family, spec))
        result.save(args.output_dir / f"{family}-selection.json")
        results.append(result)
        outcome = (
            result.selected_candidate_id
            or f"none (diagnostic: {result.diagnostic_fallback_candidate_id})"
        )
        print(
            f"{family}: {result.feasible_candidate_count}/{len(result.candidate_ids)} feasible; "
            f"selected {outcome}"
        )
    table = candidate_table(results)
    with (args.output_dir / "candidate-table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"spec {spec.spec_id}; outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
