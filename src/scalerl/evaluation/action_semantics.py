"""Action-contract experiment (#79): ``delta-v1`` vs ``desired-replicas-v1``, validation only.

The scientific variable is the **action contract**; everything else is held
fixed by the frozen experiment spec
(``benchmarks/v1/action-semantics-experiment-v1.json``), written before any
model of this experiment was trained:

* train on ``syn-train-bursty``; validate on ``syn-val-steady-high``,
  ``syn-val-ramp-down``, ``syn-val-bursty``; **no test workload**;
* the canonical simulator config, except ``action.semantics``;
* the unchanged reward (its churn term counts scaling *events*);
* the same 20 DQN and 20 PPO hyperparameter candidates for both contracts
  (``matched-action-candidates-v1``, generated before training, see
  :mod:`scalerl.tuning.candidates`), each trained with seed 0 for the same
  budget (DQN 200,000 / PPO 204,800 timesteps);
* configuration selection with the frozen #78 rule
  ``selection-v2-cost-under-sla`` (spec ``418876d6c8e9``) within each of the four
  families ``{dqn, ppo} x {delta-v1, desired-replicas-v1}``; no global winner,
  and nothing is selected when nothing is feasible;
* every successfully selected configuration retrained with seeds 0-4 (all kept);
* evaluation under nominal conditions only (dynamics seed 0, evaluation seed 0)
  of Threshold and Predictive under both contracts and every retrained model.

Phases (each resumable: completed results are files, and evaluation rows are
recovered from finished MLflow runs instead of being run twice)::

    python -m scalerl.evaluation.action_semantics freeze-spec   # before any training
    python -m scalerl.evaluation.action_semantics check
    python -m scalerl.evaluation.action_semantics screen --family dqn-delta-v1   # x4 families
    python -m scalerl.evaluation.action_semantics select
    python -m scalerl.evaluation.action_semantics retrain --family dqn-desired-replicas-v1
    python -m scalerl.evaluation.action_semantics evaluate
    python -m scalerl.evaluation.action_semantics decide [--freeze]

The decision is predeclared (``decision_principle`` in the spec): the more
faithful horizontal-autoscaling actuator, ``desired-replicas-v1``, is adopted
if its action/lifecycle invariants pass, DQN and PPO train and infer against
it, and the experiment completes without a systematic failure attributable to
the encoding. Validation metrics characterize the consequence of the change;
they do not pick the contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from scalerl.benchmarks import WorkloadEntry, build_workloads, load_benchmark_manifest
from scalerl.controllers import Controller, PredictiveController, ThresholdController
from scalerl.controllers.predictive import CAPACITY_POLICY, FORECAST_METHOD
from scalerl.environment import (
    DELTA_V1,
    DESIRED_REPLICAS_V1,
    ActionConfig,
    ActionContract,
    ActionSemantics,
    AutoscalingEnv,
    ReplicaConfig,
    SimulatorConfig,
)
from scalerl.environment.replicas import ReplicaPool
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import model_selection
from scalerl.evaluation.metrics import (
    ActionMagnitudeMetrics,
    EpisodeMetrics,
    summarize_action_magnitude,
)
from scalerl.evaluation.model_selection import (
    Candidate,
    SelectionResult,
    SelectionSpec,
    WorkloadMetrics,
)
from scalerl.evaluation.multiseed import _append_row, _repair_torn_tail, describe, read_rows
from scalerl.evaluation.robustness import NOMINAL, evaluate_robustness, robustness_run_spec
from scalerl.mlops import RunSpec, SimulatorConfigSource
from scalerl.mlops.tracking import TrackedRun
from scalerl.tuning.candidates import CANDIDATE_SET_VERSION, CandidateSet
from scalerl.workloads import WorkloadTrace

EXPERIMENT_VERSION: Final = "action-semantics-experiment-v1"
DECISION_VERSION: Final = "action-contract-v2"
EXPERIMENT_NAME: Final = "scalerl-action-semantics"
TRAIN_WORKLOAD: Final = "syn-train-bursty"
VALIDATION_WORKLOADS: Final = model_selection.VALIDATION_WORKLOADS
SELECTION_SPEC_ID: Final = "418876d6c8e9"
ALGORITHMS: Final = ("dqn", "ppo")
CONTRACTS: Final[tuple[ActionSemantics, ...]] = (DELTA_V1, DESIRED_REPLICAS_V1)
TIMESTEPS: Final = {"dqn": 200_000, "ppo": 204_800}
SCREENING_SEED: Final = 0
RETRAINING_SEEDS: Final = (0, 1, 2, 3, 4)
EVALUATION_SEED: Final = 0
DYNAMICS_SEED: Final = 0
THRESHOLD_PARAMS: Final = {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3}
PREDICTIVE_PARAMS: Final = {"history_window_ticks": 4, "target_utilization": 0.8}
METRIC_KEYS: Final = tuple(EpisodeMetrics.__dataclass_fields__)
ACTION_KEYS: Final = tuple(ActionMagnitudeMetrics.__dataclass_fields__)
PREFERRED_CONTRACT: Final = DESIRED_REPLICAS_V1

DEFAULT_SPEC = Path("benchmarks/v1/action-semantics-experiment-v1.json")
DEFAULT_CANDIDATES = Path("benchmarks/v1/action-semantics-candidates-v1.json")
DEFAULT_SELECTION_SPEC = Path("benchmarks/v1/selection-v2-cost-under-sla.json")
DEFAULT_DECISION = Path("benchmarks/v1/action-contract-v2.json")
DEFAULT_OUTPUT = Path("outputs/action-semantics-v1")

Phase = Literal["screening", "retraining"]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def family_id(algorithm: str, semantics: str) -> str:
    return f"{algorithm}-{semantics}"


FAMILIES: Final = tuple(family_id(a, c) for a in ALGORITHMS for c in CONTRACTS)


def parse_family(family: str) -> tuple[str, ActionSemantics]:
    for algorithm in ALGORITHMS:
        for semantics in CONTRACTS:
            if family == family_id(algorithm, semantics):
                return algorithm, semantics
    raise ValueError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")


def contract_config(semantics: ActionSemantics) -> tuple[SimulatorConfig, SimulatorConfigSource]:
    """The canonical simulator config under ``semantics`` (nothing else differs)."""
    if semantics == DELTA_V1:
        return SimulatorConfig(), "default"
    return SimulatorConfig(action=ActionConfig(semantics=semantics)), "predeclared"


# --- frozen experiment specification ------------------------------------------------------------


class ContractDescription(_Strict):
    semantics: ActionSemantics
    action_space: str
    code_mapping: str


class DecisionPrinciple(_Strict):
    preferred_contract: Literal["desired-replicas-v1"] = PREFERRED_CONTRACT
    rationale: str
    adopt_if: tuple[str, ...]
    if_criteria_fail: str
    not_decided_by: tuple[str, ...]


class ExperimentSpec(_Strict):
    """Everything the #79 experiment holds fixed; ``experiment_id`` hashes it."""

    experiment_version: Literal["action-semantics-experiment-v1"] = EXPERIMENT_VERSION
    benchmark_version: str
    action_contracts: tuple[ContractDescription, ...]
    base_simulator_config: dict[str, JsonValue]
    simulator_difference: Literal["action.semantics only"] = "action.semantics only"
    train_workload_id: Literal["syn-train-bursty"] = TRAIN_WORKLOAD
    validation_workload_ids: tuple[str, ...]
    test_workload_ids: tuple[str, ...] = ()
    candidate_set_version: Literal["matched-action-candidates-v1"] = CANDIDATE_SET_VERSION
    candidate_set_id: str
    candidate_generation_method: str
    candidate_generation_seed: int
    candidates_per_algorithm: int
    search_space_versions: dict[str, str]
    algorithm_config_versions: dict[str, str]
    timesteps: dict[str, int]
    screening_training_seed: Literal[0] = SCREENING_SEED
    retraining_seeds: tuple[int, ...] = RETRAINING_SEEDS
    evaluation_seed: Literal[0] = EVALUATION_SEED
    dynamics_condition: dict[str, JsonValue]
    reward_weights: dict[str, float]
    reward_definition: str
    selection_version: Literal["selection-v2-cost-under-sla"] = model_selection.SELECTION_VERSION
    selection_spec_id: Literal["418876d6c8e9"] = SELECTION_SPEC_ID
    selection_level: str
    families: tuple[str, ...] = FAMILIES
    rule_controllers: dict[str, JsonValue]
    primary_metrics: tuple[str, ...] = METRIC_KEYS
    action_magnitude_diagnostics: tuple[str, ...] = ACTION_KEYS
    decision_principle: DecisionPrinciple
    held_out_rule: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        manifest = load_benchmark_manifest()
        if self.benchmark_version != manifest.version:
            raise ValueError(f"spec benchmark {self.benchmark_version!r} is not installed")
        if self.test_workload_ids:
            raise ValueError("the #79 experiment uses no test workloads")
        for workload_id in (self.train_workload_id, *self.validation_workload_ids):
            try:
                split = manifest.get(workload_id).split
            except KeyError as error:
                raise ValueError(str(error)) from None
            if split == "test":
                raise ValueError(f"{workload_id!r} is a held-out test workload")
        if manifest.get(self.train_workload_id).split != "train":
            raise ValueError("the training workload must be a train workload")
        if tuple(self.validation_workload_ids) != VALIDATION_WORKLOADS:
            raise ValueError(f"validation workloads must be exactly {VALIDATION_WORKLOADS}")
        if self.timesteps != TIMESTEPS:
            raise ValueError(f"budgets must be {TIMESTEPS}")
        if tuple(self.retraining_seeds) != RETRAINING_SEEDS:
            raise ValueError(f"retraining seeds must be {RETRAINING_SEEDS}")
        if self.families != FAMILIES:
            raise ValueError(f"families must be {FAMILIES}")
        if self.base_simulator_config != SimulatorConfig().model_dump(mode="json"):
            raise ValueError("the base simulator config must be the canonical default")
        if self.dynamics_condition != _dynamics_condition():
            raise ValueError("selection/evaluation uses nominal conditions only")
        return self

    @property
    def experiment_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        return cls.model_validate_json(Path(path).read_text())


def _dynamics_condition() -> dict[str, JsonValue]:
    return {
        "scenario": NOMINAL.name,
        "robustness_version": NOMINAL.version,
        "dynamics_seed": DYNAMICS_SEED,
        "robustness_scenarios_used_for_selection_or_decision": False,
    }


def build_experiment_spec(candidates: CandidateSet) -> ExperimentSpec:
    """The #79 experiment specification for the frozen candidate set."""
    return ExperimentSpec(
        benchmark_version=load_benchmark_manifest().version,
        action_contracts=(
            ContractDescription(
                semantics=DELTA_V1,
                action_space="Discrete(3)",
                code_mapping=(
                    "code 0 = scale down (effect -1), 1 = hold (0), 2 = scale up (+1); "
                    "effects are not codes: step(-1) is invalid"
                ),
            ),
            ContractDescription(
                semantics=DESIRED_REPLICAS_V1,
                action_space="Discrete(max_replicas - min_replicas + 1); Discrete(10) at 1..10",
                code_mapping=(
                    "code c -> target committed replicas min_replicas + c (0..9 -> 1..10); "
                    "|target - (active + pending)| replicas started or cancelled in one step "
                    "through the startup lifecycle"
                ),
            ),
        ),
        base_simulator_config=SimulatorConfig().model_dump(mode="json"),
        validation_workload_ids=VALIDATION_WORKLOADS,
        candidate_set_id=candidates.candidate_set_id,
        candidate_generation_method=candidates.generation_method,
        candidate_generation_seed=candidates.generation_seed,
        candidates_per_algorithm=len(candidates.dqn.candidates),
        search_space_versions={
            "dqn": candidates.dqn.search_space_version,
            "ppo": candidates.ppo.search_space_version,
        },
        algorithm_config_versions={
            "dqn": candidates.dqn.config_version,
            "ppo": candidates.ppo.config_version,
        },
        timesteps=dict(TIMESTEPS),
        dynamics_condition=_dynamics_condition(),
        reward_weights=RewardWeights().model_dump(mode="json"),
        reward_definition=(
            "scalerl.environment.reward.compute_reward, unchanged by #79: the churn term is "
            "weights.churn * (applied_replica_change != 0), i.e. per scaling event, not per "
            "replica; magnitude is reported separately (action.* diagnostics); #20 owns "
            "any reward change"
        ),
        selection_level=(
            "hyperparameter-configuration candidates, within each algorithm x contract family; "
            "per-workload evidence of the seed-0 screening run of every candidate"
        ),
        rule_controllers={
            "threshold": {
                "variant": "threshold-v1",
                "controller_version": "threshold-sla-first-v1",
                "params": dict(THRESHOLD_PARAMS),
                "desired_replicas_v1_encoding": "same ±1/hold law, encoded as target committed±1",
            },
            "predictive": {
                "variant": "predictive-v1",
                "forecast_method": FORECAST_METHOD,
                "capacity_policy": CAPACITY_POLICY,
                "params": dict(PREDICTIVE_PARAMS),
                "desired_replicas_v1_encoding": "its computed desired replicas as the target",
            },
        },
        decision_principle=DecisionPrinciple(
            rationale=(
                "Horizontal autoscalers set an integer desired replica count; forcing every "
                "policy through ±1 steps makes multi-replica corrections artificially "
                "sequential. desired-replicas-v1 is the more faithful actuator model."
            ),
            adopt_if=(
                "all action/lifecycle invariants pass",
                "DQN and PPO train and infer against desired-replicas-v1 "
                "(every screening candidate trains, loads through the strict compatibility "
                "check, and validates with finite metrics)",
                "the validation experiment completes without a systematic implementation or "
                "numerical failure attributable to the action encoding",
            ),
            if_criteria_fail=(
                "do not silently revert to delta-v1: report the failure and leave the final "
                "action contract unresolved for review"
            ),
            not_decided_by=(
                "which contract has better validation metrics",
                "any held-out test outcome",
                "retraining or re-tuning until one contract wins",
            ),
        ),
        held_out_rule=(
            "no syn-test-* or azure-test-* workload, #46 output, or held-out Azure result is "
            "read; the CLI accepts no workload arguments and the spec rejects test workloads"
        ),
    )


def load_frozen_inputs(
    spec_path: Path, candidates_path: Path, selection_path: Path
) -> tuple[ExperimentSpec, CandidateSet, SelectionSpec]:
    """Load the committed spec, candidates and #78 rule; refuse any drift from the frozen plan."""
    spec = ExperimentSpec.load(spec_path)
    candidates = CandidateSet.load(candidates_path)
    selection = SelectionSpec.load(selection_path)
    if spec.candidate_set_id != candidates.candidate_set_id:
        raise ValueError("candidate set does not match the experiment spec")
    if selection.spec_id != spec.selection_spec_id:
        raise ValueError(
            f"selection spec {selection.spec_id} is not the frozen {spec.selection_spec_id}"
        )
    if spec != build_experiment_spec(candidates):
        raise ValueError("the committed experiment spec differs from this code's frozen plan")
    return spec, candidates, selection


# --- evidence records ---------------------------------------------------------------------------


class WorkloadEvidence(_Strict):
    workload_id: str
    validation_run_id: str
    metrics: dict[str, float]
    action: dict[str, float]


class TrainingEvidence(_Strict):
    """One trained model of the experiment and its per-workload validation evidence."""

    experiment_id: str
    candidate_set_id: str
    phase: Phase
    family: str
    algorithm: str
    action_semantics: ActionSemantics
    candidate_id: str
    candidate_index: int
    training_seed: int
    hyperparameters: dict[str, JsonValue]
    training_run_id: str
    model_artifact_uri: str
    compatibility: dict[str, JsonValue]
    training_episodes: int
    workloads: tuple[WorkloadEvidence, ...]
    selection_version: str | None = None
    selection_spec_id: str | None = None

    def metrics_for(self, workload_id: str) -> WorkloadEvidence:
        return next(w for w in self.workloads if w.workload_id == workload_id)

    def save(self, path: Path) -> Path:
        return _write_text(path, self.model_dump_json(indent=2) + "\n")


def study_filename(family: str) -> str:
    """``dqn-delta-study.json``, ``ppo-desired-study.json``, ..."""
    algorithm, semantics = parse_family(family)
    return f"{algorithm}-{'delta' if semantics == DELTA_V1 else 'desired'}-study.json"


def screening_path(out: Path, family: str, candidate_id: str) -> Path:
    return out / "screening" / family / f"{candidate_id}.json"


def retraining_path(out: Path, family: str, seed: int) -> Path:
    return out / "retraining" / family / f"seed{seed}.json"


def load_evidence(paths: Sequence[Path]) -> list[TrainingEvidence]:
    return [TrainingEvidence.model_validate_json(path.read_text()) for path in paths]


# --- training (screening and retraining) -------------------------------------------------------


TrainFunction = Callable[..., Any]


def _algorithm_spec(algorithm: str) -> Any:
    if algorithm == "dqn":
        from scalerl.training.dqn import DQN_ALGORITHM

        return DQN_ALGORITHM
    from scalerl.training.ppo import PPO_ALGORITHM

    return PPO_ALGORITHM


def _entries() -> tuple[WorkloadEntry, tuple[WorkloadEntry, ...]]:
    from scalerl.training.common import require_training_workload, require_validation_workloads

    training = require_training_workload(TRAIN_WORKLOAD, "#79")
    validation = require_validation_workloads(VALIDATION_WORKLOADS, "#79")
    return training, validation


def train_model(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    *,
    family: str,
    candidate_id: str,
    seed: int,
    phase: Phase,
    traces: Mapping[str, WorkloadTrace],
    track: Callable[[RunSpec], AbstractContextManager[TrackedRun]],
    timesteps: int | None = None,
) -> TrainingEvidence:
    """Train one candidate configuration under one contract and validate it on all workloads."""
    from scalerl.training.common import TrainingSettings, train_and_validate

    algorithm, semantics = parse_family(family)
    candidate = candidates.for_algorithm(algorithm).get(candidate_id)
    hyperparameters = candidates.hyperparameters(algorithm, candidate_id)
    config, source = contract_config(semantics)
    training_entry, validation_entries = _entries()
    settings = TrainingSettings(
        timesteps=timesteps or spec.timesteps[algorithm],
        seed=seed,
        config=config,
        config_source=source,
        calibration_workload_ids=(),
        calibration_note=None,
        reward_weights=RewardWeights(),
    )
    tags = {
        "scalerl.experiment_version": spec.experiment_version,
        "scalerl.experiment_id": spec.experiment_id,
        "scalerl.candidate_set_id": spec.candidate_set_id,
        "scalerl.candidate_id": candidate_id,
        "scalerl.action_family": family,
        "scalerl.experiment_phase": phase,
        "scalerl.training_seed": str(seed),
    }
    if phase == "retraining":
        tags |= {
            "scalerl.selection_version": spec.selection_version,
            "scalerl.selection_spec_id": spec.selection_spec_id,
            "scalerl.selected_candidate_id": candidate_id,
        }
    kinds: tuple[Any, Any] = ("tune", "tune") if phase == "screening" else ("train", "evaluate")
    outcome = train_and_validate(
        _algorithm_spec(algorithm),
        training_entry=training_entry,
        validation_entries=validation_entries,
        traces=traces,
        hyperparameters=hyperparameters,
        settings=settings,
        track=track,
        training_run_kind=kinds[0],
        validation_run_kind=kinds[1],
        extra_params={
            "hyperparameter_source": (
                f"{spec.candidate_set_version}:{spec.candidate_set_id}#{candidate_id}"
            )
        },
        tags=tags,
    )
    return TrainingEvidence(
        experiment_id=spec.experiment_id,
        candidate_set_id=spec.candidate_set_id,
        phase=phase,
        family=family,
        algorithm=algorithm,
        action_semantics=semantics,
        candidate_id=candidate_id,
        candidate_index=candidate.index,
        training_seed=seed,
        hyperparameters=dict(candidate.hyperparameters),
        training_run_id=outcome.training_run_id,
        model_artifact_uri=f"runs:/{outcome.training_run_id}/model",
        compatibility=outcome.compatibility.model_dump(mode="json"),
        training_episodes=outcome.training_episodes,
        workloads=tuple(
            WorkloadEvidence(
                workload_id=entry.id,
                validation_run_id=run_id,
                metrics={
                    k: float(v)
                    for k, v in outcome.validation_metrics[entry.id].as_metrics().items()
                },
                action={
                    k: float(v)
                    for k, v in vars(outcome.validation_action_metrics[entry.id]).items()
                },
            )
            for entry, run_id in zip(validation_entries, outcome.validation_run_ids, strict=True)
        ),
        selection_version=spec.selection_version if phase == "retraining" else None,
        selection_spec_id=spec.selection_spec_id if phase == "retraining" else None,
    )


def _tracker(
    tracking_uri: str | None, experiment_name: str
) -> Callable[[RunSpec], AbstractContextManager[TrackedRun]]:
    from scalerl.mlops import start_tracked_run

    def track(spec: RunSpec) -> AbstractContextManager[TrackedRun]:
        return start_tracked_run(spec, tracking_uri=tracking_uri, experiment_name=experiment_name)

    return track


def _check_evidence(evidence: TrainingEvidence, spec: ExperimentSpec) -> None:
    if evidence.experiment_id != spec.experiment_id:
        raise ValueError(f"{evidence.family}/{evidence.candidate_id}: from another experiment")


def run_screening(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    family: str,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    traces: Mapping[str, WorkloadTrace] | None = None,
    timesteps: int | None = None,
    progress: Callable[[str], None] = print,
) -> list[TrainingEvidence]:
    """Train and validate every frozen candidate of ``family`` in candidate order (resumable)."""
    algorithm, _ = parse_family(family)
    training_entry, validation_entries = _entries()
    traces = traces or build_workloads([training_entry, *validation_entries])
    track = _tracker(tracking_uri, experiment_name)
    results = []
    pool = candidates.for_algorithm(algorithm).candidates
    for candidate in pool:
        path = screening_path(out, family, candidate.candidate_id)
        if path.exists():
            evidence = TrainingEvidence.model_validate_json(path.read_text())
            _check_evidence(evidence, spec)
        else:
            evidence = train_model(
                spec,
                candidates,
                family=family,
                candidate_id=candidate.candidate_id,
                seed=spec.screening_training_seed,
                phase="screening",
                traces=traces,
                track=track,
                timesteps=timesteps,
            )
            evidence.save(path)
            progress(f"{family} {candidate.candidate_id} -> {evidence.training_run_id}")
        results.append(evidence)
    _write_json(
        out / study_filename(family),
        {
            "experiment_id": spec.experiment_id,
            "candidate_set_id": spec.candidate_set_id,
            "family": family,
            "candidates": [evidence.model_dump(mode="json") for evidence in results],
        },
    )
    return results


# --- #78 selection at the configuration level ---------------------------------------------------


def family_candidates(evidence: Sequence[TrainingEvidence], family: str) -> list[Candidate]:
    """Screening evidence as generic #78 candidates (key: candidate index)."""
    return [
        Candidate(
            candidate_id=item.candidate_id,
            family=family,
            tie_break_index=item.candidate_index,
            metrics_by_workload={
                w.workload_id: WorkloadMetrics(
                    **{metric: w.metrics[metric] for metric in model_selection.METRICS}
                )
                for w in item.workloads
            },
            provenance={
                "training_run_id": item.training_run_id,
                "training_seed": item.training_seed,
                "validation_run_ids": {w.workload_id: w.validation_run_id for w in item.workloads},
                "hyperparameters": item.hyperparameters,
            },
        )
        for item in evidence
    ]


def run_selection(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    selection: SelectionSpec,
    *,
    out: Path,
) -> dict[str, SelectionResult]:
    """Apply the frozen #78 rule within each family whose screening is complete."""
    results: dict[str, SelectionResult] = {}
    selected: dict[str, JsonValue] = {}
    for family in spec.families:
        algorithm, _ = parse_family(family)
        pool = candidates.for_algorithm(algorithm).candidates
        paths = [screening_path(out, family, c.candidate_id) for c in pool]
        if not all(path.exists() for path in paths):
            raise ValueError(f"{family}: screening incomplete; run screen --family {family}")
        evidence = load_evidence(paths)
        for item in evidence:
            _check_evidence(item, spec)
        result = model_selection.select(selection, family, family_candidates(evidence, family))
        result.save(out / f"{family}-selection.json")
        results[family] = result
        chosen = result.selected_candidate_id
        selected[family] = {
            "selection_succeeded": result.selection_succeeded,
            "feasible_candidate_count": result.feasible_candidate_count,
            "candidate_count": len(result.candidate_ids),
            "feasible_candidate_ids": list(result.feasible_candidate_ids),
            "selected_candidate_id": chosen,
            "selected_hyperparameters": (
                dict(candidates.for_algorithm(algorithm).get(chosen).hyperparameters)
                if chosen
                else None
            ),
            "selected_objective_values": (
                dict(result.selected_objective_values) if result.selected_objective_values else None
            ),
            "selected_screening_training_run_id": (
                next(e.training_run_id for e in evidence if e.candidate_id == chosen)
                if chosen
                else None
            ),
            "diagnostic_fallback_candidate_id": result.diagnostic_fallback_candidate_id,
        }
    _write_json(
        out / "selected-configs.json",
        {
            "experiment_id": spec.experiment_id,
            "candidate_set_id": spec.candidate_set_id,
            "selection_version": selection.selection_version,
            "selection_spec_id": selection.spec_id,
            "families": selected,
        },
    )
    return results


def run_retraining(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    family: str,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    traces: Mapping[str, WorkloadTrace] | None = None,
    timesteps: int | None = None,
    progress: Callable[[str], None] = print,
) -> list[TrainingEvidence]:
    """Retrain ``family``'s selected configuration with every retraining seed (all kept)."""
    result = SelectionResult.model_validate_json((out / f"{family}-selection.json").read_text())
    if not result.selection_succeeded or result.selected_candidate_id is None:
        raise ValueError(
            f"{family}: no feasible candidate was selected; there is nothing to retrain "
            "(the diagnostic fallback is never a selected model)"
        )
    training_entry, validation_entries = _entries()
    traces = traces or build_workloads([training_entry, *validation_entries])
    track = _tracker(tracking_uri, experiment_name)
    models = []
    for seed in spec.retraining_seeds:
        path = retraining_path(out, family, seed)
        if path.exists():
            evidence = TrainingEvidence.model_validate_json(path.read_text())
            _check_evidence(evidence, spec)
        else:
            evidence = train_model(
                spec,
                candidates,
                family=family,
                candidate_id=result.selected_candidate_id,
                seed=seed,
                phase="retraining",
                traces=traces,
                track=track,
                timesteps=timesteps,
            )
            evidence.save(path)
            progress(f"{family} seed{seed} -> {evidence.training_run_id}")
        models.append(evidence)
    return models


# --- nominal validation comparison ---------------------------------------------------------------


class EvaluationCase(_Strict):
    case_id: str
    controller: str
    controller_variant_id: str
    controller_version: str
    action_semantics: ActionSemantics
    workload_id: str
    training_seed: int | None = None
    training_run_id: str | None = None
    selected_candidate_id: str | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)


def evaluation_cases(
    spec: ExperimentSpec, models: Sequence[TrainingEvidence]
) -> list[EvaluationCase]:
    """Rule controllers under both contracts, then every retrained model, per workload."""
    cases = []
    for workload_id in spec.validation_workload_ids:
        for semantics in CONTRACTS:
            for controller, version, params in (
                ("threshold", "threshold-sla-first-v1", THRESHOLD_PARAMS),
                ("predictive", "predictive-v1", PREDICTIVE_PARAMS),
            ):
                variant = f"{controller}-{semantics}"
                cases.append(
                    EvaluationCase(
                        case_id=_case_id(variant, workload_id),
                        controller=controller,
                        controller_variant_id=variant,
                        controller_version=version,
                        action_semantics=semantics,
                        workload_id=workload_id,
                        params=dict(params),
                    )
                )
        for model in sorted(models, key=lambda m: (FAMILIES.index(m.family), m.training_seed)):
            variant = f"{model.family}-seed{model.training_seed}"
            cases.append(
                EvaluationCase(
                    case_id=_case_id(variant, workload_id),
                    controller=model.algorithm,
                    controller_variant_id=variant,
                    controller_version=f"{model.algorithm}-v1",
                    action_semantics=model.action_semantics,
                    workload_id=workload_id,
                    training_seed=model.training_seed,
                    training_run_id=model.training_run_id,
                    selected_candidate_id=model.candidate_id,
                    params=dict(model.hyperparameters),
                )
            )
    return cases


def _case_id(variant: str, workload_id: str) -> str:
    return f"{variant}|{workload_id}|{NOMINAL.name}|dyn{DYNAMICS_SEED}|eval{EVALUATION_SEED}"


def make_rule_controller(case: EvaluationCase, config: SimulatorConfig) -> Controller:
    contract = ActionContract.from_config(config)
    if case.controller == "threshold":
        return ThresholdController(
            low_threshold=THRESHOLD_PARAMS["low_threshold"],
            high_threshold=THRESHOLD_PARAMS["high_threshold"],
            cooldown_ticks=int(THRESHOLD_PARAMS["cooldown_ticks"]),
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            action_contract=contract,
        )
    return PredictiveController.from_config(
        config,
        history_window_ticks=int(PREDICTIVE_PARAMS["history_window_ticks"]),
        target_utilization=PREDICTIVE_PARAMS["target_utilization"],
        action_contract=contract,
    )


def _row(
    spec: ExperimentSpec,
    case: EvaluationCase,
    *,
    metrics: Mapping[str, float],
    action: Mapping[str, float],
    mlflow_run_id: str,
) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "case_id": case.case_id,
        "controller": case.controller,
        "controller_variant_id": case.controller_variant_id,
        "controller_version": case.controller_version,
        "action_semantics": case.action_semantics,
        "training_seed": case.training_seed,
        "training_run_id": case.training_run_id,
        "model_artifact_uri": (
            f"runs:/{case.training_run_id}/model" if case.training_run_id else None
        ),
        "selected_candidate_id": case.selected_candidate_id,
        "workload_id": case.workload_id,
        "workload_split": "validation",
        "robustness_scenario": NOMINAL.name,
        "dynamics_seed": DYNAMICS_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "mlflow_run_id": mlflow_run_id,
        **{key: float(metrics[key]) for key in METRIC_KEYS},
        **{f"action_{key}": float(action[key]) for key in ACTION_KEYS},
    }


def _recover(
    spec: ExperimentSpec, case: EvaluationCase, tracking_uri: str | None, experiment_name: str
) -> dict[str, Any] | None:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`scalerl.experiment_id` = '{spec.experiment_id}' and "
            f"tags.`scalerl.evaluation_case_id` = '{case.case_id}' and "
            "attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    metrics = runs[0].data.metrics
    return _row(
        spec,
        case,
        metrics=metrics,
        action={key: metrics[f"action.{key}"] for key in ACTION_KEYS},
        mlflow_run_id=runs[0].info.run_id,
    )


def run_evaluation(
    spec: ExperimentSpec,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    traces: Mapping[str, WorkloadTrace] | None = None,
    model_locator: Callable[[str], Path] | None = None,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Evaluate every case once as a tracked nominal ``evaluate`` run (resumable, deduplicated)."""
    from scalerl.mlops import start_tracked_run
    from scalerl.rl import load_sb3_controller

    models = load_evidence(sorted((out / "retraining").glob("*/seed*.json")))
    for model in models:
        _check_evidence(model, spec)
    manifest = load_benchmark_manifest()
    entries = {w: manifest.get(w) for w in spec.validation_workload_ids}
    traces = traces or build_workloads(list(entries.values()))
    locate = model_locator or _mlflow_locator(out / "models", tracking_uri)
    raw_path = out / "raw-results.jsonl"
    _repair_torn_tail(raw_path)
    completed = {
        row["case_id"] for row in read_rows(raw_path) if row["experiment_id"] == spec.experiment_id
    }
    cases = evaluation_cases(spec, models)
    _write_json(
        out / "model-manifest.json",
        {
            "experiment_id": spec.experiment_id,
            "candidate_set_id": spec.candidate_set_id,
            "selection_version": spec.selection_version,
            "selection_spec_id": spec.selection_spec_id,
            "models": [
                {
                    "variant_id": f"{m.family}-seed{m.training_seed}",
                    "family": m.family,
                    "algorithm": m.algorithm,
                    "action_semantics": m.action_semantics,
                    "training_seed": m.training_seed,
                    "selected_candidate_id": m.candidate_id,
                    "training_run_id": m.training_run_id,
                    "model_artifact_uri": m.model_artifact_uri,
                    "compatibility": m.compatibility,
                }
                for m in sorted(models, key=lambda m: (FAMILIES.index(m.family), m.training_seed))
            ],
        },
    )
    for case in cases:
        if case.case_id in completed:
            continue
        recovered = _recover(spec, case, tracking_uri, experiment_name)
        if recovered is not None:
            _append_row(raw_path, recovered)
            completed.add(case.case_id)
            continue
        config, source = contract_config(case.action_semantics)
        env = AutoscalingEnv(config, traces[case.workload_id])
        controller: Controller
        if case.training_run_id is None:
            controller = make_rule_controller(case, config)
        else:
            controller = load_sb3_controller(locate(case.training_run_id), env)
        run_spec = robustness_run_spec(
            controller=case.controller,
            entry=entries[case.workload_id],
            scenario=NOMINAL,
            dynamics_seed=DYNAMICS_SEED,
            config=config,
            base_config_source=source,
            hyperparameters=case.params,
            evaluation_seed=EVALUATION_SEED,
        )
        with start_tracked_run(
            run_spec,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"action-{case.case_id.replace('|', '-')}",
        ) as run:
            for key, value in _case_tags(spec, case).items():
                run.set_tag(key, value)
            result = evaluate_robustness(
                controller,
                traces[case.workload_id],
                scenario=NOMINAL,
                dynamics_seed=DYNAMICS_SEED,
                evaluation_seed=EVALUATION_SEED,
                config=config,
            )
            magnitude = summarize_action_magnitude(result.infos)
            run.log_metrics({**result.metrics.as_metrics(), **magnitude.as_metrics()})
        row = _row(
            spec,
            case,
            metrics=result.metrics.as_metrics(),
            action={k: float(v) for k, v in vars(magnitude).items()},
            mlflow_run_id=run.run_id,
        )
        _append_row(raw_path, row)
        completed.add(case.case_id)
        progress(f"[{len(completed)}/{len(cases)}] {case.case_id}")
    order = {case.case_id: index for index, case in enumerate(cases)}
    rows = [
        row
        for row in read_rows(raw_path)
        if row["experiment_id"] == spec.experiment_id and row["case_id"] in order
    ]
    return sorted(rows, key=lambda row: order[row["case_id"]])


def _case_tags(spec: ExperimentSpec, case: EvaluationCase) -> dict[str, str]:
    tags = {
        "scalerl.experiment_version": spec.experiment_version,
        "scalerl.experiment_id": spec.experiment_id,
        "scalerl.experiment_phase": "evaluation",
        "scalerl.evaluation_case_id": case.case_id,
        "scalerl.controller_variant_id": case.controller_variant_id,
        "scalerl.controller_version": case.controller_version,
        "scalerl.dynamics_seed": str(DYNAMICS_SEED),
        "scalerl.evaluation_seed": str(EVALUATION_SEED),
    }
    if case.training_run_id is not None:
        tags |= {
            "scalerl.model_source_run_id": case.training_run_id,
            "scalerl.model_artifact_uri": f"runs:/{case.training_run_id}/model",
            "scalerl.training_seed": str(case.training_seed),
            "scalerl.candidate_set_id": spec.candidate_set_id,
            "scalerl.selected_candidate_id": str(case.selected_candidate_id),
            "scalerl.selection_spec_id": spec.selection_spec_id,
        }
    return tags


def _mlflow_locator(cache_dir: Path, tracking_uri: str | None) -> Callable[[str], Path]:
    def locate(run_id: str) -> Path:
        target = cache_dir / run_id
        bundle = target / "model"
        if not (bundle / "model.zip").is_file():
            from mlflow import MlflowClient

            target.mkdir(parents=True, exist_ok=True)
            MlflowClient(tracking_uri).download_artifacts(run_id, "model", str(target))
        return bundle

    return locate


# --- summaries -----------------------------------------------------------------------------------


SUMMARY_METRICS: Final = (*METRIC_KEYS, *(f"action_{key}" for key in ACTION_KEYS))


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per controller x contract x workload: descriptive statistics over training seeds.

    Rule controllers are deterministic single runs (n = 1). Learned families
    summarize their five retrained seeds; nothing is pooled across workloads.
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["controller"], row["action_semantics"], row["workload_id"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (controller, semantics, workload_id), members in groups.items():
        for metric in SUMMARY_METRICS:
            stats = describe([float(member[metric]) for member in members])
            summary.append(
                {
                    "controller": controller,
                    "action_semantics": semantics,
                    "workload_id": workload_id,
                    "metric": metric,
                    "replicate_unit": "training_seed"
                    if members[0]["training_seed"] is not None
                    else "single_run",
                    "n": stats.n,
                    "mean": stats.mean,
                    "median": stats.median,
                    "std": stats.std,
                    "min": stats.minimum,
                    "max": stats.maximum,
                    "mlflow_run_ids": ",".join(member["mlflow_run_id"] for member in members),
                }
            )
    return summary


def paired_contract_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``desired-replicas-v1 − delta-v1`` for the same controller, training seed and workload."""
    index = {
        (row["controller"], row["training_seed"], row["workload_id"], row["action_semantics"]): row
        for row in rows
    }
    deltas = []
    for (controller, seed, workload_id, semantics), desired in index.items():
        if semantics != DESIRED_REPLICAS_V1:
            continue
        delta = index.get((controller, seed, workload_id, DELTA_V1))
        if delta is None:
            continue
        deltas.append(
            {
                "controller": controller,
                "training_seed": seed,
                "workload_id": workload_id,
                "delta_v1_run_id": delta["mlflow_run_id"],
                "desired_replicas_v1_run_id": desired["mlflow_run_id"],
                "delta_v1_candidate_id": delta["selected_candidate_id"],
                "desired_replicas_v1_candidate_id": desired["selected_candidate_id"],
                **{
                    f"{metric}_diff": float(desired[metric]) - float(delta[metric])
                    for metric in SUMMARY_METRICS
                },
            }
        )
    return deltas


def candidate_pairs(
    spec: ExperimentSpec, out: Path, selections: Mapping[str, SelectionResult]
) -> list[dict[str, Any]]:
    """Matched screening candidates: the same configuration under both contracts, per workload."""
    rows = []
    for algorithm in ALGORITHMS:
        delta_family = family_id(algorithm, DELTA_V1)
        desired_family = family_id(algorithm, DESIRED_REPLICAS_V1)
        delta_feasible = set(selections[delta_family].feasible_candidate_ids)
        desired_feasible = set(selections[desired_family].feasible_candidate_ids)
        for path in sorted((out / "screening" / delta_family).glob("*.json")):
            delta = TrainingEvidence.model_validate_json(path.read_text())
            desired = TrainingEvidence.model_validate_json(
                screening_path(out, desired_family, delta.candidate_id).read_text()
            )
            for workload_id in spec.validation_workload_ids:
                a, b = delta.metrics_for(workload_id), desired.metrics_for(workload_id)
                row: dict[str, Any] = {
                    "algorithm": algorithm,
                    "candidate_id": delta.candidate_id,
                    "workload_id": workload_id,
                    "delta_v1_feasible": delta.candidate_id in delta_feasible,
                    "desired_replicas_v1_feasible": delta.candidate_id in desired_feasible,
                }
                for metric in (
                    *model_selection.METRICS,
                    "mean_p95_latency_seconds",
                    "episode_reward",
                ):
                    row[f"delta_v1_{metric}"] = a.metrics[metric]
                    row[f"desired_replicas_v1_{metric}"] = b.metrics[metric]
                    row[f"{metric}_diff"] = b.metrics[metric] - a.metrics[metric]
                for key in (
                    "total_absolute_replica_change",
                    "max_absolute_replica_change_in_one_tick",
                ):
                    row[f"delta_v1_action_{key}"] = a.action[key]
                    row[f"desired_replicas_v1_action_{key}"] = b.action[key]
                rows.append(row)
    return rows


# --- invariants and decision ---------------------------------------------------------------------


def check_action_invariants() -> list[dict[str, Any]]:
    """Executable action/lifecycle invariants the decision requires (recorded in the report)."""
    from scalerl.workloads import steady_workload

    checks: list[dict[str, Any]] = []

    def record(name: str, condition: bool) -> None:
        checks.append({"check": name, "passed": bool(condition)})

    delta = ActionContract(DELTA_V1, 1, 10)
    desired = ActionContract(DESIRED_REPLICAS_V1, 1, 10)
    record("delta-v1 is Discrete(3)", delta.action_count == 3)
    record(
        "delta-v1 codes 0/1/2 -> effects -1/0/+1",
        [delta.target_for(c, 5) - 5 for c in (0, 1, 2)] == [-1, 0, 1],
    )
    record("delta-v1 rejects -1", _rejects(delta, -1))
    record(
        "desired-replicas-v1 codes 0..9 -> targets 1..10",
        [desired.target_for(c, 5) for c in range(10)] == list(range(1, 11)),
    )
    other = ActionContract(DESIRED_REPLICAS_V1, 2, 5)
    record(
        "desired-replicas-v1 arbitrary bounds 2..5",
        other.action_count == 4 and [other.target_for(c, 3) for c in range(4)] == [2, 3, 4, 5],
    )
    record(
        "desired-replicas-v1 rejects out-of-range, bool and float codes",
        _rejects(desired, 10)
        and _rejects(desired, -1)
        and _rejects(desired, True)
        and _rejects(desired, 2.0),
    )

    config, _ = contract_config(DESIRED_REPLICAS_V1)
    config = config.model_copy(update={"replicas": ReplicaConfig(initial_replicas=2)})
    trace = steady_workload(
        duration_seconds=config.timing.episode_duration_seconds,
        control_interval_seconds=config.timing.control_interval_seconds,
        rate=0.0,
    )
    env = AutoscalingEnv(config, trace)
    env.reset(seed=0)
    info = env.step(7)[4]  # target 8 from 2 committed
    record(
        "multi-replica scale-up: +6 in one step, all pending (startup delay honored)",
        info["applied_replica_change"] == 6
        and info["active_replicas"] == 2
        and info["pending_replicas"] == 6
        and info["requested_replica_target"] == 8
        and env.replica_counts["active_replicas"] == 2,  # 30 s of the 60 s startup elapsed
    )
    env.step(7)  # hold at target 8 for the second tick of startup
    record(
        "the six replicas become active only after the 60 s startup delay (two ticks)",
        env.replica_counts
        == {"active_replicas": 8, "pending_replicas": 0, "terminating_replicas": 0},
    )
    env.reset(seed=0)
    env.step(7)
    down = env.step(2)[4]  # target 3 from 8 committed, before any new replica is ready
    record(
        "multi-replica scale-down cancels the newest pending replicas first (-5)",
        down["applied_replica_change"] == -5
        and down["active_replicas"] == 2
        and down["pending_replicas"] == 1,
    )
    down = env.step(0)[4]  # target 1: cancel the last pending, then terminate one active
    record(
        "then terminates active replicas, never below min_replicas",
        down["applied_replica_change"] == -2
        and down["active_replicas"] == 1
        and down["pending_replicas"] == 0,
    )

    pool = ReplicaPool(ReplicaConfig(initial_replicas=2, startup_delay_seconds=0.0))
    record(
        "zero startup delay: multi-replica request activates immediately",
        pool.scale_up(6) == 6 and pool.active_count == 8 and pool.pending_count == 0,
    )
    return checks


def _rejects(contract: ActionContract, code: Any) -> bool:
    try:
        contract.validate_code(code)
    except ValueError:
        return True
    return False


def _finite(evidence: TrainingEvidence) -> bool:
    return all(
        math.isfinite(value)
        for w in evidence.workloads
        for value in (*w.metrics.values(), *w.action.values())
    )


def decide(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    *,
    out: Path,
) -> dict[str, Any]:
    """Apply the predeclared decision principle to the completed experiment's evidence."""
    selections = {
        family: SelectionResult.model_validate_json((out / f"{family}-selection.json").read_text())
        for family in spec.families
    }
    invariants = check_action_invariants()
    trainable: dict[str, Any] = {}
    for family in spec.families:
        algorithm, _ = parse_family(family)
        pool = candidates.for_algorithm(algorithm).candidates
        paths = [screening_path(out, family, c.candidate_id) for c in pool]
        present = [path for path in paths if path.exists()]
        evidence = load_evidence(present)
        trainable[family] = {
            "screened": len(evidence),
            "expected": len(pool),
            "all_finite": all(_finite(item) for item in evidence),
            "compatibility_semantics": sorted(
                {str(item.compatibility.get("action_semantics_version")) for item in evidence}
            ),
            "action_counts": sorted(
                {int(str(item.compatibility["action_count"])) for item in evidence}
            ),
        }
    rows = [
        row
        for row in read_rows(out / "raw-results.jsonl")
        if row["experiment_id"] == spec.experiment_id
    ]
    models = load_evidence(sorted((out / "retraining").glob("*/seed*.json")))
    expected_models = sum(
        len(spec.retraining_seeds) for result in selections.values() if result.selection_succeeded
    )
    expected_rows = len(evaluation_cases(spec, models))
    consistency = _evaluation_matches_retraining(rows, models)
    experiment_complete = (
        all(t["screened"] == t["expected"] for t in trainable.values())
        and len(models) == expected_models
        and len(rows) == expected_rows
        and consistency["mismatches"] == 0
        and all(math.isfinite(float(row[m])) for row in rows for m in SUMMARY_METRICS)
    )
    desired_families = [f for f in spec.families if f.endswith(DESIRED_REPLICAS_V1)]
    criteria = {
        "action_lifecycle_invariants_pass": all(check["passed"] for check in invariants),
        "dqn_and_ppo_train_and_infer_with_desired_replicas_v1": all(
            trainable[f]["screened"] == trainable[f]["expected"]
            and trainable[f]["all_finite"]
            and trainable[f]["compatibility_semantics"] == [DESIRED_REPLICAS_V1]
            for f in desired_families
        ),
        "experiment_completed_without_encoding_failure": experiment_complete,
    }
    adopted = all(criteria.values())
    return {
        "experiment_version": spec.experiment_version,
        "experiment_id": spec.experiment_id,
        "candidate_set_id": spec.candidate_set_id,
        "selection_version": spec.selection_version,
        "selection_spec_id": spec.selection_spec_id,
        "decision_principle": spec.decision_principle.model_dump(mode="json"),
        "criteria": criteria,
        "invariant_checks": invariants,
        "trainability": trainable,
        "evaluation_consistency": consistency,
        "evaluation_rows": len(rows),
        "expected_evaluation_rows": expected_rows,
        "retrained_models": len(models),
        "expected_retrained_models": expected_models,
        "selection": {
            family: {
                "feasible_candidate_count": result.feasible_candidate_count,
                "candidate_count": len(result.candidate_ids),
                "feasible_candidate_ids": list(result.feasible_candidate_ids),
                "selection_succeeded": result.selection_succeeded,
                "selected_candidate_id": result.selected_candidate_id,
                "selected_objective_values": result.selected_objective_values,
                "diagnostic_fallback_candidate_id": result.diagnostic_fallback_candidate_id,
            }
            for family, result in selections.items()
        },
        "decision": "adopt desired-replicas-v1" if adopted else "unresolved",
        "final_action_semantics": PREFERRED_CONTRACT if adopted else None,
        "metrics_used_to_choose_the_contract": False,
        "held_out_data_used": False,
        "reward_changed": False,
    }


def _evaluation_matches_retraining(
    rows: Sequence[Mapping[str, Any]], models: Sequence[TrainingEvidence]
) -> dict[str, Any]:
    """Each learned evaluation must equal its retraining validation run (determinism check)."""
    compared = mismatches = 0
    for model in models:
        variant = f"{model.family}-seed{model.training_seed}"
        for row in rows:
            if row["controller_variant_id"] != variant:
                continue
            expected = model.metrics_for(row["workload_id"]).metrics
            compared += 1
            if any(abs(float(row[k]) - expected[k]) > 1e-12 for k in METRIC_KEYS):
                mismatches += 1
    return {"compared": compared, "mismatches": mismatches}


class ActionContractDecision(_Strict):
    """The frozen final action contract for #20/#72/#46, with its validation evidence."""

    decision_version: Literal["action-contract-v2"] = DECISION_VERSION
    final_action_semantics: Literal["desired-replicas-v1"]
    action_space: str
    code_mapping: str
    experiment_version: str
    experiment_spec_id: str
    candidate_set_id: str
    selection_version: str
    selection_spec_id: str
    decision_rule: str
    criteria: dict[str, bool]
    selected_configurations: dict[str, JsonValue]
    retrained_models: tuple[dict[str, JsonValue], ...]
    validation_evidence_summary: dict[str, JsonValue]
    result_references: dict[str, str]
    simulator_default_note: str
    held_out_data_used: Literal[False] = False
    reward_changed: Literal[False] = False

    @property
    def decision_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def build_decision(
    spec: ExperimentSpec, report: Mapping[str, Any], *, out: Path
) -> ActionContractDecision:
    if report["decision"] != "adopt desired-replicas-v1":
        raise ValueError("the predeclared criteria did not pass; the contract stays unresolved")
    models = load_evidence(sorted((out / "retraining").glob("*/seed*.json")))
    rows = [
        row
        for row in read_rows(out / "raw-results.jsonl")
        if row["experiment_id"] == spec.experiment_id
    ]
    means: dict[str, JsonValue] = {}
    for summary in summarize_rows(rows):
        if summary["metric"] in (
            "sla_violation_rate",
            "normalized_cost",
            "queue_pressure",
            "churn_rate",
            "action_total_absolute_replica_change",
        ):
            key = f"{summary['controller']}|{summary['action_semantics']}|{summary['workload_id']}"
            entry = means.setdefault(key, {})
            assert isinstance(entry, dict)
            entry[summary["metric"]] = summary["mean"]
            entry["n"] = summary["n"]
    desired = next(c for c in spec.action_contracts if c.semantics == DESIRED_REPLICAS_V1)
    return ActionContractDecision(
        final_action_semantics=DESIRED_REPLICAS_V1,
        action_space=desired.action_space,
        code_mapping=desired.code_mapping,
        experiment_version=spec.experiment_version,
        experiment_spec_id=spec.experiment_id,
        candidate_set_id=spec.candidate_set_id,
        selection_version=spec.selection_version,
        selection_spec_id=spec.selection_spec_id,
        decision_rule=(
            "predeclared: adopt the preferred desired-replicas-v1 if invariants pass, DQN/PPO "
            "train and infer against it, and the experiment completes without an "
            "encoding failure; validation metrics characterize, they do not choose"
        ),
        criteria={k: bool(v) for k, v in report["criteria"].items()},
        selected_configurations=report["selection"],
        retrained_models=tuple(
            {
                "family": m.family,
                "training_seed": m.training_seed,
                "selected_candidate_id": m.candidate_id,
                "training_run_id": m.training_run_id,
                "model_artifact_uri": m.model_artifact_uri,
            }
            for m in sorted(models, key=lambda m: (FAMILIES.index(m.family), m.training_seed))
        ),
        validation_evidence_summary=means,
        result_references={
            "mlflow_experiment": EXPERIMENT_NAME,
            "local_outputs": str(DEFAULT_OUTPUT),
            "raw_results": "raw-results.jsonl (one nominal evaluate run per row, MLflow run IDs)",
        },
        simulator_default_note=(
            "SimulatorConfig() keeps delta-v1 as its default for historical reproducibility; "
            "downstream final experiments (#20/#72/#46) request desired-replicas-v1 explicitly"
        ),
    )


# --- helpers -------------------------------------------------------------------------------------


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(spec: ExperimentSpec, out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(out / "raw-results.csv", list(rows))
    summary = summarize_rows(rows)
    _write_json(out / "summary.json", summary)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "paired-deltas.csv", paired_contract_deltas(rows))
    selections = {
        family: SelectionResult.model_validate_json((out / f"{family}-selection.json").read_text())
        for family in spec.families
    }
    _write_csv(out / "candidate-pairs.csv", candidate_pairs(spec, out, selections))


# --- command line --------------------------------------------------------------------------------


def _describe_plan(spec: ExperimentSpec, candidates: CandidateSet, selection: SelectionSpec) -> str:
    return "\n".join(
        [
            f"experiment         {spec.experiment_version} {spec.experiment_id}",
            f"candidate set      {spec.candidate_set_version} {candidates.candidate_set_id} "
            f"({candidates.generation_method}, seed {candidates.generation_seed})",
            f"selection          {selection.selection_version} {selection.spec_id}",
            f"train workload     {spec.train_workload_id}",
            f"validation         {', '.join(spec.validation_workload_ids)}",
            f"test workloads     {list(spec.test_workload_ids) or 'none'}",
            f"action contracts   {', '.join(c.semantics for c in spec.action_contracts)}",
            f"candidates         dqn {len(candidates.dqn.candidates)}, "
            f"ppo {len(candidates.ppo.candidates)} (identical for both contracts)",
            f"budgets            {spec.timesteps}",
            f"seeds              screening {spec.screening_training_seed}; retraining "
            f"{list(spec.retraining_seeds)}; evaluation {spec.evaluation_seed}; "
            f"dynamics {DYNAMICS_SEED} (nominal)",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_VERSION} (#79), validation only.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-spec", "check", "screen", "select", "retrain", "evaluate", "decide"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        command.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
        command.add_argument("--selection-spec", type=Path, default=DEFAULT_SELECTION_SPEC)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
        command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
        if name in ("screen", "retrain"):
            command.add_argument("--family", required=True, choices=FAMILIES)
            command.add_argument("--torch-threads", type=int, default=None)
        if name == "decide":
            command.add_argument("--freeze", action="store_true", help=f"write {DEFAULT_DECISION}")
            command.add_argument("--decision-output", type=Path, default=DEFAULT_DECISION)
    args = parser.parse_args(argv)

    if args.command == "freeze-spec":
        from scalerl.tuning.candidates import generate_candidate_set

        if args.candidates.exists():
            candidates = CandidateSet.load(args.candidates)
        else:
            candidates = generate_candidate_set()
            candidates.save(args.candidates)
        spec = build_experiment_spec(candidates)
        spec.save(args.spec)
        print(f"{spec.experiment_version} {spec.experiment_id} written to {args.spec}")
        print(f"{candidates.candidate_set_version} {candidates.candidate_set_id}")
        return 0

    # Every other command verifies the frozen plan before creating any output.
    try:
        spec, candidates, selection = load_frozen_inputs(
            args.spec, args.candidates, args.selection_spec
        )
    except ValueError as error:
        parser.error(str(error))
    print(_describe_plan(spec, candidates, selection))
    if args.command == "check":
        return 0

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    spec.save(out / "experiment-spec.json")
    candidates.save(out / "candidate-set.json")
    if getattr(args, "torch_threads", None):
        import torch

        torch.set_num_threads(args.torch_threads)

    if args.command == "screen":
        run_screening(
            spec,
            candidates,
            args.family,
            out=out,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    elif args.command == "select":
        for family, result in run_selection(spec, candidates, selection, out=out).items():
            chosen = result.selected_candidate_id or (
                f"none (diagnostic only: {result.diagnostic_fallback_candidate_id})"
            )
            count = f"{result.feasible_candidate_count}/{len(result.candidate_ids)}"
            print(f"{family}: {count} feasible; selected {chosen}")
    elif args.command == "retrain":
        run_retraining(
            spec,
            candidates,
            args.family,
            out=out,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    elif args.command == "evaluate":
        rows = run_evaluation(
            spec, out=out, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name
        )
        write_reports(spec, out, rows)
        print(f"{len(rows)} evaluation rows; reports in {out}")
    elif args.command == "decide":
        report = decide(spec, candidates, out=out)
        _write_json(out / "decision-report.json", report)
        print(f"decision: {report['decision']}; criteria {report['criteria']}")
        if args.freeze:
            decision = build_decision(spec, report, out=out)
            _write_text(args.decision_output, decision.model_dump_json(indent=2) + "\n")
            print(f"{decision.decision_version} {decision.decision_id} -> {args.decision_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
