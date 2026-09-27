"""Reward-function ablation ``reward-ablation-v1`` (#20), validation only.

Which reward components and weights drive learned behavior under the final #79
action contract ``desired-replicas-v1``? Eight predeclared variants of the
unchanged penalty terms of :mod:`scalerl.environment.reward` (normalization is
not touched; only which components are used and their weights):

======================================  =======  ====  ===  =====  =====
variant                                 latency  cost  SLA  queue  churn
======================================  =======  ====  ===  =====  =====
``latency-cost-v1``                     1.0      1.0   0    0      0
``latency-cost-sla-v1``                 1.0      1.0   1.0  0      0
``latency-cost-sla-queue-v1``           1.0      1.0   1.0  1.0    0
``full-default-v1`` (current reward)    1.0      1.0   1.0  1.0    0.1
``full-cost-low-v1``                    1.0      0.5   1.0  1.0    0.1
``full-cost-high-v1``                   1.0      2.0   1.0  1.0    0.1
``full-sla-low-v1``                     1.0      1.0   0.5  1.0    0.1
``full-sla-high-v1``                    1.0      1.0   2.0  1.0    0.1
======================================  =======  ====  ===  =====  =====

Per reward variant (everything else identical to #79; train on
``syn-train-bursty``, validate on the three synthetic validation workloads, no
test workload):

* **DQN**: the exact 20 #79 desired-replicas-v1 candidates
  (``matched-action-candidates-v1``), seed 0, 200,000 timesteps; the frozen #78
  rule ``selection-v2-cost-under-sla`` (``418876d6c8e9``) selects within the
  reward condition; a selected candidate is retrained on seeds 0-4, none is
  retrained otherwise (the diagnostic fallback is never promoted);
* **PPO**: the fixed #79 configuration ``ppo-c08`` trained fresh on seeds 0-4
  (204,800 timesteps); no PPO search.

Reward decision (predeclared): equal-seed means per algorithm x reward x
workload; a reward is joint-feasible when PPO and the retrained selected DQN
both meet the Threshold SLA limit on every workload; among joint-feasible
rewards minimize the equal-algorithm (50/50) mean of equal-workload means of
normalized cost, then queue pressure, churn, SLA, reward ID; if none is
joint-feasible, ``full-default-v1`` is frozen with the negative DQN result.
Episode reward is never a selection criterion.

    python -m scalerl.evaluation.reward_ablation check
    python -m scalerl.evaluation.reward_ablation run --reward full-default-v1
    python -m scalerl.evaluation.reward_ablation decide [--freeze]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.environment import DESIRED_REPLICAS_V1, SimulatorConfig
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import model_selection
from scalerl.evaluation.action_semantics import (
    ActionContractDecision,
    WorkloadEvidence,
    contract_config,
    load_frozen_inputs,
)
from scalerl.evaluation.metrics import ActionMagnitudeMetrics, EpisodeMetrics
from scalerl.evaluation.model_selection import (
    Candidate,
    SelectionResult,
    SelectionSpec,
    WorkloadMetrics,
)
from scalerl.evaluation.multiseed import describe
from scalerl.mlops import RunSpec
from scalerl.mlops.tracking import TrackedRun
from scalerl.tuning.candidates import CandidateSet
from scalerl.workloads import WorkloadTrace

ABLATION_VERSION: Final = "reward-ablation-v1"
CONTRACT_VERSION: Final = "reward-contract-v1"
EXPERIMENT_NAME: Final = "scalerl-reward-ablation"
TRAIN_WORKLOAD: Final = "syn-train-bursty"
VALIDATION_WORKLOADS: Final = model_selection.VALIDATION_WORKLOADS
SEEDS: Final = (0, 1, 2, 3, 4)
SCREENING_SEED: Final = 0
EVALUATION_SEED: Final = 0
TIMESTEPS: Final = {"dqn": 200_000, "ppo": 204_800}
PPO_CANDIDATE: Final = "ppo-c08"
PPO_FAMILY_79: Final = "ppo-desired-replicas-v1"
DQN_FAMILY_79: Final = "dqn-desired-replicas-v1"
ACTION_DECISION_VERSION: Final = "action-contract-v2"
ACTION_EXPERIMENT_ID: Final = "899dfbb64217"
CANDIDATE_SET_ID: Final = "cbe3c1c0719b"
SELECTION_SPEC_ID: Final = "418876d6c8e9"
FALLBACK_REWARD: Final = "full-default-v1"
DECISION_METRICS: Final = ("normalized_cost", "queue_pressure", "churn_rate", "sla_violation_rate")
METRIC_KEYS: Final = tuple(EpisodeMetrics.__dataclass_fields__)
ACTION_KEYS: Final = tuple(ActionMagnitudeMetrics.__dataclass_fields__)

REWARD_VARIANTS: Final[dict[str, RewardWeights]] = {
    "latency-cost-v1": RewardWeights(latency=1.0, cost=1.0, sla=0.0, queue=0.0, churn=0.0),
    "latency-cost-sla-v1": RewardWeights(latency=1.0, cost=1.0, sla=1.0, queue=0.0, churn=0.0),
    "latency-cost-sla-queue-v1": RewardWeights(
        latency=1.0, cost=1.0, sla=1.0, queue=1.0, churn=0.0
    ),
    "full-default-v1": RewardWeights(latency=1.0, cost=1.0, sla=1.0, queue=1.0, churn=0.1),
    "full-cost-low-v1": RewardWeights(latency=1.0, cost=0.5, sla=1.0, queue=1.0, churn=0.1),
    "full-cost-high-v1": RewardWeights(latency=1.0, cost=2.0, sla=1.0, queue=1.0, churn=0.1),
    "full-sla-low-v1": RewardWeights(latency=1.0, cost=1.0, sla=0.5, queue=1.0, churn=0.1),
    "full-sla-high-v1": RewardWeights(latency=1.0, cost=1.0, sla=2.0, queue=1.0, churn=0.1),
}

# Diagnostic flags (predeclared; they never influence selection or the decision).
PATHOLOGY_RULES: Final = {
    "full_fleet": "equal-seed mean normalized cost >= 0.9 on a workload",
    "underprovisioning": "mean SLA above the Threshold limit and mean normalized cost < 0.6",
    "thrashing": "mean churn rate >= 0.3 on a workload",
    "churn_aversion": "SLA above the limit with <= 2 mean scaling actions per episode",
    "persistent_backlog": "mean queue pressure >= 0.2 on a workload",
    "reward_mismatch": (
        "DQN screening: the candidate with the highest mean episode reward is not #78-feasible, "
        "or episode reward rank-correlates positively with mean SLA violation"
    ),
}

DEFAULT_SPEC = Path("benchmarks/v1/reward-ablation-v1.json")
DEFAULT_CONTRACT = Path("benchmarks/v1/reward-contract-v1.json")
DEFAULT_ACTION_DECISION = Path("benchmarks/v1/action-contract-v2.json")
DEFAULT_ACTION_SPEC = Path("benchmarks/v1/action-semantics-experiment-v1.json")
DEFAULT_CANDIDATES = Path("benchmarks/v1/action-semantics-candidates-v1.json")
DEFAULT_SELECTION_SPEC = Path("benchmarks/v1/selection-v2-cost-under-sla.json")
DEFAULT_OUTPUT = Path("outputs/reward-ablation-v1")

Phase = Literal["dqn-screening", "dqn-retraining", "ppo-training"]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def final_config() -> SimulatorConfig:
    """The canonical config under the frozen #79 contract, requested explicitly."""
    config, _ = contract_config(DESIRED_REPLICAS_V1)
    if config.action.semantics != DESIRED_REPLICAS_V1:
        raise ValueError("#20 runs must use desired-replicas-v1")
    return config


def reward_weights(variant: str) -> RewardWeights:
    try:
        return REWARD_VARIANTS[variant]
    except KeyError:
        known = ", ".join(REWARD_VARIANTS)
        raise ValueError(f"unknown reward variant {variant!r}; known: {known}") from None


# --- frozen experiment specification ------------------------------------------------------------


class ExperimentSpec(_Strict):
    """Everything #20 holds fixed; ``experiment_id`` hashes it."""

    experiment_version: Literal["reward-ablation-v1"] = ABLATION_VERSION
    benchmark_version: str
    reward_variants: dict[str, dict[str, float]]
    reference_reward_variant: Literal["full-default-v1"] = FALLBACK_REWARD
    reward_semantics: str
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    action_experiment_spec_id: Literal["899dfbb64217"] = ACTION_EXPERIMENT_ID
    simulator_config: dict[str, JsonValue]
    training_workload_ids: tuple[str, ...]
    validation_workload_ids: tuple[str, ...]
    test_workload_ids: tuple[str, ...] = ()
    selection_version: Literal["selection-v2-cost-under-sla"] = model_selection.SELECTION_VERSION
    selection_spec_id: Literal["418876d6c8e9"] = SELECTION_SPEC_ID
    sla_thresholds: dict[str, float]
    candidate_set_id: Literal["cbe3c1c0719b"] = CANDIDATE_SET_ID
    ppo: dict[str, JsonValue]
    dqn: dict[str, JsonValue]
    training_seeds: tuple[int, ...] = SEEDS
    evaluation_seed: Literal[0] = EVALUATION_SEED
    evaluation_condition: str
    reward_level_rule: dict[str, JsonValue]
    no_joint_feasible_fallback: Literal["full-default-v1"] = FALLBACK_REWARD
    pathology_rules: dict[str, str]
    metrics: dict[str, JsonValue]
    robustness: str
    held_out_data_used: Literal[False] = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        manifest = load_benchmark_manifest()
        if self.benchmark_version != manifest.version:
            raise ValueError(f"spec benchmark {self.benchmark_version!r} is not installed")
        if self.test_workload_ids:
            raise ValueError("#20 uses no test workloads")
        for ids, split in (
            (self.training_workload_ids, "train"),
            (self.validation_workload_ids, "validation"),
        ):
            for workload_id in ids:
                try:
                    actual = manifest.get(workload_id).split
                except KeyError as error:
                    raise ValueError(str(error)) from None
                if actual != split:
                    kind = "held-out test" if actual == "test" else actual
                    raise ValueError(f"{workload_id!r} is a {kind} workload, not {split}")
        if self.training_workload_ids != (TRAIN_WORKLOAD,):
            raise ValueError(f"training uses exactly {TRAIN_WORKLOAD}")
        if self.validation_workload_ids != VALIDATION_WORKLOADS:
            raise ValueError(f"validation uses exactly {VALIDATION_WORKLOADS}")
        if self.reward_variants != _variants():
            raise ValueError("reward variants differ from the predeclared eight")
        if self.training_seeds != SEEDS:
            raise ValueError(f"training seeds must be {SEEDS}")
        if self.simulator_config != final_config().model_dump(mode="json"):
            raise ValueError("the simulator config must be the canonical desired-replicas-v1 one")
        return self

    @property
    def experiment_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        return cls.model_validate_json(Path(path).read_text())


def _variants() -> dict[str, dict[str, float]]:
    return {name: weights.model_dump() for name, weights in REWARD_VARIANTS.items()}


def verify_upstream(
    decision_path: Path, action_spec_path: Path, candidates_path: Path, selection_path: Path
) -> tuple[ActionContractDecision, CandidateSet, SelectionSpec]:
    """The frozen #78/#79 inputs, checked against the facts #20 depends on."""
    action_spec, candidates, selection = load_frozen_inputs(
        action_spec_path, candidates_path, selection_path
    )
    decision = ActionContractDecision.model_validate_json(decision_path.read_text())
    if (
        decision.decision_version != ACTION_DECISION_VERSION
        or decision.final_action_semantics != DESIRED_REPLICAS_V1
        or decision.experiment_spec_id != ACTION_EXPERIMENT_ID
        or action_spec.experiment_id != ACTION_EXPERIMENT_ID
        or candidates.candidate_set_id != CANDIDATE_SET_ID
        or selection.spec_id != SELECTION_SPEC_ID
    ):
        raise ValueError("the #78/#79 inputs differ from the frozen ones #20 depends on")
    ppo = decision.selected_configurations[PPO_FAMILY_79]
    dqn = decision.selected_configurations[DQN_FAMILY_79]
    if not isinstance(ppo, dict) or ppo.get("selected_candidate_id") != PPO_CANDIDATE:
        raise ValueError(f"#79 selected {PPO_CANDIDATE} for {PPO_FAMILY_79}")
    if not isinstance(dqn, dict) or dqn.get("selected_candidate_id") is not None:
        raise ValueError(f"#79 selected no {DQN_FAMILY_79} candidate")
    return decision, candidates, selection


def build_experiment_spec(candidates: CandidateSet, selection: SelectionSpec) -> ExperimentSpec:
    ppo = candidates.ppo.get(PPO_CANDIDATE)
    return ExperimentSpec(
        benchmark_version=load_benchmark_manifest().version,
        reward_variants=_variants(),
        reward_semantics=(
            "reward = -(w_latency*p95/(p95+target) + w_cost*tick_cost/max_tick_cost + "
            "w_sla*violated + w_queue*queued/(queued+max_tick_capacity) + "
            "w_churn*(applied_replica_change != 0)); component normalization unchanged "
            "(scalerl.environment.reward); only component inclusion/weights vary; "
            "RewardWeights() defaults stay full-default-v1"
        ),
        simulator_config=final_config().model_dump(mode="json"),
        training_workload_ids=(TRAIN_WORKLOAD,),
        validation_workload_ids=VALIDATION_WORKLOADS,
        sla_thresholds={t.workload_id: t.sla_violation_rate for t in selection.sla_thresholds},
        ppo={
            "source": f"#79 {PPO_FAMILY_79} selected candidate {PPO_CANDIDATE}",
            "candidate_id": PPO_CANDIDATE,
            "config_version": candidates.ppo.config_version,
            "hyperparameters": dict(ppo.hyperparameters),
            "timesteps": TIMESTEPS["ppo"],
            "training_seeds": list(SEEDS),
            "search": "none: the same configuration is retrained fresh under every reward",
        },
        dqn={
            "source": f"#79 {DQN_FAMILY_79} matched candidates (0/20 feasible under #79)",
            "candidate_ids": [c.candidate_id for c in candidates.dqn.candidates],
            "search_space_version": candidates.dqn.search_space_version,
            "config_version": candidates.dqn.config_version,
            "timesteps": TIMESTEPS["dqn"],
            "screening_seed": SCREENING_SEED,
            "retraining_seeds": list(SEEDS),
            "selection": "frozen #78 rule within each reward condition",
            "no_feasible": "select nothing; the diagnostic fallback is never retrained or promoted",
        },
        evaluation_condition=(
            "nominal validation (robustness-v1 nominal, fixed startup, dynamics seed 0), "
            "evaluation seed 0, deterministic policies, one tracked run per model x workload"
        ),
        reward_level_rule={
            "per_family": "equal-seed mean per validation workload over seeds 0-4 (no seed pick)",
            "joint_feasible": (
                "PPO (ppo-c08) and the retrained #78-selected DQN both meet mean SLA <= Threshold "
                "SLA + 1e-12 on every validation workload"
            ),
            "objective": (
                "among joint-feasible rewards minimize mean over {DQN, PPO} (50/50) of the "
                "equal-workload mean of equal-seed means"
            ),
            "order": ["normalized_cost", "queue_pressure", "churn_rate", "sla_violation_rate"],
            "final_tie_break": "reward variant ID",
            "episode_reward_used": False,
        },
        pathology_rules=dict(PATHOLOGY_RULES),
        metrics={
            "system": list(METRIC_KEYS),
            "action": list(ACTION_KEYS),
            "statistics": "n, mean, median, sample SD, min, max per algorithm x reward x workload",
            "episode_reward": "secondary; never a selection criterion",
        },
        robustness="none in #20: reward selection is nominal-only; no robustness run",
    )


def load_frozen(
    spec_path: Path,
    decision_path: Path = DEFAULT_ACTION_DECISION,
    action_spec_path: Path = DEFAULT_ACTION_SPEC,
    candidates_path: Path = DEFAULT_CANDIDATES,
    selection_path: Path = DEFAULT_SELECTION_SPEC,
) -> tuple[ExperimentSpec, CandidateSet, SelectionSpec]:
    _, candidates, selection = verify_upstream(
        decision_path, action_spec_path, candidates_path, selection_path
    )
    spec = ExperimentSpec.load(spec_path)
    if spec != build_experiment_spec(candidates, selection):
        raise ValueError("the committed #20 spec differs from this code's frozen plan")
    return spec, candidates, selection


# --- evidence ------------------------------------------------------------------------------------


class TrainingEvidence(_Strict):
    """One trained model of the ablation and its per-workload validation evidence."""

    experiment_id: str
    reward_variant: str
    reward_weights: dict[str, float]
    phase: Phase
    algorithm: Literal["dqn", "ppo"]
    candidate_id: str
    candidate_index: int
    training_seed: int
    hyperparameters: dict[str, JsonValue]
    training_run_id: str
    model_artifact_uri: str
    compatibility: dict[str, JsonValue]
    workloads: tuple[WorkloadEvidence, ...]

    def metrics_for(self, workload_id: str) -> WorkloadEvidence:
        return next(w for w in self.workloads if w.workload_id == workload_id)

    def save(self, path: Path) -> Path:
        return _write_text(path, self.model_dump_json(indent=2) + "\n")


def dqn_candidate_ids(spec: ExperimentSpec) -> list[str]:
    ids = spec.dqn["candidate_ids"]
    if not isinstance(ids, list):
        raise ValueError("spec dqn.candidate_ids must be a list")
    return [str(cid) for cid in ids]


def evidence_path(out: Path, reward: str, phase: Phase, candidate_id: str, seed: int) -> Path:
    return out / reward / phase / f"{candidate_id}-seed{seed}.json"


def _load_evidence(
    path: Path, spec: ExperimentSpec, reward: str, candidate_id: str, seed: int
) -> TrainingEvidence:
    evidence = TrainingEvidence.model_validate_json(path.read_text())
    expected = (spec.experiment_id, reward, candidate_id, seed, reward_weights(reward).model_dump())
    actual = (
        evidence.experiment_id,
        evidence.reward_variant,
        evidence.candidate_id,
        evidence.training_seed,
        evidence.reward_weights,
    )
    if actual != expected:
        raise ValueError(f"{path}: evidence from a different experiment/reward/candidate/seed")
    if evidence.compatibility.get("action_semantics_version") != DESIRED_REPLICAS_V1:
        raise ValueError(f"{path}: not trained under desired-replicas-v1")
    return evidence


# --- training ------------------------------------------------------------------------------------


def _algorithm_spec(algorithm: str) -> Any:
    if algorithm == "dqn":
        from scalerl.training.dqn import DQN_ALGORITHM

        return DQN_ALGORITHM
    from scalerl.training.ppo import PPO_ALGORITHM

    return PPO_ALGORITHM


TrackFactory = Callable[[RunSpec], AbstractContextManager[TrackedRun]]


def train_one(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    *,
    reward: str,
    phase: Phase,
    algorithm: Literal["dqn", "ppo"],
    candidate_id: str,
    seed: int,
    traces: Mapping[str, WorkloadTrace],
    track: TrackFactory,
    timesteps: int | None = None,
) -> TrainingEvidence:
    """Train one configuration under one reward and validate it on every workload."""
    from scalerl.training.common import (
        TrainingSettings,
        require_training_workload,
        require_validation_workloads,
        train_and_validate,
    )

    weights = reward_weights(reward)
    config = final_config()
    candidate = candidates.for_algorithm(algorithm).get(candidate_id)
    settings = TrainingSettings(
        timesteps=timesteps or TIMESTEPS[algorithm],
        seed=seed,
        config=config,
        config_source="predeclared",
        calibration_workload_ids=(),
        calibration_note=None,
        reward_weights=weights,
    )
    training_entry = require_training_workload(TRAIN_WORKLOAD, "#20")
    validation_entries = require_validation_workloads(VALIDATION_WORKLOADS, "#20")
    tags = {
        "scalerl.reward_ablation_version": spec.experiment_version,
        "scalerl.experiment_id": spec.experiment_id,
        "scalerl.reward_variant": reward,
        **{f"scalerl.reward_weight.{k}": repr(v) for k, v in weights.model_dump().items()},
        "scalerl.experiment_phase": phase,
        "scalerl.algorithm": algorithm,
        "scalerl.candidate_set_id": spec.candidate_set_id,
        "scalerl.candidate_id": candidate_id,
        "scalerl.training_seed": str(seed),
        "scalerl.evaluation_seed": str(EVALUATION_SEED),
        "scalerl.selection_version": spec.selection_version,
        "scalerl.selection_spec_id": spec.selection_spec_id,
    }
    kinds: tuple[Any, Any] = ("tune", "tune") if phase == "dqn-screening" else ("train", "evaluate")
    outcome = train_and_validate(
        _algorithm_spec(algorithm),
        training_entry=training_entry,
        validation_entries=validation_entries,
        traces=traces,
        hyperparameters=candidates.hyperparameters(algorithm, candidate_id),
        settings=settings,
        track=track,
        training_run_kind=kinds[0],
        validation_run_kind=kinds[1],
        extra_params={
            "hyperparameter_source": (
                f"{spec.candidate_set_id}#{candidate_id}; reward {reward} ({spec.experiment_id})"
            )
        },
        tags=tags,
    )
    return TrainingEvidence(
        experiment_id=spec.experiment_id,
        reward_variant=reward,
        reward_weights=weights.model_dump(),
        phase=phase,
        algorithm=algorithm,
        candidate_id=candidate_id,
        candidate_index=candidate.index,
        training_seed=seed,
        hyperparameters=dict(candidate.hyperparameters),
        training_run_id=outcome.training_run_id,
        model_artifact_uri=f"runs:/{outcome.training_run_id}/model",
        compatibility=outcome.compatibility.model_dump(mode="json"),
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
    )


def _tracker(tracking_uri: str | None, experiment_name: str) -> TrackFactory:
    from scalerl.mlops import start_tracked_run

    def track(run_spec: RunSpec) -> AbstractContextManager[TrackedRun]:
        return start_tracked_run(
            run_spec, tracking_uri=tracking_uri, experiment_name=experiment_name
        )

    return track


def _get_or_train(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    out: Path,
    *,
    reward: str,
    phase: Phase,
    algorithm: Literal["dqn", "ppo"],
    candidate_id: str,
    seed: int,
    traces: Mapping[str, WorkloadTrace],
    track: TrackFactory,
    timesteps: int | None,
    progress: Callable[[str], None],
) -> TrainingEvidence:
    path = evidence_path(out, reward, phase, candidate_id, seed)
    if path.exists():
        return _load_evidence(path, spec, reward, candidate_id, seed)
    evidence = train_one(
        spec, candidates, reward=reward, phase=phase, algorithm=algorithm,
        candidate_id=candidate_id, seed=seed, traces=traces, track=track, timesteps=timesteps,
    )  # fmt: skip
    evidence.save(path)
    progress(f"{reward} {phase} {candidate_id} seed{seed} -> {evidence.training_run_id}")
    return evidence


def dqn_candidates(evidence: Sequence[TrainingEvidence], family: str) -> list[Candidate]:
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
                "validation_run_ids": {w.workload_id: w.validation_run_id for w in item.workloads},
            },
        )
        for item in evidence
    ]


def run_reward(
    spec: ExperimentSpec,
    candidates: CandidateSet,
    selection: SelectionSpec,
    reward: str,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    traces: Mapping[str, WorkloadTrace] | None = None,
    timesteps: Mapping[str, int] | None = None,
    dqn_candidate_ids: Sequence[str] | None = None,
    progress: Callable[[str], None] = print,
) -> SelectionResult:
    """All training for one reward variant (resumable): DQN screen, #78, DQN retrain, PPO."""
    reward_weights(reward)
    if traces is None:
        manifest = load_benchmark_manifest()
        traces = build_workloads(
            [manifest.get(w) for w in (*spec.training_workload_ids, *spec.validation_workload_ids)]
        )
    track = _tracker(tracking_uri, experiment_name)
    budget = dict(timesteps or {})
    common = {"traces": traces, "track": track, "progress": progress}
    ids = list(dqn_candidate_ids or [c.candidate_id for c in candidates.dqn.candidates])
    screened = [
        _get_or_train(
            spec,
            candidates,
            out,
            reward=reward,
            phase="dqn-screening",
            algorithm="dqn",
            candidate_id=cid,
            seed=SCREENING_SEED,
            timesteps=budget.get("dqn"),
            **common,  # type: ignore[arg-type]
        )  # fmt: skip
        for cid in ids
    ]
    result = model_selection.select(
        selection, f"dqn-{reward}", dqn_candidates(screened, f"dqn-{reward}")
    )
    result.save(out / reward / "dqn-selection.json")
    if result.selection_succeeded and result.selected_candidate_id is not None:
        for seed in SEEDS:
            _get_or_train(
                spec, candidates, out, reward=reward, phase="dqn-retraining", algorithm="dqn",
                candidate_id=result.selected_candidate_id, seed=seed,
                timesteps=budget.get("dqn"), **common,  # type: ignore[arg-type]
            )  # fmt: skip
    for seed in SEEDS:
        _get_or_train(
            spec, candidates, out, reward=reward, phase="ppo-training", algorithm="ppo",
            candidate_id=PPO_CANDIDATE, seed=seed, timesteps=budget.get("ppo"), **common,  # type: ignore[arg-type]
        )  # fmt: skip
    return result


# --- reward-level decision -----------------------------------------------------------------------


def _equal_seed_means(models: Sequence[TrainingEvidence]) -> dict[str, dict[str, float]]:
    """``{workload: {metric: mean over seeds}}`` for one algorithm x reward."""
    return {
        workload: {
            metric: statistics.fmean(m.metrics_for(workload).metrics[metric] for m in models)
            for metric in METRIC_KEYS
        }
        for workload in VALIDATION_WORKLOADS
    }


def _passes(
    means: Mapping[str, Mapping[str, float]], thresholds: Mapping[str, float]
) -> dict[str, bool]:
    tolerance = model_selection.FEASIBILITY_TOLERANCE
    return {
        workload: means[workload]["sla_violation_rate"] <= thresholds[workload] + tolerance
        for workload in VALIDATION_WORKLOADS
    }


def _equal_workload(means: Mapping[str, Mapping[str, float]], metric: str) -> float:
    return statistics.fmean(means[w][metric] for w in VALIDATION_WORKLOADS)


def load_reward_models(
    spec: ExperimentSpec, out: Path, reward: str, dqn_ids: Sequence[str] | None = None
) -> tuple[SelectionResult, list[TrainingEvidence], list[TrainingEvidence], list[TrainingEvidence]]:
    """(DQN selection, DQN screening, DQN retrained, PPO) of one reward; exact matrix only."""
    selection = SelectionResult.model_validate_json(
        (out / reward / "dqn-selection.json").read_text()
    )
    screening = [
        _load_evidence(
            evidence_path(out, reward, "dqn-screening", cid, SCREENING_SEED),
            spec,
            reward,
            cid,
            SCREENING_SEED,
        )
        for cid in (dqn_ids if dqn_ids is not None else dqn_candidate_ids(spec))
    ]
    dqn: list[TrainingEvidence] = []
    if selection.selection_succeeded and selection.selected_candidate_id is not None:
        cid = selection.selected_candidate_id
        dqn = [
            _load_evidence(
                evidence_path(out, reward, "dqn-retraining", cid, s), spec, reward, cid, s
            )
            for s in SEEDS
        ]
    ppo = [
        _load_evidence(
            evidence_path(out, reward, "ppo-training", PPO_CANDIDATE, s),
            spec,
            reward,
            PPO_CANDIDATE,
            s,
        )
        for s in SEEDS
    ]
    unexpected = {p for p in (out / reward).glob("*/*.json")} - {
        evidence_path(out, reward, m.phase, m.candidate_id, m.training_seed)
        for m in (*screening, *dqn, *ppo)
    }
    if unexpected:
        raise ValueError(f"unexpected evidence files for {reward}: {sorted(map(str, unexpected))}")
    return selection, screening, dqn, ppo


def decide_reward(
    per_reward: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """Apply the predeclared reward-level rule to equal-seed means (pure; testable).

    ``per_reward[reward] = {"dqn_selected": bool, "dqn": means | None, "ppo": means}``
    where ``means`` is ``{workload: {metric: equal-seed mean}}``.
    """
    rows = {}
    for reward in sorted(per_reward):
        entry = per_reward[reward]
        ppo_pass = _passes(entry["ppo"], thresholds)
        dqn_pass = _passes(entry["dqn"], thresholds) if entry["dqn"] is not None else None
        joint = bool(
            entry["dqn_selected"]
            and dqn_pass is not None
            and all(dqn_pass.values())
            and all(ppo_pass.values())
        )
        row: dict[str, Any] = {
            "dqn_selected": entry["dqn_selected"],
            "ppo_sla_passes": ppo_pass,
            "dqn_sla_passes": dqn_pass,
            "joint_feasible": joint,
            "ppo_equal_workload": {m: _equal_workload(entry["ppo"], m) for m in DECISION_METRICS},
            "dqn_equal_workload": (
                {m: _equal_workload(entry["dqn"], m) for m in DECISION_METRICS}
                if entry["dqn"] is not None
                else None
            ),
        }
        if joint:
            row["equal_algorithm"] = {
                m: (row["ppo_equal_workload"][m] + row["dqn_equal_workload"][m]) / 2
                for m in DECISION_METRICS
            }
        rows[reward] = row
    joint_rewards = [r for r, row in rows.items() if row["joint_feasible"]]
    if joint_rewards:
        chosen = min(
            joint_rewards,
            key=lambda r: (*(rows[r]["equal_algorithm"][m] for m in DECISION_METRICS), r),
        )
        basis = "minimum equal-algorithm mean normalized cost among joint-feasible rewards"
    else:
        chosen, basis = FALLBACK_REWARD, "no joint-feasible reward: predeclared fallback"
    return {
        "per_reward": rows,
        "joint_feasible_rewards": joint_rewards,
        "selected_reward": chosen,
        "decision_basis": basis,
        "fallback_invoked": not joint_rewards,
    }


def _summaries(models: Sequence[TrainingEvidence]) -> dict[str, dict[str, Any]]:
    """n / mean / median / SD / min / max per workload x metric over training seeds."""
    out: dict[str, dict[str, Any]] = {}
    for workload in VALIDATION_WORKLOADS:
        out[workload] = {}
        for metric in (*METRIC_KEYS, *(f"action.{k}" for k in ACTION_KEYS)):
            values = [
                m.metrics_for(workload).metrics[metric]
                if not metric.startswith("action.")
                else m.metrics_for(workload).action[metric.removeprefix("action.")]
                for m in models
            ]
            stats = describe(values)
            out[workload][metric] = {
                "n": stats.n, "mean": stats.mean, "median": stats.median, "std": stats.std,
                "min": stats.minimum, "max": stats.maximum,
            }  # fmt: skip
    return out


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if statistics.pstdev(rx) == 0 or statistics.pstdev(ry) == 0:
        return None
    return float(statistics.correlation(rx, ry))


def pathology_flags(
    means: Mapping[str, Mapping[str, float]] | None, thresholds: Mapping[str, float]
) -> dict[str, list[str]]:
    """Predeclared diagnostic flags per workload (never used for selection)."""
    if means is None:
        return {}
    flags: dict[str, list[str]] = {}
    for workload in VALIDATION_WORKLOADS:
        m = means[workload]
        fails = (
            m["sla_violation_rate"] > thresholds[workload] + model_selection.FEASIBILITY_TOLERANCE
        )
        found = []
        if m["normalized_cost"] >= 0.9:
            found.append("full_fleet")
        if fails and m["normalized_cost"] < 0.6:
            found.append("underprovisioning")
        if m["churn_rate"] >= 0.3:
            found.append("thrashing")
        if fails and m["scaling_actions"] <= 2:
            found.append("churn_aversion")
        if m["queue_pressure"] >= 0.2:
            found.append("persistent_backlog")
        flags[workload] = found
    return flags


def reward_mismatch(
    screening: Sequence[TrainingEvidence], selection: SelectionResult
) -> dict[str, Any]:
    """Does episode reward point at the #78-preferred behavior among DQN candidates?"""
    reward_mean = {
        e.candidate_id: statistics.fmean(w.metrics["episode_reward"] for w in e.workloads)
        for e in screening
    }
    sla_mean = {
        e.candidate_id: statistics.fmean(w.metrics["sla_violation_rate"] for w in e.workloads)
        for e in screening
    }
    best = max(sorted(reward_mean), key=lambda c: reward_mean[c])
    ids = sorted(reward_mean)
    rho = _spearman([reward_mean[c] for c in ids], [sla_mean[c] for c in ids])
    return {
        "highest_episode_reward_candidate": best,
        "highest_episode_reward_is_feasible": best in selection.feasible_candidate_ids,
        "spearman_episode_reward_vs_mean_sla": rho,
        "flag": best not in selection.feasible_candidate_ids or (rho is not None and rho > 0),
    }


def build_report(spec: ExperimentSpec, out: Path) -> dict[str, Any]:
    thresholds = dict(spec.sla_thresholds)
    per_reward: dict[str, dict[str, Any]] = {}
    details: dict[str, Any] = {}
    for reward in spec.reward_variants:
        selection, screening, dqn, ppo = load_reward_models(spec, out, reward)
        dqn_means = _equal_seed_means(dqn) if dqn else None
        ppo_means = _equal_seed_means(ppo)
        per_reward[reward] = {
            "dqn_selected": selection.selection_succeeded,
            "dqn": dqn_means,
            "ppo": ppo_means,
        }
        fallback = selection.diagnostic_fallback
        details[reward] = {
            "weights": spec.reward_variants[reward],
            "dqn_selection": {
                "candidate_count": len(selection.candidate_ids),
                "feasible_candidate_count": selection.feasible_candidate_count,
                "feasible_candidate_ids": list(selection.feasible_candidate_ids),
                "selected_candidate_id": selection.selected_candidate_id,
                "selected_objective_values": selection.selected_objective_values,
                "diagnostic_fallback_candidate_id": selection.diagnostic_fallback_candidate_id,
                "diagnostic_fallback_sla_excess": (
                    {w.workload_id: w.sla_excess for w in fallback.workloads} if fallback else None
                ),
                "screening_training_run_ids": {
                    e.candidate_id: e.training_run_id for e in screening
                },
            },
            "dqn_models": [_model_ref(m) for m in dqn],
            "ppo_models": [_model_ref(m) for m in ppo],
            "dqn_equal_seed_means": dqn_means,
            "ppo_equal_seed_means": ppo_means,
            "dqn_summary": _summaries(dqn) if dqn else None,
            "ppo_summary": _summaries(ppo),
            "pathology": {
                "dqn": pathology_flags(dqn_means, thresholds),
                "ppo": pathology_flags(ppo_means, thresholds),
                "dqn_screening_reward_mismatch": reward_mismatch(screening, selection),
            },
        }
    decision = decide_reward(per_reward, thresholds)
    return {"experiment_id": spec.experiment_id, "decision": decision, "rewards": details}


def _model_ref(model: TrainingEvidence) -> dict[str, JsonValue]:
    return {
        "candidate_id": model.candidate_id,
        "training_seed": model.training_seed,
        "training_run_id": model.training_run_id,
        "model_artifact_uri": model.model_artifact_uri,
        "validation_run_ids": {w.workload_id: w.validation_run_id for w in model.workloads},
    }


class RewardContract(_Strict):
    """The frozen reward for #72/#46 and the evidence behind it."""

    contract_version: Literal["reward-contract-v1"] = CONTRACT_VERSION
    experiment_version: Literal["reward-ablation-v1"] = ABLATION_VERSION
    experiment_spec_id: str
    selected_reward_variant: str
    selected_weights: dict[str, float]
    reward_changed_from_default: bool
    default_weights: dict[str, float]
    reward_variants: dict[str, dict[str, float]]
    reward_semantics: str
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    selection_version: str
    selection_spec_id: str
    training_workload_ids: tuple[str, ...]
    validation_workload_ids: tuple[str, ...]
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    ppo_candidate_id: str
    decision: dict[str, JsonValue]
    per_reward: dict[str, JsonValue]
    mlflow_experiment: str
    run_provenance: dict[str, JsonValue]
    held_out_data_used: Literal[False] = False
    declares_algorithm_winner: Literal[False] = False

    @property
    def contract_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def build_contract(
    spec: ExperimentSpec, report: Mapping[str, Any], provenance: Mapping[str, JsonValue]
) -> RewardContract:
    decision = report["decision"]
    chosen = decision["selected_reward"]
    compact = {
        reward: {k: v for k, v in detail.items() if k not in ("dqn_summary", "ppo_summary")}
        for reward, detail in report["rewards"].items()
    }
    return RewardContract(
        experiment_spec_id=spec.experiment_id,
        selected_reward_variant=chosen,
        selected_weights=spec.reward_variants[chosen],
        reward_changed_from_default=spec.reward_variants[chosen] != RewardWeights().model_dump(),
        default_weights=RewardWeights().model_dump(),
        reward_variants=spec.reward_variants,
        reward_semantics=spec.reward_semantics,
        selection_version=spec.selection_version,
        selection_spec_id=spec.selection_spec_id,
        training_workload_ids=spec.training_workload_ids,
        validation_workload_ids=spec.validation_workload_ids,
        training_seeds=spec.training_seeds,
        evaluation_seed=spec.evaluation_seed,
        ppo_candidate_id=PPO_CANDIDATE,
        decision=_json(decision),
        per_reward=_json(compact),
        mlflow_experiment=EXPERIMENT_NAME,
        run_provenance=dict(provenance),
    )


def run_provenance(
    tracking_uri: str | None, experiment_name: str, spec: ExperimentSpec
) -> dict[str, JsonValue]:
    """Status / git state of every run of record of this spec (from MLflow)."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {"runs": 0}
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.`scalerl.experiment_id` = '{spec.experiment_id}'",
        max_results=50_000,
    )
    statuses: dict[str, JsonValue] = {}
    shas: dict[str, JsonValue] = {}
    dirty: dict[str, JsonValue] = {}
    for run in runs:
        for counts, key in (
            (statuses, run.info.status),
            (shas, run.data.tags.get("scalerl.git_sha", "unknown")),
            (dirty, run.data.tags.get("scalerl.git_dirty", "unknown")),
        ):
            counts[key] = int(str(counts.get(key, 0))) + 1
    return {"runs": len(runs), "status": statuses, "git_sha": shas, "git_dirty": dirty}


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, default=_nan_safe))


def _nan_safe(value: Any) -> Any:
    raise TypeError(f"not JSON serializable: {value!r}")


def _write_text(path: Path, content: str) -> Path:
    """Atomic write; the temporary name is per process, so parallel reward runs never race."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


# --- command line --------------------------------------------------------------------------------


def describe_plan(spec: ExperimentSpec) -> str:
    return "\n".join(
        [
            f"experiment          {spec.experiment_version} {spec.experiment_id}",
            f"action semantics    {spec.action_semantics} ({spec.action_decision_version}, "
            f"#79 {spec.action_experiment_spec_id})",
            f"selection           {spec.selection_version} {spec.selection_spec_id}",
            f"reward variants     {len(spec.reward_variants)}: {', '.join(spec.reward_variants)}",
            f"training workloads  {', '.join(spec.training_workload_ids)}",
            f"validation          {', '.join(spec.validation_workload_ids)}",
            f"test workloads      {len(spec.test_workload_ids)}",
            f"PPO                 {spec.ppo['candidate_id']} fixed, "
            f"seeds {list(spec.training_seeds)}, "
            f"{spec.ppo['timesteps']} steps",
            f"DQN                 {len(spec.dqn['candidate_ids'])} #79 candidates "  # type: ignore[arg-type]
            f"({spec.candidate_set_id}), {spec.dqn['timesteps']} steps, seed "
            f"{spec.dqn['screening_seed']}; retrain seeds {list(spec.training_seeds)} if selected",
            f"fallback            {spec.no_joint_feasible_fallback}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{ABLATION_VERSION} (#20), validation only.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-spec", "check", "run", "decide"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
        command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
        if name == "run":
            command.add_argument("--reward", required=True, choices=list(REWARD_VARIANTS))
            command.add_argument("--torch-threads", type=int, default=None)
        if name == "decide":
            command.add_argument("--freeze", action="store_true")
            command.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)

    if args.command == "freeze-spec":
        _, candidates, selection = verify_upstream(
            DEFAULT_ACTION_DECISION, DEFAULT_ACTION_SPEC, DEFAULT_CANDIDATES, DEFAULT_SELECTION_SPEC
        )
        spec = build_experiment_spec(candidates, selection)
        spec.save(args.spec)
        print(f"{spec.experiment_version} {spec.experiment_id} written to {args.spec}")
        return 0
    try:
        spec, candidates, selection = load_frozen(args.spec)
    except ValueError as error:
        parser.error(str(error))
    print(describe_plan(spec))
    if args.command == "check":
        return 0
    out: Path = args.output_dir
    if args.command == "run":
        if args.torch_threads:
            import torch

            torch.set_num_threads(args.torch_threads)
        spec.save(out / "experiment-spec.json")
        result = run_reward(
            spec, candidates, selection, args.reward, out=out,
            tracking_uri=args.tracking_uri, experiment_name=args.experiment_name,
        )  # fmt: skip
        chosen = result.selected_candidate_id or (
            f"none (diagnostic only: {result.diagnostic_fallback_candidate_id})"
        )
        print(
            f"{args.reward}: DQN {result.feasible_candidate_count}/20 feasible; selected {chosen}"
        )
        return 0
    report = build_report(spec, out)
    _write_text(
        out / "decision-report.json", json.dumps(_json(report), indent=2, sort_keys=True) + "\n"
    )
    decision = report["decision"]
    print(
        f"joint-feasible: {decision['joint_feasible_rewards']}; "
        f"selected {decision['selected_reward']}"
    )
    if args.freeze:
        provenance = run_provenance(args.tracking_uri, args.experiment_name, spec)
        contract = build_contract(spec, report, provenance)
        _write_text(args.contract_output, contract.model_dump_json(indent=2) + "\n")
        print(f"{contract.contract_version} {contract.contract_id} -> {args.contract_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
