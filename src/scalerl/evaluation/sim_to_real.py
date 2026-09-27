"""Frozen sim-to-real validation protocol ``sim-to-real-protocol-v1`` (#72).

Local systems-in-the-loop / Knative sim-to-real validation: before any live
(Knative) run and before #46 held-out results are interpreted, this module
freezes *what* will later run and *how* transfer will be reported:

* **Controllers** (``controller-deployment-manifest-v1``): tuned Threshold,
  ``predictive-v1``, ``predictive-seasonal-v1``, the #20 DQN (``dqn-c14``) and
  PPO (``ppo-c08``) under ``desired-replicas-v1`` + ``reward-contract-v1``, and
  Knative's native autoscaler as a real-system baseline. One deployable
  artifact per learned family is chosen by a predeclared, validation-only rule
  over the five #20 seeds (:func:`select_canonical_seed`).
* **Replay windows** (``live-replay-manifest-v1``): three 10-minute slices of
  the held-out Azure test workload at 20% / 50% / 80% of the usable start
  range, chosen by rule, never by traffic or controller outcome.
* **Arrival schedules** (``live-arrival-schedule-v1``): per slice and load seed
  0/1/2, exact request timestamps (the trace's per-bin counts, uniformly placed
  within each 30 s bin by a domain-separated RNG); identical for every
  controller.
* **Protocol** (``sim-to-real-protocol-v1``): control contract, the shared
  ``scalerl-observation-v1`` mapping, cost semantics, metrics and the
  descriptive transfer-reporting rule (no pass/fail, no "RL must win").

Nothing here runs a controller on held-out or live data.

    python -m scalerl.evaluation.sim_to_real freeze --azure-csv data/raw/<trace>.txt
    python -m scalerl.evaluation.sim_to_real check [--azure-csv ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from scalerl.benchmarks import AzureWorkload, build_workload, load_benchmark_manifest
from scalerl.environment import DESIRED_REPLICAS_V1, SimulatorConfig
from scalerl.environment.observation import OBSERVATION_VERSION, ObservationConstants, feature_names
from scalerl.evaluation.action_semantics import ActionContractDecision, contract_config
from scalerl.evaluation.model_selection import FEASIBILITY_TOLERANCE, SelectionSpec
from scalerl.workloads import WorkloadTrace

PROTOCOL_VERSION: Final = "sim-to-real-protocol-v1"
REPLAY_VERSION: Final = "live-replay-manifest-v1"
CONTROLLER_MANIFEST_VERSION: Final = "controller-deployment-manifest-v1"
SCHEDULE_VERSION: Final = "live-arrival-schedule-v1"
SCHEDULE_GENERATOR: Final = "binned-uniform-v1"
WINDOW_RULE_VERSION: Final = "fixed-fraction-windows-v1"
CANONICAL_SEED_RULE: Final = "canonical-seed-validation-v1"
WINDOW_SECONDS: Final = 600.0
WINDOW_FRACTIONS: Final = (0.2, 0.5, 0.8)
LOAD_SEEDS: Final = (0, 1, 2)
CONTROL_INTERVAL_SECONDS: Final = 30.0
# Fixed spawn key separating arrival-schedule randomness from every other stream
# (the ASCII bytes of "arrv"); changing it requires a new schedule version.
SCHEDULE_RNG_DOMAIN: Final = 0x61727276
TIMESTAMP_DECIMALS: Final = 6

UPSTREAM: Final = {
    "selection_spec_id": "418876d6c8e9",
    "action_decision_id": "0eb5562b01e6",
    "action_experiment_spec_id": "899dfbb64217",
    "predictive_baseline_artifact_id": "0d315559c680",
    "predictive_baseline_experiment_id": "096b04ef953a",
    "startup_robustness_spec_id": "60f1f14972a7",
    "startup_robustness_freeze_id": "25d65bbba36f",
    "reward_ablation_spec_id": "f5aff8f1e8c2",
    "reward_contract_id": "37158c261364",
}
SELECTED_REWARD: Final = "full-cost-low-v1"
DQN_CANDIDATE: Final = "dqn-c14"
PPO_CANDIDATE: Final = "ppo-c08"
PREDICTIVE_FROZEN_IN: Final = "predictive-baseline-v1 (0d315559c680)"
RULE_METRICS: Final = ("normalized_cost", "queue_pressure", "churn_rate", "sla_violation_rate")

BENCH = Path("benchmarks/v1")
DEFAULT_PROTOCOL = BENCH / "sim-to-real-protocol-v1.json"
DEFAULT_REPLAY = BENCH / "live-replay-manifest-v1.json"
DEFAULT_CONTROLLERS = BENCH / "controller-deployment-manifest-v1.json"
DEFAULT_SCHEDULES = BENCH / "live-replay-schedules-v1"
DEFAULT_REWARD_OUTPUTS = Path("outputs/reward-ablation-v1")
DEFAULT_AZURE_CSV = Path("data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")


def content_id(payload: Any) -> str:
    """Deterministic content ID: SHA-256 of canonical JSON, first 12 hex digits."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:12]


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


# --- upstream verification ------------------------------------------------------------------------


def verify_upstream(bench: Path = BENCH) -> dict[str, Any]:
    """Load every frozen upstream artifact and check its ID; returns the loaded objects."""
    from scalerl.evaluation.predictive_baseline import PredictiveBaselineArtifact
    from scalerl.evaluation.reward_ablation import ExperimentSpec as RewardSpec
    from scalerl.evaluation.reward_ablation import RewardContract
    from scalerl.evaluation.robustness import ROBUSTNESS_SCENARIO_VERSION, ROBUSTNESS_SCENARIOS
    from scalerl.evaluation.startup_robustness import ExperimentSpec as StartupSpec
    from scalerl.evaluation.startup_robustness import StartupRobustnessFreeze

    selection = SelectionSpec.load(bench / "selection-v2-cost-under-sla.json")
    decision = ActionContractDecision.model_validate_json(
        (bench / "action-contract-v2.json").read_text()
    )
    predictive = PredictiveBaselineArtifact.model_validate_json(
        (bench / "predictive-baseline-v1.json").read_text()
    )
    startup_spec = StartupSpec.load(bench / "startup-robustness-v1.json")
    startup_freeze = StartupRobustnessFreeze.model_validate_json(
        (bench / "startup-robustness-freeze-v1.json").read_text()
    )
    reward_spec = RewardSpec.load(bench / "reward-ablation-v1.json")
    contract = RewardContract.model_validate_json((bench / "reward-contract-v1.json").read_text())
    found = {
        "selection_spec_id": selection.spec_id,
        "action_decision_id": decision.decision_id,
        "action_experiment_spec_id": decision.experiment_spec_id,
        "predictive_baseline_artifact_id": predictive.artifact_id,
        "predictive_baseline_experiment_id": predictive.experiment_spec_id,
        "startup_robustness_spec_id": startup_spec.experiment_id,
        "startup_robustness_freeze_id": startup_freeze.freeze_id,
        "reward_ablation_spec_id": reward_spec.experiment_id,
        "reward_contract_id": contract.contract_id,
    }
    if found != UPSTREAM:
        drift = {k: (found[k], UPSTREAM[k]) for k in UPSTREAM if found[k] != UPSTREAM[k]}
        raise ValueError(f"frozen upstream artifacts changed: {drift}")
    if (
        decision.final_action_semantics != DESIRED_REPLICAS_V1
        or contract.selected_reward_variant != SELECTED_REWARD
        or contract.action_semantics != DESIRED_REPLICAS_V1
    ):
        raise ValueError("the #79/#20 decisions are not the ones #72 freezes against")
    if ROBUSTNESS_SCENARIO_VERSION != "robustness-v1" or list(ROBUSTNESS_SCENARIOS) != [
        "nominal",
        "capacity-jitter",
        "delayed-telemetry",
        "combined-robustness",
    ]:
        raise ValueError("robustness-v1 changed")
    return {
        "selection": selection,
        "decision": decision,
        "predictive": predictive,
        "reward_spec": reward_spec,
        "contract": contract,
    }


# --- canonical learned artifact per family -------------------------------------------------------


class SeedEvidence(_Strict):
    """Validation evidence of one retrained #20 seed (per validation workload)."""

    training_seed: int
    training_run_id: str
    model_artifact_uri: str
    validation_run_ids: dict[str, str]
    metrics: dict[str, dict[str, float]]  # workload -> {sla, cost, queue, churn}


def select_canonical_seed(
    seeds: Sequence[SeedEvidence], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    """``canonical-seed-validation-v1``: one deployable artifact from validation evidence only.

    Feasible seeds (SLA within the Threshold limit on every workload) are ranked
    by equal-workload mean normalized cost, queue pressure, churn, SLA, then
    the lowest training seed. With no feasible seed, the seed with the smallest
    maximum positive SLA excess is chosen (then cost, queue, churn, SLA, seed)
    and marked infeasible. This picks the artifact deployment needs; it does
    not represent algorithm performance (that is the five-seed result).
    """
    if not seeds:
        raise ValueError("no seed evidence")
    workloads = list(thresholds)

    def mean(seed: SeedEvidence, metric: str) -> float:
        return statistics.fmean(seed.metrics[w][metric] for w in workloads)

    def excess(seed: SeedEvidence) -> float:
        return max(0.0, *(seed.metrics[w]["sla_violation_rate"] - thresholds[w] for w in workloads))

    feasible = {
        s.training_seed: all(
            s.metrics[w]["sla_violation_rate"] <= thresholds[w] + FEASIBILITY_TOLERANCE
            for w in workloads
        )
        for s in seeds
    }
    candidates = [s for s in seeds if feasible[s.training_seed]]
    if candidates:
        chosen = min(
            candidates,
            key=lambda s: (*(mean(s, m) for m in RULE_METRICS), s.training_seed),
        )
    else:
        chosen = min(
            seeds,
            key=lambda s: (excess(s), *(mean(s, m) for m in RULE_METRICS), s.training_seed),
        )
    return {
        "rule": CANONICAL_SEED_RULE,
        "canonical_training_seed": chosen.training_seed,
        "canonical_training_run_id": chosen.training_run_id,
        "canonical_model_artifact_uri": chosen.model_artifact_uri,
        "canonical_seed_validation_feasible": feasible[chosen.training_seed],
        "individually_feasible_seeds": sorted(k for k, v in feasible.items() if v),
        "equal_workload_means": {
            str(s.training_seed): {m: mean(s, m) for m in RULE_METRICS} for s in seeds
        },
        "max_positive_sla_excess": {str(s.training_seed): excess(s) for s in seeds},
    }


def seed_evidence_from_outputs(
    outputs: Path, phase: str, candidate: str, contract: Any, family: str
) -> list[SeedEvidence]:
    """Per-seed evidence from the local #20 outputs, verified against reward-contract-v1.

    The run IDs must be the contract's, and the equal-seed means must equal the
    contract's recorded means exactly, so the committed evidence is exactly
    the evidence #20 froze.
    """
    detail = contract.per_reward[SELECTED_REWARD]
    refs = detail[f"{family}_models"]
    means = detail[f"{family}_equal_seed_means"]
    evidence = []
    for ref in refs:
        seed = int(ref["training_seed"])
        data = json.loads(
            (outputs / SELECTED_REWARD / phase / f"{candidate}-seed{seed}.json").read_text()
        )
        if data["training_run_id"] != ref["training_run_id"] or data["candidate_id"] != candidate:
            raise ValueError(f"{family} seed {seed}: evidence is not the reward-contract model")
        evidence.append(
            SeedEvidence(
                training_seed=seed,
                training_run_id=data["training_run_id"],
                model_artifact_uri=data["model_artifact_uri"],
                validation_run_ids={
                    w["workload_id"]: w["validation_run_id"] for w in data["workloads"]
                },
                metrics={
                    w["workload_id"]: {m: float(w["metrics"][m]) for m in RULE_METRICS}
                    for w in data["workloads"]
                },
            )
        )
    check_against_contract(evidence, means)
    return evidence


def check_against_contract(
    evidence: Sequence[SeedEvidence], means: Mapping[str, Mapping[str, float]]
) -> None:
    for workload, values in means.items():
        for metric in RULE_METRICS:
            recomputed = statistics.fmean(s.metrics[workload][metric] for s in evidence)
            if recomputed != values[metric]:
                raise ValueError(
                    f"seed evidence does not reproduce reward-contract-v1 {workload}/{metric}"
                )


# --- controller deployment manifest ---------------------------------------------------------------


def build_controller_manifest(upstream: Mapping[str, Any], outputs: Path) -> dict[str, Any]:
    from scalerl.controllers import predictive as predictive_v1
    from scalerl.controllers import proactive_predictive as seasonal
    from scalerl.tuning.candidates import CandidateSet

    selection: SelectionSpec = upstream["selection"]
    contract = upstream["contract"]
    predictive = upstream["predictive"]
    thresholds = {t.workload_id: t.sla_violation_rate for t in selection.sla_thresholds}
    candidates = CandidateSet.load(BENCH / "action-semantics-candidates-v1.json")
    reward = {
        "reward_contract": "reward-contract-v1",
        "reward_contract_id": UPSTREAM["reward_contract_id"],
        "reward_variant": SELECTED_REWARD,
        "weights": dict(contract.selected_weights),
    }
    learned = {}
    for family, phase, candidate, algorithm in (
        ("dqn", "dqn-retraining", DQN_CANDIDATE, "dqn"),
        ("ppo", "ppo-training", PPO_CANDIDATE, "ppo"),
    ):
        seeds = seed_evidence_from_outputs(outputs, phase, candidate, contract, family)
        learned[family] = {
            "controller": f"{family}-desired-replicas-full-cost-low-v1",
            "algorithm": algorithm,
            "source_issue": "#20",
            "candidate_id": candidate,
            "candidate_set_id": candidates.candidate_set_id,
            "hyperparameters": dict(
                candidates.for_algorithm(algorithm).get(candidate).hyperparameters
            ),
            "action_semantics": DESIRED_REPLICAS_V1,
            **reward,
            "canonical_selection": select_canonical_seed(seeds, thresholds),
            "five_seed_lineage": [s.model_dump(mode="json") for s in seeds],
            "five_seed_result_note": (
                "the #20 five-seed equal-seed means remain the algorithm evidence; the "
                "canonical seed is only the single artifact deployment requires"
            ),
            "inference": "deterministic (greedy/mode action), no exploration, no online training",
        }
    return {
        "manifest_version": CONTROLLER_MANIFEST_VERSION,
        "source_issue": "#72",
        "upstream": dict(UPSTREAM),
        "action_semantics": DESIRED_REPLICAS_V1,
        "sla_thresholds": thresholds,
        "controllers": {
            "threshold-v1": {
                "controller_version": "threshold-sla-first-v1",
                "source_issue": "#13/#19 (tuned; #78 reference)",
                "params": dict(selection.reference.params),
                "tuning_lineage": dict(selection.reference.tuning_lineage),
                "action_semantics": DESIRED_REPLICAS_V1,
                "encoding": "same ±1/hold law encoded as a desired-replicas-v1 target",
                "reward": "not applicable (rule-based)",
            },
            "predictive-v1": {
                "forecast_method": predictive_v1.FORECAST_METHOD,
                "capacity_policy": predictive_v1.CAPACITY_POLICY,
                "source_issue": "#14/#63",
                "params": {"history_window_ticks": 4, "target_utilization": 0.8},
                "action_semantics": DESIRED_REPLICAS_V1,
                "encoding": "its computed desired replicas as the target",
                "frozen_in": PREDICTIVE_FROZEN_IN,
            },
            "predictive-seasonal-v1": {
                "forecast_method": seasonal.FORECAST_METHOD,
                "capacity_policy": seasonal.CAPACITY_POLICY,
                "source_issue": "#80",
                "params": dict(predictive.fixed_parameters),
                "historical_profile": dict(predictive.historical_profile),
                "live_profile_alignment": (
                    "the Azure TRAIN profile is indexed by tick position within the 12:00-13:00 "
                    "window; a replay slice starting at bin b reads profile ticks b, b+1, ..."
                ),
                "action_semantics": DESIRED_REPLICAS_V1,
                "frozen_in": PREDICTIVE_FROZEN_IN,
            },
            **learned,
            "knative-native-v1": {
                "kind": "real-system baseline (not a simulated or MLflow-trained artifact)",
                "source_issue": "#72 (deployed in #73/#75)",
                "autoscaler_class": "kpa.autoscaling.knative.dev",
                "metric": "concurrency",
                "settings": {
                    "min-scale": 1,
                    "max-scale": 10,
                    "scale-to-zero": "disabled for the controller comparison",
                    "all other autoscaler settings": (
                        "Knative Serving defaults of the version pinned by #73, recorded "
                        "verbatim there; not tuned against any result"
                    ),
                },
                "action_semantics": "native (not desired-replicas-v1)",
            },
        },
        "excluded": {
            "q-learning": "#47 is not complete; optional and not a prerequisite",
            "static/random": "simulator reference rows only; not part of the live matrix",
            "dqn diagnostic fallbacks": "never deployable (#79/#20)",
        },
    }


# --- replay windows and arrival schedules ---------------------------------------------------------


def canonical_test_workload() -> AzureWorkload:
    """The lexicographically smallest Azure TEST workload ID (a rule, not a performance choice)."""
    manifest = load_benchmark_manifest()
    tests = sorted(
        (w for w in manifest.workloads if w.split == "test" and isinstance(w, AzureWorkload)),
        key=lambda w: w.id,
    )
    if not tests:
        raise ValueError("no Azure test workload in the benchmark manifest")
    return tests[0]


def window_offsets(duration_seconds: float, interval_seconds: float) -> list[float]:
    """``fixed-fraction-windows-v1``: starts at 20/50/80% of the usable range, aligned down."""
    usable = duration_seconds - WINDOW_SECONDS
    if usable <= 0:
        raise ValueError("the trace is shorter than one replay window")
    starts = [
        math.floor(f * usable / interval_seconds) * interval_seconds for f in WINDOW_FRACTIONS
    ]
    for a, b in zip(starts, starts[1:], strict=False):
        if b < a + WINDOW_SECONDS:
            raise ValueError("replay windows would overlap")
    return starts


def slice_counts(trace: WorkloadTrace, start_bin: int, bins: int) -> list[int]:
    """Exact per-bin request counts of a slice (Azure rates are counts / interval)."""
    counts = []
    for rate in trace.request_rates[start_bin : start_bin + bins]:
        count = rate * trace.control_interval_seconds
        if not math.isclose(count, round(count), abs_tol=1e-6):
            raise ValueError("slice rates are not whole request counts per bin")
        counts.append(int(round(count)))
    return counts


def generate_schedule(
    counts: Sequence[int], interval_seconds: float, *, slice_index: int, load_seed: int
) -> list[float]:
    """``binned-uniform-v1``: each bin's exact count, uniformly placed within the bin.

    Timestamps are seconds from the slice start, sorted, rounded to microseconds.
    The RNG is ``SeedSequence(load_seed, spawn_key=(SCHEDULE_RNG_DOMAIN, slice_index))``:
    it depends only on the slice and load seed, never on any controller.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(load_seed, spawn_key=(SCHEDULE_RNG_DOMAIN, slice_index))
    )
    times: list[float] = []
    for index, count in enumerate(counts):
        offsets = rng.random(count)
        times.extend(round((index + u) * interval_seconds, TIMESTAMP_DECIMALS) for u in offsets)
    return sorted(times)


def build_replay(
    trace: WorkloadTrace, workload: AzureWorkload, csv_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    interval = trace.control_interval_seconds
    if interval != CONTROL_INTERVAL_SECONDS:
        raise ValueError("the replay trace must use the 30 s control interval")
    duration = len(trace) * interval
    bins = int(WINDOW_SECONDS / interval)
    source_fingerprint = _sha256(
        {"control_interval_seconds": interval, "request_rates": list(trace.request_rates)}
    )
    slices, schedules = [], {}
    for index, start in enumerate(window_offsets(duration, interval)):
        start_bin = int(start / interval)
        counts = slice_counts(trace, start_bin, bins)
        slice_id = f"{workload.id}-w{index}-s{int(start)}-d{int(WINDOW_SECONDS)}"
        entry: dict[str, Any] = {
            "slice_id": slice_id,
            "slice_index": index,
            "source_workload_id": workload.id,
            "split": "test",
            "source_trace_fingerprint": source_fingerprint,
            "start_offset_seconds": start,
            "duration_seconds": WINDOW_SECONDS,
            "start_bin": start_bin,
            "end_bin_exclusive": start_bin + bins,
            "control_interval_seconds": interval,
            "slice_checksum": _sha256({"counts_per_bin": counts, "interval": interval}),
            "request_count": sum(counts),
            "window_rule": WINDOW_RULE_VERSION,
            "schedules": [],
        }
        for seed in LOAD_SEEDS:
            times = generate_schedule(counts, interval, slice_index=index, load_seed=seed)
            schedule_id = f"{slice_id}-seed{seed}"
            payload = {
                "schedule_version": SCHEDULE_VERSION,
                "generator": SCHEDULE_GENERATOR,
                "schedule_id": schedule_id,
                "slice_id": slice_id,
                "load_seed": seed,
                "rng": {
                    "seed_sequence_entropy": seed,
                    "spawn_key": [SCHEDULE_RNG_DOMAIN, index],
                },
                "source_trace_fingerprint": source_fingerprint,
                "timestamps_seconds_from_slice_start": times,
            }
            checksum = _sha256(times)
            schedules[schedule_id] = payload | {"timestamps_checksum": checksum}
            entry["schedules"].append(
                {
                    "schedule_id": schedule_id,
                    "load_seed": seed,
                    "file": f"live-replay-schedules-v1/{schedule_id}.json",
                    "request_count": len(times),
                    "timestamps_checksum": checksum,
                }
            )
        slices.append(entry)
    manifest = {
        "manifest_version": REPLAY_VERSION,
        "source_issue": "#72",
        "source_workload": {
            "workload_id": workload.id,
            "split": "test",
            "selection_rule": (
                "lexicographically smallest Azure test workload ID in benchmark v1 "
                "(deterministic, not performance-based)"
            ),
            "start_seconds": workload.parameters.start_seconds,
            "duration_seconds": duration,
            "control_interval_seconds": interval,
            "trace_fingerprint": source_fingerprint,
            "source_file": dict(csv_identity),
        },
        "window_rule": {
            "version": WINDOW_RULE_VERSION,
            "window_seconds": WINDOW_SECONDS,
            "fractions_of_usable_start_range": list(WINDOW_FRACTIONS),
            "usable_start_range": "duration - window_seconds",
            "alignment": "start offsets aligned down to the 30 s control interval",
            "non_overlapping": True,
            "controller_independent": True,
        },
        "schedule_rule": {
            "version": SCHEDULE_VERSION,
            "generator": SCHEDULE_GENERATOR,
            "load_seeds": list(LOAD_SEEDS),
            "definition": (
                "each 30 s bin's exact trace request count, placed uniformly at random within "
                "the bin; RNG SeedSequence(load_seed, spawn_key=(0x61727276, slice_index)); "
                "timestamps rounded to 1e-6 s, sorted"
            ),
            "same_schedule_for_every_controller": True,
        },
        "slices": slices,
        "controller_outcomes_used": False,
        "live_results_used": False,
    }
    return manifest, schedules


def csv_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"file_name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


# --- protocol -------------------------------------------------------------------------------------


def observation_contract() -> dict[str, Any]:
    config = final_config()
    constants = ObservationConstants.from_config(config)
    names = feature_names(constants)
    live = {
        "demand_pressure": (
            "request arrivals per second in the completed tick, counted at "
            "the ingress/load generator",
            False,
        ),
        "utilization": (
            "offered work / (ready replicas x calibrated per-replica capacity x "
            "30 s), capped at 1; needs the #73 per-replica capacity calibration",
            True,
        ),
        "queue_pressure": (
            "requests waiting/in flight beyond service at the tick boundary "
            "(Knative activator/queue-proxy)",
            True,
        ),
        "latency_pressure": (
            "p95 latency of requests completed in the tick; a real "
            "distribution rather than the simulator's p95 proxy",
            True,
        ),
        "active_replicas_fraction": ("ready pods / 10", False),
        "tick_cost_fraction": (
            "(ready + pending pods) x 30 s replica-seconds of the latest "
            "tick / (10 x 30 s); latest tick only",
            False,
        ),
        "episode_progress": (
            "(slice start bin + completed ticks) / 120: progress through the "
            "source hour, the same in simulation replay",
            False,
        ),
        "pending_ready": (
            "pods requested but not ready, bucketed by nominal age against the "
            "configured 60 s startup; never by an observed/predicted readiness",
            False,
        ),
    }

    def entry(name: str, index: int) -> dict[str, Any]:
        key = next(k for k in live if name.startswith(k))
        text, approx = live[key]
        return {"index": index, "name": name, "live_equivalent": text, "live_approximation": approx}

    return {
        "version": OBSERVATION_VERSION,
        "shared_builder": "scalerl.environment.observation.build_observation",
        "dimension": len(names),
        "features": [entry(name, i) for i, name in enumerate(names)],
        "normalization": {
            "max_service_rate_rps": constants.max_service_rate_rps,
            "max_tick_capacity_requests": constants.max_tick_capacity_requests,
            "max_tick_cost": constants.max_tick_cost,
            "latency_target_seconds": constants.latency_target_seconds,
            "max_replicas": constants.max_replicas,
            "history_ticks": constants.history_ticks,
            "pending_buckets": constants.pending_buckets,
            "episode_ticks": 120,
        },
        "clipping": "none: every feature is in [0, 1] by construction",
        "rule": (
            "simulation and the live adapter call the same builder with the same constants; a "
            "feature flagged live_approximation must be measured as documented and its "
            "approximation reported with the results; no ad-hoc substitute (in particular no "
            "accumulated-cost substitute for the latest-tick cost)"
        ),
        "history_at_slice_start": "empty (zeros), as at an episode reset, in both domains",
    }


def final_config() -> SimulatorConfig:
    config, _ = contract_config(DESIRED_REPLICAS_V1)
    return config


def build_protocol(
    upstream: Mapping[str, Any], controllers: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    config = final_config()
    dqn = controllers["controllers"]["dqn"]["canonical_selection"]
    ppo = controllers["controllers"]["ppo"]["canonical_selection"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_issue": "#72",
        "terminology": "local systems-in-the-loop / Knative sim-to-real validation",
        "research_question": (
            "do controller behavior and system trade-offs observed in simulation transfer to a "
            "real serving system? No requirement that RL wins; negative transfer is valid"
        ),
        "upstream": dict(UPSTREAM),
        "reward": {
            "contract_id": UPSTREAM["reward_contract_id"],
            "variant": SELECTED_REWARD,
            "weights": dict(upstream["contract"].selected_weights),
        },
        "controller_manifest": {
            "version": CONTROLLER_MANIFEST_VERSION,
            "id": content_id(controllers),
            "controllers": sorted(controllers["controllers"]),
            "canonical_dqn": {
                k: dqn[k]
                for k in (
                    "canonical_training_seed",
                    "canonical_training_run_id",
                    "canonical_model_artifact_uri",
                    "canonical_seed_validation_feasible",
                )
            },
            "canonical_ppo": {
                k: ppo[k]
                for k in (
                    "canonical_training_seed",
                    "canonical_training_run_id",
                    "canonical_model_artifact_uri",
                    "canonical_seed_validation_feasible",
                )
            },
            "canonical_seed_rule": CANONICAL_SEED_RULE,
        },
        "replay_manifest": {
            "version": REPLAY_VERSION,
            "id": content_id(replay),
            "slices": [s["slice_id"] for s in replay["slices"]],
            "load_seeds": list(LOAD_SEEDS),
        },
        "control_contract": {
            "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
            "min_replicas": config.replicas.min_replicas,
            "max_replicas": config.replicas.max_replicas,
            "action_semantics": DESIRED_REPLICAS_V1,
            "action_meaning": "set the desired fleet size to integer N in [1, 10]; not +1/-1",
            "step_size": "not applicable (desired-replicas-v1)",
            "deterministic_policy_inference": True,
            "online_training": False,
            "exploration": False,
            "scale_to_zero": "disabled for the controller comparison (training min_replicas = 1)",
            "initial_replicas": config.replicas.initial_replicas,
            "nominal_startup_delay_seconds": config.replicas.startup_delay_seconds,
        },
        "observation_contract": observation_contract(),
        "cost_contract": {
            "primary": "replica-seconds (ready + pending pods x seconds)",
            "normalized": (
                "the ScaleRL abstract cost contract: replica-seconds / (max_replicas x duration), "
                "comparable to the simulator's normalized cost"
            ),
            "not_claimed": "no AWS Lambda / Azure Functions / GCP pricing; real dollars are #32",
        },
        "metrics": {
            "per_run": [
                "sla_violation_rate",
                "p95_latency_seconds (simulation proxy / real client-side)",
                "replica_seconds",
                "normalized_cost",
                "queue / in-flight backlog",
                "failed_or_dropped_requests",
                "scaling_actions",
                "churn_rate",
                "replicas_moved (total absolute change)",
            ],
            "unit_of_analysis": "controller x slice x load seed (3 x 3 per controller)",
        },
        "transfer_reporting": {
            "absolute": (
                "for every controller x slice x load seed: simulation and real SLA violation, "
                "simulation p95 and real client p95, simulation cost proxy and real "
                "replica-seconds/normalized cost, queue/failure metrics, scaling churn"
            ),
            "relative": (
                "per metric: (controller - Threshold) in simulation and (controller - Threshold) "
                "on Knative, same slice and load seed"
            ),
            "diagnostics": [
                "sign agreement of the relative effect (direction preserved or not)",
                "magnitude of the simulation and real relative effects, side by side",
            ],
            "secondary": "Spearman rank correlation of controllers across domains, secondary only",
            "knative_native": "an additional real-only comparison row",
            "pass_fail": "none: v1 has no binary global transfer criterion",
            "overall_score": "none: no aggregate transfer score",
            "rl_must_win": False,
        },
        "held_out_controller_outcomes_used": False,
        "live_results_used": False,
        "not_in_scope": {
            "#46": "held-out evaluation",
            "#73": "Knative testbed",
            "#74": "HTTP replay engine",
            "#75": "live controller loop",
            "#76": "sim-vs-Knative conclusions",
        },  # fmt: skip
        "caveats": [
            "the canonical DQN comes from a five-seed result whose syn-val-bursty mean sits "
            "exactly at the SLA limit (#20); the deployed seed is one artifact, not the family",
            "Azure benchmark windows peak at a few requests per second (train/validation "
            "characterization); at the nominal 50 rps/replica the replay may not exercise "
            "scaling. Any load amplitude scaling or capacity calibration must be a versioned "
            "protocol revision committed before live runs (#73/#74), never tuned on results",
            "simulation replay of a 10-minute slice needs episode_progress offset by the slice "
            "start bin, as frozen in the observation contract",
        ],
    }


# --- freeze / check ---------------------------------------------------------------------------


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def freeze(azure_csv: Path, outputs: Path, bench: Path = BENCH) -> dict[str, str]:
    upstream = verify_upstream(bench)
    controllers = build_controller_manifest(upstream, outputs)
    workload = canonical_test_workload()
    trace = build_workload(workload, azure_csv_path=azure_csv)
    replay, schedules = build_replay(trace, workload, csv_identity(azure_csv))
    protocol = build_protocol(upstream, controllers, replay)
    schedule_dir = bench / "live-replay-schedules-v1"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    for schedule_id, payload in schedules.items():
        (schedule_dir / f"{schedule_id}.json").write_text(_dump(payload))
    (bench / "controller-deployment-manifest-v1.json").write_text(_dump(controllers))
    (bench / "live-replay-manifest-v1.json").write_text(_dump(replay))
    (bench / "sim-to-real-protocol-v1.json").write_text(_dump(protocol))
    return {
        "protocol": content_id(protocol),
        "replay": content_id(replay),
        "controllers": content_id(controllers),
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def check(bench: Path = BENCH, azure_csv: Path | None = None) -> dict[str, str]:
    """Validate the committed manifests; re-extract slices and schedules if the CSV exists."""
    upstream = verify_upstream(bench)
    controllers = load(bench / "controller-deployment-manifest-v1.json")
    replay = load(bench / "live-replay-manifest-v1.json")
    protocol = load(bench / "sim-to-real-protocol-v1.json")
    thresholds = controllers["sla_thresholds"]
    for family in ("dqn", "ppo"):
        entry = controllers["controllers"][family]
        seeds = [SeedEvidence.model_validate(s, strict=False) for s in entry["five_seed_lineage"]]
        check_against_contract(
            seeds, upstream["contract"].per_reward[SELECTED_REWARD][f"{family}_equal_seed_means"]
        )
        if select_canonical_seed(seeds, thresholds) != entry["canonical_selection"]:
            raise ValueError(f"{family}: canonical selection does not recompute")
    for entry in replay["slices"]:
        for ref in entry["schedules"]:
            payload = load(bench / ref["file"])
            times = payload["timestamps_seconds_from_slice_start"]
            if _sha256(times) != ref["timestamps_checksum"] or len(times) != ref["request_count"]:
                raise ValueError(f"{ref['schedule_id']}: schedule file does not match the manifest")
    if protocol["controller_manifest"]["id"] != content_id(controllers) or protocol[
        "replay_manifest"
    ]["id"] != content_id(replay):
        raise ValueError("protocol references stale controller/replay manifests")
    if protocol != json.loads(_dump(build_protocol(upstream, controllers, replay))):
        raise ValueError("the committed protocol differs from this code's frozen protocol")
    if azure_csv is not None and azure_csv.is_file():
        workload = canonical_test_workload()
        rebuilt, schedules = build_replay(
            build_workload(workload, azure_csv_path=azure_csv), workload, csv_identity(azure_csv)
        )
        if rebuilt != replay:
            raise ValueError("re-extracted replay manifest differs from the committed one")
        for schedule_id, payload in schedules.items():
            if load(bench / "live-replay-schedules-v1" / f"{schedule_id}.json") != payload:
                raise ValueError(f"{schedule_id}: regenerated schedule differs")
    return {
        "protocol": content_id(protocol),
        "replay": content_id(replay),
        "controllers": content_id(controllers),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{PROTOCOL_VERSION} (#72).")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_cmd = commands.add_parser("freeze")
    freeze_cmd.add_argument("--azure-csv", type=Path, default=DEFAULT_AZURE_CSV)
    freeze_cmd.add_argument("--reward-outputs", type=Path, default=DEFAULT_REWARD_OUTPUTS)
    check_cmd = commands.add_parser("check")
    check_cmd.add_argument("--azure-csv", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            ids = freeze(args.azure_csv, args.reward_outputs)
        else:
            ids = check(azure_csv=args.azure_csv)
    except ValueError as error:
        parser.error(str(error))
    for name, value in ids.items():
        print(f"{name:<12} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
