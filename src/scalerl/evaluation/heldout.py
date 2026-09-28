"""Held-out Azure TEST evaluation ``heldout-evaluation-v1`` (#46).

The final simulator evaluation of the controllers frozen by #72 on the two
frozen Azure TEST workloads. It is the **only** workflow that runs controllers
on test data; the development machinery (``multiseed-v1``, #81, #80, #20)
keeps refusing test workloads.

Everything that defines the experiment is a predeclared, content-identified
specification (``benchmarks/v1/heldout-evaluation-v1.json``) built only from
frozen upstream artifacts, never from test outcomes. It is committed on a clean
SHA before any run of record; :func:`run` refuses a dirty tree and a spec that
differs from this code's.

Three case kinds, one raw row each:

* ``primary``: the four ``robustness-v1`` scenarios (#65). Deterministic
  controllers use evaluation seed 0, Random seeds 0-4; dynamics seeds 0-4 only
  where capacity jitter exists (seed 0 otherwise: no pseudo-replication).
  2 workloads x (11 + 55 + 11 + 55) = 264 cases.
* ``startup``: the two ``startup-robustness-v1`` scenarios (#81) with their
  frozen ``(dynamics_seed, startup_delay_seed)`` pairs, applied to the #46
  controller set (the #81 spec itself is untouched).
  2 workloads x 2 scenarios x 5 pairs x 11 = 220 cases.
* ``replay``: nominal simulator references on the three frozen #72 replay
  windows for the #72 controllers with a simulator counterpart. The #72 load
  seeds only move requests inside a 30 s bin; the simulator consumes bin
  counts, so one run per slice and controller is mapped to all three schedule
  IDs (never counted as three replicates). 3 x 5 = 15 cases.

Command line::

    python -m scalerl.evaluation.heldout spec    # write the predeclared spec
    python -m scalerl.evaluation.heldout check   # committed spec == this code's
    python -m scalerl.evaluation.heldout plan    # pre-run checklist, no runs
    python -m scalerl.evaluation.heldout run --azure-csv ... --tracking-uri sqlite:///outputs/mlflow.db
    python -m scalerl.evaluation.heldout freeze --pre-run-sha <sha>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from pydantic import JsonValue

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers import (
    Controller,
    PredictiveController,
    RandomController,
    StaticController,
    ThresholdController,
)
from scalerl.controllers.proactive_predictive import (
    HistoricalDemandProfile,
    ProactivePredictiveController,
)
from scalerl.environment import (
    DESIRED_REPLICAS_V1,
    FIXED_V1,
    TRI_POINT_MULTIPLICATIVE_V1,
    AutoscalingEnv,
    DynamicsConfig,
    SimulatorConfig,
    SourceWindow,
)
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation import sim_to_real as s2r
from scalerl.evaluation import startup_robustness as s81
from scalerl.evaluation.metrics import (
    ActionMagnitudeMetrics,
    EpisodeMetrics,
    evaluate_controller_episode,
    summarize_action_magnitude,
)
from scalerl.evaluation.multiseed import (
    RANDOM_EVALUATION_SEEDS,
    STATIC_REFERENCE_TARGET,
    _append_row,
    _repair_torn_tail,
    describe,
    read_rows,
)
from scalerl.evaluation.predictive_baseline import AZURE_PROFILE_SOURCES, final_contract_config
from scalerl.evaluation.robustness import (
    ROBUSTNESS_SCENARIO_VERSION,
    ROBUSTNESS_SCENARIOS,
    summarize_dynamics,
)
from scalerl.mlops import RunSpec
from scalerl.workloads import WorkloadTrace

HELDOUT_VERSION: Final = "heldout-evaluation-v1"
RESULTS_VERSION: Final = "heldout-results-v1"
ROW_SCHEMA: Final = "heldout-raw-row-v1"
SUMMARY_METHOD: Final = "heldout-descriptive-v1"
EXPERIMENT_NAME: Final = "scalerl-heldout-v1"
AZURE_TEST_WORKLOADS: Final = ("azure-test-993600", "azure-test-1166400")
EXCLUDED_TEST_WORKLOADS: Final = ("syn-test-seasonal-shifted", "syn-test-spike-hard")
EVALUATION_SEED: Final = 0
DYNAMICS_SEEDS: Final = (0, 1, 2, 3, 4)
EXPECTED_PRIMARY: Final = 264
EXPECTED_STARTUP: Final = 220
EXPECTED_MAIN: Final = 484
EXPECTED_REPLAY: Final = 15
SOURCE_EPISODE_TICKS: Final = 120
REPLAY_CONTROLLERS: Final = (
    "threshold-v1",
    "predictive-v1",
    "predictive-seasonal-v1",
    "dqn-c14-seed0",
    "ppo-c08-seed4",
)
PROFILE_ID: Final = "d050d3b8ca0f"
DEFAULT_TRACKING_URI: Final = "sqlite:///outputs/mlflow.db"
BENCH: Final = s2r.BENCH
DEFAULT_SPEC = BENCH / "heldout-evaluation-v1.json"
DEFAULT_RESULTS = BENCH / "heldout-results-v1.json"
DEFAULT_OUTPUT = Path("outputs/heldout-evaluation-v1")
DEFAULT_AZURE_CSV = s2r.DEFAULT_AZURE_CSV

EPISODE_KEYS: Final = tuple(EpisodeMetrics.__dataclass_fields__)
ACTION_KEYS: Final = tuple(f"action.{k}" for k in ActionMagnitudeMetrics.__dataclass_fields__)
DYNAMICS_KEYS: Final = (
    "dynamics.mean_capacity_multiplier",
    "dynamics.min_capacity_multiplier",
    "dynamics.max_capacity_multiplier",
)
STARTUP_KEYS: Final = tuple(f"startup.{k}" for k in s81.StartupDiagnostics.__dataclass_fields__)
DERIVED_KEYS: Final = (
    "episode_ticks",
    "sla_violation_count",
    "arrived_requests",
    "processed_requests",
    "dropped_requests",
    "final_queued_requests",
    "replica_seconds",
)
METRIC_KEYS: Final = (*EPISODE_KEYS, *ACTION_KEYS, *DYNAMICS_KEYS, *STARTUP_KEYS, *DERIVED_KEYS)
# The metrics summarized and published; every other raw metric stays in the raw rows.
SUMMARY_METRICS: Final = (
    "sla_violation_rate",
    "sla_violation_count",
    "normalized_cost",
    "infrastructure_cost",
    "replica_seconds",
    "mean_p95_latency_seconds",
    "max_p95_latency_seconds",
    "queue_pressure",
    "mean_queue_depth",
    "max_queue_depth",
    "dropped_requests",
    "processed_requests",
    "scaling_actions",
    "churn_rate",
    "action.total_absolute_replica_change",
    "action.mean_absolute_replica_change_when_scaling",
    "action.max_absolute_replica_change_in_one_tick",
    "episode_reward",
)
DELTA_METRICS: Final = (
    "sla_violation_rate",
    "normalized_cost",
    "mean_p95_latency_seconds",
    "queue_pressure",
    "churn_rate",
)
FLAG_TOLERANCE: Final = 1e-12
FLAGS: Final = {
    "full_fleet": "mean normalized cost >= 0.9",
    "underprovisioning": "mean SLA above Threshold's (same workload/scenario) and mean normalized "
    "cost < 0.6",
    "thrashing": "mean churn rate >= 0.3",
    "churn_aversion": "mean SLA above Threshold's and mean scaling actions <= 2",
    "persistent_backlog": "mean queue pressure >= 0.2",
    "never_scaled": "no replica change in any run (stays at the initial fleet)",
}


# --- controller variants -------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One frozen controller row of the held-out matrix."""

    variant_id: str
    controller: str  # static | random | threshold | predictive | predictive-seasonal | dqn | ppo
    version: str
    params: Mapping[str, JsonValue]
    evaluation_seeds: tuple[int, ...] = (EVALUATION_SEED,)
    role: str = "primary"
    training_seed: int | None = None
    training_run_id: str | None = None
    model_artifact_uri: str | None = None
    candidate_id: str | None = None
    candidate_set_id: str | None = None
    deployment_name: str | None = None

    @property
    def learned(self) -> bool:
        return self.controller in ("dqn", "ppo")

    def as_json(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "controller": self.controller,
            "version": self.version,
            "params": dict(self.params),
            "evaluation_seeds": list(self.evaluation_seeds),
            "role": self.role,
            "training_seed": self.training_seed,
            "training_run_id": self.training_run_id,
            "model_artifact_uri": self.model_artifact_uri,
            "candidate_id": self.candidate_id,
            "candidate_set_id": self.candidate_set_id,
            "deployment_name": self.deployment_name,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Variant:
        return cls(**{**payload, "evaluation_seeds": tuple(payload["evaluation_seeds"])})


def frozen_variants(controllers: Mapping[str, Any]) -> tuple[Variant, ...]:
    """The #46 controller rows, taken from the frozen #72 controller manifest."""
    entries = controllers["controllers"]
    threshold = entries["threshold-v1"]
    predictive = entries["predictive-v1"]
    seasonal = entries["predictive-seasonal-v1"]
    learned = []
    for family in ("dqn", "ppo"):
        entry = entries[family]
        selection = entry["canonical_selection"]
        if not selection["canonical_seed_validation_feasible"]:
            raise ValueError(f"#72 canonical {family} is not validation-feasible")
        seed = int(selection["canonical_training_seed"])
        learned.append(
            Variant(
                variant_id=f"{entry['candidate_id']}-seed{seed}",
                controller=family,
                version=entry["controller"],
                params=dict(entry["hyperparameters"]),
                training_seed=seed,
                training_run_id=selection["canonical_training_run_id"],
                model_artifact_uri=selection["canonical_model_artifact_uri"],
                candidate_id=entry["candidate_id"],
                candidate_set_id=entry["candidate_set_id"],
                deployment_name=entry["controller"],
            )
        )
    return (
        Variant(
            "static-v1",
            "static",
            "static-v1",
            {"target_replicas": STATIC_REFERENCE_TARGET},
            role="fixed half-fleet reference (#19 convention), not tuned",
        ),
        Variant(
            "random-v1",
            "random",
            "random-v1",
            {},
            evaluation_seeds=RANDOM_EVALUATION_SEEDS,
            role="sanity check only, never a performance target",
        ),
        Variant(
            "threshold-v1",
            "threshold",
            threshold["controller_version"],
            dict(threshold["params"]),
            deployment_name="threshold-v1",
        ),
        Variant(
            "predictive-v1",
            "predictive",
            predictive["capacity_policy"],
            {**predictive["params"], "forecast_method": predictive["forecast_method"]},
            deployment_name="predictive-v1",
        ),
        Variant(
            "predictive-seasonal-v1",
            "predictive-seasonal",
            seasonal["capacity_policy"],
            {
                "forecast_method": seasonal["forecast_method"],
                "profile_id": seasonal["historical_profile"]["profile_id"],
                "profile_train_workload_ids": list(
                    seasonal["historical_profile"]["train_workload_ids"]
                ),
            },
            deployment_name="predictive-seasonal-v1",
        ),
        *learned,
    )


# --- the predeclared specification -------------------------------------------------------------


def build_spec(bench: Path = BENCH) -> dict[str, Any]:
    """The full ``heldout-evaluation-v1`` spec from frozen artifacts only (no test outcome)."""
    upstream = s2r.verify_upstream(bench)
    ids = s2r.check(bench)  # the #72 manifests recompute and reference each other
    controllers = s2r.load(bench / "controller-deployment-manifest-v1.json")
    replay = s2r.load(bench / "live-replay-manifest-v1.json")
    startup_spec = s81.ExperimentSpec.load(bench / "startup-robustness-v1.json")
    contract = upstream["contract"]
    manifest = load_benchmark_manifest()
    for workload_id in AZURE_TEST_WORKLOADS:
        entry = manifest.get(workload_id)
        if entry.split != "test" or entry.source != "azure":
            raise ValueError(f"{workload_id} is not an Azure TEST workload")
    azure_tests = sorted(
        w.id for w in manifest.workloads if w.split == "test" and w.source == "azure"
    )
    if azure_tests != sorted(AZURE_TEST_WORKLOADS):
        raise ValueError(f"benchmark Azure TEST workloads are {azure_tests}")
    variants = frozen_variants(controllers)
    by_id = {v.variant_id: v for v in variants}
    if set(REPLAY_CONTROLLERS) - set(by_id):
        raise ValueError("replay controllers are not all #46 variants")
    primary = {
        name: {
            "version": s.version,
            "capacity_jitter_fraction": s.capacity_jitter_fraction,
            "telemetry_delay_ticks": s.telemetry_delay_ticks,
            "dynamics_seeds": [0] if s.capacity_jitter_fraction == 0 else list(DYNAMICS_SEEDS),
        }
        for name, s in ROBUSTNESS_SCENARIOS.items()
    }
    slices = []
    for entry in replay["slices"]:
        slices.append(
            {
                key: entry[key]
                for key in (
                    "slice_id",
                    "slice_index",
                    "source_workload_id",
                    "split",
                    "source_trace_fingerprint",
                    "start_offset_seconds",
                    "duration_seconds",
                    "start_bin",
                    "end_bin_exclusive",
                    "control_interval_seconds",
                    "slice_checksum",
                    "request_count",
                )
            }
            | {
                "schedule_ids": [s["schedule_id"] for s in entry["schedules"]],
                "load_seeds": [s["load_seed"] for s in entry["schedules"]],
            }
        )
    spec: dict[str, Any] = {
        "experiment_version": HELDOUT_VERSION,
        "source_issue": "#46",
        "benchmark_version": manifest.version,
        "purpose": (
            "final held-out simulator evaluation of the #72-frozen controllers on the frozen "
            "Azure TEST workloads; descriptive, no winner, RL is not required to win"
        ),
        "workloads": {
            "primary_azure_test": list(AZURE_TEST_WORKLOADS),
            "split": "test",
            "source": "azure",
            "excluded_from_primary": {
                w: "synthetic TEST workload: not part of the primary Azure result"
                for w in EXCLUDED_TEST_WORKLOADS
            },
            "no_train_or_validation_workload": True,
        },
        "controllers": [v.as_json() for v in variants],
        "excluded_controllers": {
            "q-learning": "#47 is open when #46 is frozen; never added after results",
            "knative-native-v1": "real-system baseline only (#72/#76); no simulator counterpart",
            "other-dqn-ppo-training-seeds": (
                "#72 already chose the deployed artifacts on validation evidence; no seed is "
                "evaluated or reselected on TEST"
            ),
        },
        "reward": {
            "contract": "reward-contract-v1",
            "contract_id": contract.contract_id,
            "variant": contract.selected_reward_variant,
            "weights": dict(contract.selected_weights),
            "role": "secondary metric (episode_reward); system metrics are primary",
        },
        "action_semantics": DESIRED_REPLICAS_V1,
        "simulator_config": final_contract_config().model_dump(mode="json"),
        "upstream": {
            "sim_to_real_protocol_id": ids["protocol"],
            "live_replay_manifest_id": ids["replay"],
            "controller_deployment_manifest_id": ids["controllers"],
            **s2r.UPSTREAM,
            "candidate_set_id": s2r.CANDIDATE_SET_ID,
            "robustness_version": ROBUSTNESS_SCENARIO_VERSION,
            "robustness_scenarios": list(ROBUSTNESS_SCENARIOS),
            "startup_robustness_version": s81.STARTUP_ROBUSTNESS_VERSION,
            "historical_profile_id": PROFILE_ID,
            "historical_profile_train_workload_ids": list(AZURE_PROFILE_SOURCES),
        },
        "primary_matrix": {
            "scenarios": primary,
            "evaluation_seeds": {
                "deterministic_controllers": [EVALUATION_SEED],
                "random-v1": list(RANDOM_EVALUATION_SEEDS),
            },
            "pseudo_replication": (
                "scenarios without capacity jitter use dynamics seed 0 only; deterministic "
                "controllers use evaluation seed 0 only"
            ),
            "fairness": (
                "same workload, scenario and dynamics seed give every controller the same "
                "capacity-multiplier sequence, independent of actions and evaluation order"
            ),
            "cases": EXPECTED_PRIMARY,
        },
        "startup_matrix": {
            "startup_model": s81._startup_model(),
            "scenarios": s81._scenarios(),
            "startup_spec_id": startup_spec.experiment_id,
            "evaluation_seeds": {
                "deterministic_controllers": [EVALUATION_SEED],
                "random-v1": list(RANDOM_EVALUATION_SEEDS),
            },
            "note": (
                "the frozen #81 dynamics definitions applied to the #46 controller set; the "
                "#81 spec (PPO seeds, validation workloads) is not modified"
            ),
            "cases": EXPECTED_STARTUP,
        },
        "main_cases": EXPECTED_MAIN,
        "learned_policy_loading": (
            "the exact #72 MLflow bundles, deterministic inference; strict compatibility for "
            "nominal and capacity-jitter; the #65/#81 robustness-only path when telemetry delay "
            "> 0 or the startup model is not fixed-v1, with the perturbed fields recorded"
        ),
        "replay_references": {
            "manifest_id": ids["replay"],
            "source_workload_id": replay["source_workload"]["workload_id"],
            "trace_fingerprint": replay["source_workload"]["trace_fingerprint"],
            "slices": slices,
            "controllers": list(REPLAY_CONTROLLERS),
            "scenario": "nominal",
            "dynamics_seed": 0,
            "evaluation_seed": EVALUATION_SEED,
            "observation": (
                "SourceWindow(start_bin, 120): episode_progress = (start_bin + completed_ticks) "
                "/ 120; traffic history starts empty at the slice boundary"
            ),
            "seasonal_profile_alignment": (
                "the frozen Azure TRAIN profile is read from the slice's source position: "
                "profile ticks start_bin, start_bin + 1, ..."
            ),
            "load_seeds": (
                "one simulator run per slice and controller (the simulator consumes 30 s bin "
                "counts, identical for every load seed), mapped to all three #72 schedule IDs; "
                "never counted as three replicates"
            ),
            "cases": EXPECTED_REPLAY,
        },
        "metrics": {
            "episode": list(EPISODE_KEYS),
            "action": list(ACTION_KEYS),
            "dynamics": list(DYNAMICS_KEYS),
            "startup": list(STARTUP_KEYS),
            "derived": {
                "episode_ticks": "number of ticks",
                "sla_violation_count": "ticks with sla_violated",
                "arrived_requests": "sum of arrived requests",
                "processed_requests": "sum of processed (completed) requests",
                "dropped_requests": "sum of dropped requests",
                "final_queued_requests": "queue at the end of the episode",
                "replica_seconds": "billable (active + pending) replica-seconds = cost / "
                "cost_per_hour * 3600",
            },
            "primary": list(SUMMARY_METRICS[:-1]),
            "secondary": ["episode_reward"],
        },
        "aggregation": {
            "method": SUMMARY_METHOD,
            "group": "case kind x workload (or slice) x scenario x controller variant; never "
            "pooled across workloads, scenarios or case kinds",
            "statistics": "mean, median, sample SD (None for n = 1), min, max, q1, q3 "
            "(#19 descriptive-v1 describe); no fabricated variance for a single realization",
            "replicates": "raw runs of the group (dynamics seeds, startup replicates, Random "
            "evaluation seeds kept explicit in each row)",
            "random": "also summarized over its per-evaluation-seed means (evaluation-seed "
            "spread kept distinct from exogenous replicates)",
            "paired_deltas": "controller - threshold-v1 per matched realization (workload, "
            "scenario, dynamics seed, startup seed); descriptive only",
            "diagnostic_flags": FLAGS,
            "no_overall_score": True,
            "no_winner": True,
        },
        "mlflow": {
            "experiment": EXPERIMENT_NAME,
            "tracking_uri": DEFAULT_TRACKING_URI,
            "one_run_per_case": True,
            "required_tags": list(REQUIRED_TAGS),
        },
        "outputs": {
            "row_schema": ROW_SCHEMA,
            "raw": str(DEFAULT_OUTPUT / "raw-results.jsonl"),
            "results_artifact": str(DEFAULT_RESULTS),
            "per_run_artifacts": "heldout/row.json and heldout/step-infos.json in each run",
        },
        "execution": {
            "requires_clean_git": True,
            "single_writer": "serial, one process; append-only fsynced JSONL rows",
            "resumable": "completed cases (local row or FINISHED MLflow run with the same spec, "
            "case and input identity) are never rerun",
            "bug_policy": "a genuine software bug is fixed generically with a regression test; "
            "affected runs are superseded and rerun from a new clean SHA; nothing else changes",
        },
        "test_data_used_for_design": False,
        "test_outcomes_used_for_design": False,
        "model_selection_on_test": False,
        "q_learning_included": False,
        "declares_winner": False,
    }
    normalized: dict[str, Any] = json.loads(_dump(spec))
    return normalized


REQUIRED_TAGS: Final = (
    "scalerl.heldout_spec_id",
    "scalerl.heldout_version",
    "scalerl.git_sha",
    "scalerl.git_dirty",
    "scalerl.workload_id",
    "scalerl.workload_split",
    "scalerl.controller",
    "scalerl.controller_variant_id",
    "scalerl.action_semantics",
    "scalerl.reward_contract_id",
    "scalerl.protocol_id",
    "scalerl.replay_manifest_id",
    "scalerl.controller_manifest_id",
    "scalerl.robustness_scenario",
    "scalerl.robustness_version",
    "scalerl.dynamics_seed",
    "scalerl.evaluation_seed",
    "scalerl.startup_delay_model",
    "scalerl.startup_delay_seed",
    "scalerl.evaluation_case_id",
    "scalerl.case_kind",
    "scalerl.input_id",
)


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def spec_id(spec: Mapping[str, Any]) -> str:
    return s2r.content_id(spec)


def load_frozen_spec(path: Path = DEFAULT_SPEC, bench: Path = BENCH) -> dict[str, Any]:
    """The committed spec, required to equal the one this code builds from frozen inputs."""
    committed = json.loads(path.read_text())
    if committed != build_spec(bench):
        raise ValueError(f"{path} differs from this code's predeclared {HELDOUT_VERSION} spec")
    return dict(committed)


# --- cases ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One evaluation episode of the predeclared matrix."""

    index: int
    kind: str  # primary | startup | replay
    variant: Variant
    workload_id: str
    scenario: str
    scenario_version: str
    capacity_jitter_fraction: float
    telemetry_delay_ticks: int
    dynamics_seed: int
    startup_delay_seed: int | None
    evaluation_seed: int
    slice: Mapping[str, Any] | None = None
    input_id: str | None = None

    @property
    def startup_delay_model(self) -> str:
        return FIXED_V1 if self.startup_delay_seed is None else TRI_POINT_MULTIPLICATIVE_V1

    @property
    def case_id(self) -> str:
        where = self.slice["slice_id"] if self.slice else self.workload_id
        startup = (
            "fixed" if self.startup_delay_seed is None else f"startup{self.startup_delay_seed}"
        )
        return (
            f"{self.kind}|{self.variant.variant_id}|{where}|{self.scenario}"
            f"|dyn{self.dynamics_seed}|{startup}|eval{self.evaluation_seed}"
        )

    @property
    def robustness_path(self) -> bool:
        """Learned policies need the explicit robustness-only compatibility path."""
        return self.telemetry_delay_ticks > 0 or self.startup_delay_seed is not None

    def config(self, spec: Mapping[str, Any]) -> SimulatorConfig:
        base = SimulatorConfig.model_validate(spec["simulator_config"])
        if base != final_contract_config():
            raise ValueError("the spec's simulator config is not the canonical #79 config")
        dynamics = DynamicsConfig(
            capacity_jitter_fraction=self.capacity_jitter_fraction,
            telemetry_delay_ticks=self.telemetry_delay_ticks,
            dynamics_seed=self.dynamics_seed,
        )
        if self.startup_delay_seed is not None:
            dynamics = DynamicsConfig(
                capacity_jitter_fraction=self.capacity_jitter_fraction,
                telemetry_delay_ticks=self.telemetry_delay_ticks,
                dynamics_seed=self.dynamics_seed,
                startup_delay_model=TRI_POINT_MULTIPLICATIVE_V1,
                startup_delay_seed=self.startup_delay_seed,
            )
        return base.model_copy(update={"dynamics": dynamics})


def generate_cases(spec: Mapping[str, Any]) -> list[Case]:
    """Every case of the spec in a deterministic order (primary, startup, replay)."""
    variants = [Variant.from_json(v) for v in spec["controllers"]]
    by_id = {v.variant_id: v for v in variants}
    out: list[Case] = []

    def add(**fields: Any) -> None:
        out.append(Case(index=len(out), **fields))

    workloads = spec["workloads"]["primary_azure_test"]
    for workload_id in workloads:
        for name, s in spec["primary_matrix"]["scenarios"].items():
            for dyn in s["dynamics_seeds"]:
                for variant in variants:
                    for seed in variant.evaluation_seeds:
                        add(
                            kind="primary",
                            variant=variant,
                            workload_id=workload_id,
                            scenario=name,
                            scenario_version=s["version"],
                            capacity_jitter_fraction=s["capacity_jitter_fraction"],
                            telemetry_delay_ticks=s["telemetry_delay_ticks"],
                            dynamics_seed=dyn,
                            startup_delay_seed=None,
                            evaluation_seed=seed,
                        )
    for workload_id in workloads:
        for name, s in spec["startup_matrix"]["scenarios"].items():
            for dyn, startup in s["seed_pairs_dynamics_startup"]:
                for variant in variants:
                    for seed in variant.evaluation_seeds:
                        add(
                            kind="startup",
                            variant=variant,
                            workload_id=workload_id,
                            scenario=name,
                            scenario_version=s["version"],
                            capacity_jitter_fraction=s["capacity_jitter_fraction"],
                            telemetry_delay_ticks=s["telemetry_delay_ticks"],
                            dynamics_seed=dyn,
                            startup_delay_seed=startup,
                            evaluation_seed=seed,
                        )
    replay = spec["replay_references"]
    for entry in replay["slices"]:
        for variant_id in replay["controllers"]:
            add(
                kind="replay",
                variant=by_id[variant_id],
                workload_id=entry["source_workload_id"],
                scenario=replay["scenario"],
                scenario_version=ROBUSTNESS_SCENARIO_VERSION,
                capacity_jitter_fraction=0.0,
                telemetry_delay_ticks=0,
                dynamics_seed=replay["dynamics_seed"],
                startup_delay_seed=None,
                evaluation_seed=replay["evaluation_seed"],
                slice=entry,
            )
    return out


def case_counts(cases: Sequence[Case]) -> dict[str, int]:
    counts = {kind: sum(1 for c in cases if c.kind == kind) for kind in ("primary", "startup")}
    counts["main"] = counts["primary"] + counts["startup"]
    counts["replay"] = sum(1 for c in cases if c.kind == "replay")
    return counts


def require_expected_counts(spec: Mapping[str, Any], cases: Sequence[Case]) -> None:
    """The predeclared matrix size guard (checked before anything runs)."""
    counts = case_counts(cases)
    expected = {
        "primary": EXPECTED_PRIMARY,
        "startup": EXPECTED_STARTUP,
        "main": EXPECTED_MAIN,
        "replay": EXPECTED_REPLAY,
    }
    declared = {
        "primary": spec["primary_matrix"]["cases"],
        "startup": spec["startup_matrix"]["cases"],
        "main": spec["main_cases"],
        "replay": spec["replay_references"]["cases"],
    }
    if counts != expected or declared != expected:
        raise ValueError(f"case counts {counts} (declared {declared}) are not {expected}")
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case IDs are not unique")


# --- inputs: traces, slices, profile -------------------------------------------------------------


def trace_fingerprint(trace: WorkloadTrace) -> str:
    """The #72 source-trace fingerprint (same definition as the replay manifest)."""
    return s2r._sha256(
        {
            "control_interval_seconds": trace.control_interval_seconds,
            "request_rates": list(trace.request_rates),
        }
    )


def slice_trace(source: WorkloadTrace, entry: Mapping[str, Any]) -> WorkloadTrace:
    """The frozen #72 window of ``source``; refuses anything that is not that exact slice."""
    if trace_fingerprint(source) != entry["source_trace_fingerprint"]:
        raise ValueError(f"{entry['slice_id']}: source trace is not the frozen #72 source")
    start, end = int(entry["start_bin"]), int(entry["end_bin_exclusive"])
    interval = source.control_interval_seconds
    counts = s2r.slice_counts(source, start, end - start)
    if s2r._sha256({"counts_per_bin": counts, "interval": interval}) != entry["slice_checksum"]:
        raise ValueError(f"{entry['slice_id']}: slice checksum differs from #72")
    return WorkloadTrace(source.request_rates[start:end], interval)


def source_window(entry: Mapping[str, Any]) -> SourceWindow:
    return SourceWindow(int(entry["start_bin"]), SOURCE_EPISODE_TICKS)


def offset_profile(profile: HistoricalDemandProfile, start_tick: int) -> HistoricalDemandProfile:
    """The profile as seen from a slice starting at source tick ``start_tick``.

    ``offset_profile(p, b).rate_at(t) == p.rate_at(b + t)`` for every ``t >= 0``,
    so a slice reads profile ticks ``b, b + 1, ...`` and never restarts at 0.
    """
    if not 0 <= start_tick < profile.ticks:
        raise ValueError("slice start is outside the profile")
    return replace(profile, rates=profile.rates[start_tick:])


def input_identity(
    spec_identity: str,
    case: Case,
    trace: WorkloadTrace,
    profile: HistoricalDemandProfile | None,
) -> str:
    payload = {
        "spec_id": spec_identity,
        "case_id": case.case_id,
        "rates": list(trace.request_rates),
        "control_interval_seconds": trace.control_interval_seconds,
        "training_run_id": case.variant.training_run_id,
        "profile_id": profile.profile_id if profile is not None else None,
        "slice_start_bin": case.slice["start_bin"] if case.slice else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# --- controllers --------------------------------------------------------------------------------

LearnedFactory = Callable[[Variant, AutoscalingEnv, bool], Controller]


def make_controller(
    variant: Variant,
    env: AutoscalingEnv,
    *,
    profile: HistoricalDemandProfile | None,
    learned: LearnedFactory,
    robustness_path: bool,
) -> Controller:
    """A fresh controller for one case (no state is carried between cases)."""
    config = env.config
    contract = env.action_contract
    params = variant.params
    if contract.semantics != DESIRED_REPLICAS_V1:
        raise ValueError("#46 runs under desired-replicas-v1 only")
    if variant.controller == "static":
        return StaticController(
            int(_number(params["target_replicas"])), config.replicas, action_contract=contract
        )
    if variant.controller == "random":
        return RandomController(action_contract=contract)  # seeded by reset(evaluation_seed)
    if variant.controller == "threshold":
        return ThresholdController(
            low_threshold=_number(params["low_threshold"]),
            high_threshold=_number(params["high_threshold"]),
            cooldown_ticks=int(_number(params["cooldown_ticks"])),
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            action_contract=contract,
        )
    if variant.controller == "predictive":
        return PredictiveController.from_config(
            config,
            history_window_ticks=int(_number(params["history_window_ticks"])),
            target_utilization=_number(params["target_utilization"]),
        )
    if variant.controller == "predictive-seasonal":
        if profile is None:  # identity checked when inputs load (load_inputs)
            raise ValueError("predictive-seasonal-v1 needs the frozen Azure TRAIN profile")
        return ProactivePredictiveController.from_config(config, profile=profile)
    if variant.learned:
        return learned(variant, env, robustness_path)
    raise ValueError(f"unknown controller {variant.controller!r}")


def _number(value: JsonValue) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def mlflow_learned_factory(cache_dir: Path, tracking_uri: str | None) -> LearnedFactory:
    """Load the exact #72 bundle of a learned variant; refuse any other model."""

    def load(variant: Variant, env: AutoscalingEnv, robustness_path: bool) -> Controller:
        from scalerl.rl import load_sb3_controller, read_model_bundle

        assert variant.training_run_id is not None
        if variant.model_artifact_uri != f"runs:/{variant.training_run_id}/model":
            raise ValueError(f"{variant.variant_id}: artifact URI is not its training run's")
        target = cache_dir / variant.training_run_id
        bundle = target / "model"
        if not (bundle / "model.zip").is_file():
            from mlflow import MlflowClient

            target.mkdir(parents=True, exist_ok=True)
            MlflowClient(tracking_uri).download_artifacts(
                variant.training_run_id, "model", str(target)
            )
        metadata, compatibility = read_model_bundle(bundle)
        if (
            metadata.training_run_id != variant.training_run_id
            or metadata.seed != variant.training_seed
            or metadata.algorithm != variant.controller
            or compatibility.action_semantics_version != DESIRED_REPLICAS_V1
        ):
            raise ValueError(f"{variant.variant_id}: bundle is not the frozen #72 artifact")
        return load_sb3_controller(bundle, env, robustness_evaluation=robustness_path)

    return load


# --- one episode ---------------------------------------------------------------------------------


def evaluate_case(
    case: Case,
    spec: Mapping[str, Any],
    trace: WorkloadTrace,
    *,
    profile: HistoricalDemandProfile | None,
    learned: LearnedFactory,
) -> tuple[dict[str, float | None], tuple[Mapping[str, Any], ...], Controller]:
    """Run one case; returns every raw metric, the physical step infos and the controller."""
    config = case.config(spec)
    weights = RewardWeights(**spec["reward"]["weights"])
    window = source_window(case.slice) if case.slice else None
    env = AutoscalingEnv(config, trace, weights, source_window=window)
    controller = make_controller(
        case.variant,
        env,
        profile=profile,
        learned=learned,
        robustness_path=case.robustness_path,
    )
    evaluation = evaluate_controller_episode(env, controller, seed=case.evaluation_seed)
    infos = evaluation.infos
    startup = s81.summarize_startup(infos, config)
    values: dict[str, float | int | None] = {
        **evaluation.metrics.as_metrics(),
        **summarize_action_magnitude(infos).as_metrics(),
        **summarize_dynamics(infos).as_metrics(),
        **{key: getattr(startup, key.removeprefix("startup.")) for key in STARTUP_KEYS},
        **derived_metrics(infos, config),
    }
    metrics = {k: None if (v := values[k]) is None else float(v) for k in METRIC_KEYS}
    return metrics, infos, controller


def derived_metrics(
    infos: Sequence[Mapping[str, Any]], config: SimulatorConfig
) -> dict[str, float]:
    return {
        "episode_ticks": float(len(infos)),
        "sla_violation_count": float(sum(1 for info in infos if info["sla_violated"])),
        "arrived_requests": float(sum(info["arrived_requests"] for info in infos)),
        "processed_requests": float(sum(info["processed_requests"] for info in infos)),
        "dropped_requests": float(sum(info["dropped_requests"] for info in infos)),
        "final_queued_requests": float(infos[-1]["queued_requests"]),
        "replica_seconds": sum(info["infrastructure_cost"] for info in infos)
        / config.replicas.cost_per_hour
        * 3600.0,
    }


def infos_payload(case: Case, infos: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"case_id": case.case_id, "step_infos": [dict(info) for info in infos]}


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


# --- rows, tags, MLflow --------------------------------------------------------------------------


def case_tags(
    spec: Mapping[str, Any], spec_identity: str, case: Case, perturbed: Sequence[str]
) -> dict[str, str]:
    upstream = spec["upstream"]
    variant = case.variant
    tags = {
        "scalerl.heldout_spec_id": spec_identity,
        "scalerl.heldout_version": spec["experiment_version"],
        "scalerl.evaluation_case_id": case.case_id,
        "scalerl.case_kind": case.kind,
        "scalerl.input_id": str(case.input_id),
        "scalerl.controller_variant_id": variant.variant_id,
        "scalerl.controller_version": variant.version,
        "scalerl.reward_contract_id": spec["reward"]["contract_id"],
        "scalerl.reward_variant": spec["reward"]["variant"],
        "scalerl.protocol_id": upstream["sim_to_real_protocol_id"],
        "scalerl.replay_manifest_id": upstream["live_replay_manifest_id"],
        "scalerl.controller_manifest_id": upstream["controller_deployment_manifest_id"],
        "scalerl.dynamics_seed": str(case.dynamics_seed),
        "scalerl.evaluation_seed": str(case.evaluation_seed),
        "scalerl.startup_delay_model": case.startup_delay_model,
        "scalerl.startup_delay_seed": "none"
        if case.startup_delay_seed is None
        else str(case.startup_delay_seed),
        "scalerl.scenario_version": case.scenario_version,
    }
    if variant.learned:
        tags |= {
            "scalerl.model_source_run_id": str(variant.training_run_id),
            "scalerl.model_artifact_uri": str(variant.model_artifact_uri),
            "scalerl.training_seed": str(variant.training_seed),
            "scalerl.candidate_id": str(variant.candidate_id),
            "scalerl.candidate_set_id": str(variant.candidate_set_id),
        }
    if case.slice:
        tags |= {
            "scalerl.replay_slice_id": case.slice["slice_id"],
            "scalerl.replay_schedule_ids": ",".join(case.slice["schedule_ids"]),
            "scalerl.source_window_start_tick": str(case.slice["start_bin"]),
        }
    if perturbed:
        tags["scalerl.robustness.perturbed_compatibility"] = ",".join(perturbed)
    return tags


def result_row(
    spec_identity: str,
    case: Case,
    *,
    run_id: str,
    git_sha: str,
    git_dirty: bool | None,
    metrics: Mapping[str, float | None],
    perturbed: Sequence[str],
    infos_sha256: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    variant = case.variant
    return {
        "row_schema": ROW_SCHEMA,
        "heldout_spec_id": spec_identity,
        "case_index": case.index,
        "case_id": case.case_id,
        "case_kind": case.kind,
        "input_id": case.input_id,
        "controller": variant.controller,
        "controller_variant_id": variant.variant_id,
        "controller_version": variant.version,
        "training_seed": variant.training_seed,
        "model_source_run_id": variant.training_run_id,
        "model_artifact_uri": variant.model_artifact_uri,
        "candidate_id": variant.candidate_id,
        "workload_id": case.workload_id,
        "workload_split": "test",
        "replay_slice_id": case.slice["slice_id"] if case.slice else None,
        "replay_schedule_ids": list(case.slice["schedule_ids"]) if case.slice else None,
        "scenario": case.scenario,
        "scenario_version": case.scenario_version,
        "capacity_jitter_fraction": case.capacity_jitter_fraction,
        "telemetry_delay_ticks": case.telemetry_delay_ticks,
        "startup_delay_model": case.startup_delay_model,
        "dynamics_seed": case.dynamics_seed,
        "startup_delay_seed": case.startup_delay_seed,
        "evaluation_seed": case.evaluation_seed,
        "action_semantics": spec["action_semantics"],
        "reward_contract_id": spec["reward"]["contract_id"],
        "perturbed_compatibility": ",".join(perturbed),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "mlflow_run_id": run_id,
        "step_infos_sha256": infos_sha256,
        **{key: metrics.get(key) for key in METRIC_KEYS},
    }


ROW_ARTIFACT: Final = "row.json"
INFOS_ARTIFACT: Final = "step-infos.json"


def _run_spec(case: Case, spec: Mapping[str, Any]) -> RunSpec:
    return RunSpec(
        run_kind="evaluate",
        controller=case.variant.controller,
        workload_id=case.workload_id,
        workload_split="test",
        simulator_config=case.config(spec),
        simulator_config_source="predeclared",
        evaluation_seeds=(case.evaluation_seed,),
        reward_weights=RewardWeights(**spec["reward"]["weights"]),
        hyperparameters={"controller_variant": case.variant.variant_id, **case.variant.params},
        robustness_scenario=case.scenario,
        robustness_version=case.scenario_version,
    )


def _recover(
    spec_identity: str, case: Case, tracking_uri: str | None, experiment_name: str
) -> dict[str, Any] | None:
    """The row of a FINISHED MLflow run of this exact case whose local row was lost."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`scalerl.heldout_spec_id` = '{spec_identity}' and "
            f"tags.`scalerl.evaluation_case_id` = '{case.case_id}' and "
            f"tags.`scalerl.input_id` = '{case.input_id}' and "
            "attributes.status = 'FINISHED'"
        ),
        max_results=2,
    )
    if not runs:
        return None
    if len(runs) > 1:
        raise ValueError(f"{case.case_id}: more than one finished run; resolve before resuming")
    with tempfile.TemporaryDirectory() as directory:
        path = client.download_artifacts(runs[0].info.run_id, f"heldout/{ROW_ARTIFACT}", directory)
        row: dict[str, Any] = json.loads(Path(path).read_text())
    if row["mlflow_run_id"] != runs[0].info.run_id or row["case_id"] != case.case_id:
        raise ValueError(f"{case.case_id}: recovered row does not match its run")
    return row


def git_state() -> tuple[str, bool | None]:
    from scalerl.mlops.spec import software_metadata

    software = software_metadata()
    return str(software["git_sha"]), software["git_dirty"]


# --- the runner ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """Everything a run reads besides the spec (injectable for tests)."""

    traces: Mapping[str, WorkloadTrace]
    profile: HistoricalDemandProfile
    learned: LearnedFactory


def load_inputs(
    spec: Mapping[str, Any], *, azure_csv: Path, out: Path, tracking_uri: str | None
) -> Inputs:
    manifest = load_benchmark_manifest()
    workloads = spec["workloads"]["primary_azure_test"]
    traces = build_workloads([manifest.get(w) for w in workloads], azure_csv_path=azure_csv)
    profile = HistoricalDemandProfile.from_benchmark(
        spec["upstream"]["historical_profile_train_workload_ids"], azure_csv_path=azure_csv
    )
    if profile.profile_id != spec["upstream"]["historical_profile_id"]:
        raise ValueError(f"historical profile {profile.profile_id} is not the frozen one")
    replay = spec["replay_references"]
    if trace_fingerprint(traces[replay["source_workload_id"]]) != replay["trace_fingerprint"]:
        raise ValueError("the replay source trace is not the frozen #72 source")
    return Inputs(traces, profile, mlflow_learned_factory(out / "models", tracking_uri))


def case_inputs(case: Case, inputs: Inputs) -> tuple[WorkloadTrace, HistoricalDemandProfile | None]:
    trace = inputs.traces[case.workload_id]
    profile = inputs.profile if case.variant.controller == "predictive-seasonal" else None
    if case.slice:
        trace = slice_trace(trace, case.slice)
        if profile is not None:
            profile = offset_profile(profile, int(case.slice["start_bin"]))
    return trace, profile


def run(
    spec: Mapping[str, Any],
    inputs: Inputs,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    require_clean: bool = True,
    select: Callable[[Case], bool] | None = None,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Evaluate every predeclared case once (resumable; one MLflow run per case)."""
    from scalerl.mlops import start_tracked_run

    identity = spec_id(spec)
    all_cases = generate_cases(spec)
    require_expected_counts(spec, all_cases)
    sha, dirty = git_state()
    if require_clean and dirty is not False:
        raise RuntimeError(f"held-out runs of record need a clean git tree (dirty={dirty})")
    cases = [c for c in all_cases if select is None or select(c)]
    resolved = []
    for case in cases:
        trace, profile = case_inputs(case, inputs)
        resolved.append(replace(case, input_id=input_identity(identity, case, trace, profile)))
    by_id = {c.case_id: c for c in resolved}

    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout-evaluation-spec.json").write_text(_dump(spec))
    raw_path = out / "raw-results.jsonl"
    _repair_torn_tail(raw_path)
    done: set[str] = set()
    for row in read_rows(raw_path):
        if row["heldout_spec_id"] != identity or row["case_id"] not in by_id:
            continue
        if row["input_id"] != by_id[row["case_id"]].input_id:
            raise ValueError(f"{row['case_id']}: computed from different inputs; new output dir")
        if row["case_id"] in done:
            raise ValueError(f"{row['case_id']}: duplicate raw row")
        done.add(row["case_id"])

    for case in resolved:
        if case.case_id in done:
            continue
        recovered = _recover(identity, case, tracking_uri, experiment_name)
        if recovered is not None:
            _append_row(raw_path, recovered)
            done.add(case.case_id)
            continue
        trace, profile = case_inputs(case, inputs)
        with start_tracked_run(
            _run_spec(case, spec),
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"heldout-{case.case_id.replace('|', '-')}",
            git_sha=None,
        ) as tracked:
            metrics, infos, controller = evaluate_case(
                case, spec, trace, profile=profile, learned=inputs.learned
            )
            perturbed = sorted(getattr(controller, "perturbed_compatibility", ()))
            for key, value in case_tags(spec, identity, case, perturbed).items():
                tracked.set_tag(key, value)
            infos_json = infos_payload(case, infos)
            infos_sha = payload_sha256(infos_json)
            tracked.set_tag("scalerl.step_infos_sha256", infos_sha)
            tracked.log_metrics({k: v for k, v in metrics.items() if v is not None})
            tracked.log_artifact_dict(INFOS_ARTIFACT, infos_json, artifact_path="heldout")
            row = result_row(
                identity,
                case,
                run_id=tracked.run_id,
                git_sha=sha,
                git_dirty=dirty,
                metrics=metrics,
                perturbed=perturbed,
                infos_sha256=infos_sha,
                spec=spec,
            )
            tracked.log_artifact_dict(ROW_ARTIFACT, row, artifact_path="heldout")
        _append_row(raw_path, row)
        done.add(case.case_id)
        progress(f"[{len(done)}/{len(resolved)}] {case.case_id}")

    order = {c.case_id: c.index for c in resolved}
    rows = [
        r for r in read_rows(raw_path) if r["heldout_spec_id"] == identity and r["case_id"] in order
    ]
    rows.sort(key=lambda r: order[r["case_id"]])
    if [r["case_id"] for r in rows] != [c.case_id for c in resolved]:
        raise ValueError("raw rows do not match the predeclared cases")
    return rows


# --- aggregation ---------------------------------------------------------------------------------


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    where = row["replay_slice_id"] or row["workload_id"]
    return (row["case_kind"], where, row["scenario"], row["controller_variant_id"])


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``heldout-descriptive-v1`` summaries; one group never mixes kinds/workloads/scenarios."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: r["case_index"]):
        groups.setdefault(_group_key(row), []).append(row)
    out = []
    for (kind, where, scenario, variant), group in groups.items():
        for metric in SUMMARY_METRICS:
            out.append(_summary(kind, where, scenario, variant, metric, "raw run", group))
        if group[0]["controller"] == "random":
            for metric in SUMMARY_METRICS:
                per_seed: dict[int, list[float]] = {}
                for row in group:
                    per_seed.setdefault(row["evaluation_seed"], []).append(row[metric])
                means = [statistics.fmean(v) for _, v in sorted(per_seed.items())]
                out.append(
                    _summary(
                        kind,
                        where,
                        scenario,
                        variant,
                        metric,
                        "evaluation-seed mean",
                        group,
                        values=means,
                    )
                )
    return out


def _summary(
    kind: str,
    where: str,
    scenario: str,
    variant: str,
    metric: str,
    unit: str,
    group: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[float] | None = None,
) -> dict[str, Any]:
    data = [float(r[metric]) for r in group] if values is None else list(values)
    stats = describe(data)
    return {
        "case_kind": kind,
        "workload_or_slice": where,
        "scenario": scenario,
        "controller_variant_id": variant,
        "controller": group[0]["controller"],
        "metric": metric,
        "replicate_unit": unit,
        "n": stats.n,
        "n_raw_runs": len(group),
        "n_evaluation_seeds": len({r["evaluation_seed"] for r in group}),
        "n_dynamics_seeds": len({r["dynamics_seed"] for r in group}),
        "n_startup_seeds": len({r["startup_delay_seed"] for r in group} - {None}),
        "mean": stats.mean,
        "median": stats.median,
        "std": stats.std,
        "min": stats.minimum,
        "max": stats.maximum,
        "q1": stats.q1,
        "q3": stats.q3,
        "group": group_label(group[0]),
    }


def group_label(row: Mapping[str, Any]) -> str:
    return "|".join(_group_key(row))


def group_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Every summary group's exact raw cases and MLflow runs (the traceability map)."""
    groups: dict[str, dict[str, list[str]]] = {}
    for row in sorted(rows, key=lambda r: r["case_index"]):
        entry = groups.setdefault(group_label(row), {"case_ids": [], "mlflow_run_ids": []})
        entry["case_ids"].append(row["case_id"])
        entry["mlflow_run_ids"].append(row["mlflow_run_id"])
    return groups


def paired_deltas(
    rows: Sequence[Mapping[str, Any]], reference: str = "threshold-v1"
) -> list[dict[str, Any]]:
    """``controller - Threshold`` per matched realization; descriptive only."""

    def realization(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["case_kind"],
            row["replay_slice_id"] or row["workload_id"],
            row["scenario"],
            row["dynamics_seed"],
            row["startup_delay_seed"],
        )

    anchors = {realization(r): r for r in rows if r["controller_variant_id"] == reference}
    out = []
    for row in sorted(rows, key=lambda r: r["case_index"]):
        anchor = anchors.get(realization(row))
        if row["controller_variant_id"] == reference or anchor is None:
            continue
        out.append(
            {
                "case_id": row["case_id"],
                "reference_case_id": anchor["case_id"],
                **{f"delta_{m}": row[m] - anchor[m] for m in DELTA_METRICS},
            }
        )
    return out


def diagnostic_flags(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """The predeclared descriptive flags per group (never a ranking or a selection)."""
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    means = {
        key: {m: statistics.fmean(float(r[m]) for r in group) for m in SUMMARY_METRICS}
        for key, group in groups.items()
    }
    flags = {}
    for key, m in sorted(means.items()):
        kind, where, scenario, _ = key
        reference = means.get((kind, where, scenario, "threshold-v1"))
        worse = (
            reference is not None
            and m["sla_violation_rate"] > reference["sla_violation_rate"] + FLAG_TOLERANCE
        )
        found = []
        if m["normalized_cost"] >= 0.9:
            found.append("full_fleet")
        if worse and m["normalized_cost"] < 0.6:
            found.append("underprovisioning")
        if m["churn_rate"] >= 0.3:
            found.append("thrashing")
        if worse and m["scaling_actions"] <= 2:
            found.append("churn_aversion")
        if m["queue_pressure"] >= 0.2:
            found.append("persistent_backlog")
        if all(r["action.total_absolute_replica_change"] == 0 for r in groups[key]):
            found.append("never_scaled")
        flags[group_label(groups[key][0])] = found
    return flags


# --- the frozen result artifact ------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def build_results(
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    pre_run_sha: str,
    spec_at_pre_run: Mapping[str, Any],
    superseded_run_ids: Sequence[str] = (),
    superseded_note: str | None = None,
) -> dict[str, Any]:
    """``heldout-results-v1``: identities, counts, run IDs, summaries; no winner."""
    identity = spec_id(spec)
    if spec_id(spec_at_pre_run) != identity:
        raise ValueError("the spec committed at the pre-run SHA is not this spec")
    cases = generate_cases(spec)
    require_expected_counts(spec, cases)
    if [r["case_id"] for r in rows] != [c.case_id for c in cases]:
        raise ValueError("results need exactly one row per predeclared case, in order")
    execution = sorted({r["git_sha"] for r in rows})
    if any(r["git_dirty"] is not False for r in rows):
        raise ValueError("a run of record came from a dirty tree")
    main = [r for r in rows if r["case_kind"] != "replay"]
    replay_rows = [r for r in rows if r["case_kind"] == "replay"]
    counts = case_counts(cases)
    actual = {
        "primary": sum(1 for r in rows if r["case_kind"] == "primary"),
        "startup": sum(1 for r in rows if r["case_kind"] == "startup"),
    }
    actual |= {"main": actual["primary"] + actual["startup"], "replay": len(replay_rows)}
    references = []
    for row in replay_rows:
        references.append(
            {
                "slice_id": row["replay_slice_id"],
                "controller_variant_id": row["controller_variant_id"],
                "mlflow_run_id": row["mlflow_run_id"],
                "case_id": row["case_id"],
                "maps_to_schedule_ids": row["replay_schedule_ids"],
                "independent_simulator_replicates": 1,
                "metrics": {m: row[m] for m in SUMMARY_METRICS},
            }
        )
    results = {
        "results_version": RESULTS_VERSION,
        "source_issue": "#46",
        "heldout_spec_id": identity,
        "heldout_version": spec["experiment_version"],
        "pre_run_sha": pre_run_sha,
        "execution_shas": execution,
        "benchmark_version": spec["benchmark_version"],
        "workload_ids": spec["workloads"]["primary_azure_test"],
        "controllers": spec["controllers"],
        "reward": spec["reward"],
        "action_semantics": spec["action_semantics"],
        "upstream": spec["upstream"],
        "mlflow_experiment": spec["mlflow"]["experiment"],
        "expected_counts": counts,
        "actual_counts": actual,
        "runs": {r["case_id"]: r["mlflow_run_id"] for r in rows},
        "summary_method": SUMMARY_METHOD,
        "groups": group_runs(main),
        "summaries": summarize(main),
        "paired_deltas_vs_threshold": paired_deltas(main),
        "diagnostic_flags": diagnostic_flags(main),
        "replay_references": references,
        "superseded_run_ids": sorted(superseded_run_ids),
        "superseded_note": superseded_note,
        "test_data_used_for_training": False,
        "test_data_used_for_model_selection": False,
        "post_hoc_protocol_change": False,
        "declares_winner": False,
    }
    normalized: dict[str, Any] = json.loads(_dump(results))
    normalized["results_id"] = s2r.content_id(normalized)
    return normalized


def results_id(results: Mapping[str, Any]) -> str:
    return s2r.content_id({k: v for k, v in results.items() if k != "results_id"})


# --- outputs -------------------------------------------------------------------------------------


def write_reports(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: r["case_index"])
    if not ordered:
        return
    _write_csv(out / "raw.csv", ordered)
    _write_csv(out / "summary.csv", summarize([r for r in ordered if r["case_kind"] != "replay"]))
    deltas = paired_deltas([r for r in ordered if r["case_kind"] != "replay"])
    if deltas:
        _write_csv(out / "paired-deltas.csv", deltas)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv(row.get(k)) for k in fields})
    Path(handle.name).chmod(0o644)
    Path(handle.name).replace(path)


def _csv(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return " ".join(map(str, value))
    return "" if value is None else value


def describe_plan(spec: Mapping[str, Any], *, tracking_uri: str | None, out: Path) -> str:
    cases = generate_cases(spec)
    require_expected_counts(spec, cases)
    counts = case_counts(cases)
    sha, dirty = git_state()
    learned = [v for v in spec["controllers"] if v["controller"] in ("dqn", "ppo")]
    lines = [
        f"heldout spec        {HELDOUT_VERSION} {spec_id(spec)}",
        f"git                 {sha} git_dirty={str(dirty).lower()}",
        f"primary cases       {counts['primary']} (expected {EXPECTED_PRIMARY})",
        f"startup cases       {counts['startup']} (expected {EXPECTED_STARTUP})",
        f"main total          {counts['main']} (expected {EXPECTED_MAIN})",
        f"replay references   {counts['replay']} (expected {EXPECTED_REPLAY})",
        f"workloads           {', '.join(spec['workloads']['primary_azure_test'])}",
        f"controllers         {', '.join(v['variant_id'] for v in spec['controllers'])}",
        *(
            f"{v['controller']:<19} {v['training_run_id']} ({v['model_artifact_uri']}, "
            f"training seed {v['training_seed']})"
            for v in learned
        ),
        f"reward              {spec['reward']['variant']} {spec['reward']['contract_id']}",
        f"action semantics    {spec['action_semantics']}",
        f"mlflow              {tracking_uri} / {spec['mlflow']['experiment']}",
        f"output              {out}",
    ]
    return "\n".join(lines)


# --- command line --------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{HELDOUT_VERSION} (#46)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("spec", "check", "plan", "run", "freeze", "summarize"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--bench", type=Path, default=BENCH)
        cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        cmd.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
        if name == "run":
            cmd.add_argument("--azure-csv", type=Path, default=DEFAULT_AZURE_CSV)
        if name == "freeze":
            cmd.add_argument("--pre-run-sha", required=True)
            cmd.add_argument("--superseded-run-id", action="append", default=[])
            cmd.add_argument("--superseded-note", default=None)
    args = parser.parse_args(argv)
    spec_path = args.bench / DEFAULT_SPEC.name
    if args.command == "spec":
        spec_path.write_text(_dump(build_spec(args.bench)))
        print(f"{HELDOUT_VERSION} {spec_id(build_spec(args.bench))} -> {spec_path}")
        return 0
    spec = load_frozen_spec(spec_path, args.bench)
    if args.command == "check":
        print(f"{HELDOUT_VERSION} {spec_id(spec)}")
        return 0
    if args.command == "plan":
        print(describe_plan(spec, tracking_uri=args.tracking_uri, out=args.output_dir))
        return 0
    if args.command == "run":
        print(describe_plan(spec, tracking_uri=args.tracking_uri, out=args.output_dir))
        inputs = load_inputs(
            spec, azure_csv=args.azure_csv, out=args.output_dir, tracking_uri=args.tracking_uri
        )
        rows = run(spec, inputs, out=args.output_dir, tracking_uri=args.tracking_uri)
        write_reports(args.output_dir, rows)
        print(f"{len(rows)} rows in {args.output_dir}")
        return 0
    rows = sorted(
        (
            r
            for r in read_rows(args.output_dir / "raw-results.jsonl")
            if r["heldout_spec_id"] == spec_id(spec)
        ),
        key=lambda r: r["case_index"],
    )
    if args.command == "summarize":
        write_reports(args.output_dir, rows)
        print(f"{len(rows)} rows summarized")
        return 0
    at_pre_run = json.loads(_git("show", f"{args.pre_run_sha}:{DEFAULT_SPEC.as_posix()}"))
    results = build_results(
        spec,
        rows,
        pre_run_sha=_git("rev-parse", args.pre_run_sha).strip(),
        spec_at_pre_run=at_pre_run,
        superseded_run_ids=args.superseded_run_id,
        superseded_note=args.superseded_note,
    )
    (args.bench / DEFAULT_RESULTS.name).write_text(_dump(results))
    print(f"{RESULTS_VERSION} {results['results_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
