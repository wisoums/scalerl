"""Frozen benchmark-v2 workload-generalization methodology (#115).

This module freezes how the next ScaleRL benchmark is constructed before any
generalist DQN/PPO training begins. It intentionally does not contain concrete
future workload IDs: #116 builds source catalogs and #117 materializes the exact
train/validation/test manifest under the rules frozen here.

The project is a student learning project. The protocol is deliberately
narrow: Stage A asks whether broader workload training improves generalization
inside the current simplified ScaleRL simulator. Environment-generalization
changes (fleet size, cadence, capacity, SLA, observation/action redesign) are
owned by #120.

Commands::

    python -m scalerl.evaluation.generalization_protocol write
    python -m scalerl.evaluation.generalization_protocol check
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

from scalerl.environment import ActionConfig, DESIRED_REPLICAS_V1, SimulatorConfig
from scalerl.environment.observation import OBSERVATION_VERSION

PROTOCOL_VERSION: Final = "benchmark-v2-generalization-protocol-v1"
STAGE: Final = "generalist-workload-v1"
BENCHMARK_DIR: Final = Path("benchmarks/v2")
DEFAULT_PROTOCOL: Final = BENCHMARK_DIR / "generalization-protocol-v1.json"

SELECTION_SPEC_ID: Final = "418876d6c8e9"
ACTION_DECISION_ID: Final = "0eb5562b01e6"
ACTION_EXPERIMENT_ID: Final = "899dfbb64217"
REWARD_ABLATION_ID: Final = "f5aff8f1e8c2"
REWARD_CONTRACT_ID: Final = "37158c261364"
HELDOUT_SPEC_ID: Final = "b32b2f3d3dd6"
HELDOUT_RESULTS_ID: Final = "d9d3fb985f2d"

TRAINING_SEEDS: Final = (0, 1, 2, 3, 4)
SEARCH_TRAINING_SEED: Final = 0
EVALUATION_SEED: Final = 0

SPLIT_VERSION: Final = "grouped-hash-split-v1"
SPLIT_SALT: Final = "scalerl-benchmark-v2-v1"
GROUP_BUCKET_MODULUS: Final = 10_000
DEVELOPMENT_GROUP_MAX: Final = 7_999
VALIDATION_GROUP_MAX: Final = 8_999
TEMPORAL_TEST_BUCKET: Final = 0
WINDOW_BUCKET_MODULUS: Final = 10
MIN_COMPOSITIONAL_CELL_WINDOWS: Final = 4

SAMPLER_VERSION: Final = "hierarchical-balanced-sampler-v1"
SAMPLER_RNG_DOMAIN: Final = 0x73616D70
SYNTHETIC_RNG_DOMAIN: Final = 0x73796E74

DOMAIN_HOLDOUT_MIN_REAL_SOURCES: Final = 3
DOMAIN_HOLDOUT_PRIORITY: Final = (
    "alibaba-microservices-2021",
    "azure-functions-2019",
    "azure-functions-2021",
)

INTENSITY_LABELS: Final = (
    "near-idle",
    "low",
    "single-replica-range",
    "multi-replica-moderate",
    "multi-replica-high",
    "near-saturation",
    "overload",
)
SHAPE_LABELS: Final = (
    "steady",
    "ramp-up",
    "ramp-down",
    "periodic",
    "spike",
    "bursty",
    "regime-switch",
    "noisy-other",
)

CATASTROPHIC_SLA_EXCESS: Final = 0.05
GROUP_SLA_EXCESS: Final = 0.02
FLOAT_TOLERANCE: Final = 1e-12


def content_id(payload: Any) -> str:
    """SHA-256 of canonical JSON, first 12 hex digits."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:12]


def stable_hash(*parts: object) -> str:
    """Stable SHA-256 hex digest for split/ranking identities."""
    encoded = "|".join(str(part) for part in parts).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_bucket(*parts: object, modulus: int) -> int:
    """Map an identity deterministically to range(modulus)."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(stable_hash(*parts)[:16], 16) % modulus


def stage_a_config() -> SimulatorConfig:
    """Exact Stage-A simulator contract."""
    return SimulatorConfig(action=ActionConfig(semantics=DESIRED_REPLICAS_V1))


def classify_mean_intensity(mean_rps: float) -> str:
    """Classify mean demand under the frozen 50 rps/replica, 500 rps fleet."""
    if not math.isfinite(mean_rps) or mean_rps < 0:
        raise ValueError("mean_rps must be finite and non-negative")
    if mean_rps <= 5.0:
        return "near-idle"
    if mean_rps <= 25.0:
        return "low"
    if mean_rps <= 50.0:
        return "single-replica-range"
    if mean_rps <= 150.0:
        return "multi-replica-moderate"
    if mean_rps <= 400.0:
        return "multi-replica-high"
    if mean_rps <= 500.0:
        return "near-saturation"
    return "overload"


def classify_peak_pressure(peak_rps: float) -> str:
    """Independent peak-pressure label so low-mean spikes are not hidden."""
    if not math.isfinite(peak_rps) or peak_rps < 0:
        raise ValueError("peak_rps must be finite and non-negative")
    if peak_rps <= 50.0:
        return "at-or-below-one-replica"
    if peak_rps <= 400.0:
        return "multi-replica"
    if peak_rps <= 500.0:
        return "near-saturation"
    return "overload"


def _quarter_means(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) < 4:
        raise ValueError("traffic-shape classification needs at least four samples")
    edges = [round(i * len(values) / 4) for i in range(5)]
    means: list[float] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        chunk = values[start:stop]
        if not chunk:
            raise ValueError("traffic-shape quarter is empty")
        means.append(statistics.fmean(chunk))
    return means[0], means[1], means[2], means[3]


def _pearson_lag(values: Sequence[float], lag: int) -> float:
    left = values[:-lag]
    right = values[lag:]
    if len(left) < 3:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return 0.0
    return numerator / math.sqrt(left_ss * right_ss)


def traffic_shape_statistics(request_rates: Sequence[float]) -> dict[str, float]:
    """Workload-only statistics for traffic-shape-heuristic-v1."""
    values = tuple(float(value) for value in request_rates)
    if len(values) < 4:
        raise ValueError("traffic-shape classification needs at least four samples")
    if any((not math.isfinite(value) or value < 0.0) for value in values):
        raise ValueError("request rates must be finite and non-negative")

    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    peak = max(values)
    cv = stdev / mean if mean > 0.0 else 0.0
    peak_to_mean = peak / mean if mean > 0.0 else 0.0
    q0, q1, q2, q3 = _quarter_means(values)
    diffs = [b - a for a, b in zip(values[:-1], values[1:], strict=True)]
    positive_fraction = sum(diff > 0.0 for diff in diffs) / len(diffs)
    negative_fraction = sum(diff < 0.0 for diff in diffs) / len(diffs)
    fraction_above_2mean = (
        sum(value > 2.0 * mean for value in values) / len(values)
        if mean > 0.0
        else 0.0
    )

    max_lag = min(40, len(values) // 3)
    periodic_score = max(
        (_pearson_lag(values, lag) for lag in range(2, max_lag + 1)),
        default=0.0,
    )
    quarter_floor = max(min(q0, q1, q2, q3), 1.0)
    quarter_ratio = max(q0, q1, q2, q3) / quarter_floor

    return {
        "mean_rps": mean,
        "peak_rps": peak,
        "coefficient_of_variation": cv,
        "peak_to_mean_ratio": peak_to_mean,
        "first_quarter_mean_rps": q0,
        "second_quarter_mean_rps": q1,
        "third_quarter_mean_rps": q2,
        "last_quarter_mean_rps": q3,
        "positive_step_fraction": positive_fraction,
        "negative_step_fraction": negative_fraction,
        "fraction_above_2x_mean": fraction_above_2mean,
        "periodicity_score": periodic_score,
        "max_quarter_mean_ratio": quarter_ratio,
    }


def classify_primary_shape(request_rates: Sequence[float]) -> str:
    """Deterministic primary shape for balancing/reporting."""
    stats = traffic_shape_statistics(request_rates)
    mean = stats["mean_rps"]
    if mean == 0.0 or (
        stats["coefficient_of_variation"] <= 0.15
        and stats["peak_to_mean_ratio"] <= 1.5
    ):
        return "steady"

    first = stats["first_quarter_mean_rps"]
    last = stats["last_quarter_mean_rps"]
    if (
        last >= 2.0 * max(first, 1.0)
        and last - first >= 10.0
        and stats["positive_step_fraction"] >= 0.60
    ):
        return "ramp-up"
    if (
        first >= 2.0 * max(last, 1.0)
        and first - last >= 10.0
        and stats["negative_step_fraction"] >= 0.60
    ):
        return "ramp-down"
    if stats["periodicity_score"] >= 0.65:
        return "periodic"
    if (
        stats["peak_to_mean_ratio"] >= 4.0
        and stats["fraction_above_2x_mean"] <= 0.10
    ):
        return "spike"
    if (
        stats["coefficient_of_variation"] >= 0.75
        or stats["peak_to_mean_ratio"] >= 2.5
    ):
        return "bursty"
    if stats["max_quarter_mean_ratio"] >= 2.0:
        return "regime-switch"
    return "noisy-other"


def traffic_shape_tags(request_rates: Sequence[float]) -> tuple[str, ...]:
    """Auxiliary deterministic tags; not used as the sole sampler key."""
    stats = traffic_shape_statistics(request_rates)
    tags: list[str] = []
    if stats["first_quarter_mean_rps"] <= 5.0 and stats["peak_rps"] > 50.0:
        tags.append("quiet-to-spike")
    if (
        stats["first_quarter_mean_rps"] <= 5.0
        and max(
            stats["third_quarter_mean_rps"],
            stats["last_quarter_mean_rps"],
        )
        >= 50.0
    ):
        tags.append("flash-crowd")
    if stats["periodicity_score"] >= 0.65:
        tags.append("repeated-wave")
    if stats["peak_rps"] > 500.0:
        tags.append("fleet-overload")
    return tuple(tags)


def group_role(source_id: str, group_id: str) -> str:
    """Development/validation/application-test assignment by app/service."""
    bucket = stable_bucket(
        SPLIT_SALT,
        "group",
        source_id,
        group_id,
        modulus=GROUP_BUCKET_MODULUS,
    )
    if bucket <= DEVELOPMENT_GROUP_MAX:
        return "development"
    if bucket <= VALIDATION_GROUP_MAX:
        return "validation"
    return "test-b-application"


def development_window_role(source_id: str, group_id: str, window_id: str) -> str:
    """TRAIN vs temporal TEST-A for windows of a development group."""
    bucket = stable_bucket(
        SPLIT_SALT,
        "window",
        source_id,
        group_id,
        window_id,
        modulus=WINDOW_BUCKET_MODULUS,
    )
    return "test-a-temporal" if bucket == TEMPORAL_TEST_BUCKET else "train"


def choose_domain_holdout(
    eligible_real_request_sources: Sequence[str],
) -> str | None:
    """Choose TEST-C independently of controller outcomes."""
    eligible = set(eligible_real_request_sources)
    ordered = [source for source in DOMAIN_HOLDOUT_PRIORITY if source in eligible]
    if len(ordered) < DOMAIN_HOLDOUT_MIN_REAL_SOURCES:
        return None
    return ordered[0]


def choose_compositional_cell(
    source_id: str,
    cell_counts: Mapping[tuple[str, str], int],
) -> tuple[str, str] | None:
    """Pick one source-local TEST-B compositional cell from workload metadata."""
    cells = {
        cell: count
        for cell, count in cell_counts.items()
        if count >= MIN_COMPOSITIONAL_CELL_WINDOWS
    }
    eligible: list[tuple[str, str]] = []
    for intensity, shape in cells:
        other = set(cells) - {(intensity, shape)}
        if (
            any(other_intensity == intensity for other_intensity, _ in other)
            and any(other_shape == shape for _, other_shape in other)
        ):
            eligible.append((intensity, shape))
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda cell: stable_hash(
            SPLIT_SALT,
            "compositional",
            source_id,
            cell[0],
            cell[1],
        ),
    )


def _fixed_environment() -> dict[str, Any]:
    config = stage_a_config()
    return {
        "action_semantics": config.action.semantics,
        "observation_version": OBSERVATION_VERSION,
        "timing": {
            "control_interval_seconds": config.timing.control_interval_seconds,
            "episode_duration_seconds": config.timing.episode_duration_seconds,
            "episode_ticks": int(
                config.timing.episode_duration_seconds
                / config.timing.control_interval_seconds
            ),
        },
        "replicas": {
            "min_replicas": config.replicas.min_replicas,
            "max_replicas": config.replicas.max_replicas,
            "initial_replicas": config.replicas.initial_replicas,
            "startup_delay_seconds": config.replicas.startup_delay_seconds,
            "service_capacity_rps": config.replicas.service_capacity_rps,
            "fleet_capacity_rps": (
                config.replicas.max_replicas
                * config.replicas.service_capacity_rps
            ),
            "cost_per_hour": config.replicas.cost_per_hour,
        },
        "sla": {
            "latency_target_seconds": config.sla.latency_target_seconds,
        },
        "observation": {
            "traffic_history_ticks": config.observation.traffic_history_ticks,
        },
        "dynamics": {
            "capacity_jitter_fraction": config.dynamics.capacity_jitter_fraction,
            "dynamics_seed": config.dynamics.dynamics_seed,
            "telemetry_delay_ticks": config.dynamics.telemetry_delay_ticks,
            "startup_delay_model": config.dynamics.startup_delay_model,
            "startup_delay_seed": config.dynamics.startup_delay_seed,
        },
        "training_dynamics": (
            "nominal-only; #21 owns additional failure/non-stationarity stress"
        ),
    }


def _sources() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "synthetic-v2",
            "kind": "generated-request-rate",
            "stage_a_status": "eligible",
            "real_source": False,
            "request_trace": True,
            "license": (
                "ScaleRL-generated data; source code under repository MIT license"
            ),
            "official_url": None,
            "group_identity": "generator-family + generator-instance seed",
            "notes": (
                "deliberate coverage of regimes missing from public traces; "
                "never described as real data"
            ),
        },
        {
            "source_id": "azure-functions-2019",
            "kind": "serverless-invocation-counts",
            "stage_a_status": "eligible",
            "real_source": True,
            "request_trace": True,
            "license": "CC-BY Attribution License",
            "official_url": (
                "https://github.com/Azure/AzurePublicDataset/blob/master/"
                "AzureFunctionsDataset2019.md"
            ),
            "group_identity": (
                "HashApp (application is Azure Functions resource-allocation unit)"
            ),
            "notes": (
                "14 daily per-minute invocation-count files; aggregate functions "
                "to HashApp. One-minute rates may be repeated over two 30 s "
                "ScaleRL ticks; no sub-minute arrival pattern is invented. "
                "Duration/memory tables are auxiliary metadata."
            ),
        },
        {
            "source_id": "azure-functions-2021",
            "kind": "serverless-invocations",
            "stage_a_status": "eligible",
            "real_source": True,
            "request_trace": True,
            "license": "CC-BY Attribution License",
            "official_url": (
                "https://github.com/Azure/AzurePublicDataset/blob/master/"
                "AzureFunctionsInvocationTrace2021.md"
            ),
            "group_identity": "app",
            "notes": (
                "individual invocation end timestamp + duration; "
                "arrival=end_timestamp-duration; aggregate by app into 30 s bins"
            ),
        },
        {
            "source_id": "alibaba-microservices-2021",
            "kind": "microservice-call-rate",
            "stage_a_status": "conditional",
            "real_source": True,
            "request_trace": True,
            "license": (
                "must be verified by #116 before use; repository has no root "
                "license file"
            ),
            "official_url": (
                "https://github.com/alibaba/clusterdata/tree/master/"
                "cluster-trace-microservices-v2021"
            ),
            "group_identity": "msname",
            "notes": (
                "MS_Metrics_Table exposes MCR and RT metrics. #116 must define "
                "a documented ingress-call-rate mapping that avoids double-"
                "counting provider/consumer metrics and must verify usage/license "
                "terms before marking this source eligible."
            ),
        },
        {
            "source_id": "google-clusterdata-2019",
            "kind": "cluster-resource-usage",
            "stage_a_status": "deferred-stage-b",
            "real_source": True,
            "request_trace": False,
            "license": "CC-BY 4.0",
            "official_url": (
                "https://github.com/google/cluster-data/blob/master/"
                "ClusterData2019.md"
            ),
            "group_identity": "not applicable for Stage-A request traffic",
            "notes": (
                "resource requests/usage; explicitly not converted into a fake "
                "HTTP workload"
            ),
        },
        {
            "source_id": "sebs",
            "kind": "serverless-application-benchmark-suite",
            "stage_a_status": "deferred-sim-to-real",
            "real_source": False,
            "request_trace": False,
            "license": "BSD 3-Clause",
            "official_url": "https://github.com/spcl/serverless-benchmarks",
            "group_identity": "benchmark application",
            "notes": (
                "application/system benchmark suite, not a request-trace dataset"
            ),
        },
    ]


def build_protocol_payload() -> dict[str, Any]:
    """Protocol content excluding its self-derived protocol_id."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 115,
        "stage": STAGE,
        "research_question": (
            "Within ScaleRL's current simplified simulator and fixed system "
            "contract, does training DQN/PPO over a broader predeclared workload "
            "distribution improve held-out cost/SLA generalization relative to "
            "strong reactive and predictive baselines?"
        ),
        "project_scope": {
            "student_learning_project": True,
            "production_autoscaler_claim": False,
            "academic_completeness_claim": False,
            "universal_cloud_claim": False,
            "ai_assisted_development": True,
        },
        "upstream": {
            "selection_spec_id": SELECTION_SPEC_ID,
            "action_decision_id": ACTION_DECISION_ID,
            "action_experiment_spec_id": ACTION_EXPERIMENT_ID,
            "reward_ablation_spec_id": REWARD_ABLATION_ID,
            "reward_contract_id": REWARD_CONTRACT_ID,
            "heldout_v1_spec_id": HELDOUT_SPEC_ID,
            "heldout_v1_results_id": HELDOUT_RESULTS_ID,
        },
        "fixed_environment": _fixed_environment(),
        "reward": {
            "contract_version": "reward-contract-v1",
            "contract_id": REWARD_CONTRACT_ID,
            "variant": "full-cost-low-v1",
            "weights": {
                "latency": 1.0,
                "cost": 0.5,
                "sla": 1.0,
                "queue": 1.0,
                "churn": 0.1,
            },
            "reopened_by_stage_a": False,
        },
        "dataset_policy": {
            "sources": _sources(),
            "eligibility_rule": (
                "a Stage-A external source needs documented request/call-rate "
                "semantics, stable group identity, deterministic preparation, "
                "and verified usage/license terms; eligibility is decided "
                "without controller outcomes"
            ),
            "domain_holdout_priority": list(DOMAIN_HOLDOUT_PRIORITY),
            "domain_holdout_minimum_eligible_real_request_sources": (
                DOMAIN_HOLDOUT_MIN_REAL_SOURCES
            ),
        },
        "workload_unit": {
            "episode_duration_seconds": 3600.0,
            "control_interval_seconds": 30.0,
            "ticks": 120,
            "real_trace_amplitude_rescaled": False,
            "azure_2019_minute_mapping": (
                "aggregate function counts to application per minute; "
                "rate=count/60 rps; repeat that one-minute average rate for the "
                "two contained 30 s ticks"
            ),
            "azure_2021_mapping": (
                "arrival=end_timestamp-duration; aggregate by app into half-open "
                "30 s bins"
            ),
            "empty_window_policy": (
                "purely empty windows may be catalogued but are not required as "
                "training episodes; near-idle non-empty windows are retained"
            ),
        },
        "taxonomy": {
            "version": "workload-taxonomy-v1",
            "intensity": {
                "primary_metric": "mean_rps",
                "single_replica_capacity_rps": 50.0,
                "fleet_capacity_rps": 500.0,
                "labels_in_order": list(INTENSITY_LABELS),
                "boundaries": [
                    {"label": "near-idle", "max_mean_rps_inclusive": 5.0},
                    {"label": "low", "max_mean_rps_inclusive": 25.0},
                    {
                        "label": "single-replica-range",
                        "max_mean_rps_inclusive": 50.0,
                    },
                    {
                        "label": "multi-replica-moderate",
                        "max_mean_rps_inclusive": 150.0,
                    },
                    {
                        "label": "multi-replica-high",
                        "max_mean_rps_inclusive": 400.0,
                    },
                    {
                        "label": "near-saturation",
                        "max_mean_rps_inclusive": 500.0,
                    },
                    {"label": "overload", "max_mean_rps_inclusive": None},
                ],
                "also_record": [
                    "median_rps",
                    "p95_rps",
                    "peak_rps",
                    "mean_rps/50",
                    "mean_rps/500",
                    "peak_rps/50",
                    "peak_rps/500",
                    "fraction_above_50_rps",
                    "fraction_above_500_rps",
                ],
                "peak_pressure_labels": {
                    "<=50": "at-or-below-one-replica",
                    "(50,400]": "multi-replica",
                    "(400,500]": "near-saturation",
                    ">500": "overload",
                },
            },
            "traffic_shape": {
                "classifier_version": "traffic-shape-heuristic-v1",
                "primary_labels": list(SHAPE_LABELS),
                "ordered_rules": [
                    "steady: mean=0 OR (CV<=0.15 AND peak/mean<=1.5)",
                    (
                        "ramp-up: last-quarter>=2*max(first-quarter,1), "
                        "increase>=10 rps, positive-step fraction>=0.60"
                    ),
                    (
                        "ramp-down: first-quarter>=2*max(last-quarter,1), "
                        "decrease>=10 rps, negative-step fraction>=0.60"
                    ),
                    (
                        "periodic: max Pearson lag correlation for lags "
                        "2..min(40,n/3) >=0.65"
                    ),
                    "spike: peak/mean>=4 AND fraction above 2*mean<=0.10",
                    "bursty: CV>=0.75 OR peak/mean>=2.5",
                    (
                        "regime-switch: max/min quarter mean ratio "
                        "(floor 1 rps)>=2"
                    ),
                    "noisy-other: fallback",
                ],
                "auxiliary_tags": [
                    "quiet-to-spike",
                    "flash-crowd",
                    "repeated-wave",
                    "fleet-overload",
                    "synthetic-generator-label",
                ],
                "interpretation": (
                    "deterministic balancing/reporting strata, not semantic "
                    "ground truth"
                ),
            },
        },
        "grouping": {
            "group_id": (
                "source_id + source-native application/service identity"
            ),
            "window_id": (
                "source_id + group_id + source time/day/window identity"
            ),
            "no_cross_partition_rule": (
                "validation-only and TEST-B application-holdout groups cannot "
                "occur in TRAIN; TEST-A intentionally uses unseen windows from "
                "development groups"
            ),
            "hash": "SHA-256 over UTF-8 pipe-separated identity fields",
        },
        "split_policy": {
            "version": SPLIT_VERSION,
            "salt": SPLIT_SALT,
            "domain_holdout": {
                "test_tier": "TEST-C",
                "minimum_eligible_real_request_sources": (
                    DOMAIN_HOLDOUT_MIN_REAL_SOURCES
                ),
                "priority": list(DOMAIN_HOLDOUT_PRIORITY),
                "rule": (
                    "if at least three priority real request sources pass #116 "
                    "eligibility, hold out the first priority source entirely; "
                    "otherwise TEST-C is not materialized and the limitation is "
                    "recorded"
                ),
            },
            "non_domain_sources": {
                "group_bucket_modulus": GROUP_BUCKET_MODULUS,
                "development_buckets": [0, DEVELOPMENT_GROUP_MAX],
                "validation_buckets": [
                    DEVELOPMENT_GROUP_MAX + 1,
                    VALIDATION_GROUP_MAX,
                ],
                "test_b_application_buckets": [
                    VALIDATION_GROUP_MAX + 1,
                    GROUP_BUCKET_MODULUS - 1,
                ],
                "development_window_bucket_modulus": WINDOW_BUCKET_MODULUS,
                "test_a_temporal_bucket": TEMPORAL_TEST_BUCKET,
                "train_window_buckets": list(range(1, WINDOW_BUCKET_MODULUS)),
            },
            "compositional_holdout": {
                "test_tier": "TEST-B",
                "cell": "mean-intensity-stratum x primary-traffic-shape",
                "scope": (
                    "development groups of each non-domain-holdout source"
                ),
                "minimum_windows_in_cell": MIN_COMPOSITIONAL_CELL_WINDOWS,
                "eligibility": (
                    "after removing the cell, its intensity appears in another "
                    "development cell and its shape appears in another "
                    "development cell"
                ),
                "selection": (
                    "choose the eligible cell with smallest SHA-256 rank under "
                    "the frozen split salt; no controller metrics are used"
                ),
            },
            "test_tiers": [
                {
                    "id": "TEST-A",
                    "name": "temporal-holdout",
                    "meaning": (
                        "unseen windows from development apps/services"
                    ),
                },
                {
                    "id": "TEST-B",
                    "name": "structural-holdout",
                    "subtypes": [
                        "application-holdout",
                        "compositional-holdout",
                    ],
                },
                {
                    "id": "TEST-C",
                    "name": "domain-holdout",
                    "meaning": (
                        "entire source absent from TRAIN/VALIDATION when feasible"
                    ),
                },
            ],
        },
        "concrete_manifest_gate": {
            "owner": "#117 after #116 source catalogs",
            "required_before": "#118 first training run",
            "manifest_version": "benchmark-v2-workloads-v1",
            "must_include": [
                "exact source/workload/window IDs",
                "source and prepared-trace fingerprints",
                "TRAIN/VALIDATION/TEST membership",
                "group IDs",
                "taxonomy labels/statistics",
                "selected TEST-C source or explicit not-materialized reason",
                "selected compositional holdout cells",
                "coverage audit",
                "protocol_id",
            ],
            "freeze_rule": (
                "manifest is content-addressed, committed, and checkable before "
                "any #118 candidate training; after that its TEST membership is "
                "immutable"
            ),
        },
        "sampler": {
            "version": SAMPLER_VERSION,
            "scope": "TRAIN only",
            "hierarchy": [
                "source/domain",
                "mean-intensity stratum",
                "primary-traffic-shape",
                "workload/window",
            ],
            "weighting": (
                "equal probability among non-empty children at each hierarchy "
                "level; empty cells are omitted and their parent probability is "
                "renormalized"
            ),
            "synthetic_role": (
                "fill deliberate coverage gaps without receiving more "
                "source-level weight than any other TRAIN source"
            ),
            "training_seeds": list(TRAINING_SEEDS),
            "search_training_seed": SEARCH_TRAINING_SEED,
            "rng_domains": {
                "episode_sampler_spawn_key": SAMPLER_RNG_DOMAIN,
                "synthetic_workload_spawn_key": SYNTHETIC_RNG_DOMAIN,
                "algorithm_seed": (
                    "training seed passed separately to the RL library"
                ),
            },
            "episode_sequence_reproducible": True,
        },
        "selection": {
            "version": "benchmark-v2-cost-under-sla-v1",
            "scope": (
                "within one algorithm family; DQN and PPO are never ranked here"
            ),
            "screening_training_seed": SEARCH_TRAINING_SEED,
            "validation_condition": "nominal Stage-A environment only",
            "reference_controller": "frozen tuned threshold-v1",
            "feasibility": {
                "per_workload_catastrophic_guard": (
                    "candidate SLA violation rate <= matched Threshold rate + 0.05"
                ),
                "group": "source_id x mean-intensity stratum",
                "group_aggregation": (
                    "equal primary-shape mean of equal-workload SLA means"
                ),
                "group_guard": (
                    "candidate group SLA <= matched Threshold group SLA + 0.02"
                ),
                "floating_tolerance": FLOAT_TOLERANCE,
            },
            "primary_objective": "normalized_cost",
            "objective_aggregation": (
                "equal-source mean of equal-intensity mean of "
                "equal-primary-shape mean of equal-workload means"
            ),
            "tie_breaks": [
                "queue_pressure",
                "churn_rate",
                "sla_violation_rate",
                "candidate_id",
            ],
            "reward_used_for_selection": False,
            "no_feasible_candidate": (
                "select nothing; keep diagnostic fallback separate; never relax "
                "the SLA constraints after outcomes"
            ),
            "selected_config_retraining_seeds": list(TRAINING_SEEDS),
            "family_test_reporting": (
                "retain all five selected-config training seeds; #119 reports "
                "equal-seed family behavior rather than cherry-picking one seed"
            ),
            "canonical_deployment_seed_rule": (
                "validation only: among individually feasible seeds minimize the "
                "same balanced normalized-cost objective, then queue, churn, SLA, "
                "lowest seed; if none is feasible choose diagnostic minimum SLA "
                "excess and mark canonical_seed_validation_feasible=false"
            ),
            "search_space_and_budget_gate": (
                "#118 must commit DQN/PPO candidate spaces, trial counts and "
                "training budgets before the first candidate result; no post-hoc "
                "rescue budget"
            ),
        },
        "metrics": {
            "service": [
                "sla_violation_rate",
                "sla_violation_count",
                "mean_p95_latency_seconds",
                "max_p95_latency_seconds",
                "processed_requests",
                "dropped_requests",
                "final_queued_requests",
                "queue_pressure",
            ],
            "resource_cost": [
                "infrastructure_cost",
                "normalized_cost",
                "replica_seconds",
                "mean_replicas",
                "max_replicas_observed",
                "fraction_ticks_at_min_fleet",
                "fraction_ticks_at_max_fleet",
            ],
            "scaling": [
                "scaling_actions",
                "churn_rate",
                "total_absolute_replica_change",
                "max_absolute_replica_change_in_one_tick",
                "scale_up_actions",
                "scale_down_actions",
            ],
            "workload": [
                "request_count",
                "mean_rps",
                "median_rps",
                "p95_rps",
                "peak_rps",
                "mean_single_replica_load_ratio",
                "mean_fleet_load_ratio",
                "peak_single_replica_load_ratio",
                "peak_fleet_load_ratio",
                "source_id",
                "group_id",
                "window_id",
                "mean_intensity",
                "peak_pressure",
                "primary_shape",
                "shape_tags",
            ],
            "rl_secondary": [
                "episode_reward",
                "training_seed",
                "model_run_id",
            ],
            "overall_winner_score": None,
        },
        "aggregation": {
            "raw_unit": (
                "one controller/model-seed x workload x evaluation condition run"
            ),
            "family_hierarchy": [
                "equal training-seed mean within workload",
                "equal workload mean within primary-shape cell",
                "equal primary-shape mean within intensity stratum",
                "equal intensity-stratum mean within source",
                "equal source mean for optional overall summary",
            ],
            "report_separately": [
                "TEST-A",
                "TEST-B application-holdout",
                "TEST-B compositional-holdout",
                "TEST-C when materialized",
                "source/domain",
                "mean-intensity stratum",
                "primary-shape",
            ],
            "statistics": [
                "mean",
                "median",
                "sample_sd",
                "q1",
                "q3",
                "count",
            ],
            "grand_mean_is_secondary_only": True,
        },
        "generalization_terms": {
            "in-distribution": (
                "unseen workload samples from source/taxonomy support represented "
                "in TRAIN"
            ),
            "temporal-holdout": (
                "unseen time window from a development app/service"
            ),
            "application-holdout": (
                "entire app/service absent from TRAIN"
            ),
            "compositional-holdout": (
                "known factors, predeclared unseen joint taxonomy cell"
            ),
            "domain-holdout": (
                "entire request-trace source absent from TRAIN/VALIDATION"
            ),
        },
        "robustness_boundary": {
            "benchmark_v2_stage_a": [
                "traffic intensity diversity",
                "traffic-shape diversity",
                "source/application diversity",
            ],
            "issue_21": [
                "capacity loss/degradation",
                "additional startup-delay stress",
                "telemetry degradation",
                "explicit failure scenarios",
                (
                    "special non-stationary/distribution-shift stress beyond "
                    "ordinary workload diversity"
                ),
            ],
            "historical_robustness_artifacts_changed": False,
        },
        "stage_b_boundary": {
            "owner": "#120",
            "not_varied_in_stage_a": [
                "min/max replicas",
                "initial replicas",
                "service capacity per replica",
                "startup delay",
                "control interval",
                "SLA target",
                "observation shape/semantics",
                "action semantics",
            ],
            "possible_future": [
                "observation-v2",
                "action-v2",
                "environment context",
            ],
        },
        "leakage": {
            "held_out_test_used_for_design": False,
            "held_out_test_used_for_training": False,
            "held_out_test_used_for_selection": False,
            "v1_test_used_to_choose_specific_benchmark_v2_items": False,
            "controller_outcomes_used_for_split_or_taxonomy": False,
        },
        "future_test": {
            "unopened": True,
            "concrete_manifest_not_yet_materialized": True,
            "materialized_by": "#117 after #116",
            "immutable_before": "#118 first training run",
        },
    }


def build_protocol() -> dict[str, Any]:
    payload = build_protocol_payload()
    return {"protocol_id": content_id(payload), **payload}


def verify_upstream(bench_v1: Path = Path("benchmarks/v1")) -> None:
    """Fail if the frozen v1 IDs #115 relies on drift."""
    selection = json.loads(
        (bench_v1 / "selection-v2-cost-under-sla.json").read_text()
    )
    action = json.loads((bench_v1 / "action-contract-v2.json").read_text())
    reward = json.loads((bench_v1 / "reward-contract-v1.json").read_text())
    heldout_spec = json.loads(
        (bench_v1 / "heldout-evaluation-v1.json").read_text()
    )
    heldout_results = json.loads(
        (bench_v1 / "heldout-results-v1.json").read_text()
    )
    found = {
        "selection_spec_id": selection.get("selection_spec_id"),
        "action_decision_id": action.get("decision_id"),
        "action_experiment_spec_id": action.get("experiment_spec_id"),
        "reward_ablation_spec_id": reward.get("experiment_spec_id"),
        "reward_contract_id": reward.get("contract_id"),
        "heldout_v1_spec_id": heldout_spec.get("heldout_spec_id"),
        "heldout_v1_results_id": heldout_results.get("results_id"),
    }
    expected = build_protocol_payload()["upstream"]
    if found != expected:
        raise ValueError(
            f"frozen v1 upstream changed: found={found}, expected={expected}"
        )


def check_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Verify upstream and committed protocol against the code freeze."""
    verify_upstream()
    committed = json.loads(path.read_text())
    expected = build_protocol()
    if committed != expected:
        raise ValueError(
            "committed benchmark-v2 protocol differs from the frozen code"
        )
    payload = dict(committed)
    protocol_id = payload.pop("protocol_id")
    if protocol_id != content_id(payload):
        raise ValueError("benchmark-v2 protocol content ID is invalid")
    return committed


def write_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    """Write the pre-training methodology artifact."""
    verify_upstream()
    protocol = build_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "show-id"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write":
            protocol = write_protocol(args.protocol)
            print(f"wrote {args.protocol} ({protocol['protocol_id']})")
        elif args.command == "check":
            protocol = check_protocol(args.protocol)
            print(
                f"ok {protocol['protocol_version']} {protocol['protocol_id']}"
            )
        else:
            print(build_protocol()["protocol_id"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
