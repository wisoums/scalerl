"""Frozen benchmark-v2 workload-generalization protocol (#115).

Freezes *how* benchmark-v2 is built, trained on, selected and evaluated before
any generalist DQN/PPO model exists. It contains no concrete future workload
IDs: #116 prepares the approved sources, and #117 applies the rules frozen here
to materialize and commit the content-addressed ``benchmark-v2-workloads-v1``
manifest before #118 starts training. #119 later evaluates on its TEST tiers.

Stage A (``generalist-workload-v1``) changes only the training workload
distribution. The v1 system contract (``desired-replicas-v1``,
``scalerl-observation-v1``, 1-10 replicas, 50 rps/replica, 60 s startup, 30 s
control, 0.5 s SLA, ``reward-contract-v1`` / ``full-cost-low-v1``) is reused
unchanged; environment generalization is #120's study.

Every rule that #117-#119 must apply has a reference implementation here
(window roles, compositional cells, evaluation sampling, the TRAIN episode
sampler, validation selection, the clustered bootstrap), so the protocol is
testable rather than prose-only. No function reads a trace or a controller
outcome; all inputs are workload metadata.

#46 stays frozen v1 evidence: the v1 Azure hours are excluded from
benchmark-v2, and no benchmark-v2 item is chosen from #46 outcomes.

    python -m scalerl.evaluation.generalization_protocol write
    python -m scalerl.evaluation.generalization_protocol check
    python -m scalerl.evaluation.generalization_protocol show-id
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from scalerl.benchmarks import AzureWorkload, load_benchmark_manifest
from scalerl.environment import DESIRED_REPLICAS_V1, ActionConfig, SimulatorConfig
from scalerl.environment.observation import (
    OBSERVATION_VERSION,
    ObservationConstants,
    feature_names,
)
from scalerl.evaluation import sim_to_real as s2r

PROTOCOL_VERSION: Final = "benchmark-v2-generalization-protocol-v1"
STAGE: Final = "generalist-workload-v1"
MANIFEST_VERSION: Final = "benchmark-v2-workloads-v1"
BENCH_V1: Final = Path("benchmarks/v1")
BENCH_V2: Final = Path("benchmarks/v2")
DEFAULT_PROTOCOL: Final = BENCH_V2 / "generalization-protocol-v1.json"

UPSTREAM: Final = {
    "selection_spec_id": s2r.UPSTREAM["selection_spec_id"],
    "action_decision_id": s2r.UPSTREAM["action_decision_id"],
    "action_experiment_spec_id": s2r.UPSTREAM["action_experiment_spec_id"],
    "reward_ablation_spec_id": s2r.UPSTREAM["reward_ablation_spec_id"],
    "reward_contract_id": s2r.UPSTREAM["reward_contract_id"],
    "heldout_v1_spec_id": "b32b2f3d3dd6",
    "heldout_v1_results_id": "d9d3fb985f2d",
}
REWARD_VARIANT: Final = s2r.SELECTED_REWARD
REWARD_WEIGHTS: Final = {"latency": 1.0, "cost": 0.5, "sla": 1.0, "queue": 1.0, "churn": 0.1}
REFERENCE_CONTROLLER: Final = "threshold-v1"
REFERENCE_PARAMS: Final = {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3}

# --- sources -------------------------------------------------------------------------------------

SYNTHETIC: Final = "synthetic-v2"
AZURE_2019: Final = "azure-functions-2019"
AZURE_2021: Final = "azure-functions-2021"
ALIBABA_2021: Final = "alibaba-microservices-2021"
GOOGLE_2019: Final = "google-clusterdata-2019"
SEBS: Final = "sebs"
TRAIN_SOURCES: Final = (AZURE_2019, SYNTHETIC)  # sampler order
SOURCE_HOLDOUT: Final = AZURE_2021
STAGE_A_SOURCES: Final = (AZURE_2019, AZURE_2021, SYNTHETIC)

# --- windows and split ----------------------------------------------------------------------------

WINDOW_SECONDS: Final = 3600.0
CONTROL_INTERVAL_SECONDS: Final = 30.0
TICKS_PER_WINDOW: Final = 120
# Both Azure traces span 14 days = 336 aligned one-hour windows.
AZURE_WINDOWS: Final = 336
DEVELOPMENT_WINDOWS: Final = 264  # days 1-11
EMBARGO_WINDOWS: Final = 24  # day 12: separates TRAIN time from temporal TEST time
TEMPORAL_TEST_WINDOWS: Final = AZURE_WINDOWS - DEVELOPMENT_WINDOWS - EMBARGO_WINDOWS  # days 13-14

SPLIT_VERSION: Final = "grouped-temporal-split-v1"
SPLIT_SALT: Final = "scalerl-benchmark-v2-v1"
GROUP_BUCKET_MODULUS: Final = 10_000
DEVELOPMENT_GROUP_MAX: Final = 7_999  # 80% of groups
VALIDATION_GROUP_MAX: Final = 8_999  # 10% of groups; the last 10% are TEST-B application groups
SYNTHETIC_TRAIN_MAX: Final = 8_999  # 90% of synthetic groups train, 10% validation

EMPTY_REQUESTS: Final = 0
BEYOND_SCOPE_MEAN_RPS: Final = 1000.0  # 2x fleet capacity: no controller choice matters

COMPOSITIONAL_MAX_CELLS: Final = 2
COMPOSITIONAL_MIN_WINDOWS: Final = 20
COMPOSITIONAL_MIN_GROUPS: Final = 5
COMPOSITIONAL_SUPPORT_WINDOWS: Final = 20

VALIDATION_CAP_PER_CELL: Final = 4
TEST_CAP_PER_CELL: Final = 8

ROLE_TRAIN: Final = "train"
ROLE_VALIDATION: Final = "validation"
ROLE_TEST_A: Final = "test-a-temporal"
ROLE_TEST_B_APP: Final = "test-b-application"
ROLE_TEST_B_COMP: Final = "test-b-compositional"
ROLE_TEST_C: Final = "test-c-source"
ROLE_RESERVED: Final = "reserved"
ROLE_EXCLUDED: Final = "excluded"
TEST_ROLES: Final = (ROLE_TEST_A, ROLE_TEST_B_APP, ROLE_TEST_B_COMP, ROLE_TEST_C)
ROLES: Final = (ROLE_TRAIN, ROLE_VALIDATION, *TEST_ROLES, ROLE_RESERVED, ROLE_EXCLUDED)

# --- taxonomy -------------------------------------------------------------------------------------

TAXONOMY_VERSION: Final = "workload-taxonomy-v1"
SHAPE_CLASSIFIER_VERSION: Final = "traffic-shape-minute-v1"
SERVICE_CAPACITY_RPS: Final = 50.0
FLEET_CAPACITY_RPS: Final = 500.0
INTENSITY_BOUNDS: Final = (
    ("near-idle", 5.0),
    ("low", 25.0),
    ("single-replica", 50.0),
    ("multi-replica-moderate", 150.0),
    ("multi-replica-high", 400.0),
    ("near-saturation", 500.0),
    ("overload", BEYOND_SCOPE_MEAN_RPS),
)
INTENSITY_LABELS: Final = tuple(label for label, _ in INTENSITY_BOUNDS)
PEAK_PRESSURE_BOUNDS: Final = (
    ("within-one-replica", 50.0),
    ("multi-replica", 400.0),
    ("near-saturation", 500.0),
)
PEAK_PRESSURE_OVERLOAD: Final = "fleet-overload"
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
SHAPE_TAGS: Final = ("quiet-to-spike", "flash-crowd", "repeated-wave", "fleet-overload")
STEADY_MAX_CV: Final = 0.15
STEADY_MAX_PEAK_TO_MEAN: Final = 1.5
RAMP_MIN_RATIO: Final = 2.0
RATE_FLOOR_RPS: Final = 1.0
PERIODIC_MIN_LAG: Final = 3
PERIODIC_MIN_ACF: Final = 0.5
PERIODIC_MIN_DIP: Final = 0.3
SPIKE_MIN_PEAK_TO_MEDIAN: Final = 3.0
SPIKE_MAX_FRACTION_ABOVE_2X_MEDIAN: Final = 0.15
LEVEL_SHIFT_MIN_RATIO: Final = 2.0
LEVEL_SHIFT_MIN_R2: Final = 0.6
BURSTY_MIN_CV: Final = 0.75
BURSTY_MIN_PEAK_TO_MEAN: Final = 2.5
QUIET_MAX_RPS: Final = 5.0

# --- sampler, seeds, selection, statistics --------------------------------------------------------

SAMPLER_VERSION: Final = "hierarchical-balanced-sampler-v1"
SOURCE_WEIGHTS: Final = {AZURE_2019: 0.75, SYNTHETIC: 0.25}
# Spawn keys separating benchmark-v2 random streams (ASCII "samp", "synt", "boot").
SAMPLER_RNG_DOMAIN: Final = 0x73616D70
SYNTHETIC_RNG_DOMAIN: Final = 0x73796E74
BOOTSTRAP_RNG_DOMAIN: Final = 0x626F6F74
TRAINING_SEEDS: Final = (0, 1, 2, 3, 4)
SCREENING_TRAINING_SEED: Final = 0
EVALUATION_SEED: Final = 0
DYNAMICS_SEED: Final = 0
BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 0
BOOTSTRAP_LEVEL: Final = 0.95

SELECTION_VERSION: Final = "benchmark-v2-cost-under-sla-v1"
WINDOW_SLA_ALLOWANCE: Final = 0.05
GROUP_SLA_ALLOWANCE: Final = 0.02
FEASIBILITY_TOLERANCE: Final = 1e-12
SELECTION_METRICS: Final = ("sla_violation_rate", "normalized_cost", "queue_pressure", "churn_rate")
TIE_BREAKS: Final = ("queue_pressure", "churn_rate", "sla_violation_rate")


def content_id(payload: Any) -> str:
    """SHA-256 of canonical JSON, first 12 hex digits (the repository-wide content ID)."""
    return s2r.content_id(payload)


def stable_hash(*parts: object) -> str:
    """SHA-256 hex digest of UTF-8 pipe-joined identity fields."""
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def stable_bucket(*parts: object, modulus: int) -> int:
    """Deterministic bucket in ``range(modulus)`` from the first 64 bits of :func:`stable_hash`."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return int(stable_hash(*parts)[:16], 16) % modulus


def stage_a_config() -> SimulatorConfig:
    """The exact Stage-A simulator contract (the v1 #46 configuration)."""
    return SimulatorConfig(action=ActionConfig(semantics=DESIRED_REPLICAS_V1))


# --- identities -----------------------------------------------------------------------------------


def azure2019_window_id(hash_owner: str, hash_app: str, day: int, hour: int) -> str:
    """Window of one Azure 2019 application: trace day 1-14, hour 0-23."""
    if not 1 <= day <= 14 or not 0 <= hour <= 23:
        raise ValueError("Azure 2019 windows are day 1-14, hour 0-23")
    return f"az19:{hash_owner}:{hash_app}:d{day:02d}h{hour:02d}"


def azure2019_window_index(day: int, hour: int) -> int:
    if not 1 <= day <= 14 or not 0 <= hour <= 23:
        raise ValueError("Azure 2019 windows are day 1-14, hour 0-23")
    return (day - 1) * 24 + hour


def azure2021_window_id(app: str, window_index: int) -> str:
    """Window ``[3600 k, 3600 (k+1))`` trace-relative seconds of one Azure 2021 application."""
    _check_index(window_index)
    return f"az21:{app}:{int(window_index * WINDOW_SECONDS)}"


def synthetic_window_id(family: str, parameter_set: str, instance: int) -> str:
    return f"syn2:{family}:{parameter_set}:{instance}"


def synthetic_group_id(family: str, parameter_set: str) -> str:
    """Seeds of one parameter set are one group, so they never straddle TRAIN/VALIDATION."""
    return f"{family}:{parameter_set}"


def synthetic_instance_seed(instance: int) -> np.random.SeedSequence:
    if instance < 0:
        raise ValueError("synthetic instance index must be non-negative")
    return np.random.SeedSequence(instance, spawn_key=(SYNTHETIC_RNG_DOMAIN,))


def _check_index(window_index: int) -> None:
    if not 0 <= window_index < AZURE_WINDOWS:
        raise ValueError(f"window index must be in [0, {AZURE_WINDOWS})")


# --- taxonomy -------------------------------------------------------------------------------------


def _rates(request_rates: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in request_rates)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("request rates must be finite and non-negative")
    return values


def classify_intensity(mean_rps: float) -> str | None:
    """Mean-demand stratum relative to 50 rps/replica and a 500 rps fleet; None beyond scope."""
    if not math.isfinite(mean_rps) or mean_rps < 0.0:
        raise ValueError("mean_rps must be finite and non-negative")
    for label, upper in INTENSITY_BOUNDS:
        if mean_rps <= upper:
            return label
    return None


def classify_peak_pressure(peak_rps: float) -> str:
    """Analysis-only 30 s peak label, so a low mean does not hide a fleet-scale spike."""
    if not math.isfinite(peak_rps) or peak_rps < 0.0:
        raise ValueError("peak_rps must be finite and non-negative")
    for label, upper in PEAK_PRESSURE_BOUNDS:
        if peak_rps <= upper:
            return label
    return PEAK_PRESSURE_OVERLOAD


def minute_series(request_rates: Sequence[float]) -> tuple[float, ...]:
    """Average consecutive 30 s ticks into one-minute rates.

    Shapes are classified at one-minute resolution for every source: it is the
    native resolution of Azure 2019, whose minute rates are repeated over two
    30 s ticks, and a duplicated series would distort step and lag statistics.
    """
    values = _rates(request_rates)
    if len(values) % 2:
        raise ValueError("a 30 s series needs an even number of ticks")
    return tuple((values[i] + values[i + 1]) / 2.0 for i in range(0, len(values), 2))


def _quarter_means(values: Sequence[float]) -> tuple[float, float, float, float]:
    edges = [round(i * len(values) / 4) for i in range(5)]
    q0, q1, q2, q3 = (statistics.fmean(values[a:b]) for a, b in zip(edges, edges[1:], strict=False))
    return q0, q1, q2, q3


def autocorrelation(values: Sequence[float], lag: int) -> float:
    """Standard sample ACF (full-series mean and variance); 0 for a constant series."""
    if not 0 < lag < len(values):
        raise ValueError("lag must be in [1, len(values))")
    mean = statistics.fmean(values)
    denominator = sum((x - mean) ** 2 for x in values)
    if denominator == 0.0:
        return 0.0
    numerator = sum((values[t] - mean) * (values[t + lag] - mean) for t in range(len(values) - lag))
    return numerator / denominator


def periodic_lag(values: Sequence[float]) -> int | None:
    """Smallest lag whose ACF is a local peak >= 0.5 after a dip of >= 0.3 (at least 2 cycles)."""
    max_lag = len(values) // 2
    if max_lag < PERIODIC_MIN_LAG:
        return None
    acf = {lag: autocorrelation(values, lag) for lag in range(1, max_lag + 1)}
    for lag in range(PERIODIC_MIN_LAG, max_lag + 1):
        value = acf[lag]
        if value < PERIODIC_MIN_ACF or value < acf[lag - 1]:
            continue
        if lag < max_lag and value < acf[lag + 1]:
            continue
        if min(acf[k] for k in range(1, lag)) <= value - PERIODIC_MIN_DIP:
            return lag
    return None


def level_shift(values: Sequence[float]) -> tuple[float, float]:
    """Best single change point (each side >= n/8): (variance explained, high/low mean ratio).

    The ratio floors the lower segment mean at 1 rps; ties keep the earliest split.
    """
    n = len(values)
    mean = statistics.fmean(values)
    total = sum((x - mean) ** 2 for x in values)
    if total == 0.0:
        return 0.0, 1.0
    margin = math.ceil(n / 8)
    best_sse, best_ratio = math.inf, 1.0
    for split in range(margin, n - margin + 1):
        left, right = values[:split], values[split:]
        left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
        sse = sum((x - left_mean) ** 2 for x in left) + sum((x - right_mean) ** 2 for x in right)
        if sse < best_sse:
            low, high = sorted((left_mean, right_mean))
            best_sse, best_ratio = sse, high / max(low, RATE_FLOOR_RPS)
    return 1.0 - best_sse / total, best_ratio


def _excursion_runs(values: Sequence[float], level: float) -> int:
    runs, inside = 0, False
    for value in values:
        if value > level and not inside:
            runs += 1
        inside = value > level
    return runs


def traffic_shape_statistics(request_rates: Sequence[float]) -> dict[str, float]:
    """Workload-only statistics behind the shape label (30 s input, minute resolution)."""
    ticks = _rates(request_rates)
    minutes = minute_series(ticks)
    if len(minutes) < 8:
        raise ValueError("shape classification needs at least 8 minutes")
    mean = statistics.fmean(minutes)
    if mean == 0.0:
        raise ValueError("an empty window has no traffic shape; it is excluded")
    peak = max(minutes)
    median = max(statistics.median(minutes), RATE_FLOOR_RPS)
    q0, q1, q2, q3 = _quarter_means(minutes)
    lag = periodic_lag(minutes)
    shift_r2, shift_ratio = level_shift(minutes)
    return {
        "mean_rps": mean,
        "floored_median_rps": median,
        "peak_minute_rps": peak,
        "peak_tick_rps": max(ticks),
        "coefficient_of_variation": statistics.pstdev(minutes) / mean,
        "peak_to_mean_ratio": peak / mean,
        "peak_to_median_ratio": peak / median,
        "fraction_minutes_above_2x_median": sum(v > 2.0 * median for v in minutes) / len(minutes),
        "excursions_above_2x_median": float(_excursion_runs(minutes, 2.0 * median)),
        "quarter_1_mean_rps": q0,
        "quarter_2_mean_rps": q1,
        "quarter_3_mean_rps": q2,
        "quarter_4_mean_rps": q3,
        "level_shift_r2": shift_r2,
        "level_shift_ratio": shift_ratio,
        "periodic_lag_minutes": float(lag or 0),
    }


def classify_primary_shape(request_rates: Sequence[float]) -> str:
    """First matching rule of ``traffic-shape-minute-v1`` (a stratum, not semantic truth)."""
    s = traffic_shape_statistics(request_rates)
    q0, q1, q2, q3 = (s[f"quarter_{i}_mean_rps"] for i in range(1, 5))
    if (
        s["coefficient_of_variation"] <= STEADY_MAX_CV
        and s["peak_to_mean_ratio"] <= STEADY_MAX_PEAK_TO_MEAN
    ):
        return "steady"
    if q0 < q1 < q2 < q3 and q3 >= RAMP_MIN_RATIO * max(q0, RATE_FLOOR_RPS):
        return "ramp-up"
    if q0 > q1 > q2 > q3 and q0 >= RAMP_MIN_RATIO * max(q3, RATE_FLOOR_RPS):
        return "ramp-down"
    if s["periodic_lag_minutes"] > 0:
        return "periodic"
    if (
        s["peak_to_median_ratio"] >= SPIKE_MIN_PEAK_TO_MEDIAN
        and s["excursions_above_2x_median"] == 1
        and s["fraction_minutes_above_2x_median"] <= SPIKE_MAX_FRACTION_ABOVE_2X_MEDIAN
    ):
        return "spike"
    if (
        s["level_shift_r2"] >= LEVEL_SHIFT_MIN_R2
        and s["level_shift_ratio"] >= LEVEL_SHIFT_MIN_RATIO
    ):
        return "regime-switch"
    if (
        s["coefficient_of_variation"] >= BURSTY_MIN_CV
        or s["peak_to_mean_ratio"] >= BURSTY_MIN_PEAK_TO_MEAN
    ):
        return "bursty"
    return "noisy-other"


def traffic_shape_tags(request_rates: Sequence[float]) -> tuple[str, ...]:
    """Analysis-only auxiliary tags; never a sampler or split key."""
    s = traffic_shape_statistics(request_rates)
    quiet_start = s["quarter_1_mean_rps"] <= QUIET_MAX_RPS
    tags = {
        "quiet-to-spike": quiet_start and s["peak_tick_rps"] > SERVICE_CAPACITY_RPS,
        "flash-crowd": quiet_start
        and max(s["quarter_3_mean_rps"], s["quarter_4_mean_rps"]) >= SERVICE_CAPACITY_RPS,
        "repeated-wave": s["periodic_lag_minutes"] > 0,
        "fleet-overload": s["peak_tick_rps"] > FLEET_CAPACITY_RPS,
    }
    return tuple(tag for tag in SHAPE_TAGS if tags[tag])


# --- split ----------------------------------------------------------------------------------------


def window_period(window_index: int) -> str:
    """Contiguous time block shared by both Azure sources: development|embargo|temporal-test."""
    _check_index(window_index)
    if window_index < DEVELOPMENT_WINDOWS:
        return "development"
    if window_index < DEVELOPMENT_WINDOWS + EMBARGO_WINDOWS:
        return "embargo"
    return "temporal-test"


def group_role(source_id: str, group_id: str) -> str:
    """Azure 2019 owner group: development (80%) | validation (10%) | application-test (10%)."""
    bucket = stable_bucket(SPLIT_SALT, "group", source_id, group_id, modulus=GROUP_BUCKET_MODULUS)
    if bucket <= DEVELOPMENT_GROUP_MAX:
        return "development"
    if bucket <= VALIDATION_GROUP_MAX:
        return "validation"
    return "application-test"


def v1_azure_intervals() -> dict[str, tuple[tuple[float, float], ...]]:
    """Trace-relative ``[start, end)`` of every frozen v1 Azure 2021 window, by v1 split."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for workload in load_benchmark_manifest().workloads:
        if isinstance(workload, AzureWorkload):
            start = workload.parameters.start_seconds
            out[workload.split].append((start, start + workload.duration_seconds))
    return {split: tuple(sorted(spans)) for split, spans in sorted(out.items())}


def overlaps_v1_azure(window_index: int) -> bool:
    """True if an Azure 2021 window overlaps any v1 TRAIN/VALIDATION/TEST (#46) hour."""
    _check_index(window_index)
    start = window_index * WINDOW_SECONDS
    end = start + WINDOW_SECONDS
    return any(
        start < span_end and span_start < end
        for spans in v1_azure_intervals().values()
        for span_start, span_end in spans
    )


@dataclass(frozen=True)
class RoleDecision:
    role: str
    reason: str


def assign_role(
    source_id: str,
    group_id: str,
    window_index: int,
    *,
    total_requests: int,
    mean_rps: float,
    cell: tuple[str, str] | None,
    compositional_cells: Iterable[tuple[str, str]] = (),
) -> RoleDecision:
    """Frozen ``grouped-temporal-split-v1`` role of one catalog window.

    ``cell`` is (intensity, primary shape) and may be None only for excluded
    (empty or beyond-scope) windows. Synthetic windows use ``window_index=0``.
    Windows hashed into a role are never moved to another role: a window that a
    rule drops is excluded or reserved, not reassigned.
    """
    held_out = set(compositional_cells)
    if source_id not in STAGE_A_SOURCES:
        raise ValueError(f"{source_id} is not a Stage-A request source")
    if total_requests < 0:
        raise ValueError("total_requests must be non-negative")
    if total_requests == EMPTY_REQUESTS:
        return RoleDecision(ROLE_EXCLUDED, "empty-window")
    if classify_intensity(mean_rps) is None:
        return RoleDecision(ROLE_EXCLUDED, "beyond-scope-overload")
    if cell is None:
        raise ValueError("a non-empty in-scope window needs its taxonomy cell")
    if source_id == AZURE_2021:
        if overlaps_v1_azure(window_index):
            return RoleDecision(ROLE_EXCLUDED, "v1-benchmark-overlap")
        return RoleDecision(ROLE_TEST_C, "source-holdout")
    if source_id == SYNTHETIC:
        if cell in held_out:
            return RoleDecision(ROLE_EXCLUDED, "compositional-holdout-cell")
        bucket = stable_bucket(
            SPLIT_SALT, "synthetic", source_id, group_id, modulus=GROUP_BUCKET_MODULUS
        )
        role = ROLE_TRAIN if bucket <= SYNTHETIC_TRAIN_MAX else ROLE_VALIDATION
        return RoleDecision(role, "synthetic-group-hash")

    period = window_period(window_index)
    group = group_role(source_id, group_id)
    if period == "embargo":
        return RoleDecision(ROLE_RESERVED, "embargo")
    if group == "development":
        if cell in held_out:
            return RoleDecision(ROLE_TEST_B_COMP, f"compositional-holdout-{period}")
        if period == "development":
            return RoleDecision(ROLE_TRAIN, "development-group-development-period")
        return RoleDecision(ROLE_TEST_A, "development-group-temporal-test-period")
    if period != "development":
        return RoleDecision(ROLE_RESERVED, f"{group}-group-temporal-test-period")
    if group == "validation":
        if cell in held_out:
            return RoleDecision(ROLE_EXCLUDED, "compositional-holdout-cell")
        return RoleDecision(ROLE_VALIDATION, "validation-group-development-period")
    return RoleDecision(ROLE_TEST_B_APP, "application-test-group-development-period")


def choose_compositional_cells(
    cell_windows: Mapping[tuple[str, str], int],
    cell_groups: Mapping[tuple[str, str], int],
) -> tuple[tuple[str, str], ...]:
    """Up to two TEST-B compositional cells from Azure 2019 candidate-TRAIN window counts.

    Inputs count Azure 2019 windows that are TRAIN before the compositional rule
    (development groups, development period, non-empty, in scope). A cell is
    eligible with >= 20 windows from >= 5 groups; after removing the chosen
    cells, every chosen intensity and every chosen shape must still occur in
    another cell with >= 20 windows. Eligible cells are ranked by SHA-256 under
    the frozen salt; the second cell must differ in both intensity and shape.
    """

    def rank(cell: tuple[str, str]) -> str:
        return stable_hash(SPLIT_SALT, "compositional", cell[0], cell[1])

    def supported(chosen: Sequence[tuple[str, str]]) -> bool:
        rest = {
            cell
            for cell, count in cell_windows.items()
            if count >= COMPOSITIONAL_SUPPORT_WINDOWS and cell not in chosen
        }
        return all(
            any(other[0] == intensity for other in rest)
            and any(other[1] == shape for other in rest)
            for intensity, shape in chosen
        )

    for cell in cell_windows:
        if cell[0] not in INTENSITY_LABELS or cell[1] not in SHAPE_LABELS:
            raise ValueError(f"unknown taxonomy cell {cell}")
    eligible = sorted(
        (
            cell
            for cell, count in cell_windows.items()
            if count >= COMPOSITIONAL_MIN_WINDOWS
            and cell_groups.get(cell, 0) >= COMPOSITIONAL_MIN_GROUPS
            and supported([cell])
        ),
        key=rank,
    )
    if not eligible:
        return ()
    chosen = [eligible[0]]
    for cell in eligible[1:]:
        if len(chosen) == COMPOSITIONAL_MAX_CELLS:
            break
        if cell[0] != chosen[0][0] and cell[1] != chosen[0][1] and supported([*chosen, cell]):
            chosen.append(cell)
    return tuple(chosen)


# --- evaluation sets and TRAIN sampler ------------------------------------------------------------


@dataclass(frozen=True)
class CatalogWindow:
    """Workload metadata of one manifest window (no controller outcome)."""

    source_id: str
    group_id: str
    window_id: str
    intensity: str
    shape: str
    role: str

    def __post_init__(self) -> None:
        if self.intensity not in INTENSITY_LABELS:
            raise ValueError(f"unknown intensity {self.intensity!r}")
        if self.shape not in SHAPE_LABELS:
            raise ValueError(f"unknown shape {self.shape!r}")
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")

    @property
    def cell(self) -> tuple[str, str]:
        return (self.intensity, self.shape)


def _taxonomy_order(window: CatalogWindow) -> tuple[str, int, int, str]:
    return (
        window.source_id,
        INTENSITY_LABELS.index(window.intensity),
        SHAPE_LABELS.index(window.shape),
        window.window_id,
    )


def select_evaluation_windows(
    candidates: Iterable[CatalogWindow], *, cap_per_cell: int
) -> tuple[CatalogWindow, ...]:
    """Deterministic capped stratified sample of one VALIDATION or TEST role.

    Per (source, intensity, shape) cell, windows are taken in SHA-256 rank
    order, skipping a group already taken in that cell, up to ``cap_per_cell``.
    The result is in deterministic taxonomy order (the validation iterator).
    """
    if cap_per_cell <= 0:
        raise ValueError("cap_per_cell must be positive")
    pool = list(candidates)
    roles = {window.role for window in pool}
    if len(roles) > 1 or roles & {ROLE_TRAIN, ROLE_RESERVED, ROLE_EXCLUDED}:
        raise ValueError(f"evaluation sampling takes one VALIDATION/TEST role, got {roles}")
    if len({window.window_id for window in pool}) != len(pool):
        raise ValueError("duplicate window IDs")
    by_cell: dict[tuple[str, str, str], list[CatalogWindow]] = defaultdict(list)
    for window in pool:
        by_cell[(window.source_id, *window.cell)].append(window)
    chosen: list[CatalogWindow] = []
    for cell_windows in by_cell.values():
        ranked = sorted(
            cell_windows,
            key=lambda w: stable_hash(SPLIT_SALT, "evaluation-sample", w.role, w.window_id),
        )
        groups: set[str] = set()
        for window in ranked:
            if len(groups) == cap_per_cell:
                break
            if window.group_id not in groups:
                groups.add(window.group_id)
                chosen.append(window)
    return tuple(sorted(chosen, key=_taxonomy_order))


SamplingTree = dict[str, dict[str, dict[str, tuple[str, ...]]]]


def build_sampling_tree(windows: Iterable[CatalogWindow]) -> SamplingTree:
    """source -> intensity -> shape -> sorted window IDs; refuses any non-TRAIN window."""
    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    seen: set[str] = set()
    for window in windows:
        if window.role != ROLE_TRAIN:
            raise ValueError(
                f"the TRAIN sampler cannot see {window.role} window {window.window_id}"
            )
        if window.source_id not in SOURCE_WEIGHTS:
            raise ValueError(f"{window.source_id} is not a TRAIN source")
        if window.window_id in seen:
            raise ValueError(f"duplicate window {window.window_id}")
        seen.add(window.window_id)
        tree.setdefault(window.source_id, {}).setdefault(window.intensity, {}).setdefault(
            window.shape, []
        ).append(window.window_id)
    return {
        source: {
            intensity: {shape: tuple(sorted(ids)) for shape, ids in shapes.items()}
            for intensity, shapes in intensities.items()
        }
        for source, intensities in tree.items()
    }


def sampler_rng(training_seed: int) -> np.random.Generator:
    """The per-training-seed episode stream, independent of the RL library's own seeding."""
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(training_seed, spawn_key=(SAMPLER_RNG_DOMAIN,)))
    )


def draw_episode(rng: np.random.Generator, tree: SamplingTree) -> str:
    """One TRAIN window: source by frozen weight, then uniform intensity, shape and window.

    Each level uses exactly one draw. Empty children are absent from the tree,
    and the source weights renormalize over the sources present.
    """
    sources = [source for source in TRAIN_SOURCES if tree.get(source)]
    if not sources:
        raise ValueError("empty sampling tree")
    total = sum(SOURCE_WEIGHTS[source] for source in sources)
    u = float(rng.random()) * total
    source = sources[-1]
    cumulative = 0.0
    for candidate in sources:
        cumulative += SOURCE_WEIGHTS[candidate]
        if u < cumulative:
            source = candidate
            break
    intensities = [label for label in INTENSITY_LABELS if tree[source].get(label)]
    intensity = intensities[int(rng.integers(len(intensities)))]
    shapes = [label for label in SHAPE_LABELS if tree[source][intensity].get(label)]
    shape = shapes[int(rng.integers(len(shapes)))]
    ids = tree[source][intensity][shape]
    return ids[int(rng.integers(len(ids)))]


def episode_sequence(training_seed: int, tree: SamplingTree, episodes: int) -> list[str]:
    """The exact, reproducible TRAIN curriculum of one training seed."""
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    rng = sampler_rng(training_seed)
    return [draw_episode(rng, tree) for _ in range(episodes)]


def sampling_probabilities(tree: SamplingTree) -> dict[str, float]:
    """Exact per-window sampling probability implied by :func:`draw_episode`."""
    sources = [source for source in TRAIN_SOURCES if tree.get(source)]
    total = sum(SOURCE_WEIGHTS[source] for source in sources)
    out: dict[str, float] = {}
    for source in sources:
        intensities = {k: v for k, v in tree[source].items() if v}
        for shapes in intensities.values():
            nonempty = {k: v for k, v in shapes.items() if v}
            for ids in nonempty.values():
                p = SOURCE_WEIGHTS[source] / total / len(intensities) / len(nonempty) / len(ids)
                for window_id in ids:
                    out[window_id] = p
    return out


# --- selection ------------------------------------------------------------------------------------

Metrics = Mapping[str, Mapping[str, float]]  # window_id -> metric -> value


def balanced_mean(values: Mapping[str, float], windows: Mapping[str, CatalogWindow]) -> float:
    """Equal-source mean of equal-intensity means of equal-shape means of window values."""
    if not values:
        raise ValueError("no values to aggregate")
    tree: dict[str, dict[str, dict[str, list[float]]]] = {}
    for window_id, value in values.items():
        meta = windows[window_id]
        tree.setdefault(meta.source_id, {}).setdefault(meta.intensity, {}).setdefault(
            meta.shape, []
        ).append(value)
    return statistics.fmean(
        statistics.fmean(
            statistics.fmean(statistics.fmean(cell) for cell in shapes.values())
            for shapes in intensities.values()
        )
        for intensities in tree.values()
    )


def _group_sla(
    metrics: Metrics, windows: Mapping[str, CatalogWindow]
) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for window_id, row in metrics.items():
        meta = windows[window_id]
        groups[(meta.source_id, meta.intensity)][window_id] = row["sla_violation_rate"]
    return {key: balanced_mean(values, windows) for key, values in sorted(groups.items())}


@dataclass(frozen=True)
class SelectionOutcome:
    selected: str | None
    feasible: tuple[str, ...]
    ranking: tuple[str, ...]
    diagnostic_fallback: str | None
    validation_feasible: bool


def _check_evidence(
    candidates: Mapping[str, Metrics], reference: Metrics, windows: Mapping[str, CatalogWindow]
) -> None:
    if not candidates:
        raise ValueError("no candidates")
    expected = set(reference)
    if not expected:
        raise ValueError("no validation windows")
    for window_id in expected:
        meta = windows.get(window_id)
        if meta is None or meta.role != ROLE_VALIDATION:
            raise ValueError(f"selection uses VALIDATION windows only, got {window_id}")
    for name, rows in [(REFERENCE_CONTROLLER, reference), *candidates.items()]:
        if set(rows) != expected:
            raise ValueError(f"{name} does not cover exactly the reference validation windows")
        for row in rows.values():
            for metric in SELECTION_METRICS:
                if not math.isfinite(row[metric]):
                    raise ValueError(f"{name} has a non-finite {metric}")


def select_candidate(
    candidates: Mapping[str, Metrics], reference: Metrics, windows: Mapping[str, CatalogWindow]
) -> SelectionOutcome:
    """``benchmark-v2-cost-under-sla-v1`` within one algorithm family.

    Feasible: every validation window's SLA violation rate <= Threshold's + 0.05,
    and every (source, intensity) group's balanced SLA <= Threshold's + 0.02.
    Among feasible candidates minimize balanced normalized cost, then balanced
    queue pressure, churn rate, SLA violation rate, candidate ID. With no
    feasible candidate nothing is selected; the diagnostic fallback minimizes
    the largest group SLA excess, then the largest window excess, then the
    same objective, and is never a selection.
    """
    _check_evidence(candidates, reference, windows)
    ref_groups = _group_sla(reference, windows)

    def objective(name: str) -> tuple[float | str, ...]:
        rows = candidates[name]
        return (
            *(
                balanced_mean({w: r[m] for w, r in rows.items()}, windows)
                for m in ("normalized_cost", *TIE_BREAKS)
            ),
            name,
        )

    def excesses(name: str) -> tuple[float, float]:
        rows = candidates[name]
        groups = _group_sla(rows, windows)
        group_excess = max(groups[k] - ref_groups[k] for k in ref_groups)
        window_excess = max(
            rows[w]["sla_violation_rate"] - reference[w]["sla_violation_rate"] for w in reference
        )
        return group_excess, window_excess

    feasible = []
    for name in sorted(candidates):
        group_excess, window_excess = excesses(name)
        if (
            window_excess <= WINDOW_SLA_ALLOWANCE + FEASIBILITY_TOLERANCE
            and group_excess <= GROUP_SLA_ALLOWANCE + FEASIBILITY_TOLERANCE
        ):
            feasible.append(name)
    ranking = tuple(sorted(feasible, key=objective))
    if ranking:
        return SelectionOutcome(ranking[0], tuple(feasible), ranking, None, True)
    fallback = min(candidates, key=lambda name: (*excesses(name), *objective(name)))
    return SelectionOutcome(None, (), (), fallback, False)


def select_canonical_seed(
    seeds: Mapping[int, Metrics], reference: Metrics, windows: Mapping[str, CatalogWindow]
) -> tuple[int, bool]:
    """Validation-only deployable seed for later sim-to-real use; #119 reports all seeds.

    Same feasibility and objective as :func:`select_candidate`, lowest seed on a
    full tie; with no feasible seed the diagnostic fallback is returned with
    ``canonical_seed_validation_feasible=False``.
    """
    if set(seeds) != set(TRAINING_SEEDS):
        raise ValueError(f"canonical seed needs exactly training seeds {TRAINING_SEEDS}")
    named = {f"{seed:06d}": rows for seed, rows in seeds.items()}
    outcome = select_candidate(named, reference, windows)
    chosen = outcome.selected or outcome.diagnostic_fallback
    assert chosen is not None
    return int(chosen), outcome.validation_feasible


# --- statistics -----------------------------------------------------------------------------------


def clustered_bootstrap_ci(
    deltas: Mapping[str, float], groups: Mapping[str, str]
) -> tuple[float, float, float]:
    """Equal-window mean paired delta with a group-clustered percentile 95% interval.

    Groups (apps/owners/synthetic parameter sets) are resampled with
    replacement, ``BOOTSTRAP_RESAMPLES`` times, from the frozen bootstrap
    stream. Descriptive only: it does not declare a winner.
    """
    if not deltas:
        raise ValueError("no paired deltas")
    if any(not math.isfinite(v) for v in deltas.values()):
        raise ValueError("paired deltas must be finite")
    by_group: dict[str, list[float]] = defaultdict(list)
    for window_id, delta in sorted(deltas.items()):
        by_group[groups[window_id]].append(delta)
    names = sorted(by_group)
    sums = np.array([sum(by_group[name]) for name in names])
    counts = np.array([len(by_group[name]) for name in names])
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(BOOTSTRAP_SEED, spawn_key=(BOOTSTRAP_RNG_DOMAIN,)))
    )
    draws = rng.integers(len(names), size=(BOOTSTRAP_RESAMPLES, len(names)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - BOOTSTRAP_LEVEL) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return statistics.fmean(deltas.values()), float(low), float(high)


# --- protocol -------------------------------------------------------------------------------------


def _fixed_environment() -> dict[str, Any]:
    config = stage_a_config()
    return {
        "simulator_config": config.model_dump(mode="json"),
        "startup_delay_model": config.dynamics.startup_delay_model,
        "action_semantics": config.action.semantics,
        "observation_version": OBSERVATION_VERSION,
        "observation_features": list(feature_names(ObservationConstants.from_config(config))),
        "observation_contains_source_or_domain": False,
        "episode_ticks": TICKS_PER_WINDOW,
        "fleet_capacity_rps": FLEET_CAPACITY_RPS,
        "training_and_selection_condition": "nominal (robustness-v1 'nominal', dynamics seed 0)",
        "changed_from_v1": [],
    }


def _sources() -> list[dict[str, Any]]:
    cc_by = "CC-BY-4.0 (AzurePublicDataset LICENSE; the dataset page says 'CC-BY Attribution')"
    return [
        {
            "source_id": AZURE_2019,
            "stage_a_role": "TRAIN, VALIDATION, TEST-A, TEST-B",
            "semantics": "per-function invocation counts per minute (request-rate proxy)",
            "official_url": (
                "https://github.com/Azure/AzurePublicDataset/blob/master/"
                "AzureFunctionsDataset2019.md"
            ),
            "version": "revision 2, 2020-06-18; files invocations_per_function_md.anon.d01-d14.csv",
            "license": cc_by,
            "citation": (
                "Shahrad et al., 'Serverless in the Wild: Characterizing and Optimizing the "
                "Serverless Workload at a Large Cloud Provider', USENIX ATC 2020"
            ),
            "workload_unit": "HashApp (sum over its functions and triggers)",
            "leakage_group": "HashOwner (groups possibly related applications)",
            "mapping": (
                "rate = per-minute count / 60 rps, repeated for both contained 30 s ticks; "
                "window (day d, hour h) = minutes 60h+1..60h+60 of day file d; "
                "an app absent from a day file has zero requests that day"
            ),
            "caveats": [
                "counts are recorded after functions execute: minutes bin completions, "
                "not arrivals",
                "no sub-minute arrival pattern exists; none is invented",
                "a small unspecified per-application random sample of Azure Functions",
                "per-app rates are not rescaled to the simulator (amplitude as recorded)",
                "cross-day hash consistency is stated for file types; #116 must verify it across "
                "days and record the result",
                "duration/memory tables are auxiliary analysis metadata, never request rate; "
                "memory exists only for days 1-12",
                "trigger group is recorded per app for analysis only",
            ],
        },
        {
            "source_id": AZURE_2021,
            "stage_a_role": "TEST-C source holdout only (never TRAIN or VALIDATION)",
            "semantics": "individual invocations: end timestamp and duration in seconds",
            "official_url": (
                "https://github.com/Azure/AzurePublicDataset/blob/master/"
                "AzureFunctionsInvocationTrace2021.md"
            ),
            "version": "revision 1, 2021-11-30; AzureFunctionsInvocationTraceForTwoWeeksJan2021",
            "license": cc_by,
            "citation": (
                "Zhang et al., 'Faster and Cheaper Serverless Computing on Harvested "
                "Resources', SOSP 2021"
            ),
            "workload_unit": "app (sum over its functions)",
            "leakage_group": "app (the trace has no owner field)",
            "mapping": (
                "arrival = end_timestamp - duration, binned into half-open 30 s ticks of "
                "window k = [3600k, 3600(k+1)) trace-relative seconds, k in 0..335; arrivals "
                "outside [0, 1209600) are dropped"
            ),
            "caveats": [
                "timestamps were modified by Microsoft: trace-relative seconds only",
                "same provider/platform as Azure 2019: TEST-C is a cross-dataset, cross-year, "
                "different-sampling shift, not a cross-provider domain shift",
                "per-app demand is expected to be mostly low; TEST-C is reported per intensity",
                "windows overlapping any v1 Azure hour (including both #46 TEST hours) are "
                "excluded from benchmark-v2",
            ],
        },
        {
            "source_id": SYNTHETIC,
            "stage_a_role": "TRAIN and VALIDATION coverage supplement; never TEST",
            "semantics": "seeded generated request rates (scalerl.workloads generators)",
            "license": "generated by ScaleRL code under the repository MIT license",
            "workload_unit": "one generated window",
            "leakage_group": "generator family + parameter set (all seeds of a set share a group)",
            "caveats": ["never described or reported as real traffic"],
        },
        {
            "source_id": ALIBABA_2021,
            "stage_a_role": "excluded from Stage A",
            "official_url": (
                "https://github.com/alibaba/clusterdata/tree/master/"
                "cluster-trace-microservices-v2021"
            ),
            "license": (
                "none found: the repository has no LICENSE file and neither its README nor the "
                "trace README states usage terms (checked 2026-10-02)"
            ),
            "citation": (
                "Luo et al., 'Characterizing Microservice Dependency and Performance: Alibaba "
                "Trace Analysis', SoCC 2021"
            ),
            "reason": (
                "unverified usage terms; additionally only 12 hours, per-container MCR at 60 s "
                "with provider/consumer double-counting risk. Adding it later needs a new "
                "protocol version before any #118 training, never a silent addition"
            ),
        },
        {
            "source_id": GOOGLE_2019,
            "stage_a_role": "excluded from Stage A (possible #120 environment evidence)",
            "official_url": "https://github.com/google/cluster-data/blob/master/ClusterData2019.md",
            "license": "CC-BY-4.0 (stated in ClusterData2019.md)",
            "reason": "resource requests/usage of Borg jobs, not a request trace; never converted",
        },
        {
            "source_id": SEBS,
            "stage_a_role": "excluded from Stage A (possible sim-to-real-v2 calibration)",
            "official_url": "https://github.com/spcl/serverless-benchmarks",
            "license": "BSD-3-Clause (repository LICENSE.md)",
            "reason": "serverless application benchmark suite, not a workload trace",
        },
    ]


def build_protocol_payload() -> dict[str, Any]:
    """Protocol content excluding its self-derived ``protocol_id``."""
    intervals = v1_azure_intervals()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_issue": "#115",
        "stage": STAGE,
        "research_question": (
            "Within ScaleRL's simplified simulator and the unchanged v1 system contract, does "
            "training DQN/PPO over a broader, predeclared workload distribution improve "
            "cost/SLA generalization to unseen windows, applications, taxonomy combinations and "
            "a held-out trace source, relative to fixed reactive and predictive baselines?"
        ),
        "project_scope": {
            "student_learning_project": True,
            "production_autoscaler_claim": False,
            "universal_cloud_claim": False,
            "rl_required_to_win": False,
            "cross_algorithm_winner": False,
        },
        "upstream": dict(UPSTREAM),
        "stage_boundary": {
            "stage_a_changes": ["training workload distribution"],
            "stage_a_fixed": [
                "simulator dynamics and replica lifecycle",
                "min/max/initial replicas 1/10/1",
                "service capacity 50 rps/replica",
                "startup delay 60 s (fixed-v1)",
                "control interval 30 s",
                "SLA target 0.5 s",
                "desired-replicas-v1",
                "scalerl-observation-v1",
                "reward-contract-v1 / full-cost-low-v1",
            ],
            "stage_b_owner": "#120 (observation-v2/action-v2/environment context); never mixed in",
            "correctness_blocker_rule": (
                "a Stage-A contract change is allowed only for a demonstrated correctness bug, "
                "requires a new protocol version, and must happen before #118 training"
            ),
        },
        "fixed_environment": _fixed_environment(),
        "reward": {
            "contract_version": "reward-contract-v1",
            "contract_id": UPSTREAM["reward_contract_id"],
            "variant": REWARD_VARIANT,
            "weights": dict(REWARD_WEIGHTS),
            "reopened_by_stage_a": False,
            "used_for_selection": False,
        },
        "datasets": {
            "eligibility_rule": (
                "a Stage-A source needs request/call-rate semantics, a stable leakage group, "
                "deterministic preparation and a verified license; decided from documentation "
                "only, never from controller outcomes"
            ),
            "sources": _sources(),
            "raw_data_committed": False,
        },
        "workload_unit": {
            "window_seconds": WINDOW_SECONDS,
            "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
            "ticks": TICKS_PER_WINDOW,
            "grid": "aligned, non-overlapping one-hour windows; 336 per Azure source",
            "amplitude_rescaled": False,
            "window_ids": {
                AZURE_2019: "az19:<HashOwner>:<HashApp>:d<DD>h<HH>",
                AZURE_2021: "az21:<app>:<start_seconds>",
                SYNTHETIC: "syn2:<family>:<parameter_set>:<instance>",
            },
            "excluded_windows": {
                "empty-window": "zero requests (counted in the catalog, used nowhere)",
                "beyond-scope-overload": f"mean > {BEYOND_SCOPE_MEAN_RPS} rps (2x fleet capacity)",
                "v1-benchmark-overlap": "Azure 2021 window overlapping any v1 Azure window",
            },
            "v1_azure_intervals_seconds": {
                split: [list(span) for span in spans] for split, spans in intervals.items()
            },
        },
        "taxonomy": {
            "version": TAXONOMY_VERSION,
            "intensity": {
                "metric": "mean rps over the 120 ticks",
                "service_capacity_rps": SERVICE_CAPACITY_RPS,
                "fleet_capacity_rps": FLEET_CAPACITY_RPS,
                "upper_bounds_inclusive": [
                    {"label": label, "max_mean_rps": upper} for label, upper in INTENSITY_BOUNDS
                ],
                "beyond_scope_rule": "mean above the overload bound is excluded",
                "peak_pressure_upper_bounds_inclusive": [
                    {"label": label, "max_peak_tick_rps": upper}
                    for label, upper in PEAK_PRESSURE_BOUNDS
                ],
                "peak_pressure_above": PEAK_PRESSURE_OVERLOAD,
                "peak_pressure_role": "analysis only",
            },
            "traffic_shape": {
                "classifier_version": SHAPE_CLASSIFIER_VERSION,
                "resolution": "one-minute means of consecutive 30 s tick pairs (60 samples)",
                "labels": list(SHAPE_LABELS),
                "ordered_rules": [
                    f"steady: CV <= {STEADY_MAX_CV} and peak/mean <= {STEADY_MAX_PEAK_TO_MEAN}",
                    f"ramp-up: quarter means strictly increasing and q4 >= "
                    f"{RAMP_MIN_RATIO} * max(q1, {RATE_FLOOR_RPS})",
                    f"ramp-down: quarter means strictly decreasing and q1 >= "
                    f"{RAMP_MIN_RATIO} * max(q4, {RATE_FLOOR_RPS})",
                    f"periodic: some lag in [{PERIODIC_MIN_LAG}, n/2] minutes has a local ACF "
                    f"peak >= {PERIODIC_MIN_ACF} and a smaller lag has ACF <= that peak - "
                    f"{PERIODIC_MIN_DIP}",
                    f"spike: with m = max(median, {RATE_FLOOR_RPS}), peak >= "
                    f"{SPIKE_MIN_PEAK_TO_MEDIAN} m, exactly one contiguous run of minutes "
                    f"above 2m, and at most {SPIKE_MAX_FRACTION_ABOVE_2X_MEDIAN} of minutes "
                    "above 2m",
                    f"regime-switch: the best single change point (each side >= n/8 minutes, "
                    f"earliest on ties) explains >= {LEVEL_SHIFT_MIN_R2} of the variance and "
                    f"the segment-mean ratio (low floored at {RATE_FLOOR_RPS} rps) is >= "
                    f"{LEVEL_SHIFT_MIN_RATIO}",
                    f"bursty: CV >= {BURSTY_MIN_CV} or peak/mean >= {BURSTY_MIN_PEAK_TO_MEAN}",
                    "noisy-other: otherwise",
                ],
                "auxiliary_tags": list(SHAPE_TAGS),
                "auxiliary_tag_role": "analysis only; never a sampler or split key",
                "interpretation": "deterministic strata, not semantic ground truth",
            },
            "labels_in_observation": False,
            "labels_role": "sampling, stratified selection and reporting only",
        },
        "split": {
            "version": SPLIT_VERSION,
            "salt": SPLIT_SALT,
            "hash": "SHA-256 of UTF-8 pipe-joined fields; bucket = first 64 bits mod modulus",
            "periods": {
                "development": [0, DEVELOPMENT_WINDOWS - 1],
                "embargo": [DEVELOPMENT_WINDOWS, DEVELOPMENT_WINDOWS + EMBARGO_WINDOWS - 1],
                "temporal-test": [DEVELOPMENT_WINDOWS + EMBARGO_WINDOWS, AZURE_WINDOWS - 1],
                "meaning": "window indices: days 1-11 | day 12 | days 13-14",
            },
            "azure_2019_groups": {
                "fields": ["group", AZURE_2019, "<HashOwner>"],
                "modulus": GROUP_BUCKET_MODULUS,
                "development": [0, DEVELOPMENT_GROUP_MAX],
                "validation": [DEVELOPMENT_GROUP_MAX + 1, VALIDATION_GROUP_MAX],
                "application-test": [VALIDATION_GROUP_MAX + 1, GROUP_BUCKET_MODULUS - 1],
            },
            "synthetic_groups": {
                "fields": ["synthetic", SYNTHETIC, "<family:parameter_set>"],
                "train": [0, SYNTHETIC_TRAIN_MAX],
                "validation": [SYNTHETIC_TRAIN_MAX + 1, GROUP_BUCKET_MODULUS - 1],
            },
            "role_table": {
                "azure-2019 development group": (
                    "development period -> train; embargo -> reserved; temporal-test -> TEST-A; "
                    "compositional cell (non-embargo) -> TEST-B compositional"
                ),
                "azure-2019 validation group": (
                    "development period -> validation (compositional cell -> excluded); "
                    "otherwise reserved"
                ),
                "azure-2019 application-test group": (
                    "development period -> TEST-B application; otherwise reserved"
                ),
                "azure-2021": "TEST-C unless excluded (v1 overlap)",
                "synthetic-v2": "group hash -> train/validation; compositional cell -> excluded",
            },
            "no_reassignment": (
                "a window removed by a rule is excluded or reserved, never moved to another role"
            ),
            "reserved_windows": "never opened, sampled or evaluated in benchmark-v2-workloads-v1",
            "leakage_guarantees": [
                "no Azure 2019 owner group has windows in more than one of train, validation, "
                "TEST-B application",
                "TEST-A shares owners with train but is separated by the one-day embargo; "
                "neighbouring hours never straddle train and TEST-A",
                "TEST-B compositional cells are absent from train and validation for every source",
                "TEST-C source never appears in train or validation",
                "no synthetic parameter set appears in both train and validation; synthetic "
                "is never TEST",
            ],
        },
        "test_tiers": [
            {
                "id": "TEST-A",
                "role": ROLE_TEST_A,
                "meaning": "unseen future time (days 13-14) of development owners after an embargo",
                "kind": "in-domain temporal",
            },
            {
                "id": "TEST-B-application",
                "role": ROLE_TEST_B_APP,
                "meaning": "owners/apps never seen in train or validation, same time span as train",
                "kind": "in-domain application holdout",
            },
            {
                "id": "TEST-B-compositional",
                "role": ROLE_TEST_B_COMP,
                "meaning": "known intensity and shape, predeclared unseen joint (intensity, shape)",
                "kind": "compositional / OOD regime",
                "rule": {
                    "max_cells": COMPOSITIONAL_MAX_CELLS,
                    "min_windows": COMPOSITIONAL_MIN_WINDOWS,
                    "min_groups": COMPOSITIONAL_MIN_GROUPS,
                    "support_windows": COMPOSITIONAL_SUPPORT_WINDOWS,
                    "counted_over": (
                        "Azure 2019 candidate-train windows (development owners, development "
                        "period, non-empty, in scope), from workload statistics only"
                    ),
                    "ranking": "SHA-256(salt|compositional|intensity|shape) ascending",
                    "second_cell": "differs in both intensity and shape; support kept",
                    "none_eligible": "tier not materialized; reason recorded in the manifest",
                },
            },
            {
                "id": "TEST-C",
                "role": ROLE_TEST_C,
                "source": SOURCE_HOLDOUT,
                "meaning": "a whole trace source absent from train and validation",
                "kind": "source holdout (same-provider cross-dataset shift)",
            },
        ],
        "evaluation_sets": {
            "method": (
                "per role and (source, intensity, shape) cell, SHA-256(salt|evaluation-sample|"
                "role|window_id) rank order, at most one window per group per cell"
            ),
            "validation_cap_per_cell": VALIDATION_CAP_PER_CELL,
            "test_cap_per_cell": TEST_CAP_PER_CELL,
            "order": "source, intensity order, shape order, window_id (validation iterator)",
            "train": "the full train role; no cap (sampled by the episode sampler)",
        },
        "synthetic_generation": {
            "version": "synthetic-v2",
            "purpose": "fill train/validation taxonomy cells with < 20 Azure 2019 train windows",
            "gap_audit_input": "Azure 2019 train-role catalog counts only (no TEST, no outcomes)",
            "seed": "SeedSequence(instance_index, spawn_key=(0x73796E74,))",
            "classification": "the same taxonomy; instances in compositional cells are dropped",
            "frozen_by": "#117 in the manifest, before #118; never tuned after any outcome",
        },
        "sampler": {
            "version": SAMPLER_VERSION,
            "scope": "train role only; validation/test windows cannot enter the tree",
            "source_weights": dict(SOURCE_WEIGHTS),
            "source_weight_rationale": (
                "real data dominates; synthetic is a coverage supplement that still guarantees "
                "rare regimes appear"
            ),
            "levels": [
                "source by frozen weight (renormalized over present sources)",
                "intensity uniform over non-empty strata",
                "shape uniform over non-empty cells",
                "window uniform",
            ],
            "rng": (
                "PCG64(SeedSequence(training_seed, spawn_key=(0x73616D70,))); one draw per level"
            ),
            "source_order": list(TRAIN_SOURCES),
            "per_episode_record": ["training_seed", "episode_index", "window_id"],
            "episode_sequence_reproducible": True,
            "rl_library_seed": "the training seed, passed separately to Stable-Baselines3",
        },
        "seeds": {
            "training_seeds": list(TRAINING_SEEDS),
            "screening_training_seed": SCREENING_TRAINING_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "dynamics_seed": DYNAMICS_SEED,
            "selected_config_members": (
                "training seeds 0-4 with the screening run reused as the seed-0 member"
            ),
        },
        "selection": {
            "version": SELECTION_VERSION,
            "scope": "within one algorithm family; DQN and PPO are never ranked against each other",
            "evidence": "validation role only, nominal condition, screening seed 0",
            "reference": {"variant_id": REFERENCE_CONTROLLER, "params": dict(REFERENCE_PARAMS)},
            "feasibility": {
                "window": f"SLA violation rate <= Threshold's + {WINDOW_SLA_ALLOWANCE}",
                "group": (
                    f"(source, intensity) balanced SLA <= Threshold's + {GROUP_SLA_ALLOWANCE}"
                ),
                "tolerance": FEASIBILITY_TOLERANCE,
            },
            "objective": "balanced normalized_cost (equal source > intensity > shape > window)",
            "tie_breaks": [*TIE_BREAKS, "candidate_id"],
            "no_feasible_candidate": (
                "select nothing; report the diagnostic fallback (min max group excess, min max "
                "window excess, objective); never relax the constraint after outcomes"
            ),
            "canonical_seed": (
                "same rule over the five selected-config seeds, lowest seed on a full tie; "
                "for later deployment only; #119 reports all five seeds"
            ),
            "adaptation_from_78": (
                "#78 required per-workload SLA no worse than Threshold on 3 validation "
                "workloads; with hundreds of heterogeneous windows an exact per-window rule "
                "makes feasibility hinge on single noisy windows, so v2 keeps SLA as a "
                "constraint with a small per-window allowance plus a tighter per-stratum "
                "allowance, and keeps cost as the objective. Fixed before any v2 outcome."
            ),
            "search_gate": (
                "#118 commits DQN/PPO candidate spaces, trial counts and timesteps before the "
                "first candidate result; no rescue budget"
            ),
            "reward_used": False,
        },
        "comparators": {
            "baselines": {
                "threshold-v1": "frozen #78 reference params, not re-tuned on benchmark-v2",
                "predictive-v1": "frozen predictive-baseline-v1",
                "predictive-seasonal-v1": (
                    "only where the same app has a prior day in its source: profile = median of "
                    "the same hour on up to 7 previous days of that app (demand history only); "
                    "reported as a separate subset; not applicable to synthetic"
                ),
                "static-v1, random-v1": "sanity references only",
            },
            "baselines_retuned": False,
            "baseline_caveat": (
                "baselines keep their v1 tuning, so they are not re-optimized for benchmark-v2 "
                "while RL is retrained on it"
            ),
            "matched_conditions": "identical window, config, dynamics seed and evaluation seed",
        },
        "metrics": {
            "primary": [
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
                "processed_requests",
                "dropped_requests",
                "final_queued_requests",
                "scaling_actions",
                "churn_rate",
                "action.total_absolute_replica_change",
                "action.max_absolute_replica_change_in_one_tick",
            ],
            "new_diagnostics": {
                "overprovision_ratio": (
                    "sum over ticks of (active + pending) / sum of max(1, ceil(request_rate / 50))"
                ),
                "fraction_ticks_at_min_fleet": "ticks with active + pending == 1, / 120",
                "fraction_ticks_at_max_fleet": "ticks with active + pending == 10, / 120",
            },
            "diagnostic_flags": "heldout-descriptive-v1 flags (full_fleet, never_scaled, ...)",
            "workload_descriptors": [
                "request_count",
                "mean_rps",
                "peak_tick_rps",
                "mean_rps/50",
                "mean_rps/500",
                "intensity",
                "peak_pressure",
                "shape",
                "shape_tags",
            ],
            "secondary": ["episode_reward"],
            "overall_winner_score": None,
        },
        "aggregation": {
            "raw_unit": "controller (and training seed) x window x condition",
            "learned_family": "equal training-seed mean per window, all five seeds retained",
            "hierarchy": "window -> shape -> intensity -> source (equal weights)",
            "report_separately": [
                "each TEST tier",
                "source",
                "intensity stratum",
                "shape",
                "low-load strata (near-idle, low, single-replica)",
                "high-load strata (multi-replica-high, near-saturation, overload)",
            ],
            "statistics": ["mean", "median", "sample_sd", "q1", "q3", "min", "max", "count"],
            "paired_deltas": (
                "controller - threshold-v1 per matched window; equal-window mean with "
                f"group-clustered percentile bootstrap {BOOTSTRAP_LEVEL:.0%} interval, "
                f"{BOOTSTRAP_RESAMPLES} resamples, SeedSequence({BOOTSTRAP_SEED}, "
                "spawn_key=(0x626F6F74,))"
            ),
            "grand_mean": "secondary only; never hides a tier, source or stratum",
        },
        "generalization_terms": {
            "in-distribution": "validation: unseen owners, train time span, train cells",
            "temporal": "TEST-A",
            "application": "TEST-B application",
            "compositional": "TEST-B compositional",
            "source": "TEST-C; not called a cross-provider domain shift",
        },
        "robustness_boundary": {
            "stage_a_training": "nominal only; diversity comes from workloads",
            "issue_21": {
                "scenarios": "separately versioned benchmark-v2 scenarios; robustness-v1 unchanged",
                "train_role": "none in Stage A",
                "validation_role": "diagnostic only; never part of selection",
                "test_role": (
                    "a separate stress tier applied only to frozen TEST windows, materialized "
                    "only if #21 commits scenario specs and seeds before #118's first training "
                    "run; otherwise #119 runs no stress tier"
                ),
            },
            "historical_robustness_artifacts_changed": False,
        },
        "heldout_v1_relationship": {
            "issue_46": "frozen v1 evidence; not rerun, overwritten or reinterpreted",
            "v1_test_hours_in_benchmark_v2": False,
            "v1_outcomes_used_to_choose_items": False,
            "low_load_coverage_source": "Azure 2019 near-idle/low strata and synthetic windows",
            "reporting": (
                "#119 may show #46 v1 results next to benchmark-v2 results, labelled as v1 "
                "evidence; benchmark-v2 is not a rerun of #46"
            ),
        },
        "manifest_gate": {
            "owner": "#117 (after #116 prepares azure-functions-2019 and azure-functions-2021)",
            "manifest_version": MANIFEST_VERSION,
            "required_before": "#118 first training run",
            "order": [
                "catalog both Azure sources with workload statistics only",
                "assign roles without compositional cells",
                "choose compositional cells from candidate-train counts",
                "re-assign roles with the chosen cells",
                "generate synthetic-v2 from the train gap audit",
                "sample validation and TEST sets with the frozen caps",
                "write coverage audit and leakage checks",
                "commit the content-addressed manifest",
            ],
            "must_include": [
                "protocol_id",
                "source file fingerprints and preparation versions",
                "every window id, group id, role, reason, statistics, intensity, shape, tags",
                "chosen compositional cells or the not-materialized reason",
                "validation and TEST sample membership",
                "excluded/reserved counts by reason",
                "coverage audit by source x intensity x shape x role",
            ],
            "contingency": (
                "if a Stage-A source cannot be prepared or a tier is empty, record it; the fix "
                "is a new protocol version before #118, never substituting other data"
            ),
            "immutable_after": "#118 first training run",
        },
        "leakage": {
            "held_out_test_used_for_design": False,
            "held_out_test_used_for_training": False,
            "held_out_test_used_for_selection": False,
            "controller_outcomes_used_for_split_or_taxonomy": False,
            "benchmark_v2_test_opened": False,
            "generalist_model_trained": False,
        },
        "caveats": [
            "all evidence remains simulator evidence; no real-cloud claim",
            "TEST-C is a same-provider source shift, not a cross-provider domain shift",
            "Azure 2019 minute counts bin completions, a proxy for arrivals",
            "real traces are not rescaled, so intensity coverage depends on what the sampled "
            "apps contain; empty cells are reported, not filled from TEST",
            "baselines are not re-tuned for benchmark-v2",
            "taxonomy thresholds are heuristic strata fixed before data inspection",
        ],
    }


def build_protocol() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(build_protocol_payload(), allow_nan=False))
    return {"protocol_id": content_id(payload), **payload}


def verify_upstream(bench: Path = BENCH_V1) -> None:
    """Fail if any frozen v1 artifact this protocol builds on has drifted."""
    from scalerl.evaluation import heldout

    loaded = s2r.verify_upstream(bench)
    spec = heldout.load_frozen_spec(bench / "heldout-evaluation-v1.json", bench)
    results = json.loads((bench / "heldout-results-v1.json").read_text())
    found = {
        **{key: s2r.UPSTREAM[key] for key in UPSTREAM if key in s2r.UPSTREAM},
        "heldout_v1_spec_id": heldout.spec_id(spec),
        "heldout_v1_results_id": heldout.results_id(results),
    }
    if found != UPSTREAM:
        drift = {k: (found[k], UPSTREAM[k]) for k in UPSTREAM if found[k] != UPSTREAM[k]}
        raise ValueError(f"frozen v1 upstream changed: {drift}")
    if results.get("results_id") != UPSTREAM["heldout_v1_results_id"]:
        raise ValueError("heldout-results-v1 stored results_id changed")
    if results.get("heldout_spec_id") != UPSTREAM["heldout_v1_spec_id"]:
        raise ValueError("heldout-results-v1 no longer references the frozen spec")
    if spec["simulator_config"] != stage_a_config().model_dump(mode="json"):
        raise ValueError("Stage-A simulator contract differs from the #46 configuration")
    contract = loaded["contract"]
    if contract.selected_reward_variant != REWARD_VARIANT or dict(
        contract.selected_weights
    ) != dict(REWARD_WEIGHTS):
        raise ValueError("reward-contract-v1 selection changed")
    selection = loaded["selection"]
    if dict(selection.reference.params) != REFERENCE_PARAMS:
        raise ValueError("frozen Threshold reference changed")


def check_protocol(path: Path = DEFAULT_PROTOCOL, bench: Path = BENCH_V1) -> dict[str, Any]:
    """Verify upstream, the committed artifact against this code, and its content ID."""
    verify_upstream(bench)
    committed: dict[str, Any] = json.loads(path.read_text())
    payload = {k: v for k, v in committed.items() if k != "protocol_id"}
    if committed.get("protocol_id") != content_id(payload):
        raise ValueError(f"{path} protocol_id does not match its content")
    if committed != build_protocol():
        raise ValueError(f"{path} differs from this code's frozen {PROTOCOL_VERSION}")
    return committed


def write_protocol(path: Path = DEFAULT_PROTOCOL, bench: Path = BENCH_V1) -> dict[str, Any]:
    verify_upstream(bench)
    protocol = build_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="benchmark-v2 generalization protocol (#115)")
    parser.add_argument("command", choices=("write", "check", "show-id"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bench", type=Path, default=BENCH_V1)
    args = parser.parse_args(argv)
    try:
        if args.command == "write":
            protocol = write_protocol(args.protocol, args.bench)
            print(f"wrote {args.protocol} {protocol['protocol_id']}")
        elif args.command == "check":
            protocol = check_protocol(args.protocol, args.bench)
            print(f"ok {PROTOCOL_VERSION} {protocol['protocol_id']}")
        else:
            print(build_protocol()["protocol_id"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
