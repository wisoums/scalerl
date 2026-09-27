"""Tests for the frozen sim-to-real protocol (#72). No held-out outcome, no live data."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import RandomController
from scalerl.environment import (
    ActionConfig,
    ActionContract,
    AutoscalingEnv,
    SimulatorConfig,
)
from scalerl.environment.observation import (
    OBSERVATION_VERSION,
    Measurement,
    ObservationConstants,
    build_observation,
    feature_names,
)
from scalerl.evaluation import sim_to_real as s2r
from scalerl.workloads import WorkloadTrace

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "v1"
THRESHOLDS = {
    "syn-val-steady-high": 0.275,
    "syn-val-ramp-down": 0.35,
    "syn-val-bursty": 0.20833333333333334,
}
DESIRED = SimulatorConfig(action=ActionConfig(semantics="desired-replicas-v1"))


# --- upstream -----------------------------------------------------------------------------------


def test_upstream_ids_are_the_frozen_ones() -> None:
    loaded = s2r.verify_upstream(BENCH)
    assert s2r.UPSTREAM["reward_contract_id"] == "37158c261364"
    assert loaded["contract"].selected_reward_variant == "full-cost-low-v1"
    assert loaded["contract"].selected_weights == {
        "latency": 1.0, "cost": 0.5, "sla": 1.0, "queue": 1.0, "churn": 0.1
    }  # fmt: skip
    assert loaded["decision"].final_action_semantics == "desired-replicas-v1"
    assert s2r.PREDICTIVE_FROZEN_IN.endswith(f"({s2r.UPSTREAM['predictive_baseline_artifact_id']})")


def test_tampered_upstream_artifact_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "v1"
    shutil.copytree(BENCH, copy)
    payload = json.loads((copy / "reward-contract-v1.json").read_text())
    payload["decision"]["decision_basis"] = "edited"
    (copy / "reward-contract-v1.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reward_contract_id"):
        s2r.verify_upstream(copy)


# --- canonical learned artifact rule -----------------------------------------------------------


def seed(
    n: int,
    sla: dict[str, float] | float,
    cost: float = 0.6,
    queue: float = 0.05,
    churn: float = 0.1,
) -> s2r.SeedEvidence:
    per = sla if isinstance(sla, dict) else dict.fromkeys(THRESHOLDS, sla)
    return s2r.SeedEvidence(
        training_seed=n,
        training_run_id=f"run{n}",
        model_artifact_uri=f"runs:/run{n}/model",
        validation_run_ids=dict.fromkeys(THRESHOLDS, "v"),
        metrics={
            w: {
                "sla_violation_rate": per[w],
                "normalized_cost": cost,
                "queue_pressure": queue,
                "churn_rate": churn,
            }
            for w in THRESHOLDS
        },
    )


def test_canonical_seed_prefers_feasible_then_cost_then_ties() -> None:
    bursty_miss = {**THRESHOLDS, "syn-val-bursty": 0.3}
    chosen = s2r.select_canonical_seed(
        [seed(0, 0.1, cost=0.7), seed(1, bursty_miss, cost=0.2), seed(2, 0.1, cost=0.5)], THRESHOLDS
    )
    assert chosen["canonical_training_seed"] == 2  # seed 1 is cheaper but infeasible
    assert chosen["canonical_seed_validation_feasible"] is True
    assert chosen["individually_feasible_seeds"] == [0, 2]
    queue = s2r.select_canonical_seed(
        [seed(0, 0.1, queue=0.2), seed(1, 0.1, queue=0.1)], THRESHOLDS
    )
    churn = s2r.select_canonical_seed(
        [seed(0, 0.1, churn=0.2), seed(1, 0.1, churn=0.1)], THRESHOLDS
    )
    sla = s2r.select_canonical_seed([seed(0, 0.2), seed(1, 0.1)], THRESHOLDS)
    lowest = s2r.select_canonical_seed([seed(3, 0.1), seed(1, 0.1)], THRESHOLDS)
    assert [r["canonical_training_seed"] for r in (queue, churn, sla, lowest)] == [1, 1, 1, 1]
    equal = s2r.select_canonical_seed([seed(0, dict(THRESHOLDS))], THRESHOLDS)
    assert equal["canonical_seed_validation_feasible"] is True  # equality passes (#78)


def test_no_feasible_seed_picks_smallest_excess_and_says_so() -> None:
    result = s2r.select_canonical_seed(
        [
            seed(0, {**THRESHOLDS, "syn-val-bursty": 0.40}, cost=0.1),
            seed(1, {**THRESHOLDS, "syn-val-bursty": 0.25}, cost=0.9),
            seed(2, {**THRESHOLDS, "syn-val-ramp-down": 0.38}, cost=0.9),
        ],
        THRESHOLDS,
    )
    assert result["canonical_training_seed"] == 2  # excess 0.03 < 0.0417 < 0.19
    assert result["canonical_seed_validation_feasible"] is False
    assert result["individually_feasible_seeds"] == []


# --- replay windows and schedules --------------------------------------------------------------


def test_canonical_test_workload_rule() -> None:
    workload = s2r.canonical_test_workload()
    tests = sorted(
        w.id
        for w in load_benchmark_manifest().workloads
        if w.split == "test" and w.source == "azure"
    )
    assert workload.id == tests[0] == "azure-test-1166400"
    assert workload.split == "test"


def test_window_rule() -> None:
    assert s2r.window_offsets(3600.0, 30.0) == [600.0, 1500.0, 2400.0]
    assert s2r.window_offsets(3615.0, 30.0) == [600.0, 1500.0, 2400.0]  # aligned down
    with pytest.raises(ValueError, match="shorter"):
        s2r.window_offsets(600.0, 30.0)
    with pytest.raises(ValueError, match="overlap"):
        s2r.window_offsets(1300.0, 30.0)
    assert (s2r.WINDOW_SECONDS, s2r.WINDOW_FRACTIONS, s2r.LOAD_SEEDS) == (
        600.0,
        (0.2, 0.5, 0.8),
        (0, 1, 2),
    )


def test_slice_counts_require_whole_counts() -> None:
    trace = WorkloadTrace([1.0, 0.5, 2 / 30, 0.0], 30.0)
    assert s2r.slice_counts(trace, 1, 3) == [15, 2, 0]
    with pytest.raises(ValueError, match="whole request counts"):
        s2r.slice_counts(WorkloadTrace([0.01], 30.0), 0, 1)


def test_schedule_is_deterministic_seeded_and_controller_independent() -> None:
    counts = [3, 0, 5, 2]
    a = s2r.generate_schedule(counts, 30.0, slice_index=1, load_seed=0)
    b = s2r.generate_schedule(counts, 30.0, slice_index=1, load_seed=0)
    assert a == b and len(a) == 10 and a == sorted(a)
    for index, count in enumerate(counts):  # every bin keeps its exact count
        assert sum(1 for t in a if index * 30.0 <= t < (index + 1) * 30.0) == count
    assert s2r.generate_schedule(counts, 30.0, slice_index=1, load_seed=1) != a
    assert s2r.generate_schedule(counts, 30.0, slice_index=2, load_seed=0) != a
    # No controller input exists, and other RNG use or call order cannot shift it.
    np.random.default_rng(0).random(1000)
    s2r.generate_schedule(counts, 30.0, slice_index=0, load_seed=2)
    assert s2r.generate_schedule(counts, 30.0, slice_index=1, load_seed=0) == a
    # Domain-separated from a plain default_rng(load_seed) stream.
    plain = np.random.default_rng(0).random(3)
    ours = np.random.default_rng(
        np.random.SeedSequence(0, spawn_key=(s2r.SCHEDULE_RNG_DOMAIN, 1))
    ).random(3)
    assert not np.any(plain == ours)


# --- shared observation contract ----------------------------------------------------------------


def test_observation_feature_order_and_dimension() -> None:
    constants = ObservationConstants.from_config(DESIRED)
    assert OBSERVATION_VERSION == "scalerl-observation-v1"
    assert feature_names(constants) == (
        "demand_pressure_t-0", "demand_pressure_t-1", "demand_pressure_t-2", "demand_pressure_t-3",
        "utilization", "queue_pressure", "latency_pressure", "active_replicas_fraction",
        "tick_cost_fraction", "episode_progress", "pending_ready_in_1", "pending_ready_in_2",
    )  # fmt: skip
    assert (constants.max_service_rate_rps, constants.max_tick_capacity_requests) == (
        500.0,
        15000.0,
    )
    assert constants.max_tick_cost == pytest.approx(10 * 0.1 * 30 / 3600)
    assert AutoscalingEnv(DESIRED, idle()).observation_space.shape == (12,)


def idle() -> WorkloadTrace:
    return WorkloadTrace([0.0] * 120, 30.0)


def test_shared_builder_reproduces_the_simulator_feature_by_feature() -> None:
    """Independent re-derivation of every feature from physical info (no delay)."""
    env = AutoscalingEnv(DESIRED, build_workload(load_benchmark_manifest().get("syn-val-bursty")))
    controller = RandomController(seed=3, action_contract=ActionContract.from_config(DESIRED))
    observation, info = env.reset(seed=0)
    controller.reset(seed=0)
    assert not observation[[0, 1, 2, 3, 4, 5, 6, 8, 9]].any()  # nothing measured yet
    assert observation[7] == np.float32(0.1)  # one active replica of ten
    infos: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    for t in range(120):
        observation, _, _, _, info = env.step(controller.act(observation, env.decision_info(info)))
        infos.append(info)
        cumulative_cost += info["infrastructure_cost"]
        rates = [i["request_rate"] for i in reversed(infos[-4:])] + [0.0] * max(0, 4 - len(infos))
        q, p95 = info["queued_requests"], info["p95_latency_seconds"]
        expected = np.array(
            [
                *(r / (r + 500.0) for r in rates),
                info["utilization"],
                q / (q + 15000.0),
                p95 / (p95 + 0.5),
                env.replica_counts["active_replicas"] / 10,
                info["infrastructure_cost"] / (10 * 0.1 * 30 / 3600),  # latest tick, not cumulative
                (t + 1) / 120,
                *(c / 10 for c in env._pool.pending_by_ticks_until_active(30.0)),
            ],
            dtype=np.float32,
        )
        assert np.array_equal(observation, expected), t
    if t > 1:
        assert observation[8] != np.float32(cumulative_cost / (10 * 0.1 * 30 / 3600))


def test_build_observation_validates_its_inputs() -> None:
    constants = ObservationConstants.from_config(DESIRED)
    base: dict[str, Any] = {
        "request_rates_newest_first": [1.0],
        "latest": Measurement(0.5, 0.0, 0.1, 0.001),
        "active_replicas": 1,
        "pending_by_ticks": [0, 0],
        "completed_ticks": 1,
        "episode_ticks": 120,
    }
    assert build_observation(constants, **base).shape == (12,)
    with pytest.raises(ValueError, match="history"):
        build_observation(constants, **(base | {"request_rates_newest_first": [1.0] * 5}))
    with pytest.raises(ValueError, match="buckets"):
        build_observation(constants, **(base | {"pending_by_ticks": [0]}))


def test_observation_contract_documents_every_feature() -> None:
    contract = s2r.observation_contract()
    assert contract["version"] == "scalerl-observation-v1" and contract["dimension"] == 12
    assert [f["name"] for f in contract["features"]] == list(
        feature_names(ObservationConstants.from_config(DESIRED))
    )
    flagged = {f["name"] for f in contract["features"] if f["live_approximation"]}
    assert flagged == {"utilization", "queue_pressure", "latency_pressure"}
    cost = next(f for f in contract["features"] if f["name"] == "tick_cost_fraction")
    assert "latest tick" in cost["live_equivalent"]
    assert contract["normalization"]["episode_ticks"] == 120
