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


# --- the frozen manifests ----------------------------------------------------------------------


def manifest(name: str) -> Any:
    return json.loads((BENCH / name).read_text())


def test_committed_manifests_check_and_ids_are_pinned() -> None:
    ids = s2r.check(BENCH)  # recomputes seeds, verifies schedules and protocol references
    assert ids == {
        "protocol": "b7bf45c109f1",
        "replay": "7803b71fdfe8",
        "controllers": "553ddf3e512b",
    }


def test_control_contract_is_frozen() -> None:
    control = manifest("sim-to-real-protocol-v1.json")["control_contract"]
    assert control["control_interval_seconds"] == 30.0
    assert (control["min_replicas"], control["max_replicas"]) == (1, 10)
    assert control["action_semantics"] == "desired-replicas-v1"
    assert control["deterministic_policy_inference"] is True
    assert control["online_training"] is False and control["exploration"] is False
    assert control["scale_to_zero"].startswith("disabled")
    assert "not +1/-1" in control["action_meaning"]


def test_transfer_reporting_rule_is_descriptive_only() -> None:
    protocol = manifest("sim-to-real-protocol-v1.json")
    rule = protocol["transfer_reporting"]
    assert rule["rl_must_win"] is False
    assert rule["pass_fail"].startswith("none") and rule["overall_score"].startswith("none")
    assert "Threshold" in rule["relative"] and "secondary" in rule["secondary"]
    assert protocol["held_out_controller_outcomes_used"] is False
    assert protocol["live_results_used"] is False
    assert protocol["reward"]["contract_id"] == "37158c261364"
    assert protocol["reward"]["variant"] == "full-cost-low-v1"
    assert protocol["observation_contract"]["version"] == "scalerl-observation-v1"
    assert "no AWS Lambda" in protocol["cost_contract"]["not_claimed"]
    assert protocol["terminology"] == "local systems-in-the-loop / Knative sim-to-real validation"


def test_replay_windows_and_schedules_are_frozen() -> None:
    replay = manifest("live-replay-manifest-v1.json")
    assert replay["source_workload"]["workload_id"] == "azure-test-1166400"
    slices = replay["slices"]
    assert [(s["start_offset_seconds"], s["duration_seconds"]) for s in slices] == [
        (600.0, 600.0), (1500.0, 600.0), (2400.0, 600.0)
    ]  # fmt: skip
    assert [(s["start_bin"], s["end_bin_exclusive"]) for s in slices] == [
        (20, 40),
        (50, 70),
        (80, 100),
    ]
    assert all(s["split"] == "test" and s["control_interval_seconds"] == 30.0 for s in slices)
    assert len({s["source_trace_fingerprint"] for s in slices}) == 1
    for entry in slices:
        assert [x["load_seed"] for x in entry["schedules"]] == [0, 1, 2]
        counts = {x["request_count"] for x in entry["schedules"]}
        assert counts == {entry["request_count"]}  # every seed keeps the exact trace counts
        files = [json.loads((BENCH / x["file"]).read_text()) for x in entry["schedules"]]
        assert len({json.dumps(f["timestamps_seconds_from_slice_start"]) for f in files}) == 3
        assert all(
            f["rng"]["spawn_key"] == [s2r.SCHEDULE_RNG_DOMAIN, entry["slice_index"]] for f in files
        )
        assert all(0 <= t < 600 for f in files for t in f["timestamps_seconds_from_slice_start"])
    assert replay["controller_outcomes_used"] is False and replay["live_results_used"] is False


def test_canonical_learned_artifacts_recompute_from_committed_evidence() -> None:
    controllers = manifest("controller-deployment-manifest-v1.json")
    contract = s2r.verify_upstream(BENCH)["contract"]
    expected = {"dqn": (0, "920b0ddb58df4d2b9f550431d2f8ceeb", [0, 1, 2, 4]),
                "ppo": (4, "9bac13a9447b4c64a44d9732568c949e", [0, 1, 2, 3, 4])}  # fmt: skip
    for family, (seed_id, run_id, feasible) in expected.items():
        entry = controllers["controllers"][family]
        seeds = [
            s2r.SeedEvidence.model_validate(s, strict=False) for s in entry["five_seed_lineage"]
        ]
        assert [s.training_seed for s in seeds] == [0, 1, 2, 3, 4]
        # only validation workloads participate; no held-out metric exists in the evidence
        assert all(set(s.metrics) == set(THRESHOLDS) for s in seeds)
        s2r.check_against_contract(
            seeds, contract.per_reward["full-cost-low-v1"][f"{family}_equal_seed_means"]
        )
        result = s2r.select_canonical_seed(seeds, THRESHOLDS)
        assert result == entry["canonical_selection"]
        assert (result["canonical_training_seed"], result["canonical_training_run_id"]) == (
            seed_id,
            run_id,
        )
        assert result["canonical_seed_validation_feasible"] is True
        assert result["individually_feasible_seeds"] == feasible
        assert entry["action_semantics"] == "desired-replicas-v1"
        assert entry["reward_contract_id"] == "37158c261364"
    assert controllers["controllers"]["dqn"]["candidate_id"] == "dqn-c14"
    assert controllers["controllers"]["ppo"]["candidate_id"] == "ppo-c08"


def test_controller_set() -> None:
    controllers = manifest("controller-deployment-manifest-v1.json")
    assert sorted(controllers["controllers"]) == [
        "dqn", "knative-native-v1", "ppo", "predictive-seasonal-v1", "predictive-v1", "threshold-v1"
    ]  # fmt: skip
    assert controllers["controllers"]["threshold-v1"]["params"] == {
        "high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3
    }  # fmt: skip
    assert controllers["controllers"]["predictive-v1"]["forecast_method"] == "linear-trend"
    seasonal = controllers["controllers"]["predictive-seasonal-v1"]
    assert seasonal["historical_profile"]["profile_id"] == "d050d3b8ca0f"
    native = controllers["controllers"]["knative-native-v1"]["settings"]
    assert (native["min-scale"], native["max-scale"]) == (1, 10)
    assert "q-learning" in controllers["excluded"]


# --- the requested bench directory is the only source ------------------------------------------


def outputs_from_lineage(root: Path) -> Path:
    """Minimal #20 per-seed evidence files rebuilt from the committed lineage."""
    controllers = manifest("controller-deployment-manifest-v1.json")["controllers"]
    for family, phase in (("dqn", "dqn-retraining"), ("ppo", "ppo-training")):
        entry = controllers[family]
        directory = root / s2r.SELECTED_REWARD / phase
        directory.mkdir(parents=True)
        for seed in entry["five_seed_lineage"]:
            payload = {
                "candidate_id": entry["candidate_id"],
                "training_run_id": seed["training_run_id"],
                "model_artifact_uri": seed["model_artifact_uri"],
                "workloads": [
                    {"workload_id": w, "validation_run_id": run, "metrics": seed["metrics"][w]}
                    for w, run in seed["validation_run_ids"].items()
                ],
            }
            name = f"{entry['candidate_id']}-seed{seed['training_seed']}.json"
            (directory / name).write_text(json.dumps(payload))
    return root


def test_controller_manifest_uses_only_the_requested_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "staging" / "bench"
    shutil.copytree(BENCH, custom)
    outputs = outputs_from_lineage(tmp_path / "outputs")
    elsewhere = tmp_path / "elsewhere"  # no benchmarks/v1 below the working directory
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    upstream = s2r.verify_upstream(custom)
    built = json.loads(s2r._dump(s2r.build_controller_manifest(upstream, outputs)))
    assert built == manifest("controller-deployment-manifest-v1.json")
    assert s2r.content_id(built) == "553ddf3e512b"
    assert s2r.check(custom)["controllers"] == "553ddf3e512b"

    # the candidate set is read from the custom bench, and a changed one is refused
    candidates = json.loads((custom / "action-semantics-candidates-v1.json").read_text())
    candidates["dqn"]["candidates"][14]["hyperparameters"]["gamma"] = 0.5
    (custom / "action-semantics-candidates-v1.json").write_text(json.dumps(candidates))
    with pytest.raises(ValueError, match="candidate set changed"):
        s2r.verify_upstream(custom)


# --- frozen schedule validation ----------------------------------------------------------------

SCHEDULE_FILE = "live-replay-schedules-v1/azure-test-1166400-w1-s1500-d600-seed1.json"


def slice_and_ref(replay: Any, file: str = SCHEDULE_FILE) -> tuple[Any, Any]:
    for entry in replay["slices"]:
        for ref in entry["schedules"]:
            if ref["file"] == file:
                return entry, ref
    raise AssertionError(file)


def test_every_committed_schedule_validates() -> None:
    replay = manifest("live-replay-manifest-v1.json")
    for entry in replay["slices"]:
        for ref in entry["schedules"]:
            assert s2r.schedule_errors(manifest(ref["file"]), ref, entry) == []


def bench_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "v1"
    shutil.copytree(BENCH, copy)
    return copy


def edit_json(path: Path, change: Any) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def set_rng(key: str, value: Any) -> Any:
    return lambda p: p["rng"].__setitem__(key, value)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            lambda p: p.__setitem__("schedule_version", "live-arrival-schedule-v2"),
            "schedule_version",
        ),
        (lambda p: p.__setitem__("generator", "poisson-v1"), "generator"),
        (lambda p: p.__setitem__("schedule_id", "other"), "schedule_id"),
        (lambda p: p.__setitem__("slice_id", "azure-test-1166400-w0-s600-d600"), "slice_id"),
        (lambda p: p.__setitem__("load_seed", 2), "load_seed"),
        (lambda p: p.__setitem__("source_trace_fingerprint", "0" * 64), "fingerprint"),
        (set_rng("seed_sequence_entropy", 7), "rng"),
        (set_rng("spawn_key", [s2r.SCHEDULE_RNG_DOMAIN, 0]), "rng"),
        (lambda p: p.__setitem__("timestamps_checksum", "0" * 64), "embedded timestamps_checksum"),
        (lambda p: p.__setitem__("note", "extra"), "schedule keys"),
        (lambda p: p["timestamps_seconds_from_slice_start"].pop(), "timestamps_checksum"),
    ],
)
def test_check_refuses_tampered_schedule_file(tmp_path: Path, change: Any, reason: str) -> None:
    copy = bench_copy(tmp_path)
    edit_json(copy / SCHEDULE_FILE, change)
    with pytest.raises(ValueError, match=reason):
        s2r.check(copy)


def manifest_ref(key: str, value: Any) -> Any:
    def change(replay: Any) -> None:
        slice_and_ref(replay)[1][key] = value

    return change


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (manifest_ref("timestamps_checksum", "0" * 64), "manifest timestamps_checksum"),
        (manifest_ref("request_count", 1), "request count"),
        (manifest_ref("schedule_id", "other"), "manifest schedule_id"),
        (manifest_ref("load_seed", 5), "load seeds"),
        (manifest_ref("extra", 1), "manifest reference keys"),
        (
            manifest_ref(
                "file", "live-replay-schedules-v1/azure-test-1166400-w1-s1500-d600-seed2.json"
            ),
            "manifest file",
        ),
        (lambda r: r["slices"][1].__setitem__("source_trace_fingerprint", "0" * 64), "fingerprint"),
        (lambda r: r["schedule_rule"].__setitem__("generator", "poisson-v1"), "schedule rule"),
    ],
)
def test_check_refuses_tampered_manifest_reference(
    tmp_path: Path, change: Any, reason: str
) -> None:
    copy = bench_copy(tmp_path)
    edit_json(copy / "live-replay-manifest-v1.json", change)
    with pytest.raises(ValueError, match=reason):
        s2r.check(copy)


def test_check_refuses_missing_and_unreferenced_schedule_files(tmp_path: Path) -> None:
    copy = bench_copy(tmp_path)
    (copy / "live-replay-schedules-v1" / "stray.json").write_text("{}")
    with pytest.raises(ValueError, match="unreferenced schedule files"):
        s2r.check(copy)
    (copy / "live-replay-schedules-v1" / "stray.json").unlink()
    (copy / SCHEDULE_FILE).unlink()
    with pytest.raises(ValueError, match="missing schedule file"):
        s2r.check(copy)


def resealed(times: list[float]) -> tuple[Any, Any, Any]:
    """A schedule whose checksums and counts are consistent with ``times``."""
    replay = manifest("live-replay-manifest-v1.json")
    entry, ref = slice_and_ref(replay)
    payload = manifest(SCHEDULE_FILE)
    checksum = s2r._sha256(times)
    payload["timestamps_seconds_from_slice_start"] = times
    payload["timestamps_checksum"] = ref["timestamps_checksum"] = checksum
    ref["request_count"] = entry["request_count"] = len(times)
    return payload, ref, entry


def test_schedule_errors_require_sorted_in_range_timestamps() -> None:
    times = manifest(SCHEDULE_FILE)["timestamps_seconds_from_slice_start"]
    assert s2r.schedule_errors(*resealed(times)) == []
    unsorted = [times[1], times[0], *times[2:]]
    assert s2r.schedule_errors(*resealed(unsorted)) == ["timestamps are not sorted"]
    assert s2r.schedule_errors(*resealed([*times[:-1], 600.0])) == [
        "timestamps outside [0, slice duration)"
    ]
    assert s2r.schedule_errors(*resealed([-0.5, *times[1:]])) == [
        "timestamps outside [0, slice duration)"
    ]
    payload, ref, entry = resealed(times)
    payload["timestamps_seconds_from_slice_start"] = ["1.0"]
    assert s2r.schedule_errors(payload, ref, entry)[-1] == "timestamps are not a list of numbers"
