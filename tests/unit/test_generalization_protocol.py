"""Tests for the frozen benchmark-v2 generalization protocol (#115).

Methodology only: no trace is downloaded, no controller is run, no model is trained.
"""

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scalerl.environment import SimulatorConfig
from scalerl.evaluation import generalization_protocol as gp
from scalerl.evaluation.generalization_protocol import CatalogWindow

REPO = Path(__file__).resolve().parents[2]
BENCH_V1 = REPO / "benchmarks" / "v1"
PROTOCOL = REPO / "benchmarks" / "v2" / "generalization-protocol-v1.json"
PROTOCOL_ID = "f8eb4bfa9c42"


def committed() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(PROTOCOL.read_text())
    return payload


def group_with_role(role: str, source: str = gp.AZURE_2019) -> str:
    for i in range(10_000):
        if gp.group_role(source, f"owner-{i}") == role:
            return f"owner-{i}"
    raise AssertionError(role)


# --- artifact -------------------------------------------------------------------------------------


def test_committed_protocol_checks_and_id_is_pinned() -> None:
    protocol = gp.check_protocol(PROTOCOL, BENCH_V1)
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["protocol_version"] == "benchmark-v2-generalization-protocol-v1"
    payload = {k: v for k, v in protocol.items() if k != "protocol_id"}
    assert gp.content_id(payload) == PROTOCOL_ID


def test_write_is_byte_identical_to_the_committed_artifact(tmp_path: Path) -> None:
    out = tmp_path / "protocol.json"
    gp.write_protocol(out, BENCH_V1)
    assert out.read_bytes() == PROTOCOL.read_bytes()
    assert gp.build_protocol() == gp.build_protocol()


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda p: p["sampler"].__setitem__("source_weights", {"x": 1.0}), "protocol_id"),
        (lambda p: p.__setitem__("protocol_id", "000000000000"), "protocol_id"),
    ],
)
def test_check_refuses_a_tampered_artifact(tmp_path: Path, edit: Any, reason: str) -> None:
    protocol = committed()
    edit(protocol)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match=reason):
        gp.check_protocol(path, BENCH_V1)


def test_check_refuses_a_self_consistent_artifact_that_differs_from_code(tmp_path: Path) -> None:
    protocol = committed()
    protocol["seeds"]["training_seeds"] = [0, 1, 2]
    protocol["protocol_id"] = gp.content_id(
        {k: v for k, v in protocol.items() if k != "protocol_id"}
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="differs from this code"):
        gp.check_protocol(path, BENCH_V1)


def test_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert gp.main(["check", "--protocol", str(PROTOCOL), "--bench", str(BENCH_V1)]) == 0
    assert PROTOCOL_ID in capsys.readouterr().out
    assert gp.main(["show-id"]) == 0
    assert capsys.readouterr().out.strip() == PROTOCOL_ID
    assert gp.main(["check", "--protocol", str(tmp_path / "missing.json")]) == 2
    assert "error" in capsys.readouterr().err


# --- upstream and fixed Stage-A contract ----------------------------------------------------------


def test_upstream_v1_artifacts_are_the_frozen_ones() -> None:
    gp.verify_upstream(BENCH_V1)
    assert committed()["upstream"] == {
        "selection_spec_id": "418876d6c8e9",
        "action_decision_id": "0eb5562b01e6",
        "action_experiment_spec_id": "899dfbb64217",
        "reward_ablation_spec_id": "f5aff8f1e8c2",
        "reward_contract_id": "37158c261364",
        "heldout_v1_spec_id": "b32b2f3d3dd6",
        "heldout_v1_results_id": "d9d3fb985f2d",
    }


def test_tampered_heldout_results_are_refused(tmp_path: Path) -> None:
    copy = tmp_path / "v1"
    shutil.copytree(BENCH_V1, copy)
    results = json.loads((copy / "heldout-results-v1.json").read_text())
    results["post_hoc_protocol_change"] = True
    (copy / "heldout-results-v1.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="heldout_v1_results_id"):
        gp.verify_upstream(copy)


def test_stage_a_contract_is_exactly_v1() -> None:
    config = gp.stage_a_config()
    assert config.action.semantics == "desired-replicas-v1"
    assert (config.replicas.min_replicas, config.replicas.max_replicas) == (1, 10)
    assert config.replicas.initial_replicas == 1
    assert config.replicas.service_capacity_rps == 50.0
    assert config.replicas.startup_delay_seconds == 60.0
    assert config.timing.control_interval_seconds == 30.0
    assert config.timing.episode_duration_seconds == 3600.0
    assert config.sla.latency_target_seconds == 0.5
    assert config.dynamics.startup_delay_model == "fixed-v1"
    heldout_spec = json.loads((BENCH_V1 / "heldout-evaluation-v1.json").read_text())
    assert heldout_spec["simulator_config"] == config.model_dump(mode="json")
    env = committed()["fixed_environment"]
    assert env["changed_from_v1"] == []
    assert env["observation_version"] == "scalerl-observation-v1"
    reward = committed()["reward"]
    assert reward["variant"] == "full-cost-low-v1"
    assert reward["weights"] == {
        "latency": 1.0,
        "cost": 0.5,
        "sla": 1.0,
        "queue": 1.0,
        "churn": 0.1,
    }
    assert reward["reopened_by_stage_a"] is False


def test_observation_carries_no_source_or_domain_label() -> None:
    features = committed()["fixed_environment"]["observation_features"]
    assert len(features) == 12
    for feature in features:
        assert not re.search(r"source|domain|dataset|azure|synthetic|app|intensity|shape", feature)
    assert committed()["taxonomy"]["labels_in_observation"] is False


def test_leakage_flags_and_no_concrete_future_ids() -> None:
    protocol = committed()
    assert set(protocol["leakage"].values()) == {False}
    assert protocol["leakage"]["held_out_test_used_for_design"] is False
    text = PROTOCOL.read_text()
    assert not re.search(r"[0-9a-f]{32,}", text), "no concrete hashed app/window identity"
    assert not re.search(r"az19:[0-9a-f]|az21:[0-9a-f]", text)
    assert protocol["manifest_gate"]["required_before"] == "#118 first training run"


# --- sources --------------------------------------------------------------------------------------


def test_source_roles_and_licenses() -> None:
    sources = {s["source_id"]: s for s in committed()["datasets"]["sources"]}
    assert set(sources) == {
        gp.AZURE_2019, gp.AZURE_2021, gp.SYNTHETIC, gp.ALIBABA_2021, gp.GOOGLE_2019, gp.SEBS
    }  # fmt: skip
    assert "CC-BY-4.0" in sources[gp.AZURE_2019]["license"]
    assert "CC-BY-4.0" in sources[gp.AZURE_2021]["license"]
    assert "HashOwner" in sources[gp.AZURE_2019]["leakage_group"]
    assert any("completions" in c for c in sources[gp.AZURE_2019]["caveats"])
    assert sources[gp.AZURE_2021]["stage_a_role"].startswith("TEST-C")
    for excluded in (gp.ALIBABA_2021, gp.GOOGLE_2019, gp.SEBS):
        assert sources[excluded]["stage_a_role"].startswith("excluded")
    assert "none found" in sources[gp.ALIBABA_2021]["license"]
    assert "not a request trace" in sources[gp.GOOGLE_2019]["reason"]
    assert "not a workload trace" in sources[gp.SEBS]["reason"]
    assert set(gp.SOURCE_WEIGHTS) == {gp.AZURE_2019, gp.SYNTHETIC}
    assert gp.SOURCE_HOLDOUT not in gp.SOURCE_WEIGHTS


def test_identity_formats() -> None:
    assert gp.azure2019_window_id("o", "a", 3, 7) == "az19:o:a:d03h07"
    assert gp.azure2019_window_index(1, 0) == 0
    assert gp.azure2019_window_index(14, 23) == 335
    assert gp.azure2021_window_id("a", 276) == "az21:a:993600"
    assert gp.synthetic_window_id("spike", "p1", 4) == "syn2:spike:p1:4"
    assert gp.synthetic_group_id("spike", "p1") == "spike:p1"
    for bad in [(0, 0), (15, 0), (1, 24)]:
        with pytest.raises(ValueError):
            gp.azure2019_window_index(*bad)
        with pytest.raises(ValueError):
            gp.azure2019_window_id("o", "a", *bad)
    with pytest.raises(ValueError):
        gp.azure2021_window_id("a", 336)
    first = gp.synthetic_instance_seed(3).generate_state(4)
    assert np.array_equal(first, gp.synthetic_instance_seed(3).generate_state(4))
    assert not np.array_equal(first, gp.synthetic_instance_seed(4).generate_state(4))
    with pytest.raises(ValueError):
        gp.synthetic_instance_seed(-1)


# --- taxonomy -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mean", "label"),
    [
        (0.0, "near-idle"),
        (5.0, "near-idle"),
        (5.01, "low"),
        (25.0, "low"),
        (50.0, "single-replica"),
        (150.0, "multi-replica-moderate"),
        (400.0, "multi-replica-high"),
        (500.0, "near-saturation"),
        (500.1, "overload"),
        (1000.0, "overload"),
        (1000.1, None),
    ],
)
def test_intensity_boundaries_are_inclusive(mean: float, label: str | None) -> None:
    assert gp.classify_intensity(mean) == label


@pytest.mark.parametrize(
    ("peak", "label"),
    [(50.0, "within-one-replica"), (50.1, "multi-replica"), (500.0, "near-saturation"),
     (500.1, "fleet-overload")],
)  # fmt: skip
def test_peak_pressure(peak: float, label: str) -> None:
    assert gp.classify_peak_pressure(peak) == label


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_taxonomy_rejects_invalid_rates(bad: float) -> None:
    with pytest.raises(ValueError):
        gp.classify_intensity(bad)
    with pytest.raises(ValueError):
        gp.classify_peak_pressure(bad)
    with pytest.raises(ValueError):
        gp.classify_primary_shape([1.0] * 119 + [bad])


def test_minute_series() -> None:
    assert gp.minute_series([1.0, 3.0, 5.0, 5.0]) == (2.0, 5.0)
    with pytest.raises(ValueError, match="even"):
        gp.minute_series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="8 minutes"):
        gp.classify_primary_shape([1.0] * 14)
    with pytest.raises(ValueError, match="empty"):
        gp.classify_primary_shape([0.0] * 120)


RNG = np.random.default_rng(1)


def _dup(minutes: Any) -> list[float]:
    return [float(x) for v in minutes for x in (v, v)]


SHAPES = {
    "steady": [100.0] * 120,
    "ramp-up": list(np.clip(np.linspace(40, 360, 120) + RNG.normal(0, 30, 120), 0, None)),
    "ramp-down": list(np.linspace(350, 60, 120)),
    "periodic": [150 + 100 * math.sin(2 * math.pi * i / 20) for i in range(120)],
    "spike": [80.0] * 50 + [420.0] * 10 + [80.0] * 60,
    "regime-switch": [1.0] * 60 + [300.0] * 60,
    # four irregular 2-minute bursts: several excursions (not a spike), no period
    "bursty": [390.0 if i // 4 in (2, 9, 20, 26) else 90.0 for i in range(120)],
    "noisy-other": list(RNG.poisson(15, 120) / 30),
}


@pytest.mark.parametrize(("label", "rates"), SHAPES.items())
def test_primary_shape_rules(label: str, rates: list[float]) -> None:
    assert gp.classify_primary_shape(rates) == label


def test_persistent_random_walk_is_not_periodic() -> None:
    # The pre-review classifier called any persistent series periodic (high lag-2 correlation).
    rng = np.random.default_rng(7)
    for _ in range(50):
        walk = np.abs(100 + np.cumsum(rng.normal(0, 8, 120)))
        assert gp.periodic_lag(gp.minute_series(walk)) is None


def test_duplicated_minute_data_is_classified_like_its_minutes() -> None:
    # Azure 2019 repeats each minute over two ticks; ramps must stay detectable.
    minutes = list(np.linspace(10, 80, 60) + np.random.default_rng(3).normal(0, 2, 60))
    assert gp.classify_primary_shape(_dup(minutes)) == "ramp-up"
    assert gp.minute_series(_dup(minutes)) == pytest.approx(minutes)


def test_shape_tags() -> None:
    assert gp.traffic_shape_tags([1.0] * 60 + [300.0] * 60) == ("quiet-to-spike", "flash-crowd")
    assert gp.traffic_shape_tags(SHAPES["periodic"]) == ("repeated-wave",)
    assert gp.traffic_shape_tags([100.0] * 119 + [600.0]) == ("fleet-overload",)
    assert gp.traffic_shape_tags(SHAPES["steady"]) == ()


def test_autocorrelation_and_level_shift_edges() -> None:
    assert gp.autocorrelation([2.0] * 10, 1) == 0.0
    with pytest.raises(ValueError):
        gp.autocorrelation([1.0, 2.0], 2)
    assert gp.level_shift([5.0] * 16) == (0.0, 1.0)
    r2, ratio = gp.level_shift([10.0] * 8 + [40.0] * 8)
    assert r2 == pytest.approx(1.0) and ratio == pytest.approx(4.0)


# --- split ----------------------------------------------------------------------------------------


def test_periods_are_contiguous_with_a_one_day_embargo() -> None:
    periods = [gp.window_period(i) for i in range(336)]
    assert periods.count("development") == 264
    assert periods.count("embargo") == 24
    assert periods.count("temporal-test") == 48
    assert periods == sorted(periods, key=["development", "embargo", "temporal-test"].index)
    assert gp.window_period(gp.azure2019_window_index(12, 0)) == "embargo"
    assert gp.window_period(gp.azure2019_window_index(13, 0)) == "temporal-test"
    for bad in (-1, 336):
        with pytest.raises(ValueError):
            gp.window_period(bad)


def test_group_role_is_deterministic_and_close_to_80_10_10() -> None:
    roles = Counter(gp.group_role(gp.AZURE_2019, f"g{i}") for i in range(20_000))
    assert abs(roles["development"] / 20_000 - 0.8) < 0.02
    assert abs(roles["validation"] / 20_000 - 0.1) < 0.01
    assert abs(roles["application-test"] / 20_000 - 0.1) < 0.01
    assert gp.group_role(gp.AZURE_2019, "g1") == gp.group_role(gp.AZURE_2019, "g1")
    assert gp.stable_bucket("a", modulus=10) == int(gp.stable_hash("a")[:16], 16) % 10
    with pytest.raises(ValueError):
        gp.stable_bucket("a", modulus=0)


def test_v1_azure_hours_are_derived_from_the_frozen_manifest() -> None:
    intervals = gp.v1_azure_intervals()
    assert intervals["test"] == ((993600.0, 997200.0), (1166400.0, 1170000.0))
    assert len(intervals["train"]) == 3 and len(intervals["validation"]) == 1
    flagged = [i for i in range(336) if gp.overlaps_v1_azure(i)]
    assert flagged == [36, 84, 132, 204, 276, 324]
    recorded = committed()["workload_unit"]["v1_azure_intervals_seconds"]
    assert recorded["test"] == [[993600.0, 997200.0], [1166400.0, 1170000.0]]


CELL = ("low", "bursty")
HELD = ("near-idle", "spike")


def role(source: str, group: str, index: int, **kw: Any) -> str:
    kw.setdefault("total_requests", 100)
    kw.setdefault("mean_rps", 10.0)
    kw.setdefault("cell", CELL)
    return gp.assign_role(source, group, index, **kw).role


def test_role_table_for_azure_2019() -> None:
    dev, val, app = (group_with_role(r) for r in ("development", "validation", "application-test"))
    assert role(gp.AZURE_2019, dev, 0) == gp.ROLE_TRAIN
    assert role(gp.AZURE_2019, dev, 263) == gp.ROLE_TRAIN
    assert role(gp.AZURE_2019, dev, 264) == gp.ROLE_RESERVED
    assert role(gp.AZURE_2019, dev, 288) == gp.ROLE_TEST_A
    assert role(gp.AZURE_2019, val, 10) == gp.ROLE_VALIDATION
    assert role(gp.AZURE_2019, val, 300) == gp.ROLE_RESERVED
    assert role(gp.AZURE_2019, app, 10) == gp.ROLE_TEST_B_APP
    assert role(gp.AZURE_2019, app, 300) == gp.ROLE_RESERVED
    held = {"cell": HELD, "compositional_cells": [HELD]}
    assert role(gp.AZURE_2019, dev, 10, **held) == gp.ROLE_TEST_B_COMP
    assert role(gp.AZURE_2019, dev, 300, **held) == gp.ROLE_TEST_B_COMP
    assert role(gp.AZURE_2019, dev, 270, **held) == gp.ROLE_RESERVED
    assert role(gp.AZURE_2019, val, 10, **held) == gp.ROLE_EXCLUDED
    assert role(gp.AZURE_2019, app, 10, **held) == gp.ROLE_TEST_B_APP


def test_role_exclusions_and_other_sources() -> None:
    assert role(gp.AZURE_2019, "g", 0, total_requests=0, cell=None) == gp.ROLE_EXCLUDED
    beyond = gp.assign_role(gp.AZURE_2019, "g", 0, total_requests=9, mean_rps=1001.0, cell=None)
    assert beyond == gp.RoleDecision(gp.ROLE_EXCLUDED, "beyond-scope-overload")
    assert role(gp.AZURE_2021, "app", 0) == gp.ROLE_TEST_C
    assert gp.assign_role(
        gp.AZURE_2021, "app", 276, total_requests=5, mean_rps=1.0, cell=CELL
    ) == gp.RoleDecision(gp.ROLE_EXCLUDED, "v1-benchmark-overlap")
    synthetic = Counter(role(gp.SYNTHETIC, f"family:{i}", 0) for i in range(2000))
    assert set(synthetic) == {gp.ROLE_TRAIN, gp.ROLE_VALIDATION}
    assert role(gp.SYNTHETIC, "x:1", 0, cell=HELD, compositional_cells=[HELD]) == gp.ROLE_EXCLUDED
    with pytest.raises(ValueError, match="not a Stage-A"):
        role(gp.ALIBABA_2021, "g", 0)
    with pytest.raises(ValueError, match="cell"):
        role(gp.AZURE_2019, "g", 0, cell=None)
    with pytest.raises(ValueError):
        role(gp.AZURE_2019, "g", 0, total_requests=-1)


def test_split_is_leakage_safe_over_a_synthetic_catalog() -> None:
    rng = np.random.default_rng(0)
    cells = [(i, s) for i in gp.INTENSITY_LABELS[:4] for s in gp.SHAPE_LABELS[:4]]
    held = [cells[5]]
    by_group: dict[str, set[str]] = {}
    for g in range(400):
        group = f"owner-{g}"
        for index in range(336):
            cell = cells[int(rng.integers(len(cells)))]
            r = role(gp.AZURE_2019, group, index, cell=cell, compositional_cells=held)
            by_group.setdefault(group, set()).add(r)
            if r in (gp.ROLE_TRAIN, gp.ROLE_VALIDATION):
                assert cell not in held
            if r == gp.ROLE_TRAIN:
                assert index < 264
            if r == gp.ROLE_TEST_A:
                assert index >= 288  # >= 24 windows after the last train window
    exclusive = {gp.ROLE_TRAIN, gp.ROLE_VALIDATION, gp.ROLE_TEST_B_APP}
    for roles in by_group.values():
        assert len(roles & exclusive) <= 1


def test_choose_compositional_cells() -> None:
    windows = {
        ("low", "bursty"): 100,
        ("low", "steady"): 100,
        ("near-idle", "bursty"): 100,
        ("near-idle", "steady"): 100,
        ("multi-replica-high", "spike"): 50,
        ("multi-replica-high", "ramp-up"): 50,
        ("overload", "spike"): 30,
    }
    groups = dict.fromkeys(windows, 10)
    chosen = gp.choose_compositional_cells(windows, groups)
    assert 1 <= len(chosen) <= 2
    assert chosen == gp.choose_compositional_cells(dict(reversed(windows.items())), groups)
    if len(chosen) == 2:
        assert chosen[0][0] != chosen[1][0] and chosen[0][1] != chosen[1][1]
    for intensity, shape in chosen:
        rest = {c for c in windows if c not in chosen and windows[c] >= 20}
        assert any(c[0] == intensity for c in rest) and any(c[1] == shape for c in rest)
    # ("overload", "ramp-up") would be unsupported; a lone cell never qualifies.
    assert gp.choose_compositional_cells({("low", "bursty"): 100}, {("low", "bursty"): 9}) == ()
    few_groups = dict.fromkeys(windows, 4)
    assert gp.choose_compositional_cells(windows, few_groups) == ()
    with pytest.raises(ValueError, match="unknown"):
        gp.choose_compositional_cells({("x", "bursty"): 1}, {})


# --- evaluation sets and sampler ------------------------------------------------------------------


def window(i: int, *, role: str = gp.ROLE_TRAIN, source: str = gp.AZURE_2019, **kw: Any) -> Any:
    return CatalogWindow(
        source_id=source,
        group_id=kw.get("group", f"g{i % 7}"),
        window_id=f"w{i:04d}",
        intensity=kw.get("intensity", gp.INTENSITY_LABELS[i % 3]),
        shape=kw.get("shape", gp.SHAPE_LABELS[i % 2]),
        role=role,
    )


def test_catalog_window_validates_labels() -> None:
    with pytest.raises(ValueError):
        window(0, intensity="huge")
    with pytest.raises(ValueError):
        window(0, shape="wobbly")
    with pytest.raises(ValueError):
        window(0, role="test")


def test_evaluation_sample_is_capped_group_diverse_and_order_free() -> None:
    pool = [window(i, role=gp.ROLE_TEST_A) for i in range(300)]
    chosen = gp.select_evaluation_windows(pool, cap_per_cell=4)
    assert chosen == gp.select_evaluation_windows(list(reversed(pool)), cap_per_cell=4)
    per_cell = Counter((w.source_id, w.cell) for w in chosen)
    assert set(per_cell.values()) == {4}
    seen = Counter((w.source_id, w.cell, w.group_id) for w in chosen)
    assert set(seen.values()) == {1}
    one_group = [window(i, role=gp.ROLE_VALIDATION, group="same") for i in range(20)]
    assert len(gp.select_evaluation_windows(one_group, cap_per_cell=4)) == 6  # one per cell
    with pytest.raises(ValueError, match="role"):
        gp.select_evaluation_windows([window(0)], cap_per_cell=4)
    with pytest.raises(ValueError, match="role"):
        gp.select_evaluation_windows(
            [window(0, role=gp.ROLE_TEST_A), window(1, role=gp.ROLE_TEST_C)], cap_per_cell=4
        )
    with pytest.raises(ValueError, match="duplicate"):
        gp.select_evaluation_windows([window(0, role=gp.ROLE_TEST_A)] * 2, cap_per_cell=4)
    with pytest.raises(ValueError):
        gp.select_evaluation_windows(pool, cap_per_cell=0)


def train_pool() -> list[CatalogWindow]:
    real = [window(i) for i in range(60)]
    synthetic = [
        CatalogWindow(gp.SYNTHETIC, f"s{i}", f"s{i:03d}", "overload", "spike", gp.ROLE_TRAIN)
        for i in range(5)
    ]
    return real + synthetic


def test_sampler_refuses_anything_but_train() -> None:
    for bad in (gp.ROLE_VALIDATION, *gp.TEST_ROLES, gp.ROLE_RESERVED, gp.ROLE_EXCLUDED):
        with pytest.raises(ValueError, match="TRAIN sampler"):
            gp.build_sampling_tree([window(0), window(1, role=bad)])
    with pytest.raises(ValueError, match="TRAIN source"):
        gp.build_sampling_tree([window(0, source=gp.AZURE_2021)])
    with pytest.raises(ValueError, match="duplicate"):
        gp.build_sampling_tree([window(0), window(0)])
    with pytest.raises(ValueError, match="empty"):
        gp.draw_episode(gp.sampler_rng(0), {})


def test_episode_sequence_is_reproducible_per_seed() -> None:
    tree = gp.build_sampling_tree(train_pool())
    first = gp.episode_sequence(0, tree, 200)
    assert first == gp.episode_sequence(0, tree, 200)
    assert first != gp.episode_sequence(1, tree, 200)
    assert gp.episode_sequence(0, tree, 50) == first[:50]
    assert gp.episode_sequence(0, gp.build_sampling_tree(reversed(train_pool())), 200) == first
    with pytest.raises(ValueError):
        gp.episode_sequence(0, tree, -1)
    with pytest.raises(ValueError):
        gp.sampler_rng(-1)


def test_sampler_matches_its_frozen_probabilities() -> None:
    tree = gp.build_sampling_tree(train_pool())
    probabilities = gp.sampling_probabilities(tree)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    synthetic = sum(p for w, p in probabilities.items() if w.startswith("s"))
    assert synthetic == pytest.approx(0.25)
    counts = Counter(gp.episode_sequence(3, tree, 40_000))
    for window_id, p in probabilities.items():
        assert counts[window_id] / 40_000 == pytest.approx(p, abs=0.01)
    # 6 real cells: each (intensity, shape) cell gets the same share regardless of size.
    real_only = gp.sampling_probabilities(gp.build_sampling_tree(train_pool()[:60]))
    assert sum(real_only.values()) == pytest.approx(1.0)


# --- selection ------------------------------------------------------------------------------------


VALIDATION = {
    f"v{i}": CatalogWindow(
        gp.AZURE_2019 if i < 4 else gp.SYNTHETIC,
        f"g{i}",
        f"v{i}",
        "low" if i % 2 == 0 else "multi-replica-high",
        "bursty",
        gp.ROLE_VALIDATION,
    )
    for i in range(6)
}


def metrics(
    sla: float | dict[str, float], cost: float, queue: float = 0.1, churn: float = 0.1
) -> Any:
    per = sla if isinstance(sla, dict) else dict.fromkeys(VALIDATION, sla)
    return {
        w: {"sla_violation_rate": per[w], "normalized_cost": cost, "queue_pressure": queue,
            "churn_rate": churn}
        for w in VALIDATION
    }  # fmt: skip


REFERENCE = metrics(0.10, 0.5)


def test_balanced_mean_weights_sources_then_strata_equally() -> None:
    values = {"v0": 1.0, "v2": 1.0, "v1": 0.0, "v3": 0.0, "v4": 10.0, "v5": 20.0}
    # azure: low 1.0, high 0.0 -> 0.5; synthetic: low 10, high 20 -> 15; overall 7.75
    assert gp.balanced_mean(values, VALIDATION) == pytest.approx(7.75)
    with pytest.raises(ValueError):
        gp.balanced_mean({}, VALIDATION)


def test_selection_prefers_feasible_then_cost_then_tie_breaks() -> None:
    outcome = gp.select_candidate(
        {
            "cheap-but-violating": metrics(0.20, 0.2),
            "a": metrics(0.11, 0.4, queue=0.2),
            "b": metrics(0.11, 0.4, queue=0.1),
            "expensive": metrics(0.0, 0.9),
        },
        REFERENCE,
        VALIDATION,
    )
    assert outcome.selected == "b"
    assert outcome.ranking == ("b", "a", "expensive")
    assert "cheap-but-violating" not in outcome.feasible
    assert outcome.validation_feasible and outcome.diagnostic_fallback is None


def test_window_and_group_allowances() -> None:
    one_bad = dict.fromkeys(VALIDATION, 0.10) | {"v0": 0.16}
    assert gp.select_candidate({"x": metrics(one_bad, 0.1)}, REFERENCE, VALIDATION).selected is None
    at_window_limit = dict.fromkeys(VALIDATION, 0.10) | {"v0": 0.15}
    # group (azure, low) = mean(0.15, 0.10) = 0.125 -> excess 0.025 > 0.02
    assert (
        gp.select_candidate({"x": metrics(at_window_limit, 0.1)}, REFERENCE, VALIDATION).selected
        is None
    )
    within = dict.fromkeys(VALIDATION, 0.10) | {"v0": 0.14}
    assert gp.select_candidate({"x": metrics(within, 0.1)}, REFERENCE, VALIDATION).selected == "x"


def test_no_feasible_candidate_selects_nothing() -> None:
    outcome = gp.select_candidate(
        {"bad": metrics(0.5, 0.1), "less-bad": metrics(0.3, 0.9)}, REFERENCE, VALIDATION
    )
    assert outcome.selected is None and outcome.ranking == ()
    assert outcome.diagnostic_fallback == "less-bad"
    assert outcome.validation_feasible is False


def test_selection_evidence_must_be_complete_validation_only() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        gp.select_candidate({}, REFERENCE, VALIDATION)
    partial = {k: v for k, v in metrics(0.1, 0.1).items() if k != "v0"}
    with pytest.raises(ValueError, match="cover exactly"):
        gp.select_candidate({"x": partial}, REFERENCE, VALIDATION)
    nan = metrics(0.1, math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        gp.select_candidate({"x": nan}, REFERENCE, VALIDATION)
    test_window = {**VALIDATION, "v0": window(0, role=gp.ROLE_TEST_A)}
    with pytest.raises(ValueError, match="VALIDATION windows only"):
        gp.select_candidate({"x": metrics(0.1, 0.1)}, REFERENCE, test_window)


def test_canonical_seed_rule() -> None:
    seeds = {s: metrics(0.1, 0.5) for s in gp.TRAINING_SEEDS}
    assert gp.select_canonical_seed(seeds, REFERENCE, VALIDATION) == (0, True)
    seeds[3] = metrics(0.1, 0.3)
    assert gp.select_canonical_seed(seeds, REFERENCE, VALIDATION) == (3, True)
    infeasible = {s: metrics(0.5 - 0.01 * s, 0.5) for s in gp.TRAINING_SEEDS}
    assert gp.select_canonical_seed(infeasible, REFERENCE, VALIDATION) == (4, False)
    with pytest.raises(ValueError, match="training seeds"):
        gp.select_canonical_seed({0: metrics(0.1, 0.5)}, REFERENCE, VALIDATION)


# --- statistics -----------------------------------------------------------------------------------


def test_clustered_bootstrap_is_deterministic_and_clustered() -> None:
    deltas = {f"w{i}": float(i % 5) - 2.0 for i in range(40)}
    groups = {f"w{i}": f"g{i % 8}" for i in range(40)}
    first = gp.clustered_bootstrap_ci(deltas, groups)
    assert first == gp.clustered_bootstrap_ci(dict(reversed(deltas.items())), groups)
    point, low, high = first
    assert point == pytest.approx(0.0) and low <= point <= high
    assert gp.clustered_bootstrap_ci(dict.fromkeys(deltas, 1.5), groups) == (1.5, 1.5, 1.5)
    clustered = {f"w{i}": 1.0 if i < 20 else -1.0 for i in range(40)}
    two_groups = {f"w{i}": "a" if i < 20 else "b" for i in range(40)}
    many_groups = {f"w{i}": f"g{i}" for i in range(40)}
    wide = gp.clustered_bootstrap_ci(clustered, two_groups)
    narrow = gp.clustered_bootstrap_ci(clustered, many_groups)
    assert wide[2] - wide[1] > narrow[2] - narrow[1]
    with pytest.raises(ValueError):
        gp.clustered_bootstrap_ci({}, {})
    with pytest.raises(ValueError):
        gp.clustered_bootstrap_ci({"w": math.inf}, {"w": "g"})


# --- protocol content -----------------------------------------------------------------------------


def test_protocol_freezes_every_required_decision() -> None:
    p = committed()
    assert p["stage"] == "generalist-workload-v1"
    assert p["stage_boundary"]["stage_a_changes"] == ["training workload distribution"]
    assert p["stage_boundary"]["stage_b_owner"].startswith("#120")
    assert [t["id"] for t in p["test_tiers"]] == [
        "TEST-A", "TEST-B-application", "TEST-B-compositional", "TEST-C"
    ]  # fmt: skip
    assert p["seeds"]["training_seeds"] == [0, 1, 2, 3, 4]
    assert p["sampler"]["source_weights"] == {gp.AZURE_2019: 0.75, gp.SYNTHETIC: 0.25}
    assert p["selection"]["reward_used"] is False
    assert p["selection"]["reference"]["params"] == {
        "high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3
    }  # fmt: skip
    assert p["comparators"]["baselines_retuned"] is False
    assert p["metrics"]["overall_winner_score"] is None
    assert p["project_scope"]["rl_required_to_win"] is False
    assert p["heldout_v1_relationship"]["v1_test_hours_in_benchmark_v2"] is False
    assert p["robustness_boundary"]["issue_21"]["train_role"] == "none in Stage A"
    assert p["evaluation_sets"]["validation_cap_per_cell"] == 4
    assert p["evaluation_sets"]["test_cap_per_cell"] == 8
    assert p["taxonomy"]["intensity"]["upper_bounds_inclusive"][-1] == {
        "label": "overload", "max_mean_rps": 1000.0
    }  # fmt: skip
    assert len(p["taxonomy"]["traffic_shape"]["ordered_rules"]) == len(gp.SHAPE_LABELS)


def test_default_simulator_config_is_not_stage_a() -> None:
    # Stage A must opt into desired-replicas-v1 explicitly; the library default is delta-v1.
    assert SimulatorConfig().action.semantics != gp.stage_a_config().action.semantics
