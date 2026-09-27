"""Tests for the #20 reward ablation: frozen spec, lineage, decision rule, guards.

Synthetic evidence and tiny budgets only; nothing reads outputs/ or test workloads.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from scalerl.environment import DESIRED_REPLICAS_V1, SimulatorConfig
from scalerl.environment.reward import RewardWeights, compute_reward
from scalerl.evaluation import reward_ablation as ra
from scalerl.evaluation.action_semantics import ExperimentSpec as ActionSpec
from scalerl.evaluation.model_selection import SelectionSpec
from scalerl.tuning.candidates import CandidateSet

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "benchmarks" / "v1" / "reward-ablation-v1.json"
CANDIDATES = REPO / "benchmarks" / "v1" / "action-semantics-candidates-v1.json"
SELECTION = REPO / "benchmarks" / "v1" / "selection-v2-cost-under-sla.json"
ACTION_SPEC = REPO / "benchmarks" / "v1" / "action-semantics-experiment-v1.json"
DECISION = REPO / "benchmarks" / "v1" / "action-contract-v2.json"
SPEC_ID = "f5aff8f1e8c2"
THRESHOLDS = {
    "syn-val-steady-high": 0.275,
    "syn-val-ramp-down": 0.35,
    "syn-val-bursty": 0.20833333333333334,
}


def frozen() -> tuple[ra.ExperimentSpec, CandidateSet, SelectionSpec]:
    return ra.load_frozen(SPEC, DECISION, ACTION_SPEC, CANDIDATES, SELECTION)


# --- predeclared variants and spec ----------------------------------------------------------------


def test_the_eight_reward_variants_are_pinned() -> None:
    expected = {
        "latency-cost-v1": (1.0, 1.0, 0.0, 0.0, 0.0),
        "latency-cost-sla-v1": (1.0, 1.0, 1.0, 0.0, 0.0),
        "latency-cost-sla-queue-v1": (1.0, 1.0, 1.0, 1.0, 0.0),
        "full-default-v1": (1.0, 1.0, 1.0, 1.0, 0.1),
        "full-cost-low-v1": (1.0, 0.5, 1.0, 1.0, 0.1),
        "full-cost-high-v1": (1.0, 2.0, 1.0, 1.0, 0.1),
        "full-sla-low-v1": (1.0, 1.0, 0.5, 1.0, 0.1),
        "full-sla-high-v1": (1.0, 1.0, 2.0, 1.0, 0.1),
    }
    actual = {
        name: (w.latency, w.cost, w.sla, w.queue, w.churn) for name, w in ra.REWARD_VARIANTS.items()
    }
    assert actual == expected
    # The default reward is the reference condition and stays the package default.
    assert ra.REWARD_VARIANTS["full-default-v1"] == RewardWeights()


def test_reward_normalization_is_unchanged() -> None:
    from scalerl.environment.metrics import TickMetrics
    from scalerl.environment.queue import QueueStepResult

    config = SimulatorConfig()
    metrics = TickMetrics(
        utilization=0.5, p95_latency_seconds=0.5, sla_violated=True, infrastructure_cost=0.004
    )
    queue = QueueStepResult(
        arrived_requests=10.0, processed_requests=10.0, queued_requests=1500.0, dropped_requests=0.0
    )
    parts = compute_reward(
        metrics,
        queue,
        applied_replica_change=3,
        config=config,
        weights=RewardWeights(latency=1, cost=1, sla=1, queue=1, churn=1),
    )
    assert parts.latency_penalty == pytest.approx(0.5)
    assert parts.cost_penalty == pytest.approx(0.004 / (10 * 0.1 * 30 / 3600))
    assert parts.sla_penalty == 1.0 and parts.churn_penalty == 1.0
    assert parts.queue_penalty == pytest.approx(1500 / (1500 + 10 * 50 * 30))
    scaled = compute_reward(
        metrics,
        queue,
        applied_replica_change=3,
        config=config,
        weights=ra.REWARD_VARIANTS["full-cost-high-v1"],
    )
    assert scaled.cost_penalty == pytest.approx(2 * parts.cost_penalty)
    assert scaled.churn_penalty == pytest.approx(0.1)


def test_committed_spec_is_the_frozen_plan() -> None:
    spec, candidates, selection = frozen()
    assert spec.experiment_id == SPEC_ID
    assert spec == ra.build_experiment_spec(candidates, selection)
    assert list(spec.reward_variants) == list(ra.REWARD_VARIANTS)
    assert spec.action_semantics == DESIRED_REPLICAS_V1
    assert spec.simulator_config["action"] == {"semantics": DESIRED_REPLICAS_V1}
    assert (spec.action_experiment_spec_id, spec.candidate_set_id, spec.selection_spec_id) == (
        "899dfbb64217",
        "cbe3c1c0719b",
        "418876d6c8e9",
    )
    assert spec.training_workload_ids == ("syn-train-bursty",)
    assert spec.validation_workload_ids == tuple(THRESHOLDS)
    assert spec.test_workload_ids == () and spec.held_out_data_used is False
    assert spec.sla_thresholds == THRESHOLDS
    assert spec.training_seeds == (0, 1, 2, 3, 4) and spec.evaluation_seed == 0
    assert spec.no_joint_feasible_fallback == "full-default-v1"
    assert spec.reward_level_rule["episode_reward_used"] is False
    assert spec.reward_level_rule["order"] == [
        "normalized_cost", "queue_pressure", "churn_rate", "sla_violation_rate"
    ]  # fmt: skip


def test_ppo_is_the_exact_79_selected_configuration() -> None:
    spec, candidates, _ = frozen()
    decision = json.loads(DECISION.read_text())
    assert (
        decision["selected_configurations"]["ppo-desired-replicas-v1"]["selected_candidate_id"]
        == "ppo-c08"
    )
    assert spec.ppo["candidate_id"] == "ppo-c08"
    assert spec.ppo["hyperparameters"] == dict(candidates.ppo.get("ppo-c08").hyperparameters)
    assert spec.ppo["timesteps"] == 204_800 and spec.ppo["training_seeds"] == [0, 1, 2, 3, 4]


def test_dqn_is_the_exact_79_candidate_space_and_budget() -> None:
    spec, candidates, _ = frozen()
    action_spec = ActionSpec.load(ACTION_SPEC)
    assert spec.dqn["candidate_ids"] == [c.candidate_id for c in candidates.dqn.candidates]
    assert len(spec.dqn["candidate_ids"]) == 20  # type: ignore[arg-type]
    assert spec.dqn["timesteps"] == action_spec.timesteps["dqn"] == 200_000
    assert spec.dqn["screening_seed"] == action_spec.screening_training_seed == 0
    assert spec.dqn["search_space_version"] == "dqn-search-v1"
    decision = json.loads(DECISION.read_text())
    assert (
        decision["selected_configurations"]["dqn-desired-replicas-v1"]["selected_candidate_id"]
        is None
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("test_workload_ids", ["syn-test-spike-hard"], "no test workloads"),
        ("validation_workload_ids", ["syn-val-bursty", "syn-test-spike-hard"], "held-out"),
        ("training_workload_ids", ["azure-test-993600"], "held-out"),
        ("training_workload_ids", ["syn-train-spike"], "exactly"),
        ("validation_workload_ids", ["syn-val-bursty"], "exactly"),
        ("training_seeds", [0, 1, 2], "seeds"),
        ("simulator_config", SimulatorConfig().model_dump(mode="json"), "desired-replicas-v1"),
        ("action_semantics", "delta-v1", None),
        ("held_out_data_used", True, None),
        ("no_joint_feasible_fallback", "full-cost-low-v1", None),
    ],
)
def test_spec_rejects_departures(field: str, value: Any, message: str | None) -> None:
    payload = json.loads(SPEC.read_text())
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        ra.ExperimentSpec.model_validate_json(json.dumps(payload))


def test_changed_variants_or_edits_change_identity_and_are_refused(tmp_path: Path) -> None:
    payload = json.loads(SPEC.read_text())
    payload["reward_variants"]["full-sla-high-v1"]["sla"] = 3.0
    with pytest.raises(ValidationError, match="predeclared"):
        ra.ExperimentSpec.model_validate_json(json.dumps(payload))
    payload = json.loads(SPEC.read_text())
    payload["robustness"] = "edited"
    edited = tmp_path / "spec.json"
    edited.write_text(json.dumps(payload))
    assert ra.ExperimentSpec.load(edited).experiment_id != SPEC_ID
    with pytest.raises(ValueError, match="differs"):
        ra.load_frozen(edited, DECISION, ACTION_SPEC, CANDIDATES, SELECTION)


def test_final_config_is_explicitly_desired_replicas() -> None:
    assert SimulatorConfig().action.semantics == "delta-v1"  # package default unchanged
    assert ra.final_config().action.semantics == DESIRED_REPLICAS_V1


def test_cli_has_no_test_or_workload_escape(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (
        ["run", "--reward", "full-default-v1", "--include-test"],
        ["run", "--reward", "full-default-v1", "--workload", "syn-test-spike-hard"],
        ["run", "--reward", "made-up-v1"],
    ):
        with pytest.raises(SystemExit):
            ra.main(argv)
    assert ra.main(["check", "--spec", str(SPEC)]) == 0
    printed = capsys.readouterr().out
    for text in (SPEC_ID, "desired-replicas-v1", "418876d6c8e9", "test workloads      0"):
        assert text in printed


# --- the reward-level decision rule ------------------------------------------------------------


def means(
    sla: float | dict[str, float], cost: float = 0.5, queue: float = 0.05, churn: float = 0.05
) -> dict[str, dict[str, float]]:
    per = sla if isinstance(sla, dict) else dict.fromkeys(THRESHOLDS, sla)
    return {
        w: {
            "sla_violation_rate": per[w],
            "normalized_cost": cost,
            "queue_pressure": queue,
            "churn_rate": churn,
        }
        for w in THRESHOLDS
    }


def test_no_joint_feasible_reward_falls_back_to_full_default() -> None:
    rows = {
        "latency-cost-v1": {"dqn_selected": False, "dqn": None, "ppo": means(0.1, cost=0.3)},
        "full-default-v1": {"dqn_selected": False, "dqn": None, "ppo": means(0.1)},
    }
    decision = ra.decide_reward(rows, THRESHOLDS)
    assert decision["joint_feasible_rewards"] == []
    assert decision["selected_reward"] == "full-default-v1" and decision["fallback_invoked"]
    # PPO alone never selects the reward, however cheap.
    assert decision["per_reward"]["latency-cost-v1"]["joint_feasible"] is False


def test_joint_feasibility_is_per_workload_for_both_families() -> None:
    bursty_miss = {**THRESHOLDS, "syn-val-bursty": 0.25}
    rows = {
        "a": {"dqn_selected": True, "dqn": means(bursty_miss), "ppo": means(0.1)},  # DQN fails one
        "b": {"dqn_selected": True, "dqn": means(0.1), "ppo": means(bursty_miss)},  # PPO fails one
        "c": {"dqn_selected": True, "dqn": means(dict(THRESHOLDS)), "ppo": means(0.1)},  # equality
    }
    decision = ra.decide_reward(rows, THRESHOLDS)
    assert decision["joint_feasible_rewards"] == ["c"]
    assert decision["per_reward"]["a"]["dqn_sla_passes"]["syn-val-bursty"] is False
    assert decision["selected_reward"] == "c" and not decision["fallback_invoked"]


def test_equal_algorithm_cost_then_tie_breaks() -> None:
    rows = {
        # a: DQN 0.9, PPO 0.3 -> 0.6; b: DQN 0.5, PPO 0.8 -> 0.65 -> a wins on cost
        "b": {"dqn_selected": True, "dqn": means(0.1, cost=0.5), "ppo": means(0.1, cost=0.8)},
        "a": {"dqn_selected": True, "dqn": means(0.1, cost=0.9), "ppo": means(0.1, cost=0.3)},
    }
    decision = ra.decide_reward(rows, THRESHOLDS)
    assert decision["selected_reward"] == "a"
    assert decision["per_reward"]["a"]["equal_algorithm"]["normalized_cost"] == pytest.approx(0.6)
    tied = {
        "x": {"dqn_selected": True, "dqn": means(0.1, queue=0.2), "ppo": means(0.1)},
        "y": {"dqn_selected": True, "dqn": means(0.1, queue=0.1), "ppo": means(0.1)},
    }
    assert ra.decide_reward(tied, THRESHOLDS)["selected_reward"] == "y"  # queue breaks the tie
    churn = {
        "x": {"dqn_selected": True, "dqn": means(0.1, churn=0.1), "ppo": means(0.1)},
        "y": {"dqn_selected": True, "dqn": means(0.1, churn=0.2), "ppo": means(0.1)},
    }
    assert ra.decide_reward(churn, THRESHOLDS)["selected_reward"] == "x"
    sla = {
        "x": {"dqn_selected": True, "dqn": means(0.2), "ppo": means(0.1)},
        "y": {"dqn_selected": True, "dqn": means(0.1), "ppo": means(0.1)},
    }
    assert ra.decide_reward(sla, THRESHOLDS)["selected_reward"] == "y"
    identical = {
        r: {"dqn_selected": True, "dqn": means(0.1), "ppo": means(0.1)} for r in ("m", "k")
    }
    assert ra.decide_reward(identical, THRESHOLDS)["selected_reward"] == "k"  # reward ID


def test_decision_is_order_independent_and_ignores_episode_reward() -> None:
    rows = {
        "b": {"dqn_selected": True, "dqn": means(0.1, cost=0.5), "ppo": means(0.1, cost=0.5)},
        "a": {"dqn_selected": True, "dqn": means(0.1, cost=0.6), "ppo": means(0.1, cost=0.6)},
    }
    rows["a"]["dqn"]["syn-val-bursty"]["episode_reward"] = 1e9  # never consulted
    forward = ra.decide_reward(rows, THRESHOLDS)
    backward = ra.decide_reward(dict(reversed(list(rows.items()))), THRESHOLDS)
    assert forward == backward and forward["selected_reward"] == "b"


def test_equal_seed_means_and_summaries(tmp_path: Path) -> None:
    models = [
        evidence("full-default-v1", "ppo-training", "ppo-c08", seed, sla=0.1 * (seed + 1))
        for seed in range(5)
    ]
    got = ra._equal_seed_means(models)
    assert got["syn-val-bursty"]["sla_violation_rate"] == pytest.approx(0.3)
    summary = ra._summaries(models)["syn-val-bursty"]["sla_violation_rate"]
    assert (summary["n"], summary["min"], summary["max"]) == (
        5,
        pytest.approx(0.1),
        pytest.approx(0.5),
    )
    assert summary["median"] == pytest.approx(0.3) and summary["std"] is not None


# --- pathology diagnostics --------------------------------------------------------------------


def test_pathology_flags_are_diagnostic() -> None:
    m = means({**THRESHOLDS, "syn-val-bursty": 0.5}, cost=0.95, queue=0.25, churn=0.4)
    for w in THRESHOLDS:
        m[w]["scaling_actions"] = 1.0
    flags = ra.pathology_flags(m, THRESHOLDS)
    assert set(flags["syn-val-bursty"]) == {
        "full_fleet",
        "thrashing",
        "churn_aversion",
        "persistent_backlog",
    }
    cheap = means(0.9, cost=0.2)
    for w in THRESHOLDS:
        cheap[w]["scaling_actions"] = 20.0
    assert ra.pathology_flags(cheap, THRESHOLDS)["syn-val-ramp-down"] == ["underprovisioning"]
    assert ra.pathology_flags(None, THRESHOLDS) == {}


def test_reward_mismatch_flag() -> None:
    from scalerl.evaluation import model_selection

    selection_spec = SelectionSpec.load(SELECTION)
    screening = [
        evidence("full-default-v1", "dqn-screening", f"dqn-c{i:02d}", 0, sla=s, episode_reward=r)
        for i, (s, r) in enumerate([(0.5, -10.0), (0.1, -20.0), (0.3, -15.0)])
    ]
    result = model_selection.select(selection_spec, "dqn-x", ra.dqn_candidates(screening, "dqn-x"))
    mismatch = ra.reward_mismatch(screening, result)
    assert mismatch["highest_episode_reward_candidate"] == "dqn-c00"
    assert mismatch["highest_episode_reward_is_feasible"] is False and mismatch["flag"] is True
    assert mismatch["spearman_episode_reward_vs_mean_sla"] == pytest.approx(1.0)


# --- evidence handling and a tiny end-to-end run ------------------------------------------------


def evidence(
    variant: str,
    phase: Any,
    candidate: str,
    seed: int,
    *,
    sla: float,
    episode_reward: float = -100.0,
    experiment_id: str = SPEC_ID,
) -> ra.TrainingEvidence:
    from scalerl.evaluation.action_semantics import WorkloadEvidence

    return ra.TrainingEvidence(
        experiment_id=experiment_id,
        reward_variant=variant,
        reward_weights=ra.REWARD_VARIANTS[variant].model_dump(),
        phase=phase,
        algorithm="ppo" if candidate.startswith("ppo") else "dqn",
        candidate_id=candidate,
        candidate_index=int(candidate[-2:]),
        training_seed=seed,
        hyperparameters={},
        training_run_id=f"run-{candidate}-{seed}",
        model_artifact_uri=f"runs:/run-{candidate}-{seed}/model",
        compatibility={"action_semantics_version": DESIRED_REPLICAS_V1},
        workloads=tuple(
            WorkloadEvidence(
                workload_id=w,
                validation_run_id=f"val-{candidate}-{seed}-{w}",
                metrics=dict.fromkeys(ra.METRIC_KEYS, 0.1)
                | {"sla_violation_rate": sla, "episode_reward": episode_reward},
                action=dict.fromkeys(ra.ACTION_KEYS, 1.0),
            )
            for w in THRESHOLDS
        ),
    )


def test_evidence_from_another_reward_or_experiment_is_refused(tmp_path: Path) -> None:
    spec, _, _ = frozen()
    path = ra.evidence_path(tmp_path, "full-default-v1", "ppo-training", "ppo-c08", 0)
    evidence("full-default-v1", "ppo-training", "ppo-c08", 0, sla=0.1).save(path)
    assert ra._load_evidence(path, spec, "full-default-v1", "ppo-c08", 0).training_seed == 0
    with pytest.raises(ValueError, match="different"):
        ra._load_evidence(path, spec, "full-sla-high-v1", "ppo-c08", 0)
    evidence("full-default-v1", "ppo-training", "ppo-c08", 0, sla=0.1, experiment_id="0" * 12).save(
        path
    )
    with pytest.raises(ValueError, match="different"):
        ra._load_evidence(path, spec, "full-default-v1", "ppo-c08", 0)


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_run_reward_trains_fresh_desired_models_and_resumes(
    tmp_path: Path, tracking_uri: str
) -> None:
    spec, candidates, selection = frozen()
    out = tmp_path / "out"
    quiet = lambda _: None  # noqa: E731
    kwargs: dict[str, Any] = {
        "out": out, "tracking_uri": tracking_uri, "timesteps": {"dqn": 64, "ppo": 1024},
        "dqn_candidate_ids": ["dqn-c00", "dqn-c01"], "progress": quiet,
    }  # fmt: skip
    result = ra.run_reward(spec, candidates, selection, "full-cost-high-v1", **kwargs)
    assert result.spec_id == "418876d6c8e9" and len(result.candidate_ids) == 2
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*-seed*.json"))
    ppo_files = [f for f in files if "/ppo-training/" in f]
    assert ppo_files == [f"full-cost-high-v1/ppo-training/ppo-c08-seed{s}.json" for s in range(5)]
    retrained = [f for f in files if "/dqn-retraining/" in f]
    assert len(retrained) == (5 if result.selection_succeeded else 0)  # fallback never retrained
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(ra.EXPERIMENT_NAME)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    for run in runs:
        tags, params = run.data.tags, run.data.params
        assert tags["scalerl.reward_variant"] == "full-cost-high-v1"
        assert tags["scalerl.experiment_id"] == SPEC_ID
        assert tags["scalerl.action_semantics"] == DESIRED_REPLICAS_V1
        assert params["reward.cost"] == "2.0" and params["reward.churn"] == "0.1"
    ppo = ra.TrainingEvidence.model_validate_json((out / ppo_files[0]).read_text())
    assert ppo.hyperparameters == dict(candidates.ppo.get("ppo-c08").hyperparameters)
    assert ppo.compatibility["action_semantics_version"] == DESIRED_REPLICAS_V1
    assert ppo.compatibility["action_count"] == 10
    count = len(runs)
    ra.run_reward(spec, candidates, selection, "full-cost-high-v1", **kwargs)
    assert len(client.search_runs([experiment.experiment_id], max_results=1000)) == count
    ids = ["dqn-c00", "dqn-c01"]
    loaded = ra.load_reward_models(spec, out, "full-cost-high-v1", ids)
    assert [m.training_seed for m in loaded[3]] == [0, 1, 2, 3, 4]
    stray = out / "full-cost-high-v1" / "ppo-training" / "ppo-c08-seed7.json"
    stray.write_text((out / ppo_files[0]).read_text())
    with pytest.raises(ValueError, match="unexpected evidence"):
        ra.load_reward_models(spec, out, "full-cost-high-v1", ids)
