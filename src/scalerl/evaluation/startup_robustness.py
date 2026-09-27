"""Seeded stochastic startup-delay robustness extension ``startup-robustness-v1`` (#81).

A separate, versioned extension; the four #65 ``robustness-v1`` scenarios are
unchanged and ``robustness-v1 / nominal`` stays the reference. Two predeclared
scenarios, both using the ``tri-point-multiplicative-v1`` startup model (per new
replica: 0.5x / 1.0x / 1.5x nominal startup with probabilities 0.25 / 0.50 /
0.25; see :mod:`scalerl.environment.startup`):

====================================  ==========  ================  ==========================
scenario                              jitter      telemetry delay   startup model
====================================  ==========  ================  ==========================
``startup-delay-jitter``              0           0 ticks           tri-point-multiplicative-v1
``combined-startup-robustness``       0.10        1 tick            tri-point-multiplicative-v1
====================================  ==========  ================  ==========================

Seeds: ``startup-delay-jitter`` uses startup seeds 0-4 (dynamics seed 0, no
jitter drawn); ``combined-startup-robustness`` uses matched indices
``dynamics_seed = startup_delay_seed = s`` for ``s`` in 0-4. The capacity and
startup RNGs are separate streams.

**Fairness.** For the same workload, scenario, startup seed and request
sequence, a controller gets exactly the same startup realization, whatever the
evaluation order (local RNGs, restarted on reset, one draw per new replica in
request order). Controllers that request different replicas at different times
consume the stream differently, so their realized delay lists legitimately
diverge; the guarantee is not "identical delay lists across controllers".

**Information.** Controllers know the nominal startup delay and see pending
replicas by their nominal readiness; a sampled realization is physical
evaluation truth only (step ``info``, never ``decision_info`` or the
observation).

    python -m scalerl.evaluation.startup_robustness check
    python -m scalerl.evaluation.startup_robustness run
    python -m scalerl.evaluation.startup_robustness freeze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers import Controller
from scalerl.environment import (
    DESIRED_REPLICAS_V1,
    TRI_POINT_EXPECTED_MULTIPLIER,
    TRI_POINT_MULTIPLICATIVE_V1,
    TRI_POINT_MULTIPLIERS,
    TRI_POINT_PROBABILITIES,
    AutoscalingEnv,
    DynamicsConfig,
    SimulatorConfig,
)
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation.action_semantics import ActionContractDecision
from scalerl.evaluation.metrics import (
    ActionMagnitudeMetrics,
    EpisodeMetrics,
    evaluate_controller_episode,
    summarize_action_magnitude,
)
from scalerl.evaluation.multiseed import _append_row, _repair_torn_tail, describe, read_rows
from scalerl.evaluation.predictive_baseline import final_contract_config, make_controller
from scalerl.evaluation.robustness import (
    NOMINAL,
    ROBUSTNESS_SCENARIO_VERSION,
    ROBUSTNESS_SCENARIOS,
    DynamicsSummary,
    summarize_dynamics,
)
from scalerl.mlops import RunSpec
from scalerl.workloads import WorkloadTrace

STARTUP_ROBUSTNESS_VERSION: Final = "startup-robustness-v1"
EXPERIMENT_VERSION: Final = "startup-robustness-experiment-v1"
FREEZE_VERSION: Final = "startup-robustness-freeze-v1"
EXPERIMENT_NAME: Final = "scalerl-startup-robustness"
ACTION_DECISION_VERSION: Final = "action-contract-v2"
STARTUP_SEEDS: Final = (0, 1, 2, 3, 4)
EVALUATION_SEED: Final = 0
VALIDATION_WORKLOADS: Final = ("syn-val-steady-high", "syn-val-ramp-down", "syn-val-bursty")
RULE_CONTROLLERS: Final = ("threshold-v1", "predictive-v1", "predictive-seasonal-v1")
PPO_FAMILY: Final = "ppo-desired-replicas-v1"
PPO_CANDIDATE: Final = "ppo-c08"
METRIC_KEYS: Final = tuple(EpisodeMetrics.__dataclass_fields__)
ACTION_KEYS: Final = tuple(ActionMagnitudeMetrics.__dataclass_fields__)

DEFAULT_SPEC = Path("benchmarks/v1/startup-robustness-v1.json")
DEFAULT_ACTION_DECISION = Path("benchmarks/v1/action-contract-v2.json")
DEFAULT_FREEZE = Path("benchmarks/v1/startup-robustness-freeze-v1.json")
DEFAULT_OUTPUT = Path("outputs/startup-robustness-v1")


# --- scenarios -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupScenario:
    """One named ``startup-robustness-v1`` condition."""

    name: str
    capacity_jitter_fraction: float
    telemetry_delay_ticks: int
    startup_delay_model: Literal["tri-point-multiplicative-v1"] = TRI_POINT_MULTIPLICATIVE_V1
    version: str = STARTUP_ROBUSTNESS_VERSION

    def dynamics(self, *, startup_delay_seed: int, dynamics_seed: int) -> DynamicsConfig:
        return DynamicsConfig(
            capacity_jitter_fraction=self.capacity_jitter_fraction,
            telemetry_delay_ticks=self.telemetry_delay_ticks,
            dynamics_seed=dynamics_seed,
            startup_delay_model=self.startup_delay_model,
            startup_delay_seed=startup_delay_seed,
        )

    def seed_pairs(self) -> tuple[tuple[int, int], ...]:
        """``(dynamics_seed, startup_delay_seed)`` pairs of the predeclared seed plan."""
        if self.capacity_jitter_fraction == 0:
            return tuple((0, seed) for seed in STARTUP_SEEDS)  # no jitter: dynamics seed unused
        return tuple((seed, seed) for seed in STARTUP_SEEDS)  # matched indices, separate streams


STARTUP_DELAY_JITTER = StartupScenario("startup-delay-jitter", 0.0, 0)
COMBINED_STARTUP_ROBUSTNESS = StartupScenario("combined-startup-robustness", 0.10, 1)
STARTUP_ROBUSTNESS_SCENARIOS: dict[str, StartupScenario] = {
    scenario.name: scenario for scenario in (STARTUP_DELAY_JITTER, COMBINED_STARTUP_ROBUSTNESS)
}


def apply_startup_scenario(
    config: SimulatorConfig,
    scenario: StartupScenario,
    *,
    startup_delay_seed: int,
    dynamics_seed: int,
) -> SimulatorConfig:
    """A copy of ``config`` whose dynamics are ``scenario`` with the given seeds."""
    dynamics = scenario.dynamics(startup_delay_seed=startup_delay_seed, dynamics_seed=dynamics_seed)
    return config.model_copy(update={"dynamics": dynamics})


# --- startup diagnostics -------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupDiagnostics:
    """Realized startup delays of one episode (diagnostics, never objectives).

    Under ``fixed-v1`` every started replica takes exactly the nominal delay
    (multiplier 1.0). Delay/multiplier/spread statistics are ``None`` (not
    logged) when no replica was started, and batch spreads are ``None`` when
    no scale-up requested more than one replica.
    """

    replicas_requested: int
    mean_realized_delay_seconds: float | None
    min_realized_delay_seconds: float | None
    max_realized_delay_seconds: float | None
    mean_multiplier: float | None
    min_multiplier: float | None
    max_multiplier: float | None
    multi_replica_batches: int
    mean_batch_readiness_spread_seconds: float | None
    max_batch_readiness_spread_seconds: float | None

    def as_metrics(self) -> dict[str, float]:
        return {
            f"startup.{name}": float(value)
            for name, value in vars(self).items()
            if value is not None
        }


def startup_batches(
    infos: Sequence[Mapping[str, Any]], config: SimulatorConfig
) -> list[list[tuple[float, float]]]:
    """Per scale-up tick, the ``(realized_delay, multiplier)`` of each replica it started."""
    nominal = config.replicas.startup_delay_seconds
    batches = []
    for info in infos:
        if "startup_realized_delays_seconds" in info:
            delays = [float(d) for d in info["startup_realized_delays_seconds"]]
            multipliers = [float(m) for m in info["startup_multipliers"]]
            if delays:
                batches.append(list(zip(delays, multipliers, strict=True)))
        elif int(info["applied_replica_change"]) > 0:  # fixed-v1: nominal, exactly
            count = int(info["applied_replica_change"])
            batches.append([(nominal, 1.0)] * count)
    return batches


def summarize_startup(
    infos: Sequence[Mapping[str, Any]], config: SimulatorConfig
) -> StartupDiagnostics:
    batches = startup_batches(infos, config)
    started = [pair for batch in batches for pair in batch]
    delays = [delay for delay, _ in started]
    multipliers = [multiplier for _, multiplier in started]
    spreads = [
        max(d for d, _ in batch) - min(d for d, _ in batch) for batch in batches if len(batch) > 1
    ]
    return StartupDiagnostics(
        replicas_requested=len(started),
        mean_realized_delay_seconds=statistics.fmean(delays) if delays else None,
        min_realized_delay_seconds=min(delays) if delays else None,
        max_realized_delay_seconds=max(delays) if delays else None,
        mean_multiplier=statistics.fmean(multipliers) if multipliers else None,
        min_multiplier=min(multipliers) if multipliers else None,
        max_multiplier=max(multipliers) if multipliers else None,
        multi_replica_batches=len(spreads),
        mean_batch_readiness_spread_seconds=statistics.fmean(spreads) if spreads else None,
        max_batch_readiness_spread_seconds=max(spreads) if spreads else None,
    )


@dataclass(frozen=True)
class StartupRobustnessResult:
    """One controller on one workload under one condition; seed meanings kept distinct."""

    scenario_name: str
    scenario_version: str
    dynamics_seed: int
    startup_delay_seed: int | None  # None: the fixed-startup nominal reference
    evaluation_seed: int
    metrics: EpisodeMetrics
    action: ActionMagnitudeMetrics
    startup: StartupDiagnostics
    dynamics: DynamicsSummary
    infos: tuple[Mapping[str, Any], ...]  # raw physical step infos


def evaluate_startup_robustness(
    controller: Controller,
    trace: WorkloadTrace,
    *,
    scenario: StartupScenario,
    startup_delay_seed: int,
    dynamics_seed: int,
    evaluation_seed: int = EVALUATION_SEED,
    config: SimulatorConfig | None = None,
    reward_weights: RewardWeights | None = None,
) -> StartupRobustnessResult:
    """Evaluate a fixed controller for one episode under a ``startup-robustness-v1`` scenario."""
    scenario_config = apply_startup_scenario(
        config or SimulatorConfig(),
        scenario,
        startup_delay_seed=startup_delay_seed,
        dynamics_seed=dynamics_seed,
    )
    return _evaluate(
        controller,
        AutoscalingEnv(scenario_config, trace, reward_weights),
        scenario_name=scenario.name,
        scenario_version=scenario.version,
        dynamics_seed=dynamics_seed,
        startup_delay_seed=startup_delay_seed,
        evaluation_seed=evaluation_seed,
    )


def _evaluate(
    controller: Controller,
    env: AutoscalingEnv,
    *,
    scenario_name: str,
    scenario_version: str,
    dynamics_seed: int,
    startup_delay_seed: int | None,
    evaluation_seed: int,
) -> StartupRobustnessResult:
    evaluation = evaluate_controller_episode(env, controller, seed=evaluation_seed)
    return StartupRobustnessResult(
        scenario_name=scenario_name,
        scenario_version=scenario_version,
        dynamics_seed=dynamics_seed,
        startup_delay_seed=startup_delay_seed,
        evaluation_seed=evaluation_seed,
        metrics=evaluation.metrics,
        action=summarize_action_magnitude(evaluation.infos),
        startup=summarize_startup(evaluation.infos, env.config),
        dynamics=summarize_dynamics(evaluation.infos),
        infos=evaluation.infos,
    )


# --- frozen experiment specification ------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def verify_action_decision(path: Path) -> ActionContractDecision:
    decision = ActionContractDecision.model_validate_json(path.read_text())
    if decision.decision_version != ACTION_DECISION_VERSION:
        raise ValueError(
            f"#79 decision is {decision.decision_version}, not {ACTION_DECISION_VERSION}"
        )
    if decision.final_action_semantics != DESIRED_REPLICAS_V1:
        raise ValueError("#81 needs the frozen desired-replicas-v1 action contract")
    if decision.held_out_data_used or decision.reward_changed:
        raise ValueError("the #79 decision must not have used held-out data or changed reward")
    return decision


def ppo_models(decision: ActionContractDecision) -> tuple[dict[str, JsonValue], ...]:
    """The frozen #79 PPO desired-replicas-v1 models (ppo-c08, seeds 0-4), by seed."""
    models = sorted(
        (dict(m) for m in decision.retrained_models if m["family"] == PPO_FAMILY),
        key=lambda m: int(str(m["training_seed"])),
    )
    if [m["training_seed"] for m in models] != list(STARTUP_SEEDS) or {
        m["selected_candidate_id"] for m in models
    } != {PPO_CANDIDATE}:
        raise ValueError(f"expected {PPO_FAMILY} {PPO_CANDIDATE} seeds 0-4 in the #79 decision")
    return tuple(models)


class ExperimentSpec(_Strict):
    """Everything #81 holds fixed; ``experiment_id`` hashes it."""

    experiment_version: Literal["startup-robustness-experiment-v1"] = EXPERIMENT_VERSION
    startup_robustness_version: Literal["startup-robustness-v1"] = STARTUP_ROBUSTNESS_VERSION
    benchmark_version: str
    startup_model: dict[str, JsonValue]
    fixed_model: str
    scenarios: dict[str, dict[str, JsonValue]]
    reference: dict[str, JsonValue]
    robustness_v1_scenarios: tuple[str, ...]
    startup_seeds: tuple[int, ...]
    combined_dynamics_seeds: tuple[int, ...]
    evaluation_seed: Literal[0] = EVALUATION_SEED
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    simulator_config: dict[str, JsonValue]
    validation_workload_ids: tuple[str, ...]
    test_workload_ids: tuple[str, ...] = ()
    controller_variants: tuple[str, ...]
    ppo_models: tuple[dict[str, JsonValue], ...]
    dqn: str
    metrics: dict[str, JsonValue]
    compatibility: dict[str, JsonValue]
    fairness: str
    information_contract: str
    reward_weights: dict[str, float]
    reward_changed: Literal[False] = False
    held_out_data_used: Literal[False] = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        manifest = load_benchmark_manifest()
        if self.benchmark_version != manifest.version:
            raise ValueError(f"spec benchmark {self.benchmark_version!r} is not installed")
        if self.test_workload_ids:
            raise ValueError("#81 uses no test workloads")
        for workload_id in self.validation_workload_ids:
            try:
                split = manifest.get(workload_id).split
            except KeyError as error:
                raise ValueError(str(error)) from None
            if split != "validation":
                kind = "held-out test" if split == "test" else split
                raise ValueError(f"{workload_id!r} is a {kind} workload, not validation")
        if self.validation_workload_ids != VALIDATION_WORKLOADS:
            raise ValueError(f"validation workloads must be exactly {VALIDATION_WORKLOADS}")
        if self.startup_seeds != STARTUP_SEEDS or self.combined_dynamics_seeds != STARTUP_SEEDS:
            raise ValueError(f"startup and combined dynamics seeds must be {STARTUP_SEEDS}")
        if self.startup_model != _startup_model():
            raise ValueError("the startup model is the predeclared tri-point-multiplicative-v1")
        if self.scenarios != _scenarios() or self.robustness_v1_scenarios != tuple(
            ROBUSTNESS_SCENARIOS
        ):
            raise ValueError("scenario definitions differ from the predeclared ones")
        if self.simulator_config != final_contract_config().model_dump(mode="json"):
            raise ValueError("the base config must be the canonical desired-replicas-v1 config")
        if self.reward_weights != RewardWeights().model_dump(mode="json"):
            raise ValueError("the reward must be the unchanged default")
        return self

    @property
    def experiment_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        return cls.model_validate_json(Path(path).read_text())


def _startup_model() -> dict[str, JsonValue]:
    return {
        "id": TRI_POINT_MULTIPLICATIVE_V1,
        "multipliers": list(TRI_POINT_MULTIPLIERS),
        "probabilities": list(TRI_POINT_PROBABILITIES),
        "expected_multiplier": TRI_POINT_EXPECTED_MULTIPLIER,
        "support": [min(TRI_POINT_MULTIPLIERS), max(TRI_POINT_MULTIPLIERS)],
        "per_replica": "one independent draw per newly requested replica, in request order",
        "rounding": (
            "no pre-rounding of realized seconds; activation only through the lifecycle's "
            "per-tick advance"
        ),
        "canonical_example": "60 s nominal -> 30 / 60 / 90 s: ready after 1 / 2 / 3 ticks of 30 s",
        "rng": "dedicated per-environment RNG seeded by startup_delay_seed; reset restarts it",
        "calibration": "robustness stress model, not a provider-calibrated distribution",
    }


def _scenarios() -> dict[str, dict[str, JsonValue]]:
    return {
        name: {
            "version": s.version,
            "capacity_jitter_fraction": s.capacity_jitter_fraction,
            "telemetry_delay_ticks": s.telemetry_delay_ticks,
            "startup_delay_model": s.startup_delay_model,
            "seed_pairs_dynamics_startup": [list(pair) for pair in s.seed_pairs()],
        }
        for name, s in STARTUP_ROBUSTNESS_SCENARIOS.items()
    }


def build_experiment_spec(decision: ActionContractDecision) -> ExperimentSpec:
    return ExperimentSpec(
        benchmark_version=load_benchmark_manifest().version,
        startup_model=_startup_model(),
        fixed_model="fixed-v1: realized delay = nominal delay exactly, no RNG draw (default)",
        scenarios=_scenarios(),
        reference={
            "robustness_version": ROBUSTNESS_SCENARIO_VERSION,
            "scenario": NOMINAL.name,
            "dynamics_seed": 0,
            "startup_delay_model": "fixed-v1",
        },
        robustness_v1_scenarios=tuple(ROBUSTNESS_SCENARIOS),
        startup_seeds=STARTUP_SEEDS,
        combined_dynamics_seeds=STARTUP_SEEDS,
        simulator_config=final_contract_config().model_dump(mode="json"),
        validation_workload_ids=VALIDATION_WORKLOADS,
        controller_variants=(
            *RULE_CONTROLLERS,
            *(f"{PPO_FAMILY}-seed{s}" for s in STARTUP_SEEDS),
        ),
        ppo_models=ppo_models(decision),
        dqn=(
            "no canonical desired-replicas-v1 DQN exists (#79: 0/20 feasible); the diagnostic "
            "fallback is not used and no DQN is run"
        ),
        metrics={
            "system": list(METRIC_KEYS),
            "action": list(ACTION_KEYS),
            "startup": list(StartupDiagnostics.__dataclass_fields__),
            "dynamics": ["mean/min/max capacity multiplier"],
            "reporting": (
                "descriptive per controller/workload/scenario over startup seeds; PPO training "
                "seeds summarized separately; no significance tests, no ranking"
            ),
        },
        compatibility={
            "compatibility_defining": "startup_delay_model",
            "evaluation_metadata": "startup_delay_seed",
            "robustness_perturbable": ["telemetry_delay_ticks", "startup_delay_model"],
            "ppo_loading": "strict for nominal; robustness-only path for startup scenarios",
        },
        fairness=(
            "same workload, scenario, startup seed and request sequence -> identical startup "
            "realization, independent of evaluation order; divergent request histories "
            "legitimately consume the stream differently"
        ),
        information_contract=(
            "controllers know the nominal startup delay and pending replicas by nominal "
            "readiness; sampled realizations are physical-only info, never decision_info or "
            "the observation"
        ),
        reward_weights=RewardWeights().model_dump(mode="json"),
    )


def load_frozen_spec(spec_path: Path, decision_path: Path) -> ExperimentSpec:
    decision = verify_action_decision(decision_path)
    spec = ExperimentSpec.load(spec_path)
    if spec != build_experiment_spec(decision):
        raise ValueError("the committed #81 spec differs from this code's frozen plan")
    return spec


# --- cases ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    case_id: str
    variant: str
    workload_id: str
    scenario_name: str
    scenario_version: str
    dynamics_seed: int
    startup_delay_seed: int | None
    training_run_id: str | None = None
    training_seed: int | None = None
    input_id: str | None = None


def cases(spec: ExperimentSpec) -> list[Case]:
    ppo = {f"{PPO_FAMILY}-seed{m['training_seed']}": m for m in spec.ppo_models}
    result = []
    for workload_id in spec.validation_workload_ids:
        for variant in spec.controller_variants:
            model = ppo.get(variant)
            conditions: list[tuple[str, str, int, int | None]] = [
                (NOMINAL.name, ROBUSTNESS_SCENARIO_VERSION, 0, None)
            ]
            for scenario in STARTUP_ROBUSTNESS_SCENARIOS.values():
                conditions += [
                    (scenario.name, scenario.version, dyn, startup)
                    for dyn, startup in scenario.seed_pairs()
                ]
            for name, version, dyn, startup in conditions:
                startup_label = "fixed" if startup is None else f"startup{startup}"
                result.append(
                    Case(
                        case_id=(
                            f"{variant}|{workload_id}|{name}|dyn{dyn}|{startup_label}"
                            f"|eval{EVALUATION_SEED}"
                        ),
                        variant=variant,
                        workload_id=workload_id,
                        scenario_name=name,
                        scenario_version=version,
                        dynamics_seed=dyn,
                        startup_delay_seed=startup,
                        training_run_id=str(model["training_run_id"]) if model else None,
                        training_seed=int(str(model["training_seed"])) if model else None,
                    )
                )
    return result


def case_config(case: Case) -> SimulatorConfig:
    base = final_contract_config()
    if case.startup_delay_seed is None:  # robustness-v1 nominal reference, fixed startup
        return base.model_copy(update={"dynamics": NOMINAL.dynamics(case.dynamics_seed)})
    return apply_startup_scenario(
        base,
        STARTUP_ROBUSTNESS_SCENARIOS[case.scenario_name],
        startup_delay_seed=case.startup_delay_seed,
        dynamics_seed=case.dynamics_seed,
    )


def input_identity(case: Case, trace: WorkloadTrace, spec: ExperimentSpec) -> str:
    payload = {
        "experiment_id": spec.experiment_id,
        "case_id": case.case_id,
        "rates": list(trace.request_rates),
        "control_interval_seconds": trace.control_interval_seconds,
        "training_run_id": case.training_run_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# --- running ----------------------------------------------------------------------------------


REALIZATIONS_ARTIFACT: Final = "startup-realizations.json"


def _controller_name(variant: str) -> str:
    if variant.startswith("ppo"):
        return "ppo"
    return {"threshold-v1": "threshold", "predictive-v1": "predictive"}.get(
        variant, "predictive-seasonal"
    )


def _build_controller(case: Case, env: AutoscalingEnv, locate: Callable[[str], Path]) -> Controller:
    if case.training_run_id is None:
        return make_controller(case.variant, final_contract_config(), None)
    from scalerl.rl import load_sb3_controller

    return load_sb3_controller(
        locate(case.training_run_id),
        env,
        robustness_evaluation=case.startup_delay_seed is not None,
    )


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


def _tags(spec: ExperimentSpec, case: Case, perturbed: Sequence[str]) -> dict[str, str]:
    config = case_config(case)
    tags = {
        "scalerl.experiment_version": spec.experiment_version,
        "scalerl.experiment_id": spec.experiment_id,
        "scalerl.evaluation_case_id": case.case_id,
        "scalerl.input_id": str(case.input_id),
        "scalerl.controller_variant_id": case.variant,
        "scalerl.action_decision_version": spec.action_decision_version,
        "scalerl.startup_robustness_version": spec.startup_robustness_version,
        "scalerl.startup_scenario": case.scenario_name,
        "scalerl.startup_delay_model": config.dynamics.startup_delay_model,
        "scalerl.startup_delay_seed": "none"
        if case.startup_delay_seed is None
        else str(case.startup_delay_seed),
        "scalerl.dynamics_seed": str(case.dynamics_seed),
        "scalerl.evaluation_seed": str(EVALUATION_SEED),
    }
    if case.training_run_id is not None:
        tags |= {
            "scalerl.model_source_run_id": case.training_run_id,
            "scalerl.model_artifact_uri": f"runs:/{case.training_run_id}/model",
            "scalerl.training_seed": str(case.training_seed),
        }
    if perturbed:
        tags["scalerl.robustness.perturbed_compatibility"] = ",".join(perturbed)
    return tags


def _row(
    spec: ExperimentSpec,
    case: Case,
    run_id: str,
    metrics: Mapping[str, float],
    realizations: Sequence[Sequence[Sequence[float]]],
    perturbed: str,
) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "case_id": case.case_id,
        "input_id": case.input_id,
        "controller_variant": case.variant,
        "controller": _controller_name(case.variant),
        "training_seed": case.training_seed,
        "training_run_id": case.training_run_id,
        "workload_id": case.workload_id,
        "workload_split": "validation",
        "scenario": case.scenario_name,
        "scenario_version": case.scenario_version,
        "dynamics_seed": case.dynamics_seed,
        "startup_delay_seed": case.startup_delay_seed,
        "evaluation_seed": EVALUATION_SEED,
        "action_semantics": spec.action_semantics,
        "perturbed_compatibility": perturbed,
        "mlflow_run_id": run_id,
        "startup_realizations": [list(map(list, batch)) for batch in realizations],
        **{key: float(value) for key, value in sorted(metrics.items())},
    }


def _run_spec(case: Case, entry: Any, hyperparameters: Mapping[str, JsonValue]) -> RunSpec:
    return RunSpec(
        run_kind="evaluate",
        controller=_controller_name(case.variant),
        workload_id=entry.id,
        workload_split=entry.split,
        simulator_config=case_config(case),
        simulator_config_source="predeclared",
        evaluation_seeds=(EVALUATION_SEED,),
        reward_weights=RewardWeights(),
        hyperparameters=dict(hyperparameters),
        robustness_scenario=case.scenario_name,
        robustness_version=case.scenario_version,
    )


def _recover(
    spec: ExperimentSpec, case: Case, tracking_uri: str | None, experiment_name: str
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
            f"tags.`scalerl.input_id` = '{case.input_id}' and "
            "attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    with tempfile.TemporaryDirectory() as directory:
        path = client.download_artifacts(
            run.info.run_id, f"startup/{REALIZATIONS_ARTIFACT}", directory
        )
        realizations = json.loads(Path(path).read_text())["batches"]
    perturbed = run.data.tags.get("scalerl.robustness.perturbed_compatibility", "")
    return _row(spec, case, run.info.run_id, run.data.metrics, realizations, perturbed)


def run_experiment(
    spec: ExperimentSpec,
    *,
    out: Path,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    progress: Callable[[str], None] = print,
    locate: Callable[[str], Path] | None = None,
    only_variants: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every predeclared case once (resumable, input-identified)."""
    from scalerl.mlops import start_tracked_run
    from scalerl.mlops.tracking import TrackedRun

    out.mkdir(parents=True, exist_ok=True)
    manifest = load_benchmark_manifest()
    entries = {w: manifest.get(w) for w in spec.validation_workload_ids}
    traces = build_workloads(list(entries.values()))
    all_cases = [c for c in cases(spec) if only_variants is None or c.variant in only_variants]
    all_cases = [
        Case(**{**vars(c), "input_id": input_identity(c, traces[c.workload_id], spec)})
        for c in all_cases
    ]
    by_id = {c.case_id: c for c in all_cases}
    locate = locate or _mlflow_locator(out / "models", tracking_uri)
    raw_path = out / "raw-results.jsonl"
    _repair_torn_tail(raw_path)
    done: set[str] = set()
    for row in read_rows(raw_path):
        if row["experiment_id"] != spec.experiment_id or row["case_id"] not in by_id:
            continue
        if row.get("input_id") != by_id[row["case_id"]].input_id:
            raise ValueError(
                f"{raw_path}: {row['case_id']} was computed from different inputs; "
                "use a new output directory"
            )
        done.add(row["case_id"])
    for case in all_cases:
        if case.case_id in done:
            continue
        recovered = _recover(spec, case, tracking_uri, experiment_name)
        if recovered is not None:
            _append_row(raw_path, recovered)
            done.add(case.case_id)
            continue
        config = case_config(case)
        if config.action.semantics != DESIRED_REPLICAS_V1:
            raise ValueError("#81 cases must run under desired-replicas-v1")
        env = AutoscalingEnv(config, traces[case.workload_id])
        controller = _build_controller(case, env, locate)
        perturbed = sorted(getattr(controller, "perturbed_compatibility", ()))
        hyperparameters: dict[str, JsonValue] = {"controller_variant": case.variant}
        run: TrackedRun
        with start_tracked_run(
            _run_spec(case, entries[case.workload_id], hyperparameters),
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"startup-{case.case_id.replace('|', '-')}",
        ) as run:
            for key, value in _tags(spec, case, perturbed).items():
                run.set_tag(key, value)
            result = _evaluate(
                controller,
                env,
                scenario_name=case.scenario_name,
                scenario_version=case.scenario_version,
                dynamics_seed=case.dynamics_seed,
                startup_delay_seed=case.startup_delay_seed,
                evaluation_seed=EVALUATION_SEED,
            )
            realizations = startup_batches(result.infos, config)
            metrics = {
                **result.metrics.as_metrics(),
                **result.action.as_metrics(),
                **result.dynamics.as_metrics(),
                **result.startup.as_metrics(),
            }
            run.log_metrics(metrics)
            run.log_artifact_dict(
                REALIZATIONS_ARTIFACT,
                {"case_id": case.case_id, "batches": [list(map(list, b)) for b in realizations]},
                artifact_path="startup",
            )
        _append_row(
            raw_path,
            _row(spec, case, run.run_id, metrics, realizations, ",".join(perturbed)),
        )
        done.add(case.case_id)
        progress(f"[{len(done)}/{len(all_cases)}] {case.case_id}")
    order = {c.case_id: i for i, c in enumerate(all_cases)}
    rows = [r for r in read_rows(raw_path) if r["experiment_id"] == spec.experiment_id]
    rows = sorted((r for r in rows if r["case_id"] in order), key=lambda r: order[r["case_id"]])
    if sorted(r["case_id"] for r in rows) != sorted(order):
        raise ValueError("evaluation rows do not match the predeclared cases")
    return rows


# --- reporting and the freeze artifact --------------------------------------------------------


REPORT_METRICS: Final = (
    "sla_violation_rate",
    "normalized_cost",
    "infrastructure_cost",
    "mean_p95_latency_seconds",
    "max_p95_latency_seconds",
    "mean_queue_depth",
    "max_queue_depth",
    "queue_pressure",
    "scaling_actions",
    "churn_rate",
    "episode_reward",
    "action.total_absolute_replica_change",
    "action.max_absolute_replica_change_in_one_tick",
    "action.max_pending_replicas",
    "startup.replicas_requested",
    "startup.mean_realized_delay_seconds",
    "startup.min_realized_delay_seconds",
    "startup.max_realized_delay_seconds",
    "startup.mean_multiplier",
    "startup.mean_batch_readiness_spread_seconds",
    "startup.max_batch_readiness_spread_seconds",
)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    summary = describe(values)
    return {
        "n": summary.n,
        "mean": summary.mean,
        "median": summary.median,
        "std": summary.std,
        "min": summary.minimum,
        "max": summary.maximum,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive statistics, never pooled across workloads or scenarios.

    ``level=variant``: one controller variant over its startup (and matched
    dynamics) seeds. ``level=ppo-training-seeds``: the five PPO seeds, each
    represented by its mean over startup seeds, so startup-realization
    variation and training-seed variation stay distinct.
    """
    summary = []
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["controller_variant"], row["workload_id"], row["scenario"]), []
        ).append(row)
    for (variant, workload, scenario), members in groups.items():
        for metric in REPORT_METRICS:
            values = [float(m[metric]) for m in members if metric in m]
            if values:
                summary.append(
                    {
                        "level": "variant",
                        "replicate_unit": "startup_seed" if len(members) > 1 else "single_run",
                        "controller_variant": variant,
                        "workload_id": workload,
                        "scenario": scenario,
                        "metric": metric,
                        **_stats(values),
                    }
                )
    ppo: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if row["controller"] == "ppo":
            ppo.setdefault((row["workload_id"], row["scenario"]), {}).setdefault(
                row["controller_variant"], []
            ).append(row)
    for (workload, scenario), by_seed in ppo.items():
        for metric in REPORT_METRICS:
            means = [
                statistics.fmean(float(m[metric]) for m in members)
                for members in by_seed.values()
                if all(metric in m for m in members)
            ]
            if len(means) == len(by_seed):
                summary.append(
                    {
                        "level": "ppo-training-seeds",
                        "replicate_unit": "training_seed (each averaged over startup seeds)",
                        "controller_variant": PPO_FAMILY,
                        "workload_id": workload,
                        "scenario": scenario,
                        "metric": metric,
                        **_stats(means),
                    }
                )
    return summary


class StartupRobustnessFreeze(_Strict):
    """What #72/#46 consume: the frozen startup-robustness-v1 definition and its evidence.

    ``validation_evidence`` is compact: rule controllers per variant (over startup
    seeds) and PPO at its training-seed level (each seed averaged over startup
    seeds); per-seed PPO rows are in the local outputs and MLflow runs.
    """

    freeze_version: Literal["startup-robustness-freeze-v1"] = FREEZE_VERSION
    startup_robustness_version: Literal["startup-robustness-v1"] = STARTUP_ROBUSTNESS_VERSION
    experiment_spec_id: str
    startup_model: dict[str, JsonValue]
    scenarios: dict[str, dict[str, JsonValue]]
    reference: dict[str, JsonValue]
    startup_seeds: tuple[int, ...]
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    validation_workload_ids: tuple[str, ...]
    controller_variants: tuple[str, ...]
    dqn: str
    mlflow_experiment: str
    mlflow_run_ids: dict[str, str]
    validation_evidence: dict[str, JsonValue]
    declares_overall_winner: Literal[False] = False
    held_out_data_used: Literal[False] = False
    reward_changed: Literal[False] = False

    @property
    def freeze_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


EVIDENCE_METRICS: Final = (
    "sla_violation_rate",
    "normalized_cost",
    "queue_pressure",
    "churn_rate",
    "action.total_absolute_replica_change",
    "startup.mean_realized_delay_seconds",
)


def build_freeze(
    spec: ExperimentSpec, rows: Sequence[Mapping[str, Any]]
) -> StartupRobustnessFreeze:
    evidence: dict[str, JsonValue] = {}
    for item in summarize(rows):
        if item["metric"] not in EVIDENCE_METRICS:
            continue
        if item["level"] == "variant" and str(item["controller_variant"]).startswith("ppo"):
            continue  # PPO enters as its training-seed level; per-seed rows stay in outputs/
        key = (
            f"{item['level']}|{item['controller_variant']}|{item['workload_id']}|{item['scenario']}"
        )
        entry = evidence.setdefault(key, {"n": item["n"]})
        assert isinstance(entry, dict)
        entry[item["metric"]] = {k: item[k] for k in ("mean", "median", "std", "min", "max")}
    return StartupRobustnessFreeze(
        experiment_spec_id=spec.experiment_id,
        startup_model=spec.startup_model,
        scenarios=spec.scenarios,
        reference=spec.reference,
        startup_seeds=spec.startup_seeds,
        validation_workload_ids=spec.validation_workload_ids,
        controller_variants=spec.controller_variants,
        dqn=spec.dqn,
        mlflow_experiment=EXPERIMENT_NAME,
        mlflow_run_ids={r["case_id"]: r["mlflow_run_id"] for r in rows},
        validation_evidence=json.loads(json.dumps(evidence, allow_nan=False)),
    )


def write_reports(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat = [{k: v for k, v in r.items() if k != "startup_realizations"} for r in rows]
    _write_csv(out / "raw-results.csv", flat)
    summary = summarize(rows)
    _write_text(out / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(out / "summary.csv", summary)
    _write_text(
        out / "startup-realizations.jsonl",
        "".join(
            json.dumps({"case_id": r["case_id"], "batches": r["startup_realizations"]}) + "\n"
            for r in rows
        ),
    )


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --- command line -----------------------------------------------------------------------------


def describe_plan(spec: ExperimentSpec) -> str:
    return "\n".join(
        [
            f"experiment                 {spec.experiment_version} {spec.experiment_id}",
            f"action semantics           {spec.action_semantics}",
            f"action decision            {spec.action_decision_version}",
            f"startup robustness version {spec.startup_robustness_version}",
            f"startup model              {spec.startup_model['id']}",
            f"multipliers                {spec.startup_model['multipliers']}",
            f"probabilities              {spec.startup_model['probabilities']}",
            f"robustness-v1 scenarios    {len(spec.robustness_v1_scenarios)}: "
            f"{', '.join(spec.robustness_v1_scenarios)}",
            f"startup scenarios          {', '.join(spec.scenarios)}",
            f"startup seeds              {list(spec.startup_seeds)}",
            f"validation workloads       {', '.join(spec.validation_workload_ids)}",
            f"controllers                {', '.join(spec.controller_variants)}",
            f"test workload count        {len(spec.test_workload_ids)}",
            f"reward changed             {str(spec.reward_changed).lower()}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"{STARTUP_ROBUSTNESS_VERSION} (#81), validation only."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-spec", "check", "run", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        command.add_argument("--action-decision", type=Path, default=DEFAULT_ACTION_DECISION)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
        command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
        if name == "freeze":
            command.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args(argv)

    if args.command == "freeze-spec":
        spec = build_experiment_spec(verify_action_decision(args.action_decision))
        spec.save(args.spec)
        print(f"{spec.experiment_version} {spec.experiment_id} written to {args.spec}")
        return 0
    try:
        spec = load_frozen_spec(args.spec, args.action_decision)
    except ValueError as error:
        parser.error(str(error))
    print(describe_plan(spec))
    if args.command == "check":
        return 0
    out: Path = args.output_dir
    spec.save(out / "experiment-spec.json")
    rows = run_experiment(
        spec, out=out, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name
    )
    write_reports(out, rows)
    print(f"{len(rows)} cases; reports in {out}")
    if args.command == "freeze":
        freeze = build_freeze(spec, rows)
        _write_text(args.freeze_output, freeze.model_dump_json(indent=2) + "\n")
        print(f"{freeze.freeze_version} {freeze.freeze_id} -> {args.freeze_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
