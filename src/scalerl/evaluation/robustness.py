"""Predeclared robustness scenarios (#65) and their evaluation.

Four named conditions, frozen as ``robustness-v1`` before any held-out
evaluation (#46) and never tuned to favor a controller:

====================  ========================  =====================
scenario              capacity_jitter_fraction  telemetry_delay_ticks
====================  ========================  =====================
nominal               0.00                      0
capacity-jitter       0.10                      0
delayed-telemetry     0.00                      1
combined-robustness   0.10                      1
====================  ========================  =====================

* **nominal**: the perfectly repeatable reference simulator;
* **capacity-jitter**: seeded ±10% per-tick variation in *actual* service
  throughput (model ``uniform-multiplicative-v1``);
* **delayed-telemetry**: the manager sees load/queue/latency measurements one
  tick (30 s at the benchmark cadence) late, but knows its current replicas;
* **combined-robustness**: both at once.

These are robustness tests, not models of a particular cloud provider.

Fixed controllers and models are evaluated as they are: nothing is retrained
or retuned per scenario. A ``dynamics_seed`` names one stochastic realization;
every controller evaluated on the same workload, scenario, and seed gets the
same capacity-multiplier sequence (it depends only on the seed and tick, not
on actions or evaluation order), so #19 can compare them seed by seed.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from scalerl.benchmarks import WorkloadEntry
from scalerl.controllers import Controller
from scalerl.environment import (
    CAPACITY_JITTER_MODEL,
    AutoscalingEnv,
    DynamicsConfig,
    SimulatorConfig,
)
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation.metrics import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import RunSpec, SimulatorConfigSource
from scalerl.workloads import WorkloadTrace

ROBUSTNESS_SCENARIO_VERSION = "robustness-v1"
# Optional, predeclared jitter levels for a later sensitivity plot; the main
# robustness comparison uses only the scenarios below (0.10).
JITTER_SENSITIVITY_LEVELS = (0.0, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class RobustnessScenario:
    """One named, versioned robustness condition."""

    name: str
    capacity_jitter_fraction: float
    telemetry_delay_ticks: int
    version: str = ROBUSTNESS_SCENARIO_VERSION

    def dynamics(self, dynamics_seed: int) -> DynamicsConfig:
        return DynamicsConfig(
            capacity_jitter_fraction=self.capacity_jitter_fraction,
            telemetry_delay_ticks=self.telemetry_delay_ticks,
            dynamics_seed=dynamics_seed,
        )

    @property
    def is_nominal(self) -> bool:
        return self.capacity_jitter_fraction == 0 and self.telemetry_delay_ticks == 0


NOMINAL = RobustnessScenario("nominal", 0.0, 0)
CAPACITY_JITTER = RobustnessScenario("capacity-jitter", 0.10, 0)
DELAYED_TELEMETRY = RobustnessScenario("delayed-telemetry", 0.0, 1)
COMBINED_ROBUSTNESS = RobustnessScenario("combined-robustness", 0.10, 1)
ROBUSTNESS_SCENARIOS: dict[str, RobustnessScenario] = {
    scenario.name: scenario
    for scenario in (NOMINAL, CAPACITY_JITTER, DELAYED_TELEMETRY, COMBINED_ROBUSTNESS)
}


def get_scenario(name: str) -> RobustnessScenario:
    try:
        return ROBUSTNESS_SCENARIOS[name]
    except KeyError:
        known = ", ".join(ROBUSTNESS_SCENARIOS)
        raise ValueError(f"unknown robustness scenario {name!r}; known: {known}") from None


def apply_scenario(
    config: SimulatorConfig, scenario: RobustnessScenario, *, dynamics_seed: int = 0
) -> SimulatorConfig:
    """A copy of ``config`` whose dynamics are ``scenario`` with ``dynamics_seed``.

    Everything else (timing, replicas, SLA, observation) is kept; ``config`` is
    not modified.
    """
    dynamics = scenario.dynamics(dynamics_seed)  # validated here
    return config.model_copy(update={"dynamics": dynamics})


@dataclass(frozen=True)
class DynamicsSummary:
    """Realized capacity multipliers of one episode (diagnostics, never objectives)."""

    mean_capacity_multiplier: float
    min_capacity_multiplier: float
    max_capacity_multiplier: float

    def as_metrics(self) -> dict[str, float]:
        return {f"dynamics.{name}": float(value) for name, value in vars(self).items()}


def summarize_dynamics(infos: Sequence[Mapping[str, Any]]) -> DynamicsSummary:
    """Summarize the physical ``capacity_multiplier`` of every step ``info``."""
    if not infos:
        raise ValueError("cannot summarize an empty episode")
    multipliers = [float(info["capacity_multiplier"]) for info in infos]
    return DynamicsSummary(
        mean_capacity_multiplier=statistics.fmean(multipliers),
        min_capacity_multiplier=min(multipliers),
        max_capacity_multiplier=max(multipliers),
    )


@dataclass(frozen=True)
class RobustnessResult:
    """One controller on one workload under one scenario and dynamics seed."""

    scenario: RobustnessScenario
    dynamics_seed: int
    metrics: EpisodeMetrics
    dynamics: DynamicsSummary
    infos: tuple[Mapping[str, Any], ...]  # raw physical step infos, for per-seed storage


def evaluate_robustness(
    controller: Controller,
    trace: WorkloadTrace,
    *,
    scenario: RobustnessScenario,
    dynamics_seed: int,
    config: SimulatorConfig | None = None,
    reward_weights: RewardWeights | None = None,
) -> RobustnessResult:
    """Evaluate a fixed controller for one episode under ``scenario``.

    Metrics are computed from the physical step infos (never from the stale
    telemetry the controller saw). The controller is reset by the runner.
    """
    scenario_config = apply_scenario(
        config or SimulatorConfig(), scenario, dynamics_seed=dynamics_seed
    )
    env = AutoscalingEnv(scenario_config, trace, reward_weights)
    evaluation = evaluate_controller_episode(env, controller, seed=0)
    return RobustnessResult(
        scenario=scenario,
        dynamics_seed=dynamics_seed,
        metrics=evaluation.metrics,
        dynamics=summarize_dynamics(evaluation.infos),
        infos=evaluation.infos,
    )


def robustness_run_spec(
    *,
    controller: str,
    entry: WorkloadEntry,
    scenario: RobustnessScenario,
    dynamics_seed: int,
    config: SimulatorConfig | None = None,
    base_config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    hyperparameters: Mapping[str, JsonValue] | None = None,
) -> RunSpec:
    """The ``evaluate`` RunSpec of a robustness evaluation.

    A non-nominal scenario is not the default simulator, so a ``default`` base
    config is recorded as ``predeclared`` (the scenario is predeclared); the
    scenario name/version become params and tags.
    """
    scenario_config = apply_scenario(
        config or SimulatorConfig(), scenario, dynamics_seed=dynamics_seed
    )
    source: SimulatorConfigSource = base_config_source
    if source == "default" and scenario_config != SimulatorConfig():
        source = "predeclared"
    return RunSpec(
        run_kind="evaluate",
        controller=controller,
        workload_id=entry.id,
        workload_split=entry.split,
        simulator_config=scenario_config,
        simulator_config_source=source,
        calibration_workload_ids=tuple(calibration_workload_ids),
        calibration_note=calibration_note,
        evaluation_seeds=(0,),
        hyperparameters=dict(hyperparameters or {}),
        robustness_scenario=scenario.name,
        robustness_version=scenario.version,
    )


def evaluate_robustness_tracked(
    controller: Controller,
    trace: WorkloadTrace,
    *,
    controller_name: str,
    entry: WorkloadEntry,
    scenario: RobustnessScenario,
    dynamics_seed: int,
    config: SimulatorConfig | None = None,
    base_config_source: SimulatorConfigSource = "default",
    calibration_workload_ids: Sequence[str] = (),
    calibration_note: str | None = None,
    hyperparameters: Mapping[str, JsonValue] | None = None,
    model_source_run_id: str | None = None,
    perturbed_compatibility: Sequence[str] = (),
    tracking_uri: str | None = None,
    experiment_name: str = "scalerl-robustness",
    raw_infos_path: str | Path | None = None,
) -> tuple[RobustnessResult, str]:
    """:func:`evaluate_robustness` as one tracked MLflow ``evaluate`` run.

    Logs the shared ``EpisodeMetrics``, the realized-capacity summary
    (``dynamics.*``), the scenario and model identity as params/tags
    (``sim.dynamics.*``, ``compat.capacity_jitter_model``, and the
    ``scalerl.robustness_*`` tags), and, for learned policies evaluated under
    a deliberately changed observation contract, the perturbed compatibility
    fields. Returns the result and the MLflow run ID.

    Perturbed compatibility fields are taken from the controller itself (an
    ``SB3Controller`` loaded with ``robustness_evaluation=True`` records them)
    and merged with any given explicitly, so the tag cannot be forgotten.
    ``calibration_workload_ids``/``calibration_note`` carry the lineage of a
    ``calibrated_train_validation`` base config.
    """
    from scalerl.mlops import start_tracked_run

    perturbed = sorted(
        set(perturbed_compatibility) | set(getattr(controller, "perturbed_compatibility", ()))
    )
    spec = robustness_run_spec(
        controller=controller_name,
        entry=entry,
        scenario=scenario,
        dynamics_seed=dynamics_seed,
        config=config,
        base_config_source=base_config_source,
        calibration_workload_ids=calibration_workload_ids,
        calibration_note=calibration_note,
        hyperparameters=hyperparameters,
    )
    with start_tracked_run(
        spec,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=f"robustness-{controller_name}-{entry.id}-{scenario.name}-seed{dynamics_seed}",
    ) as run:
        run.set_tag("scalerl.dynamics_seed", str(dynamics_seed))
        run.set_tag("scalerl.capacity_jitter_model", CAPACITY_JITTER_MODEL)
        if model_source_run_id is not None:
            run.set_tag("scalerl.model_source_run_id", model_source_run_id)
        if perturbed:
            run.set_tag("scalerl.robustness.perturbed_compatibility", ",".join(perturbed))
        result = evaluate_robustness(
            controller,
            trace,
            scenario=scenario,
            dynamics_seed=dynamics_seed,
            config=config,
        )
        run.log_metrics({**result.metrics.as_metrics(), **result.dynamics.as_metrics()})
        if raw_infos_path is not None:
            Path(raw_infos_path).write_text(_infos_json(result.infos))
            run.log_artifact(raw_infos_path, artifact_path="robustness")
    return result, run.run_id


def _infos_json(infos: Sequence[Mapping[str, Any]]) -> str:
    return json.dumps([dict(info) for info in infos], indent=None, sort_keys=True) + "\n"
