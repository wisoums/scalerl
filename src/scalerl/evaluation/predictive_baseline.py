"""Validation-only predictive-baseline experiment (#80): predictive-v1 vs predictive-seasonal-v1.

Everything is predeclared in ``benchmarks/v1/predictive-baseline-experiment-v1.json``
before any validation run; nothing is tuned (no Optuna, no grid). Controllers,
all under the frozen #79 contract ``desired-replicas-v1`` (requested explicitly:
``SimulatorConfig()`` still defaults to ``delta-v1``):

* ``threshold-v1`` (0.6 / 0.2 / cooldown 3), a reactive reference;
* ``predictive-v1`` (``linear-trend`` + ``forecast-plus-backlog-v1``), unchanged;
* ``predictive-seasonal-v1`` (``historical-profile-plus-linear-v1`` +
  ``proactive-scaleout-conservative-scalein-v1``).

Workloads: the three synthetic validation workloads, where no recurring history
exists, so the new baseline runs **without** a profile (this isolates its
actuation policy); and ``azure-val-734400`` when the Azure CSV is available
locally, where the new baseline uses the TRAIN-only profile of
``azure-train-129600/302400/475200`` (the same hour of earlier days). Nominal
dynamics, dynamics seed 0, evaluation seed 0. No test workload is read.

    python -m scalerl.evaluation.predictive_baseline check
    python -m scalerl.evaluation.predictive_baseline run --azure-csv data/raw/<trace>.txt
    python -m scalerl.evaluation.predictive_baseline freeze

Each case is one tracked MLflow ``evaluate`` run (experiment
``scalerl-predictive-baseline``); rows are resumable and recovered from finished
runs instead of being run twice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers import (
    Controller,
    HistoricalDemandProfile,
    PredictiveController,
    ProactivePredictiveController,
    ThresholdController,
)
from scalerl.controllers import predictive as predictive_v1
from scalerl.controllers import proactive_predictive as seasonal
from scalerl.environment import (
    DESIRED_REPLICAS_V1,
    ActionConfig,
    ActionContract,
    SimulatorConfig,
)
from scalerl.environment.reward import RewardWeights
from scalerl.evaluation.action_semantics import ActionContractDecision
from scalerl.evaluation.forecast import score_forecasts
from scalerl.evaluation.metrics import (
    ActionMagnitudeMetrics,
    EpisodeMetrics,
    summarize_action_magnitude,
)
from scalerl.evaluation.multiseed import _append_row, _repair_torn_tail, read_rows
from scalerl.evaluation.robustness import NOMINAL, evaluate_robustness, robustness_run_spec

EXPERIMENT_VERSION: Final = "predictive-baseline-experiment-v1"
ARTIFACT_VERSION: Final = "predictive-baseline-v1"
EXPERIMENT_NAME: Final = "scalerl-predictive-baseline"
ACTION_DECISION_VERSION: Final = "action-contract-v2"
ACTION_EXPERIMENT_ID: Final = "899dfbb64217"
ACTION_CANDIDATE_SET_ID: Final = "cbe3c1c0719b"
SELECTION_SPEC_ID: Final = "418876d6c8e9"
SYNTHETIC_VALIDATION: Final = ("syn-val-steady-high", "syn-val-ramp-down", "syn-val-bursty")
AZURE_PROFILE_SOURCES: Final = ("azure-train-129600", "azure-train-302400", "azure-train-475200")
AZURE_VALIDATION: Final = "azure-val-734400"
AZURE_PROFILE_NAME: Final = "azure-historical-profile-v1"
EVALUATION_SEED: Final = 0
DYNAMICS_SEED: Final = 0
THRESHOLD_PARAMS: Final = {"high_threshold": 0.6, "low_threshold": 0.2, "cooldown_ticks": 3}
CONTROLLERS: Final = ("threshold-v1", "predictive-v1", "predictive-seasonal-v1")
METRIC_KEYS: Final = tuple(EpisodeMetrics.__dataclass_fields__)
ACTION_KEYS: Final = tuple(ActionMagnitudeMetrics.__dataclass_fields__)
FORECAST_KEYS: Final = (
    "forecast_count",
    "forecast_mae_rps",
    "forecast_rmse_rps",
    "forecast_bias_rps",
)
AzureStatus = Literal["available", "unavailable_external_data"]

DEFAULT_SPEC = Path("benchmarks/v1/predictive-baseline-experiment-v1.json")
DEFAULT_ACTION_DECISION = Path("benchmarks/v1/action-contract-v2.json")
DEFAULT_ARTIFACT = Path("benchmarks/v1/predictive-baseline-v1.json")
DEFAULT_OUTPUT = Path("outputs/predictive-baseline-v1")
DEFAULT_AZURE_CSV = Path("data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt")


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


def final_contract_config() -> SimulatorConfig:
    """The canonical simulator config under the frozen #79 contract, requested explicitly."""
    return SimulatorConfig(action=ActionConfig(semantics=DESIRED_REPLICAS_V1))


def require_final_contract(config: SimulatorConfig) -> None:
    """Refuse any config that is not the canonical config under ``desired-replicas-v1``."""
    if config.action.semantics != DESIRED_REPLICAS_V1:
        raise ValueError(
            f"#80 runs under the frozen #79 contract {DESIRED_REPLICAS_V1}, not "
            f"{config.action.semantics} (SimulatorConfig() defaults to delta-v1)"
        )
    if config != final_contract_config():
        raise ValueError("#80 uses the canonical simulator config (only the action contract set)")


def verify_action_decision(path: Path) -> ActionContractDecision:
    """The committed #79 decision, checked against the facts #80 depends on."""
    decision = ActionContractDecision.model_validate_json(path.read_text())
    expected = {
        "decision_version": ACTION_DECISION_VERSION,
        "final_action_semantics": DESIRED_REPLICAS_V1,
        "experiment_spec_id": ACTION_EXPERIMENT_ID,
        "candidate_set_id": ACTION_CANDIDATE_SET_ID,
        "selection_spec_id": SELECTION_SPEC_ID,
    }
    for field, value in expected.items():
        if getattr(decision, field) != value:
            raise ValueError(f"#79 decision {field} is {getattr(decision, field)!r}, not {value!r}")
    if decision.held_out_data_used or decision.reward_changed:
        raise ValueError("the #79 decision must not have used held-out data or changed reward")
    return decision


# --- frozen experiment specification ------------------------------------------------------------


class BaselineIdentity(_Strict):
    variant: str
    forecast_method: str
    capacity_policy: str


class ExperimentSpec(_Strict):
    """Everything #80 holds fixed; ``experiment_id`` hashes it."""

    experiment_version: Literal["predictive-baseline-experiment-v1"] = EXPERIMENT_VERSION
    benchmark_version: str
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_experiment_spec_id: Literal["899dfbb64217"] = ACTION_EXPERIMENT_ID
    action_candidate_set_id: Literal["cbe3c1c0719b"] = ACTION_CANDIDATE_SET_ID
    selection_spec_id: Literal["418876d6c8e9"] = SELECTION_SPEC_ID
    simulator_config: dict[str, JsonValue]
    existing_baseline: BaselineIdentity
    new_baseline: BaselineIdentity
    reference_controllers: dict[str, JsonValue]
    fixed_policy: dict[str, JsonValue]
    forecast_horizon_rule: str
    synthetic_validation_workload_ids: tuple[str, ...]
    synthetic_profile: Literal["none: no recurring history exists for these workloads"] = (
        "none: no recurring history exists for these workloads"
    )
    azure_profile_name: Literal["azure-historical-profile-v1"] = AZURE_PROFILE_NAME
    azure_profile_train_workload_ids: tuple[str, ...]
    azure_validation_workload_ids: tuple[str, ...]
    azure_capacity_note: str
    test_workload_ids: tuple[str, ...] = ()
    dynamics: dict[str, JsonValue]
    evaluation_seed: Literal[0] = EVALUATION_SEED
    reward_weights: dict[str, float]
    reward_changed: Literal[False] = False
    tuning: Literal["none: every value is predeclared; no Optuna, grid, or post-hoc change"] = (
        "none: every value is predeclared; no Optuna, grid, or post-hoc change"
    )
    system_metrics: tuple[str, ...] = METRIC_KEYS
    action_magnitude_metrics: tuple[str, ...] = ACTION_KEYS
    forecast_metrics: tuple[str, ...]
    learned_context: dict[str, JsonValue]
    interpretation_rules: tuple[str, ...]
    held_out_rule: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        manifest = load_benchmark_manifest()
        if self.benchmark_version != manifest.version:
            raise ValueError(f"spec benchmark {self.benchmark_version!r} is not installed")
        if self.test_workload_ids:
            raise ValueError("#80 uses no test workloads")
        workloads = (
            *self.synthetic_validation_workload_ids,
            *self.azure_profile_train_workload_ids,
            *self.azure_validation_workload_ids,
        )
        for workload_id in workloads:
            try:
                split = manifest.get(workload_id).split
            except KeyError as error:
                raise ValueError(str(error)) from None
            if split == "test":
                raise ValueError(f"{workload_id!r} is a held-out test workload")
        if self.synthetic_validation_workload_ids != SYNTHETIC_VALIDATION:
            raise ValueError(f"synthetic validation must be exactly {SYNTHETIC_VALIDATION}")
        if self.azure_profile_train_workload_ids != AZURE_PROFILE_SOURCES:
            raise ValueError(f"the Azure profile must use exactly {AZURE_PROFILE_SOURCES}")
        if self.azure_validation_workload_ids != (AZURE_VALIDATION,):
            raise ValueError(f"Azure validation must be exactly {AZURE_VALIDATION}")
        if self.simulator_config != final_contract_config().model_dump(mode="json"):
            raise ValueError("the simulator config must be the canonical desired-replicas-v1 one")
        if self.reward_weights != RewardWeights().model_dump(mode="json"):
            raise ValueError("the reward must be the unchanged default")
        if self.dynamics != _dynamics():
            raise ValueError("#80 evaluates nominal dynamics only (dynamics seed 0)")
        return self

    @property
    def experiment_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]

    def save(self, path: str | Path) -> Path:
        return _write_text(Path(path), self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        return cls.model_validate_json(Path(path).read_text())


def _dynamics() -> dict[str, JsonValue]:
    return {
        "scenario": NOMINAL.name,
        "robustness_version": NOMINAL.version,
        "capacity_jitter_fraction": 0.0,
        "telemetry_delay_ticks": 0,
        "dynamics_seed": DYNAMICS_SEED,
    }


def build_experiment_spec() -> ExperimentSpec:
    """The #80 experiment specification (frozen before any validation run)."""
    return ExperimentSpec(
        benchmark_version=load_benchmark_manifest().version,
        simulator_config=final_contract_config().model_dump(mode="json"),
        existing_baseline=BaselineIdentity(
            variant="predictive-v1",
            forecast_method=predictive_v1.FORECAST_METHOD,
            capacity_policy=predictive_v1.CAPACITY_POLICY,
        ),
        new_baseline=BaselineIdentity(
            variant=seasonal.VARIANT,
            forecast_method=seasonal.FORECAST_METHOD,
            capacity_policy=seasonal.CAPACITY_POLICY,
        ),
        reference_controllers={
            "threshold-v1": {
                "controller_version": "threshold-sla-first-v1",
                "params": dict(THRESHOLD_PARAMS),
                "encoding": "same ±1/hold law encoded as a desired-replicas-v1 target",
            },
            "predictive-v1": {
                "history_window_ticks": 4,
                "target_utilization": 0.8,
                "encoding": "its computed desired replicas as the target (unchanged since #79)",
            },
        },
        fixed_policy={
            "recent_history_window_ticks": seasonal.HISTORY_WINDOW_TICKS,
            "target_utilization": seasonal.TARGET_UTILIZATION,
            "profile_aggregation": seasonal.PROFILE_AGGREGATION,
            "profile_alignment": "tick position within equally aligned windows",
            "level_adaptation": (
                "median of observed/profile over the latest <=4 completed samples whose "
                "profile value >= min_profile_rps_for_ratio; 1.0 when none"
            ),
            "min_profile_rps_for_ratio": seasonal.MIN_PROFILE_RPS_FOR_RATIO,
            "proactive_forecast": "max(linear, profile * level_factor); linear when no profile",
            "profile_beyond_its_ticks": "no profile value: linear only for that decision",
            "backlog_recovery": "queued_requests / control_interval_seconds (#63)",
            "sizing": "ceil((proactive + backlog_recovery) / (service_capacity_rps * 0.8))",
            "scale_out": "direct desired target when desired > active + pending",
            "scale_in": (
                "never with queued requests; else target max(max(observed_desired, desired), "
                "committed - 1) when that is below committed: at most one replica per decision"
            ),
            "blend_weight": None,
            "tunable_scale_in_step": None,
        },
        forecast_horizon_rule=(
            "1 + startup_ticks(startup_delay_seconds, control_interval_seconds), the "
            "predictive-v1 arithmetic: 3 ticks (90 s) at 60 s startup and 30 s ticks; linear and "
            "profile forecasts target the same tick"
        ),
        synthetic_validation_workload_ids=SYNTHETIC_VALIDATION,
        azure_profile_train_workload_ids=AZURE_PROFILE_SOURCES,
        azure_validation_workload_ids=(AZURE_VALIDATION,),
        azure_capacity_note=(
            "Azure windows are the same hour (12:00-13:00) of different days, so tick position is "
            "a recurring time of day. Under the canonical config one replica serves 50 rps; the "
            "train/validation characterization peaks at 2.5-4.9 rps, so every controller is "
            "expected to stay at min_replicas and Azure system metrics are not expected to "
            "discriminate. Azure evidence is primarily forecast quality. No Azure capacity "
            "calibration is done in #80. If the Azure CSV is absent, Azure is reported as "
            "unavailable_external_data and never substituted."
        ),
        dynamics=_dynamics(),
        reward_weights=RewardWeights().model_dump(mode="json"),
        forecast_metrics=(
            "forecast_count",
            "forecast_mae_rps",
            "forecast_rmse_rps",
            "forecast_bias_rps (forecast - actual)",
            "horizon_ticks",
            "horizon_seconds",
            "profile_used_count",
            "profile_used_fraction",
        ),
        learned_context={
            "ppo": (
                "#79 PPO desired-replicas-v1 (ppo-c08, seeds 0-4) is not evaluated in #80; no RL "
                "retraining"
            ),
            "dqn": (
                "no selected desired-replicas-v1 DQN exists from #79 (0/20 feasible); the "
                "diagnostic fallback dqn-c01 is not a model of record and is not used"
            ),
        },
        interpretation_rules=(
            "random bursts are not inherently predictable",
            "forecast accuracy is reported separately from control quality",
            "no global winner from one metric or workload",
            "synthetic validation has no recurring history: no seasonal claim from it",
        ),
        held_out_rule=(
            "no syn-test-* or azure-test-* workload is read, run, plotted, or used; the CLI "
            "accepts no workload arguments and the spec rejects test workloads"
        ),
    )


def load_frozen_spec(spec_path: Path, decision_path: Path) -> ExperimentSpec:
    spec = ExperimentSpec.load(spec_path)
    verify_action_decision(decision_path)
    if spec != build_experiment_spec():
        raise ValueError("the committed #80 spec differs from this code's frozen plan")
    return spec


# --- cases and controllers -----------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    case_id: str
    controller_variant: str
    workload_id: str
    profile_id: str | None


def cases(
    spec: ExperimentSpec, *, azure: bool, profile: HistoricalDemandProfile | None
) -> list[Case]:
    workloads = [*spec.synthetic_validation_workload_ids]
    if azure:
        workloads += list(spec.azure_validation_workload_ids)
    result = []
    for workload_id in workloads:
        for variant in CONTROLLERS:
            profile_id = (
                profile.profile_id
                if variant == seasonal.VARIANT and workload_id == AZURE_VALIDATION and profile
                else None
            )
            result.append(
                Case(
                    case_id=f"{variant}|{workload_id}|nominal|dyn{DYNAMICS_SEED}|eval{EVALUATION_SEED}",
                    controller_variant=variant,
                    workload_id=workload_id,
                    profile_id=profile_id,
                )
            )
    return result


def make_controller(
    variant: str, config: SimulatorConfig, profile: HistoricalDemandProfile | None
) -> Controller:
    require_final_contract(config)
    if variant == "threshold-v1":
        return ThresholdController(
            low_threshold=THRESHOLD_PARAMS["low_threshold"],
            high_threshold=THRESHOLD_PARAMS["high_threshold"],
            cooldown_ticks=int(THRESHOLD_PARAMS["cooldown_ticks"]),
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            action_contract=ActionContract.from_config(config),
        )
    if variant == "predictive-v1":
        return PredictiveController.from_config(config)
    if variant == seasonal.VARIANT:
        return ProactivePredictiveController.from_config(config, profile=profile)
    raise ValueError(f"unknown controller variant {variant!r}")


def _identity(variant: str) -> dict[str, str]:
    if variant == "predictive-v1":
        return {
            "controller": "predictive",
            "forecast_method": predictive_v1.FORECAST_METHOD,
            "capacity_policy": predictive_v1.CAPACITY_POLICY,
        }
    if variant == seasonal.VARIANT:
        return {
            "controller": "predictive-seasonal",
            "forecast_method": seasonal.FORECAST_METHOD,
            "capacity_policy": seasonal.CAPACITY_POLICY,
        }
    return {"controller": "threshold", "forecast_method": "none", "capacity_policy": "none"}


# --- forecast diagnostics -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Component:
    target_tick: int
    forecast_rps: float


def forecast_metrics(
    controller: Controller, infos: Sequence[Mapping[str, Any]], control_interval_seconds: float
) -> dict[str, float]:
    """Forecast diagnostics scored after the episode (empty for non-forecasting controllers).

    Unavailable errors (no scored forecast) are omitted, never logged as zero.
    """
    horizon = getattr(controller, "forecast_horizon_ticks", None)
    if horizon is None:
        return {}
    records = tuple(getattr(controller, "forecasts", ()))
    accuracy = score_forecasts(records, infos)
    metrics = accuracy.as_metrics()
    metrics |= {
        "horizon_ticks": float(horizon),
        "horizon_seconds": horizon * control_interval_seconds,
    }
    if isinstance(controller, ProactivePredictiveController):
        used = [r for r in records if r.profile_used]
        metrics["profile_used_count"] = float(len(used))
        if records:
            metrics["profile_used_fraction"] = len(used) / len(records)
        linear = score_forecasts(
            [_Component(r.target_tick, r.linear_forecast_rps) for r in records], infos
        )
        metrics |= {f"linear_component_{k}": v for k, v in linear.as_metrics().items()}
        profile = score_forecasts(
            [
                _Component(r.target_tick, r.historical_profile_forecast_rps)
                for r in used
                if r.historical_profile_forecast_rps is not None
            ],
            infos,
        )
        metrics |= {f"profile_component_{k}": v for k, v in profile.as_metrics().items()}
    return {f"forecast.{key}": value for key, value in metrics.items()}


def forecast_record_rows(
    case: Case, controller: Controller, infos: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Forecast records with the actual target-tick demand joined after the episode."""
    actual = {int(info["tick"]): float(info["request_rate"]) for info in infos}
    rows = []
    for record in getattr(controller, "forecasts", ()):
        values = asdict(record)
        rows.append(
            {
                "case_id": case.case_id,
                "controller_variant": case.controller_variant,
                "workload_id": case.workload_id,
                **values,
                "actual_target_rps": actual.get(int(record.target_tick)),
            }
        )
    return rows


# --- running ----------------------------------------------------------------------------------


def _tags(spec: ExperimentSpec, case: Case) -> dict[str, str]:
    identity = _identity(case.controller_variant)
    return {
        "scalerl.experiment_version": spec.experiment_version,
        "scalerl.experiment_id": spec.experiment_id,
        "scalerl.evaluation_case_id": case.case_id,
        "scalerl.controller_variant_id": case.controller_variant,
        "scalerl.forecast_method": identity["forecast_method"],
        "scalerl.capacity_policy": identity["capacity_policy"],
        "scalerl.action_decision_version": spec.action_decision_version,
        "scalerl.profile_id": case.profile_id or "none",
        "scalerl.dynamics_seed": str(DYNAMICS_SEED),
        "scalerl.evaluation_seed": str(EVALUATION_SEED),
    }


def _row(
    spec: ExperimentSpec, case: Case, split: str, run_id: str, metrics: Mapping[str, float]
) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "case_id": case.case_id,
        "controller_variant": case.controller_variant,
        **_identity(case.controller_variant),
        "action_semantics": spec.action_semantics,
        "workload_id": case.workload_id,
        "workload_split": split,
        "profile_id": case.profile_id,
        "robustness_scenario": NOMINAL.name,
        "dynamics_seed": DYNAMICS_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "mlflow_run_id": run_id,
        **{key: float(value) for key, value in sorted(metrics.items())},
    }


def _recover(
    spec: ExperimentSpec, case: Case, split: str, tracking_uri: str | None, experiment_name: str
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
            "attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if not runs:
        return None
    return _row(spec, case, split, runs[0].info.run_id, runs[0].data.metrics)


def run_experiment(
    spec: ExperimentSpec,
    *,
    out: Path,
    azure_csv_path: Path | None,
    tracking_uri: str | None,
    experiment_name: str = EXPERIMENT_NAME,
    progress: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], AzureStatus, HistoricalDemandProfile | None]:
    """Evaluate every case once (resumable); returns rows, Azure status and the profile."""
    from scalerl.mlops import start_tracked_run

    config = final_contract_config()
    require_final_contract(config)
    out.mkdir(parents=True, exist_ok=True)
    azure: AzureStatus = (
        "available"
        if azure_csv_path is not None and Path(azure_csv_path).is_file()
        else "unavailable_external_data"
    )
    profile: HistoricalDemandProfile | None = None
    if azure == "available":
        profile = HistoricalDemandProfile.from_benchmark(
            spec.azure_profile_train_workload_ids, azure_csv_path=azure_csv_path
        )
        _write_text(out / "historical-profile.json", profile.to_json() + "\n")
    manifest = load_benchmark_manifest()
    all_cases = cases(spec, azure=azure == "available", profile=profile)
    entries = {w: manifest.get(w) for w in dict.fromkeys(c.workload_id for c in all_cases)}
    traces = build_workloads(
        list(entries.values()), azure_csv_path=azure_csv_path if azure == "available" else None
    )
    raw_path, records_path = out / "raw-results.jsonl", out / "forecast-records.jsonl"
    _repair_torn_tail(raw_path)
    _repair_torn_tail(records_path)
    done = {r["case_id"] for r in read_rows(raw_path) if r["experiment_id"] == spec.experiment_id}
    for case in all_cases:
        if case.case_id in done:
            continue
        entry = entries[case.workload_id]
        recovered = _recover(spec, case, entry.split, tracking_uri, experiment_name)
        if recovered is not None:
            _append_row(raw_path, recovered)
            done.add(case.case_id)
            continue
        case_profile = profile if case.profile_id is not None else None
        controller = make_controller(case.controller_variant, config, case_profile)
        run_spec = robustness_run_spec(
            controller=_identity(case.controller_variant)["controller"],
            entry=entry,
            scenario=NOMINAL,
            dynamics_seed=DYNAMICS_SEED,
            config=config,
            base_config_source="predeclared",
            hyperparameters=spec.fixed_policy
            if case.controller_variant == seasonal.VARIANT
            else spec.reference_controllers[case.controller_variant],  # type: ignore[arg-type]
            evaluation_seed=EVALUATION_SEED,
        )
        with start_tracked_run(
            run_spec,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=f"predictive-baseline-{case.case_id.replace('|', '-')}",
        ) as run:
            for key, value in _tags(spec, case).items():
                run.set_tag(key, value)
            result = evaluate_robustness(
                controller,
                traces[case.workload_id],
                scenario=NOMINAL,
                dynamics_seed=DYNAMICS_SEED,
                evaluation_seed=EVALUATION_SEED,
                config=config,
            )
            metrics = {
                **result.metrics.as_metrics(),
                **summarize_action_magnitude(result.infos).as_metrics(),
                **forecast_metrics(
                    controller, result.infos, config.timing.control_interval_seconds
                ),
            }
            run.log_metrics(metrics)
        for record in forecast_record_rows(case, controller, result.infos):
            _append_row(records_path, record | {"mlflow_run_id": run.run_id})
        _append_row(raw_path, _row(spec, case, entry.split, run.run_id, metrics))
        done.add(case.case_id)
        progress(f"[{len(done)}/{len(all_cases)}] {case.case_id}")
    order = {case.case_id: index for index, case in enumerate(all_cases)}
    rows = [
        r
        for r in read_rows(raw_path)
        if r["experiment_id"] == spec.experiment_id and r["case_id"] in order
    ]
    rows.sort(key=lambda r: order[r["case_id"]])
    if sorted(r["case_id"] for r in rows) != sorted(order):
        raise ValueError("evaluation rows do not match the predeclared cases")
    return rows, azure, profile


# --- reports and the frozen artifact ----------------------------------------------------------


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
    *(f"action.{key}" for key in ACTION_KEYS),
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{workload: {controller: metrics + run ID}}``; nothing pooled across workloads."""
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        values: dict[str, Any] = {k: row[k] for k in REPORT_METRICS}
        values |= {k: v for k, v in row.items() if k.startswith("forecast.")}
        values["mlflow_run_id"] = row["mlflow_run_id"]
        values["profile_id"] = row["profile_id"]
        summary.setdefault(row["workload_id"], {})[row["controller_variant"]] = values
    return summary


def seasonal_minus_v1(summary: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Descriptive ``predictive-seasonal-v1 − predictive-v1`` per workload (no ranking)."""
    return {
        workload: {
            key: float(by[seasonal.VARIANT][key]) - float(by["predictive-v1"][key])
            for key in REPORT_METRICS
        }
        for workload, by in summary.items()
    }


class PredictiveBaselineArtifact(_Strict):
    """The frozen ``predictive-seasonal-v1`` baseline that #72/#46 must include."""

    artifact_version: Literal["predictive-baseline-v1"] = ARTIFACT_VERSION
    frozen_baseline: BaselineIdentity
    existing_baseline: BaselineIdentity
    frozen_by_predeclaration: Literal[True] = True
    declares_overall_winner: Literal[False] = False
    action_semantics: Literal["desired-replicas-v1"] = DESIRED_REPLICAS_V1
    action_decision_version: Literal["action-contract-v2"] = ACTION_DECISION_VERSION
    experiment_version: str
    experiment_spec_id: str
    fixed_parameters: dict[str, JsonValue]
    forecast_horizon_rule: str
    historical_profile: dict[str, JsonValue]
    synthetic_validation_workload_ids: tuple[str, ...]
    azure_validation_workload_ids: tuple[str, ...]
    azure_validation_status: AzureStatus
    validation_results: dict[str, JsonValue]
    seasonal_minus_predictive_v1: dict[str, JsonValue]
    mlflow_experiment: str
    learned_context: dict[str, JsonValue]
    held_out_data_used: Literal[False] = False
    reward_changed: Literal[False] = False

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def build_artifact(
    spec: ExperimentSpec,
    rows: Sequence[Mapping[str, Any]],
    azure: AzureStatus,
    profile: HistoricalDemandProfile | None,
) -> PredictiveBaselineArtifact:
    summary = summarize(rows)
    profile_definition: dict[str, JsonValue] = {
        "name": spec.azure_profile_name,
        "version": seasonal.PROFILE_VERSION,
        "aggregation": seasonal.PROFILE_AGGREGATION,
        "formula": "profile[tick] = median(train_1[tick], train_2[tick], train_3[tick])",
        "train_workload_ids": list(spec.azure_profile_train_workload_ids),
        "profile_id": profile.profile_id if profile else None,
        "ticks": profile.ticks if profile else None,
        "synthetic_validation_profile": None,
    }
    return PredictiveBaselineArtifact(
        frozen_baseline=spec.new_baseline,
        existing_baseline=spec.existing_baseline,
        experiment_version=spec.experiment_version,
        experiment_spec_id=spec.experiment_id,
        fixed_parameters=spec.fixed_policy,
        forecast_horizon_rule=spec.forecast_horizon_rule,
        historical_profile=profile_definition,
        synthetic_validation_workload_ids=spec.synthetic_validation_workload_ids,
        azure_validation_workload_ids=spec.azure_validation_workload_ids,
        azure_validation_status=azure,
        validation_results=_json(summary),
        seasonal_minus_predictive_v1=_json(seasonal_minus_v1(summary)),
        mlflow_experiment=EXPERIMENT_NAME,
        learned_context=spec.learned_context,
    )


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def write_reports(
    spec: ExperimentSpec,
    out: Path,
    rows: Sequence[Mapping[str, Any]],
    azure: AzureStatus,
    profile: HistoricalDemandProfile | None,
) -> dict[str, Any]:
    _write_csv(out / "raw-results.csv", rows)
    records = read_rows(out / "forecast-records.jsonl")
    _write_csv(out / "forecast-records.csv", records)
    summary = summarize(rows)
    _write_text(out / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    flat = [
        {"workload_id": w, "controller_variant": c, **values}
        for w, by in summary.items()
        for c, values in by.items()
    ]
    _write_csv(out / "summary.csv", flat)
    report = {
        "experiment_id": spec.experiment_id,
        "azure_validation_status": azure,
        "azure_profile_id": profile.profile_id if profile else None,
        "cases": len(rows),
        "seasonal_minus_predictive_v1": seasonal_minus_v1(summary),
        "frozen_baseline": spec.new_baseline.model_dump(mode="json"),
        "declares_overall_winner": False,
        "held_out_data_used": False,
        "reward_changed": False,
    }
    _write_text(out / "decision-report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --- command line -----------------------------------------------------------------------------


def describe_plan(
    spec: ExperimentSpec, decision: ActionContractDecision, azure_csv: Path | None
) -> str:
    csv_state = azure_csv if azure_csv is not None and azure_csv.is_file() else "absent"
    return "\n".join(
        [
            f"experiment              {spec.experiment_version} {spec.experiment_id}",
            f"action semantics        {spec.action_semantics}",
            f"action decision         {decision.decision_version} {decision.decision_id}",
            f"#79 experiment spec     {decision.experiment_spec_id}",
            f"#79 candidate set       {decision.candidate_set_id}",
            f"#78 selection spec      {decision.selection_spec_id}",
            f"existing predictive     {spec.existing_baseline.variant} / "
            f"{spec.existing_baseline.forecast_method} / {spec.existing_baseline.capacity_policy}",
            f"new predictive          {spec.new_baseline.variant} / "
            f"{spec.new_baseline.forecast_method} / {spec.new_baseline.capacity_policy}",
            f"synthetic validation    {', '.join(spec.synthetic_validation_workload_ids)}",
            f"Azure profile sources   {', '.join(spec.azure_profile_train_workload_ids)}",
            f"Azure validation        {', '.join(spec.azure_validation_workload_ids)}",
            f"Azure CSV               {csv_state}",
            f"test workload count     {len(spec.test_workload_ids)}",
            f"dynamics                {spec.dynamics['scenario']}",
            f"dynamics_seed           {spec.dynamics['dynamics_seed']}",
            f"evaluation_seed         {spec.evaluation_seed}",
            f"reward changed          {str(spec.reward_changed).lower()}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_VERSION} (#80), validation only.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-spec", "check", "run", "freeze"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
        command.add_argument("--action-decision", type=Path, default=DEFAULT_ACTION_DECISION)
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--azure-csv", type=Path, default=DEFAULT_AZURE_CSV)
        command.add_argument("--tracking-uri", default=None, help="defaults to MLFLOW_TRACKING_URI")
        command.add_argument("--experiment-name", default=EXPERIMENT_NAME)
        if name == "freeze":
            command.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)

    if args.command == "freeze-spec":
        verify_action_decision(args.action_decision)
        spec = build_experiment_spec()
        spec.save(args.spec)
        print(f"{spec.experiment_version} {spec.experiment_id} written to {args.spec}")
        return 0
    try:
        spec = load_frozen_spec(args.spec, args.action_decision)
        decision = verify_action_decision(args.action_decision)
    except ValueError as error:
        parser.error(str(error))
    print(describe_plan(spec, decision, args.azure_csv))
    if args.command == "check":
        return 0
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    spec.save(out / "experiment-spec.json")
    rows, azure, profile = run_experiment(
        spec,
        out=out,
        azure_csv_path=args.azure_csv,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    report = write_reports(spec, out, rows, azure, profile)
    print(f"{len(rows)} cases; Azure {azure}; profile {report['azure_profile_id']}")
    if args.command == "freeze":
        artifact = build_artifact(spec, rows, azure, profile)
        _write_text(args.artifact, artifact.model_dump_json(indent=2) + "\n")
        print(f"{artifact.artifact_version} {artifact.artifact_id} -> {args.artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
