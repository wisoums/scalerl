"""Interactive simulation session behind the City View; no Streamlit dependency.

The session only drives the real ``AutoscalingEnv`` and controllers and keeps
presentation state (history, cumulative cost). All simulator and controller
behavior lives in the backend modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scalerl.benchmarks import (
    AzureWorkload,
    BenchmarkManifest,
    WorkloadEntry,
    build_workload,
    load_benchmark_manifest,
)
from scalerl.controllers import (
    Controller,
    PredictiveController,
    PredictiveDecision,
    RandomController,
    StaticController,
    ThresholdController,
    ThresholdDecision,
    decision_info,
)
from scalerl.environment import ActionContract, AutoscalingEnv, SimulatorConfig
from scalerl.environment.gym_env import HOLD, SCALE_DOWN, SCALE_UP, Observation
from scalerl.workloads import (
    WorkloadTrace,
    bursty_workload,
    diurnal_workload,
    ramp_workload,
    spike_workload,
    steady_workload,
)

ManagerKind = Literal["manual", "random", "static", "threshold", "predictive"]
DEFAULT_AZURE_PATH = "data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
ACTION_LABELS = {SCALE_DOWN: "SCALE DOWN ↓", HOLD: "HOLD —", SCALE_UP: "SCALE UP ↑"}
HELD_OUT_WARNING = (
    "Held-out test workloads are for final evaluation only. Do not use this view to "
    "tune controller, model, reward, or simulator settings."
)

CUSTOM_GENERATORS: dict[str, Callable[..., WorkloadTrace]] = {
    "steady": steady_workload,
    "diurnal": diurnal_workload,
    "ramp": ramp_workload,
    "spike": spike_workload,
    "bursty": bursty_workload,
}


@dataclass(frozen=True)
class ScenarioInfo:
    """What the running scenario is, for display."""

    label: str
    source: Literal["synthetic", "azure"]
    split: str  # "train", "validation", "test", or "custom"
    duration_seconds: float
    control_interval_seconds: float


@dataclass(frozen=True)
class Scenario:
    trace: WorkloadTrace
    info: ScenarioInfo


def benchmark_choices(
    manifest: BenchmarkManifest, *, include_test: bool
) -> tuple[WorkloadEntry, ...]:
    """Train and validation workloads, plus held-out test ones only when asked."""
    return manifest.tuning + (manifest.test if include_test else ())


def benchmark_scenario(
    workload_id: str,
    *,
    azure_csv_path: str | Path | None = None,
    manifest: BenchmarkManifest | None = None,
) -> Scenario:
    """Build a frozen benchmark workload with the existing benchmark API."""
    entry = (manifest or load_benchmark_manifest()).get(workload_id)
    if isinstance(entry, AzureWorkload):
        if azure_csv_path is None or not str(azure_csv_path).strip():
            raise ValueError("Azure scenarios need the local path to the extracted trace")
        path = Path(azure_csv_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Azure trace not found at {path}. Extract the dataset locally "
                "(see data/README.md) and enter its path."
            )
        trace = build_workload(entry, azure_csv_path=path)
    else:
        trace = build_workload(entry)
    return Scenario(
        trace,
        ScenarioInfo(
            label=entry.id,
            source=entry.source,
            split=entry.split,
            duration_seconds=entry.duration_seconds,
            control_interval_seconds=entry.control_interval_seconds,
        ),
    )


def custom_scenario(
    generator: str,
    *,
    duration_seconds: float,
    control_interval_seconds: float,
    parameters: Mapping[str, Any],
) -> Scenario:
    """Build an exploratory synthetic workload with an existing generator."""
    if generator not in CUSTOM_GENERATORS:
        raise ValueError(f"unknown generator {generator!r}")
    trace = CUSTOM_GENERATORS[generator](
        duration_seconds=duration_seconds,
        control_interval_seconds=control_interval_seconds,
        **parameters,
    )
    return Scenario(
        trace,
        ScenarioInfo(
            label=f"custom {generator}",
            source="synthetic",
            split="custom",
            duration_seconds=float(duration_seconds),
            control_interval_seconds=float(control_interval_seconds),
        ),
    )


@dataclass(frozen=True)
class ManagerSpec:
    """Which manager runs the city and its settings."""

    kind: ManagerKind
    seed: int | None = None
    target_replicas: int | None = None
    low_threshold: float = 0.3
    high_threshold: float = 0.8
    cooldown_ticks: int = 0
    history_window_ticks: int = 4
    target_utilization: float = 0.8

    def build(self, config: SimulatorConfig) -> Controller | None:
        """Construct the real controller (``None`` for manual control).

        Replica bounds, capacity, and timing come from ``config``, never from
        duplicated manager settings.
        """
        replicas = config.replicas
        contract = ActionContract.from_config(config)
        if self.kind == "manual":
            return None
        if self.kind == "random":
            return RandomController(seed=self.seed, action_contract=contract)
        if self.kind == "predictive":
            return PredictiveController.from_config(
                config,
                history_window_ticks=self.history_window_ticks,
                target_utilization=self.target_utilization,
            )
        if self.kind == "static":
            if self.target_replicas is None:
                raise ValueError("static manager needs target_replicas")
            return StaticController(self.target_replicas, replicas, action_contract=contract)
        return ThresholdController(
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            min_replicas=replicas.min_replicas,
            max_replicas=replicas.max_replicas,
            cooldown_ticks=self.cooldown_ticks,
            action_contract=contract,
        )


class ScenarioSession:
    """One interactive episode, stepped one tick at a time.

    Construction validates everything (config, trace timing, manager) before
    any state exists, so a failed build never replaces a working session.
    """

    def __init__(self, scenario: Scenario, config: SimulatorConfig, manager: ManagerSpec) -> None:
        self._env = AutoscalingEnv(config, scenario.trace)
        self._controller = manager.build(config)
        self._scenario = scenario
        self._manager = manager
        self.reset()

    # --- state ---------------------------------------------------------------

    @property
    def env(self) -> AutoscalingEnv:
        return self._env

    @property
    def scenario(self) -> ScenarioInfo:
        return self._scenario.info

    @property
    def trace(self) -> WorkloadTrace:
        return self._scenario.trace

    @property
    def config(self) -> SimulatorConfig:
        return self._env.config

    @property
    def manager(self) -> ManagerSpec:
        return self._manager

    @property
    def controller(self) -> Controller | None:
        return self._controller

    @property
    def observation(self) -> Observation:
        return self._observation

    @property
    def latest_info(self) -> dict[str, Any]:
        """The latest ``reset()``/``step()`` info: values for the last completed tick."""
        return self._info

    @property
    def fleet(self) -> dict[str, int]:
        """Current replica counts, i.e. what the next decision applies to."""
        return self._env.replica_counts

    @property
    def history(self) -> list[dict[str, Any]]:
        """One row per completed tick: the step's raw ``info`` plus presentation fields."""
        return list(self._history)

    @property
    def tick(self) -> int:
        return len(self._history)

    @property
    def episode_ticks(self) -> int:
        return self._env.episode_ticks

    @property
    def simulated_seconds(self) -> float:
        return self.tick * self.config.timing.control_interval_seconds

    @property
    def done(self) -> bool:
        return self._done

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def predictive_decision(self) -> PredictiveDecision | None:
        """The predictive controller's own diagnostics for its latest decision."""
        if isinstance(self._controller, PredictiveController):
            return self._controller.last_decision
        return None

    @property
    def threshold_decision(self) -> ThresholdDecision | None:
        """The threshold controller's own diagnostics for its latest decision."""
        if isinstance(self._controller, ThresholdController):
            return self._controller.last_decision
        return None

    # --- actions -------------------------------------------------------------

    def reset(self) -> None:
        """Restart the episode from tick 0 and clear all presentation state."""
        self._observation, self._info = self._env.reset(seed=self._manager.seed)
        if self._controller is not None:
            self._controller.reset(seed=self._manager.seed)
        self._history: list[dict[str, Any]] = []
        self._cumulative_cost = 0.0
        self._done = False

    def step_manual(self, action: int) -> dict[str, Any]:
        """Apply one manual action; only valid in manual mode."""
        if self._controller is not None:
            raise RuntimeError(f"{self._manager.kind} manager is driving this session")
        return self._apply(action)

    def step_controller(self) -> dict[str, Any]:
        """Let the controller decide one tick using the runner's decision contract."""
        if self._controller is None:
            raise RuntimeError("manual session: use step_manual")
        self._require_running()
        action = self._controller.act(self._observation, decision_info(self._env, self._info))
        return self._apply(action)

    def run_to_end(self) -> None:
        """Let the controller decide every remaining tick."""
        while not self._done:
            self.step_controller()

    def _apply(self, action: int) -> dict[str, Any]:
        self._require_running()
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._observation, self._info = observation, info
        self._cumulative_cost += info["infrastructure_cost"]
        self._done = terminated or truncated
        fleet = self._env.replica_counts
        row = {
            **info,
            "sla_target_seconds": self.config.sla.latency_target_seconds,
            "cumulative_cost": self._cumulative_cost,
            "next_active_replicas": fleet["active_replicas"],
            "next_pending_replicas": fleet["pending_replicas"],
        }
        self._history.append(row)
        return row

    def _require_running(self) -> None:
        if self._done:
            raise RuntimeError("episode complete; build or reset the scenario")


def azure_capacity_note(session: ScenarioSession) -> str | None:
    """Explain when one replica already covers an Azure trace's peak demand."""
    if session.scenario.source != "azure":
        return None
    peak = max(session.trace.request_rates)
    capacity = session.config.replicas.service_capacity_rps
    if peak > capacity:
        return None
    return (
        f"This Azure window peaks at {peak:.2f} RPS, while one replica serves "
        f"{capacity:g} RPS, so a single replica handles all traffic and scaling may look "
        "uneventful. Demand is shown unscaled. Lowering service capacity here is exploratory "
        "only; formal calibration must use train/validation workloads (see docs/EXPERIMENTS.md)."
    )


def icon_row(icon: str, count: float, *, cap: int = 10) -> str:
    """Up to ``cap`` icons for ``count`` (rounded), with the remainder as ``+N``."""
    whole = max(0, round(count))
    shown = min(whole, cap)
    extra = whole - shown
    return icon * shown + (f" +{extra}" if extra else "")
