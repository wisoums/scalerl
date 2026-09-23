"""Tests for per-tick latency, SLA, and infrastructure-cost metrics."""

import math
from typing import Any

import pytest

from scalerl.environment import (
    QueueStepResult,
    ReplicaConfig,
    RequestQueue,
    SimulatorConfig,
    SlaConfig,
    TickMetrics,
    TimingConfig,
    compute_tick_metrics,
)

# 50 rps per replica -> service time 0.02 s; 10 s ticks -> 500 requests per replica per tick.
SERVICE_TIME = 0.02


def make_config(
    *,
    service_capacity_rps: float = 50,
    control_interval_seconds: float = 10,
    cost_per_hour: float = 0.36,
    latency_target_seconds: float = 0.5,
) -> SimulatorConfig:
    return SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=control_interval_seconds),
        replicas=ReplicaConfig(
            service_capacity_rps=service_capacity_rps, cost_per_hour=cost_per_hour
        ),
        sla=SlaConfig(latency_target_seconds=latency_target_seconds),
    )


def measure(
    processed: float,
    queued: float,
    active: int,
    pending: int = 0,
    config: SimulatorConfig | None = None,
) -> TickMetrics:
    result = QueueStepResult(
        arrived_requests=processed + queued,
        processed_requests=processed,
        queued_requests=queued,
        dropped_requests=0.0,
    )
    return compute_tick_metrics(
        result,
        active_replicas=active,
        pending_replicas=pending,
        config=config or make_config(),
    )


# --- latency ----------------------------------------------------------------


def test_underloaded_tick_has_low_latency_and_meets_sla() -> None:
    metrics = measure(processed=200, queued=0, active=2)  # 20% of 1000 capacity

    assert metrics.utilization == pytest.approx(0.2)
    assert metrics.p95_latency_seconds == pytest.approx(SERVICE_TIME / 0.8)
    assert not metrics.sla_violated


def test_idle_tick_latency_is_nominal_service_time() -> None:
    metrics = measure(processed=0, queued=0, active=3)

    assert metrics.utilization == 0.0
    assert metrics.p95_latency_seconds == pytest.approx(SERVICE_TIME)


def test_latency_rises_as_utilization_approaches_saturation() -> None:
    latencies = [
        measure(processed=p, queued=0, active=2).p95_latency_seconds for p in (0, 500, 900, 990)
    ]

    assert latencies == sorted(latencies)
    assert len(set(latencies)) == len(latencies)


def test_saturation_factor_is_capped_to_stay_finite() -> None:
    at_capacity = measure(processed=1000, queued=0, active=2)

    assert at_capacity.utilization == 1.0
    assert at_capacity.p95_latency_seconds == pytest.approx(SERVICE_TIME * 20)
    assert measure(processed=960, queued=0, active=2).p95_latency_seconds == pytest.approx(
        at_capacity.p95_latency_seconds
    )


def test_queued_work_adds_drain_time_and_violates_sla() -> None:
    metrics = measure(processed=1000, queued=500, active=2)  # drains at 100 rps -> 5 s

    assert metrics.utilization == 1.0
    assert metrics.p95_latency_seconds == pytest.approx(SERVICE_TIME * 20 + 5.0)
    assert metrics.sla_violated


def test_latency_grows_with_sustained_overload() -> None:
    config = make_config()
    queue = RequestQueue(config.replicas, config.timing)

    latencies = []
    for _ in range(4):
        result = queue.step(request_rate=150, active_replicas=2)
        latencies.append(
            compute_tick_metrics(result, active_replicas=2, pending_replicas=0, config=config)
        )

    values = [m.p95_latency_seconds for m in latencies]
    assert values == sorted(values) and len(set(values)) == 4
    assert all(m.sla_violated for m in latencies)


def test_latency_recovers_once_backlog_drains() -> None:
    config = make_config()
    queue = RequestQueue(config.replicas, config.timing)
    queue.step(request_rate=300, active_replicas=2)  # 2000 queued

    history = []
    for _ in range(3):
        result = queue.step(request_rate=50, active_replicas=4)  # spare capacity 1500 per tick
        history.append(
            compute_tick_metrics(result, active_replicas=4, pending_replicas=0, config=config)
        )

    assert history[0].sla_violated
    assert history[0].p95_latency_seconds > history[1].p95_latency_seconds
    assert not history[1].sla_violated and not history[2].sla_violated
    assert history[2].p95_latency_seconds == pytest.approx(SERVICE_TIME / (1 - 0.25))


# --- zero active replicas ---------------------------------------------------


def test_zero_active_replicas_while_idle_is_a_valid_idle_state() -> None:
    metrics = measure(processed=0, queued=0, active=0, pending=1)

    assert metrics.utilization == 0.0
    assert metrics.p95_latency_seconds == pytest.approx(SERVICE_TIME)
    assert not metrics.sla_violated


def test_zero_active_replicas_with_work_is_finite_saturated_and_violating() -> None:
    metrics = measure(processed=0, queued=100, active=0)

    # Saturated service time + one full tick of waiting + drain at one replica's rate.
    assert metrics.utilization == 1.0
    assert metrics.p95_latency_seconds == pytest.approx(SERVICE_TIME * 20 + 10 + 100 / 50)
    assert math.isfinite(metrics.p95_latency_seconds)
    assert metrics.sla_violated


def test_zero_active_replicas_with_work_violates_even_a_loose_sla() -> None:
    config = make_config(latency_target_seconds=1000)

    assert measure(processed=0, queued=0.001, active=0, config=config).sla_violated


# --- SLA boundary -----------------------------------------------------------


def test_latency_equal_to_target_is_not_a_violation() -> None:
    at_target = make_config(latency_target_seconds=SERVICE_TIME)
    below_target = make_config(latency_target_seconds=SERVICE_TIME - 1e-6)

    assert not measure(processed=0, queued=0, active=1, config=at_target).sla_violated
    assert measure(processed=0, queued=0, active=1, config=below_target).sla_violated


# --- cost -------------------------------------------------------------------


def test_cost_bills_active_and_pending_replica_hours() -> None:
    # 0.36 per replica-hour * 10 s / 3600 s = 0.001 per replica per tick.
    active_only = measure(processed=0, queued=0, active=3)
    with_pending = measure(processed=0, queued=0, active=3, pending=2)

    assert active_only.infrastructure_cost == pytest.approx(0.003)
    assert with_pending.infrastructure_cost == pytest.approx(0.005)


def test_cost_over_an_hour_equals_hourly_price() -> None:
    config = make_config(control_interval_seconds=3600, cost_per_hour=0.36)

    assert measure(processed=0, queued=0, active=1, config=config).infrastructure_cost == 0.36


def test_cost_does_not_depend_on_load() -> None:
    idle = measure(processed=0, queued=0, active=2)
    overloaded = measure(processed=1000, queued=5000, active=2)

    assert idle.infrastructure_cost == overloaded.infrastructure_cost


@pytest.mark.parametrize(("cost_per_hour", "active", "pending"), [(0.0, 5, 3), (0.36, 0, 0)])
def test_cost_is_exactly_zero_without_price_or_billable_replicas(
    cost_per_hour: float, active: int, pending: int
) -> None:
    config = make_config(cost_per_hour=cost_per_hour)

    assert measure(0, 0, active=active, pending=pending, config=config).infrastructure_cost == 0.0


# --- invariants and validation ----------------------------------------------


@pytest.mark.parametrize("active", [0, 1, 4])
@pytest.mark.parametrize(("processed", "queued"), [(0, 0), (0, 300), (400, 0), (2000, 10_000)])
def test_outputs_are_finite_and_non_negative(active: int, processed: float, queued: float) -> None:
    processed = min(processed, active * 500)
    metrics = measure(processed=processed, queued=queued, active=active, pending=2)

    for value in (metrics.utilization, metrics.p95_latency_seconds, metrics.infrastructure_cost):
        assert math.isfinite(value) and value >= 0
    assert 0 <= metrics.utilization <= 1


def test_metrics_are_deterministic() -> None:
    args: dict[str, Any] = {"processed": 700, "queued": 250, "active": 2, "pending": 1}

    assert measure(**args) == measure(**args)


@pytest.mark.parametrize("field", ["active_replicas", "pending_replicas"])
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (-1, ValueError, "must be non-negative"),
        (1.0, TypeError, "must be an integer"),
        (True, TypeError, "must be an integer"),
    ],
)
def test_invalid_replica_counts_are_rejected(
    field: str, value: object, error: type[Exception], message: str
) -> None:
    counts: dict[str, Any] = {"active_replicas": 1, "pending_replicas": 0, field: value}
    result = QueueStepResult(0.0, 0.0, 0.0, 0.0)
    config = make_config()

    with pytest.raises(error, match=f"{field} {message}"):
        compute_tick_metrics(result, config=config, **counts)


def test_cost_overflow_is_rejected() -> None:
    config = make_config(cost_per_hour=1e308, control_interval_seconds=3600)

    with pytest.raises(ValueError, match="tick metrics overflow"):
        measure(processed=0, queued=0, active=2, config=config)
