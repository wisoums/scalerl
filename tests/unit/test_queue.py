"""Tests for the request queue and service-capacity model."""

import math

import pytest

from scalerl.environment import QueueStepResult, ReplicaConfig, RequestQueue, TimingConfig


def make_queue(
    service_capacity_rps: float = 50, control_interval_seconds: float = 10
) -> RequestQueue:
    """Default: each active replica serves 50 rps * 10 s = 500 requests per tick."""
    return RequestQueue(
        ReplicaConfig(service_capacity_rps=service_capacity_rps),
        TimingConfig(control_interval_seconds=control_interval_seconds),
    )


def test_queue_starts_empty() -> None:
    assert make_queue().queue_depth == 0.0


def test_under_capacity_demand_is_fully_processed() -> None:
    queue = make_queue()

    result = queue.step(request_rate=80, active_replicas=2)  # 800 arrive, capacity 1000

    assert result == QueueStepResult(
        arrived_requests=800.0, processed_requests=800.0, queued_requests=0.0, dropped_requests=0.0
    )
    assert queue.queue_depth == 0.0


def test_demand_exactly_at_capacity_leaves_no_queue() -> None:
    result = make_queue().step(request_rate=100, active_replicas=2)

    assert (result.processed_requests, result.queued_requests) == (1000.0, 0.0)


def test_capacity_scales_with_config_and_active_replicas() -> None:
    queue = make_queue(service_capacity_rps=20, control_interval_seconds=30)  # 600 per replica

    result = queue.step(request_rate=1000, active_replicas=3)

    assert result.arrived_requests == 30_000.0
    assert result.processed_requests == 1800.0


def test_overload_queues_excess_without_dropping() -> None:
    queue = make_queue()

    result = queue.step(request_rate=130, active_replicas=2)  # 1300 arrive, capacity 1000

    assert result.processed_requests == 1000.0
    assert result.queued_requests == 300.0
    assert result.dropped_requests == 0.0
    assert queue.queue_depth == 300.0


def test_sustained_overload_grows_queue_linearly() -> None:
    queue = make_queue()

    depths = [queue.step(request_rate=150, active_replicas=2).queued_requests for _ in range(5)]

    assert depths == [500.0, 1000.0, 1500.0, 2000.0, 2500.0]


def test_backlog_is_served_before_it_can_grow_and_recovers_with_spare_capacity() -> None:
    queue = make_queue()
    queue.step(request_rate=250, active_replicas=2)  # queue 1500

    results = [queue.step(request_rate=50, active_replicas=4) for _ in range(3)]

    # 500 new arrivals per tick against capacity 2000: 1500 spare, then empty.
    assert [r.processed_requests for r in results] == [2000.0, 500.0, 500.0]
    assert [r.queued_requests for r in results] == [0.0, 0.0, 0.0]


def test_partial_recovery_drains_backlog_over_several_ticks() -> None:
    queue = make_queue()
    queue.step(request_rate=400, active_replicas=1)  # queue 3500

    depths = [queue.step(request_rate=0, active_replicas=2).queued_requests for _ in range(4)]

    assert depths == [2500.0, 1500.0, 500.0, 0.0]


def test_zero_active_replicas_queue_all_arrivals() -> None:
    queue = make_queue()

    first = queue.step(request_rate=10, active_replicas=0)
    second = queue.step(request_rate=10, active_replicas=0)

    assert first.processed_requests == second.processed_requests == 0.0
    assert (first.queued_requests, second.queued_requests) == (100.0, 200.0)


def test_zero_demand_with_empty_queue_is_idle() -> None:
    result = make_queue().step(request_rate=0, active_replicas=3)

    assert result == QueueStepResult(0.0, 0.0, 0.0, 0.0)


def test_request_volumes_are_continuous() -> None:
    queue = make_queue(service_capacity_rps=1.5, control_interval_seconds=1)

    result = queue.step(request_rate=2.25, active_replicas=1)

    assert result.processed_requests == 1.5
    assert result.queued_requests == pytest.approx(0.75)


def test_work_is_conserved_across_ticks() -> None:
    queue = make_queue()
    rates_and_replicas = [(300, 1), (20, 3), (180, 2), (0, 0), (90, 1), (10, 5)]

    results = [queue.step(rate, replicas) for rate, replicas in rates_and_replicas]

    arrived = sum(r.arrived_requests for r in results)
    processed = sum(r.processed_requests for r in results)
    assert arrived == pytest.approx(processed + queue.queue_depth)
    for result in results:
        values = (
            result.arrived_requests,
            result.processed_requests,
            result.queued_requests,
            result.dropped_requests,
        )
        assert all(math.isfinite(v) and v >= 0 for v in values)


def test_reset_clears_queued_work() -> None:
    queue = make_queue()
    queue.step(request_rate=500, active_replicas=0)

    queue.reset()

    assert queue.queue_depth == 0.0
    assert queue.step(request_rate=10, active_replicas=1).queued_requests == 0.0


def test_identical_inputs_produce_identical_results() -> None:
    inputs = [(120, 2), (300, 1), (0, 3), (75.5, 2), (999, 0), (50, 10)]

    def run(queue: RequestQueue) -> list[QueueStepResult]:
        return [queue.step(rate, replicas) for rate, replicas in inputs]

    queue = make_queue()
    first = run(queue)
    queue.reset()

    assert run(queue) == first == run(make_queue())


@pytest.mark.parametrize("rate", [-1, -1e-9, math.nan, math.inf, -math.inf])
def test_invalid_request_rate_is_rejected_without_changing_state(rate: float) -> None:
    queue = make_queue()
    queue.step(request_rate=200, active_replicas=1)

    with pytest.raises(ValueError, match="request_rate must be finite and non-negative"):
        queue.step(request_rate=rate, active_replicas=1)

    assert queue.queue_depth == 1500.0


def test_negative_active_replicas_are_rejected() -> None:
    queue = make_queue()

    with pytest.raises(ValueError, match="active_replicas must be non-negative"):
        queue.step(request_rate=10, active_replicas=-1)


@pytest.mark.parametrize("replicas", [1.0, "1", True])
def test_active_replicas_must_be_an_integer(replicas: object) -> None:
    queue = make_queue()

    with pytest.raises(TypeError, match="active_replicas must be an integer"):
        queue.step(request_rate=10, active_replicas=replicas)  # type: ignore[arg-type]
