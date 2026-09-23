"""Tests for workload traces and deterministic replay."""

import dataclasses

import pytest

from scalerl.workloads import WorkloadReplay, WorkloadTrace


def test_trace_stores_one_float_sample_per_tick() -> None:
    trace = WorkloadTrace([10, 20.5, 0], control_interval_seconds=30)

    assert trace.request_rates == (10.0, 20.5, 0.0)
    assert all(type(rate) is float for rate in trace.request_rates)
    assert len(trace) == 3
    assert trace.control_interval_seconds == 30.0
    assert trace.duration_seconds == 90.0


def test_trace_accepts_any_iterable_and_copies_it() -> None:
    rates = [1.0, 2.0]
    trace = WorkloadTrace((rate for rate in rates), control_interval_seconds=1)
    rates.append(3.0)

    assert trace.request_rates == (1.0, 2.0)


def test_trace_is_immutable() -> None:
    trace = WorkloadTrace([1.0], control_interval_seconds=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        trace.control_interval_seconds = 2.0  # type: ignore[misc]


def test_equal_traces_compare_equal() -> None:
    assert WorkloadTrace([1, 2], 5) == WorkloadTrace((1.0, 2.0), 5.0)
    assert WorkloadTrace([1, 2], 5) != WorkloadTrace([1, 2], 10)


def test_empty_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        WorkloadTrace([], control_interval_seconds=1)


@pytest.mark.parametrize("rate", [-1.0, -1e-9, float("nan"), float("inf"), float("-inf")])
def test_invalid_request_rates_are_rejected_with_tick(rate: float) -> None:
    with pytest.raises(ValueError, match="tick 1 must be finite and non-negative"):
        WorkloadTrace([5.0, rate, 5.0], control_interval_seconds=1)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_invalid_control_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="control_interval_seconds must be finite"):
        WorkloadTrace([1.0], control_interval_seconds=interval)


def test_demand_at_returns_sample_for_each_tick() -> None:
    trace = WorkloadTrace([3.0, 1.0, 4.0], control_interval_seconds=1)

    assert [trace.demand_at(tick) for tick in range(len(trace))] == [3.0, 1.0, 4.0]


@pytest.mark.parametrize("tick", [-1, 3, 100])
def test_demand_at_rejects_out_of_range_ticks_without_wrapping(tick: int) -> None:
    trace = WorkloadTrace([3.0, 1.0, 4.0], control_interval_seconds=1)

    with pytest.raises(IndexError, match=f"tick {tick} is outside trace of length 3"):
        trace.demand_at(tick)


@pytest.mark.parametrize("tick", [1.0, "1", True])
def test_demand_at_requires_integer_tick(tick: object) -> None:
    trace = WorkloadTrace([3.0, 1.0], control_interval_seconds=1)

    with pytest.raises(TypeError, match="tick must be an integer"):
        trace.demand_at(tick)  # type: ignore[arg-type]


def test_replay_yields_trace_in_order_then_is_exhausted() -> None:
    trace = WorkloadTrace([3.0, 1.0, 4.0], control_interval_seconds=1)
    replay = WorkloadReplay(trace)

    assert replay.trace is trace
    assert (replay.position, replay.remaining, replay.is_exhausted) == (0, 3, False)

    assert replay.next_demand() == 3.0
    assert (replay.position, replay.remaining) == (1, 2)

    assert [replay.next_demand(), replay.next_demand()] == [1.0, 4.0]
    assert (replay.position, replay.remaining, replay.is_exhausted) == (3, 0, True)


def test_replay_past_end_fails_without_changing_state() -> None:
    replay = WorkloadReplay(WorkloadTrace([7.0], control_interval_seconds=1))
    replay.next_demand()

    for _ in range(2):
        with pytest.raises(IndexError, match="exhausted"):
            replay.next_demand()

    assert replay.position == 1
    assert replay.is_exhausted


@pytest.mark.parametrize("consumed", [0, 2, 4])
def test_reset_restores_exact_initial_state(consumed: int) -> None:
    trace = WorkloadTrace([3.0, 1.0, 4.0, 1.0], control_interval_seconds=1)
    replay = WorkloadReplay(trace)
    for _ in range(consumed):
        replay.next_demand()

    replay.reset()

    assert (replay.position, replay.remaining, replay.is_exhausted) == (0, 4, False)
    assert replay.trace == WorkloadTrace([3.0, 1.0, 4.0, 1.0], control_interval_seconds=1)


def test_repeated_replays_after_reset_are_identical() -> None:
    trace = WorkloadTrace([0.0, 12.5, 50.0, 12.5, 3.25], control_interval_seconds=15)
    replay = WorkloadReplay(trace)

    runs = []
    for _ in range(3):
        runs.append([replay.next_demand() for _ in range(len(trace))])
        replay.reset()

    assert runs == [list(trace.request_rates)] * 3


def test_independent_replays_of_one_trace_do_not_interfere() -> None:
    trace = WorkloadTrace([1.0, 2.0, 3.0], control_interval_seconds=1)
    first = WorkloadReplay(trace)
    second = WorkloadReplay(trace)

    first.next_demand()
    first.next_demand()

    assert second.position == 0
    assert second.next_demand() == 1.0
    assert first.next_demand() == 3.0
