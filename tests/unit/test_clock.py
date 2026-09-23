"""Tests for the deterministic simulation clock."""

import pytest

from scalerl.environment import SimulationClock


def test_clock_starts_at_zero() -> None:
    clock = SimulationClock(tick_seconds=30)

    assert clock.step_count == 0
    assert clock.time_seconds == 0
    assert clock.tick_seconds == 30


def test_step_advances_by_one_tick_by_default() -> None:
    clock = SimulationClock(tick_seconds=30)

    result = clock.step()

    assert result == 30
    assert clock.step_count == 1
    assert clock.time_seconds == 30


def test_step_can_advance_multiple_ticks() -> None:
    clock = SimulationClock(tick_seconds=2.5)

    result = clock.step(4)

    assert result == 10
    assert clock.step_count == 4
    assert clock.time_seconds == 10


def test_reset_returns_clock_to_zero_without_changing_tick_duration() -> None:
    clock = SimulationClock(tick_seconds=15)
    clock.step(3)

    clock.reset()

    assert clock.step_count == 0
    assert clock.time_seconds == 0
    assert clock.tick_seconds == 15


@pytest.mark.parametrize("tick_seconds", [0, -1, -0.5, float("nan"), float("inf"), float("-inf")])
def test_tick_duration_must_be_finite_and_positive(tick_seconds: float) -> None:
    with pytest.raises(ValueError, match="tick_seconds must be finite and greater than zero"):
        SimulationClock(tick_seconds=tick_seconds)


@pytest.mark.parametrize("ticks", [0, -1, -10])
def test_step_count_must_be_positive(ticks: int) -> None:
    clock = SimulationClock(tick_seconds=1)

    with pytest.raises(ValueError, match="ticks must be greater than zero"):
        clock.step(ticks)


@pytest.mark.parametrize("ticks", [1.5, "2", True])
def test_step_requires_integer_ticks(ticks: object) -> None:
    clock = SimulationClock(tick_seconds=1)

    with pytest.raises(TypeError, match="ticks must be an integer"):
        clock.step(ticks)  # type: ignore[arg-type]


def test_clock_changes_only_when_explicitly_stepped() -> None:
    clock = SimulationClock(tick_seconds=30)

    before = clock.time_seconds
    _ = clock.tick_seconds
    _ = clock.step_count
    after = clock.time_seconds

    assert before == after == 0


def test_identical_operation_sequences_are_deterministic() -> None:
    first = SimulationClock(tick_seconds=7.5)
    second = SimulationClock(tick_seconds=7.5)

    first_times = [first.step(), first.step(2), first.step()]
    second_times = [second.step(), second.step(2), second.step()]

    assert first_times == second_times == [7.5, 22.5, 30.0]
    assert first.step_count == second.step_count == 4
