"""Tests for the deterministic replica lifecycle."""

from typing import Any

import pytest

from scalerl.environment import ReplicaConfig, ReplicaPool, startup_ticks


def make_pool(**overrides: Any) -> ReplicaPool:
    params: dict[str, Any] = {
        "min_replicas": 1,
        "max_replicas": 5,
        "initial_replicas": 2,
        "startup_delay_seconds": 60.0,
    }
    return ReplicaPool(ReplicaConfig(**{**params, **overrides}))


def counts(pool: ReplicaPool) -> tuple[int, int, int]:
    """Return (active, pending, terminating)."""
    return pool.active_count, pool.pending_count, pool.terminating_count


# --- initialization ---------------------------------------------------------


def test_pool_starts_with_initial_replicas_active() -> None:
    pool = make_pool(initial_replicas=3)

    assert counts(pool) == (3, 0, 0)
    assert pool.desired_count == 3


# --- scale-up and delayed activation ----------------------------------------


def test_scale_up_creates_pending_replicas_that_do_not_serve_yet() -> None:
    pool = make_pool()

    assert pool.scale_up(2) == 2
    assert counts(pool) == (2, 2, 0)
    assert pool.desired_count == 4


def test_pending_replicas_activate_only_after_startup_delay() -> None:
    pool = make_pool(startup_delay_seconds=60)
    pool.scale_up()

    pool.advance(30)
    assert counts(pool) == (2, 1, 0)

    pool.advance(29.999)
    assert counts(pool) == (2, 1, 0)

    pool.advance(0.001)
    assert counts(pool) == (3, 0, 0)


def test_one_long_advance_activates_pending_replicas() -> None:
    pool = make_pool(startup_delay_seconds=60)
    pool.scale_up(2)

    pool.advance(3600)

    assert counts(pool) == (4, 0, 0)


def test_replicas_requested_at_different_times_activate_in_order() -> None:
    pool = make_pool(startup_delay_seconds=60)
    pool.scale_up()
    pool.advance(30)
    pool.scale_up()

    pool.advance(30)
    assert counts(pool) == (3, 1, 0)

    pool.advance(30)
    assert counts(pool) == (4, 0, 0)


def test_activation_tolerates_float_rounding_of_short_ticks() -> None:
    pool = make_pool(startup_delay_seconds=0.3)
    pool.scale_up()

    for _ in range(3):
        pool.advance(0.1)  # 0.1 * 3 is slightly below 0.3 in floating point

    assert counts(pool) == (3, 0, 0)


def test_zero_startup_delay_activates_immediately() -> None:
    pool = make_pool(startup_delay_seconds=0)

    assert pool.scale_up(2) == 2
    assert counts(pool) == (4, 0, 0)


@pytest.mark.parametrize("delay", [0, 60])
def test_scale_up_is_clamped_to_max_replicas(delay: float) -> None:
    pool = make_pool(max_replicas=5, initial_replicas=2, startup_delay_seconds=delay)

    assert pool.scale_up(10) == 3
    assert pool.desired_count == 5
    assert pool.scale_up() == 0
    assert pool.desired_count == 5


# --- scale-down -------------------------------------------------------------


def test_scale_down_cancels_most_recent_pending_replica_first() -> None:
    pool = make_pool(startup_delay_seconds=60)
    pool.scale_up()  # older: will be ready at t=60
    pool.advance(30)
    pool.scale_up()  # newer: would be ready at t=90

    assert pool.scale_down() == 1
    assert counts(pool) == (2, 1, 0)

    pool.advance(30)  # the older request survives and activates on schedule
    assert counts(pool) == (3, 0, 0)


def test_cancelled_pending_replica_never_becomes_active() -> None:
    pool = make_pool()
    pool.scale_up()
    pool.scale_down()

    pool.advance(3600)

    assert counts(pool) == (2, 0, 0)


def test_scale_down_terminates_active_replicas_after_pending_are_cancelled() -> None:
    pool = make_pool(initial_replicas=3)
    pool.scale_up()

    assert pool.scale_down(2) == 2
    assert counts(pool) == (2, 0, 1)
    assert pool.desired_count == 2


def test_terminating_replicas_are_finalized_on_next_advance() -> None:
    pool = make_pool(initial_replicas=3)
    pool.scale_down()
    assert counts(pool) == (2, 0, 1)

    pool.advance(1)

    assert counts(pool) == (2, 0, 0)


@pytest.mark.parametrize("delay", [0, 60])
def test_scale_down_is_clamped_to_min_replicas(delay: float) -> None:
    pool = make_pool(min_replicas=2, initial_replicas=4, startup_delay_seconds=delay)

    assert pool.scale_down(10) == 2
    assert pool.desired_count == 2
    assert pool.scale_down() == 0
    assert counts(pool) == (2, 0, 2)


def test_scale_up_after_scale_down_requests_new_replicas() -> None:
    pool = make_pool(initial_replicas=3)
    pool.scale_down()
    pool.scale_up()

    assert counts(pool) == (2, 1, 1)


# --- bounds and invariants --------------------------------------------------


def test_bounds_hold_through_an_arbitrary_operation_sequence() -> None:
    pool = make_pool(min_replicas=2, max_replicas=6, initial_replicas=3)
    operations = [(pool.scale_up, 5), (pool.scale_down, 1), (pool.scale_down, 9)]
    operations += [(pool.scale_up, 2), (pool.scale_down, 1), (pool.scale_up, 7)]

    for operation, count in operations * 3:
        operation(count)
        pool.advance(30)
        assert 2 <= pool.desired_count <= 6
        assert min(counts(pool)) >= 0
        assert pool.desired_count == pool.active_count + pool.pending_count


@pytest.mark.parametrize("count", [0, -1])
def test_scale_counts_must_be_positive(count: int) -> None:
    pool = make_pool()

    for operation in (pool.scale_up, pool.scale_down):
        with pytest.raises(ValueError, match="count must be greater than zero"):
            operation(count)


@pytest.mark.parametrize("count", [1.0, "1", True])
def test_scale_counts_must_be_integers(count: object) -> None:
    pool = make_pool()

    for operation in (pool.scale_up, pool.scale_down):
        with pytest.raises(TypeError, match="count must be an integer"):
            operation(count)  # type: ignore[arg-type]


@pytest.mark.parametrize("elapsed", [0, -1, float("nan"), float("inf")])
def test_advance_requires_positive_finite_time(elapsed: float) -> None:
    pool = make_pool()
    pool.scale_up()

    with pytest.raises(ValueError, match="elapsed_seconds must be finite"):
        pool.advance(elapsed)

    assert counts(pool) == (2, 1, 0)


# --- readiness --------------------------------------------------------------


def test_pending_replicas_are_bucketed_by_advances_until_active() -> None:
    pool = make_pool(startup_delay_seconds=90)
    pool.scale_up()
    pool.advance(30)
    pool.scale_up(2)

    assert pool.pending_by_ticks_until_active(30) == (0, 1, 2)

    pool.advance(30)
    assert pool.pending_by_ticks_until_active(30) == (1, 2, 0)


def test_readiness_buckets_round_up_partial_intervals() -> None:
    pool = make_pool(startup_delay_seconds=45)
    pool.scale_up()

    assert pool.pending_by_ticks_until_active(30) == (0, 1)
    pool.advance(30)
    assert pool.pending_by_ticks_until_active(30) == (1, 0)
    pool.advance(30)
    assert pool.active_count == 3


def test_readiness_buckets_match_activation_under_float_rounding() -> None:
    pool = make_pool(startup_delay_seconds=0.3)
    pool.scale_up()
    pool.advance(0.1)
    pool.advance(0.1)

    assert pool.pending_by_ticks_until_active(0.1) == (1, 0, 0)
    pool.advance(0.1)
    assert pool.active_count == 3


def test_zero_startup_delay_has_no_readiness_buckets() -> None:
    pool = make_pool(startup_delay_seconds=0)
    pool.scale_up()

    assert pool.pending_by_ticks_until_active(30) == ()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_readiness_buckets_require_positive_finite_interval(interval: float) -> None:
    pool = make_pool()

    with pytest.raises(ValueError, match="interval_seconds must be finite"):
        pool.pending_by_ticks_until_active(interval)


@pytest.mark.parametrize(
    ("delay", "interval", "ticks"),
    [(0.0, 30.0, 0), (1.0, 30.0, 1), (30.0, 30.0, 1), (45.0, 30.0, 2), (60.0, 30.0, 2)],
)
def test_startup_ticks_matches_pending_buckets(delay: float, interval: float, ticks: int) -> None:
    assert startup_ticks(delay, interval) == ticks
    pool = make_pool(startup_delay_seconds=delay)
    assert len(pool.pending_by_ticks_until_active(interval)) == ticks


@pytest.mark.parametrize(
    ("delay", "interval", "message"),
    [
        (-1.0, 30.0, "startup_delay_seconds must be finite and non-negative"),
        (float("nan"), 30.0, "startup_delay_seconds must be finite and non-negative"),
        (float("inf"), 30.0, "startup_delay_seconds must be finite and non-negative"),
        (0.0, 0.0, "control_interval_seconds must be finite and greater than zero"),
        (60.0, 0.0, "control_interval_seconds must be finite and greater than zero"),
        (60.0, -30.0, "control_interval_seconds must be finite and greater than zero"),
        (60.0, float("inf"), "control_interval_seconds must be finite and greater than zero"),
    ],
)
def test_startup_ticks_rejects_invalid_timing(delay: float, interval: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        startup_ticks(delay, interval)


# --- reset and determinism --------------------------------------------------


def test_reset_restores_initial_state() -> None:
    pool = make_pool(initial_replicas=3)
    pool.scale_up(2)
    pool.scale_down(4)

    pool.reset()

    assert counts(pool) == (3, 0, 0)


def test_identical_operation_sequences_evolve_identically() -> None:
    def run(pool: ReplicaPool) -> list[tuple[int, int, int]]:
        history = []
        for step in range(40):
            if step % 7 == 0:
                pool.scale_up(3)
            elif step % 5 == 0:
                pool.scale_down(2)
            pool.advance(15)
            history.append(counts(pool))
        return history

    pool = make_pool(max_replicas=8, startup_delay_seconds=45)
    first = run(pool)
    pool.reset()

    assert run(pool) == first == run(make_pool(max_replicas=8, startup_delay_seconds=45))
