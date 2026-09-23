"""Tests for synthetic workload generators."""

import math
import random
from collections.abc import Callable
from itertools import groupby
from typing import Any

import numpy as np
import pytest

from scalerl.workloads import (
    WorkloadTrace,
    bursty_workload,
    diurnal_workload,
    ramp_workload,
    spike_workload,
    steady_workload,
)

Generator = Callable[..., WorkloadTrace]

# Minimal valid shape parameters for each generator, independent of timing.
GENERATORS: dict[str, tuple[Generator, dict[str, Any]]] = {
    "steady": (steady_workload, {"rate": 10.0}),
    "diurnal": (diurnal_workload, {"baseline": 50.0, "amplitude": 50.0, "period_seconds": 600}),
    "ramp": (ramp_workload, {"start_rate": 5.0, "end_rate": 100.0}),
    "spike": (
        spike_workload,
        {
            "baseline": 10.0,
            "spike_rate": 80.0,
            "spike_start_seconds": 60,
            "spike_duration_seconds": 60,
        },
    ),
    "bursty": (
        bursty_workload,
        {
            "baseline": 2.0,
            "burst_rate": 40.0,
            "burst_probability": 0.2,
            "burst_duration_seconds": 60,
            "noise_std": 5.0,
            "seed": 7,
        },
    ),
}


def generate(
    name: str, duration: float = 1200, interval: float = 30, **overrides: Any
) -> WorkloadTrace:
    generator, params = GENERATORS[name]
    return generator(
        duration_seconds=duration, control_interval_seconds=interval, **{**params, **overrides}
    )


# --- common behavior --------------------------------------------------------


@pytest.mark.parametrize("name", GENERATORS)
def test_generators_return_one_valid_sample_per_tick(name: str) -> None:
    trace = generate(name, duration=1200, interval=30)

    assert isinstance(trace, WorkloadTrace)
    assert len(trace) == 40
    assert trace.control_interval_seconds == 30
    assert all(math.isfinite(rate) and rate >= 0 for rate in trace.request_rates)


@pytest.mark.parametrize("name", GENERATORS)
def test_generators_are_repeatable(name: str) -> None:
    assert generate(name) == generate(name)


def test_tick_count_tolerates_float_rounding() -> None:
    trace = steady_workload(duration_seconds=0.3, control_interval_seconds=0.1, rate=1)

    assert len(trace) == 3


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("duration", [0, -60, math.nan, math.inf])
def test_invalid_duration_is_rejected(name: str, duration: float) -> None:
    with pytest.raises(ValueError, match="duration_seconds must be finite"):
        generate(name, duration=duration)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("duration", [1210, 15])
def test_partial_tick_duration_is_rejected(name: str, duration: float) -> None:
    with pytest.raises(ValueError, match="duration_seconds must be a whole number"):
        generate(name, duration=duration, interval=30)


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("interval", [0, -30, math.nan, math.inf])
def test_invalid_control_interval_is_rejected(name: str, interval: float) -> None:
    with pytest.raises(ValueError, match="control_interval_seconds must be finite"):
        generate(name, interval=interval)


@pytest.mark.parametrize(
    ("name", "overrides", "message"),
    [
        ("steady", {"rate": -1}, "rate must be finite and non-negative"),
        ("steady", {"rate": math.nan}, "rate must be finite and non-negative"),
        ("diurnal", {"baseline": -1, "amplitude": 0}, "baseline must be finite"),
        ("diurnal", {"amplitude": -1}, "amplitude must be finite"),
        ("diurnal", {"amplitude": 60}, "amplitude must not exceed baseline"),
        ("diurnal", {"period_seconds": 0}, "period_seconds must be finite"),
        ("diurnal", {"phase_seconds": math.inf}, "phase_seconds must be finite"),
        ("ramp", {"start_rate": -1}, "start_rate must be finite"),
        ("ramp", {"end_rate": math.inf}, "end_rate must be finite"),
        ("spike", {"baseline": -1}, "baseline must be finite"),
        ("spike", {"spike_rate": -1}, "spike_rate must be finite"),
        ("spike", {"spike_start_seconds": -30}, "spike_start_seconds must be finite"),
        ("spike", {"spike_start_seconds": 45}, "spike_start_seconds must be a whole number"),
        ("spike", {"spike_duration_seconds": 0}, "spike_duration_seconds must be finite"),
        ("spike", {"spike_duration_seconds": 45}, "spike_duration_seconds must be a whole"),
        ("spike", {"spike_start_seconds": 1170}, "spike window must end within"),
        ("bursty", {"baseline": -1}, "baseline must be finite"),
        ("bursty", {"burst_rate": -1}, "burst_rate must be finite"),
        ("bursty", {"burst_probability": -0.1}, "burst_probability must be between"),
        ("bursty", {"burst_probability": 1.1}, "burst_probability must be between"),
        ("bursty", {"burst_probability": math.nan}, "burst_probability must be finite"),
        ("bursty", {"burst_duration_seconds": 0}, "burst_duration_seconds must be finite"),
        ("bursty", {"burst_duration_seconds": 45}, "burst_duration_seconds must be a whole"),
        ("bursty", {"noise_std": -1}, "noise_std must be finite"),
        ("bursty", {"seed": -1}, "seed must be a non-negative integer"),
        ("bursty", {"seed": 1.5}, "seed must be a non-negative integer"),
        ("bursty", {"seed": True}, "seed must be a non-negative integer"),
    ],
)
def test_invalid_shape_parameters_are_rejected(
    name: str, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate(name, **overrides)


# --- steady -----------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 42.5])
def test_steady_is_constant(rate: float) -> None:
    assert set(generate("steady", rate=rate).request_rates) == {rate}


# --- diurnal ----------------------------------------------------------------


def test_diurnal_oscillates_around_baseline_with_given_period_and_amplitude() -> None:
    # 2 periods of 3600 s at 60 s ticks; the sine peaks at t = 900 s (tick 15).
    trace = generate(
        "diurnal", duration=7200, interval=60, baseline=100, amplitude=40, period_seconds=3600
    )
    rates = np.array(trace.request_rates)

    assert rates.mean() == pytest.approx(100)
    assert rates.max() == pytest.approx(140)
    assert rates.min() == pytest.approx(60)
    assert int(rates.argmax()) == 15
    np.testing.assert_allclose(rates[:60], rates[60:])


def test_diurnal_phase_shifts_the_cycle() -> None:
    trace = generate(
        "diurnal",
        duration=3600,
        interval=60,
        baseline=100,
        amplitude=40,
        period_seconds=3600,
        phase_seconds=900,
    )

    assert trace.demand_at(0) == pytest.approx(140)


def test_diurnal_full_amplitude_touches_zero_without_going_negative() -> None:
    trace = generate(
        "diurnal", duration=3600, interval=60, baseline=50, amplitude=50, period_seconds=3600
    )

    assert min(trace.request_rates) == pytest.approx(0, abs=1e-9)
    assert min(trace.request_rates) >= 0


# --- ramp -------------------------------------------------------------------


@pytest.mark.parametrize(("start", "end"), [(5.0, 100.0), (100.0, 5.0), (20.0, 20.0)])
def test_ramp_is_linear_between_endpoints(start: float, end: float) -> None:
    rates = np.array(generate("ramp", start_rate=start, end_rate=end).request_rates)

    assert rates[0] == start
    assert rates[-1] == end
    np.testing.assert_allclose(np.diff(rates), (end - start) / (len(rates) - 1), atol=1e-9)


def test_single_tick_ramp_uses_start_rate() -> None:
    trace = generate("ramp", duration=30, interval=30, start_rate=5, end_rate=100)

    assert trace.request_rates == (5.0,)


# --- spike ------------------------------------------------------------------


def test_spike_raises_demand_only_inside_window() -> None:
    trace = generate(
        "spike", duration=300, interval=30, spike_start_seconds=90, spike_duration_seconds=60
    )

    assert trace.request_rates == (10.0, 10.0, 10.0, 80.0, 80.0, 10.0, 10.0, 10.0, 10.0, 10.0)


@pytest.mark.parametrize(("start", "length"), [(0, 60), (240, 60), (0, 300)])
def test_spike_window_may_touch_trace_edges(start: float, length: float) -> None:
    trace = generate(
        "spike", duration=300, interval=30, spike_start_seconds=start, spike_duration_seconds=length
    )

    spike_ticks = [tick for tick, rate in enumerate(trace.request_rates) if rate == 80.0]
    assert spike_ticks == list(range(int(start // 30), int((start + length) // 30)))


# --- bursty -----------------------------------------------------------------


def test_bursty_same_seed_is_identical_and_different_seeds_differ() -> None:
    train = generate("bursty", duration=3600, seed=1)

    assert generate("bursty", duration=3600, seed=1) == train
    assert generate("bursty", duration=3600, seed=2) != train


def test_bursty_does_not_touch_global_random_state() -> None:
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), np.random.random())

    random.seed(123)
    np.random.seed(123)
    generate("bursty", seed=99)

    assert (random.random(), np.random.random()) == expected


def test_bursty_without_noise_alternates_between_baseline_and_bursts() -> None:
    trace = generate(
        "bursty",
        duration=6000,
        interval=30,
        noise_std=0,
        burst_probability=0.05,
        burst_duration_seconds=90,
        seed=3,
    )

    assert set(trace.request_rates) == {2.0, 42.0}
    burst_runs = [len(list(run)) for rate, run in groupby(trace.request_rates) if rate == 42.0]
    # Every burst lasts at least its duration, unless the trace ends mid-burst.
    assert all(length >= 3 for length in burst_runs[:-1])


@pytest.mark.parametrize(("probability", "expected"), [(0.0, 2.0), (1.0, 42.0)])
def test_bursty_probability_extremes(probability: float, expected: float) -> None:
    trace = generate("bursty", noise_std=0, burst_probability=probability)

    assert set(trace.request_rates) == {expected}


def test_bursty_intensity_grows_with_burst_probability() -> None:
    def mean_rate(probability: float) -> float:
        trace = generate("bursty", duration=36_000, burst_probability=probability, seed=5)
        return float(np.mean(trace.request_rates))

    assert mean_rate(0.01) < mean_rate(0.1) < mean_rate(0.5)


def test_bursty_noise_is_clipped_at_zero() -> None:
    trace = generate("bursty", duration=3600, baseline=1, noise_std=50, burst_probability=0)

    assert min(trace.request_rates) == 0.0
    assert len(set(trace.request_rates)) > 1
