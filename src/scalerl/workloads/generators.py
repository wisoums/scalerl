"""Synthetic request-rate workload generators.

Every generator returns a :class:`WorkloadTrace` with exactly
``duration_seconds / control_interval_seconds`` ticks. Tick ``i`` starts at
``i * control_interval_seconds``. Rates are in requests per second.
"""

from __future__ import annotations

import math
import random

from scalerl.workloads.trace import WorkloadTrace


def steady_workload(
    *, duration_seconds: float, control_interval_seconds: float, rate: float
) -> WorkloadTrace:
    """Constant demand of ``rate`` at every tick."""
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    rate = _non_negative("rate", rate)
    return WorkloadTrace([rate] * ticks, control_interval_seconds)


def diurnal_workload(
    *,
    duration_seconds: float,
    control_interval_seconds: float,
    baseline: float,
    amplitude: float,
    period_seconds: float = 86_400.0,
    phase_seconds: float = 0.0,
) -> WorkloadTrace:
    """Sinusoidal demand: ``baseline + amplitude * sin(2π (t + phase) / period)``.

    ``amplitude`` may not exceed ``baseline``, so demand never goes negative.
    """
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    baseline = _non_negative("baseline", baseline)
    amplitude = _non_negative("amplitude", amplitude)
    period_seconds = _positive("period_seconds", period_seconds)
    phase_seconds = _finite("phase_seconds", phase_seconds)
    if amplitude > baseline:
        raise ValueError("amplitude must not exceed baseline")

    interval = float(control_interval_seconds)
    rates = [
        baseline
        + amplitude * math.sin(2 * math.pi * (tick * interval + phase_seconds) / period_seconds)
        for tick in range(ticks)
    ]
    # Clamp float rounding at the trough when amplitude == baseline.
    return WorkloadTrace([max(rate, 0.0) for rate in rates], control_interval_seconds)


def ramp_workload(
    *,
    duration_seconds: float,
    control_interval_seconds: float,
    start_rate: float,
    end_rate: float,
) -> WorkloadTrace:
    """Linear demand from ``start_rate`` at the first tick to ``end_rate`` at the last.

    ``end_rate`` may be lower than ``start_rate`` for a ramp-down.
    """
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    start_rate = _non_negative("start_rate", start_rate)
    end_rate = _non_negative("end_rate", end_rate)
    if ticks == 1:
        return WorkloadTrace([start_rate], control_interval_seconds)

    step = (end_rate - start_rate) / (ticks - 1)
    rates = [start_rate + tick * step for tick in range(ticks - 1)] + [end_rate]
    return WorkloadTrace(rates, control_interval_seconds)


def spike_workload(
    *,
    duration_seconds: float,
    control_interval_seconds: float,
    baseline: float,
    spike_rate: float,
    spike_start_seconds: float,
    spike_duration_seconds: float,
) -> WorkloadTrace:
    """Demand of ``baseline`` except ``spike_rate`` during one window.

    The window ``[spike_start, spike_start + spike_duration)`` must align with
    tick boundaries and lie inside the trace.
    """
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    baseline = _non_negative("baseline", baseline)
    spike_rate = _non_negative("spike_rate", spike_rate)
    start = _whole_ticks(
        "spike_start_seconds", spike_start_seconds, control_interval_seconds, allow_zero=True
    )
    length = _whole_ticks(
        "spike_duration_seconds", spike_duration_seconds, control_interval_seconds
    )
    if start + length > ticks:
        raise ValueError("spike window must end within duration_seconds")

    rates = [baseline] * ticks
    rates[start : start + length] = [spike_rate] * length
    return WorkloadTrace(rates, control_interval_seconds)


def bursty_workload(
    *,
    duration_seconds: float,
    control_interval_seconds: float,
    baseline: float,
    burst_rate: float,
    burst_probability: float,
    burst_duration_seconds: float,
    seed: int,
    noise_std: float = 0.0,
) -> WorkloadTrace:
    """Noisy baseline with randomly timed bursts, drawn from a local seeded RNG.

    Each tick, a burst starts with probability ``burst_probability`` and adds
    ``burst_rate`` for ``burst_duration_seconds``; overlapping bursts do not
    stack. Gaussian noise with standard deviation ``noise_std`` is added to
    the baseline, and the noisy baseline is clipped at zero.
    """
    ticks = _tick_count(duration_seconds, control_interval_seconds)
    baseline = _non_negative("baseline", baseline)
    burst_rate = _non_negative("burst_rate", burst_rate)
    burst_probability = _finite("burst_probability", burst_probability)
    if not 0 <= burst_probability <= 1:
        raise ValueError("burst_probability must be between 0 and 1")
    burst_ticks = _whole_ticks(
        "burst_duration_seconds", burst_duration_seconds, control_interval_seconds
    )
    noise_std = _non_negative("noise_std", noise_std)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = random.Random(seed)
    rates = []
    burst_ticks_left = 0
    for _ in range(ticks):
        if rng.random() < burst_probability:
            burst_ticks_left = burst_ticks
        rate = max(baseline + rng.gauss(0.0, noise_std), 0.0)
        if burst_ticks_left > 0:
            rate += burst_rate
            burst_ticks_left -= 1
        rates.append(rate)
    return WorkloadTrace(rates, control_interval_seconds)


def _tick_count(duration_seconds: float, control_interval_seconds: float) -> int:
    interval = _positive("control_interval_seconds", control_interval_seconds)
    return _whole_ticks("duration_seconds", duration_seconds, interval)


def _whole_ticks(
    name: str, seconds: float, control_interval_seconds: float, *, allow_zero: bool = False
) -> int:
    """Convert ``seconds`` to a tick count, rejecting partial ticks."""
    seconds = _non_negative(name, seconds) if allow_zero else _positive(name, seconds)
    ratio = seconds / float(control_interval_seconds)
    ticks = round(ratio)
    if not math.isclose(ratio, ticks, rel_tol=1e-9) or (ticks == 0 and not allow_zero):
        raise ValueError(f"{name} must be a whole number of control intervals")
    return ticks


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value
