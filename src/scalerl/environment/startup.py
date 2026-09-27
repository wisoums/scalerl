"""Versioned replica startup-delay models (#81).

``fixed-v1`` (the default, and the only model before #81): every new replica
becomes ready after exactly ``startup_delay_seconds``; no random number is drawn.

``tri-point-multiplicative-v1``: a bounded, seeded **robustness stress model**, not a
provider-calibrated distribution. Each newly requested replica independently
draws a multiplier

    0.5 with probability 0.25, 1.0 with probability 0.50, 1.5 with probability 0.25

(expected multiplier exactly 1.0) and its realized startup delay is
``startup_delay_seconds * multiplier``; realized seconds are not rounded, and a
replica still only activates through the lifecycle's per-tick ``advance``. At
the benchmark's 60 s nominal delay and 30 s ticks the three outcomes are 30 /
60 / 90 s, i.e. ready after 1, 2, or 3 ticks. A continuous ±25% model would
mostly collapse into 2-vs-3 ticks at that cadence; three explicit points make
early / nominal / late readiness visible without claiming sub-tick calibration.

Draws come from a dedicated per-environment RNG seeded by
``DynamicsConfig.startup_delay_seed`` (separate from the capacity-jitter RNG),
consumed one draw per new replica in request order.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np

FIXED_V1: Final = "fixed-v1"
TRI_POINT_MULTIPLICATIVE_V1: Final = "tri-point-multiplicative-v1"
StartupDelayModel = Literal["fixed-v1", "tri-point-multiplicative-v1"]
STARTUP_DELAY_MODELS: Final[tuple[StartupDelayModel, ...]] = (FIXED_V1, TRI_POINT_MULTIPLICATIVE_V1)

TRI_POINT_MULTIPLIERS: Final = (0.5, 1.0, 1.5)
TRI_POINT_PROBABILITIES: Final = (0.25, 0.50, 0.25)
TRI_POINT_EXPECTED_MULTIPLIER: Final = sum(
    m * p for m, p in zip(TRI_POINT_MULTIPLIERS, TRI_POINT_PROBABILITIES, strict=True)
)


def draw_tri_point_multiplier(rng: np.random.Generator) -> float:
    """One ``tri-point-multiplicative-v1`` multiplier from exactly one uniform draw."""
    u = float(rng.random())
    cumulative = 0.0
    for multiplier, probability in zip(TRI_POINT_MULTIPLIERS, TRI_POINT_PROBABILITIES, strict=True):
        cumulative += probability
        if u < cumulative:
            return multiplier
    return TRI_POINT_MULTIPLIERS[-1]
