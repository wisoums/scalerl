"""Versioned action contracts: action codes vs. their effect on replica count (#79).

``AutoscalingEnv`` always takes an integer **action code** from a Gymnasium
``Discrete`` space. What a code *means* is the environment's action contract
(``SimulatorConfig.action.semantics``):

``delta-v1`` (the default, and the only contract before #79)
    ``Discrete(3)``; code 0 = scale down, 1 = hold, 2 = scale up. Their
    replica-count **effects** are -1 / 0 / +1. Effects are not codes:
    ``step(-1)`` is invalid.

``desired-replicas-v1``
    ``Discrete(max_replicas - min_replicas + 1)``; code ``c`` requests the
    integer fleet size ``min_replicas + c`` (benchmark defaults: codes 0..9 ->
    targets 1..10). Replica counts stay discrete.

Under both contracts the environment turns the code into a **requested target**
for committed capacity (``active + pending``, i.e. ``ReplicaPool.desired_count``)
and starts or cancels ``|target - committed|`` replicas in one step through the
normal lifecycle: new replicas start pending and wait out the startup delay,
and a reduction cancels the newest pending replicas before terminating active
ones. For ``delta-v1`` the target is ``committed + effect`` clipped to the
replica bounds, so it reproduces the pre-#79 behavior exactly.

:class:`ActionContract` is the one place that encodes and decodes codes, for
the environment and for rule-based controllers alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from scalerl.environment.config import (
    DELTA_V1,
    DESIRED_REPLICAS_V1,
    ActionSemantics,
    SimulatorConfig,
)

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
DELTA_EFFECTS: Final = {SCALE_DOWN: -1, HOLD: 0, SCALE_UP: 1}
ACTION_SEMANTICS: Final[tuple[ActionSemantics, ...]] = (DELTA_V1, DESIRED_REPLICAS_V1)


@dataclass(frozen=True, slots=True)
class ActionContract:
    """Encoding of one action contract for fixed replica bounds."""

    semantics: ActionSemantics
    min_replicas: int
    max_replicas: int

    def __post_init__(self) -> None:
        if self.semantics not in ACTION_SEMANTICS:
            raise ValueError(f"unknown action semantics {self.semantics!r}")
        for name in ("min_replicas", "max_replicas"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.min_replicas <= self.max_replicas:
            raise ValueError("replica bounds must satisfy 1 <= min_replicas <= max_replicas")

    @classmethod
    def from_config(cls, config: SimulatorConfig) -> ActionContract:
        return cls(
            config.action.semantics, config.replicas.min_replicas, config.replicas.max_replicas
        )

    @property
    def action_count(self) -> int:
        """Size of the ``Discrete`` action space."""
        if self.semantics == DELTA_V1:
            return 3
        return self.max_replicas - self.min_replicas + 1

    def validate_code(self, code: Any) -> int:
        """Return ``code`` as ``int`` if it is a valid action code, else raise ``ValueError``.

        Only integers (Python or NumPy) are accepted; ``bool``, floats and
        out-of-range values are rejected.
        """
        if isinstance(code, bool | np.bool_) or not isinstance(code, int | np.integer):
            raise ValueError(f"invalid action {code!r}; {self._expected()}")
        value = int(code)
        if not 0 <= value < self.action_count:
            raise ValueError(f"invalid action {code!r}; {self._expected()}")
        return value

    def target_for(self, code: int, committed: int) -> int:
        """Committed capacity that ``code`` requests when ``committed`` is current.

        Always within ``[min_replicas, max_replicas]``: ``delta-v1`` clips
        ``committed + effect``; ``desired-replicas-v1`` is ``min_replicas + code``.
        """
        value = self.validate_code(code)
        if self.semantics == DELTA_V1:
            return self._clip(committed + DELTA_EFFECTS[value])
        return self.min_replicas + value

    def code_for_target(self, target: int, committed: int) -> int:
        """The code that moves committed capacity toward ``target`` (clipped to bounds).

        ``delta-v1`` can move one replica per decision, so it returns the
        direction; ``desired-replicas-v1`` encodes the target itself.
        """
        if isinstance(target, bool) or not isinstance(target, int | np.integer):
            raise TypeError("target must be an integer replica count")
        bounded = self._clip(int(target))
        if self.semantics == DELTA_V1:
            if bounded > committed:
                return SCALE_UP
            if bounded < committed:
                return SCALE_DOWN
            return HOLD
        return bounded - self.min_replicas

    def code_for_step(self, delta_code: int, committed: int) -> int:
        """Express a one-replica decision (a ``delta-v1`` code) in this contract.

        ``delta-v1`` returns ``delta_code`` unchanged (so historical controllers
        emit exactly the same codes); ``desired-replicas-v1`` returns the code of
        the target ``committed + effect``, clipped to the replica bounds.
        """
        if delta_code not in DELTA_EFFECTS or isinstance(delta_code, bool):
            raise ValueError(f"invalid delta-v1 action code {delta_code!r}")
        if self.semantics == DELTA_V1:
            return delta_code
        return self.code_for_target(committed + DELTA_EFFECTS[delta_code], committed)

    def _clip(self, replicas: int) -> int:
        return min(max(replicas, self.min_replicas), self.max_replicas)

    def _expected(self) -> str:
        if self.semantics == DELTA_V1:
            return "delta-v1 expects 0 (scale down), 1 (hold), or 2 (scale up)"
        return f"{self.semantics} expects an integer code in 0..{self.action_count - 1}"
