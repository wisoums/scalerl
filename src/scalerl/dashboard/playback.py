"""Live City playback scheduling; no Streamlit dependency.

Playback is presentation only. It decides *when* the Scenario Lab shows the
next tick; every tick is still one real ``ScenarioSession.step_controller()``
call on the same session, controller, environment, and history as Inspect mode.
Speed changes only the real-time cadence (1x ≈ one simulator tick per real
second), never the simulated control interval, workload, or controller.

Timing uses injected monotonic timestamps so it is testable without sleeping.
There is no catch-up: however late a check is, it advances at most one tick
and schedules the next one a full period later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

PlaybackMode = Literal["inspect", "live"]
SPEEDS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
BASE_PERIOD_SECONDS = 1.0  # real seconds per simulator tick at 1x
# A timer firing slightly early still counts; without it, jitter could skip a whole period.
DUE_TOLERANCE = 0.1  # fraction of the period

MANUAL_AUTOPLAY_MESSAGE = (
    "Live City autoplay requires a controller-driven manager. "
    "Use Inspect mode to choose Manual actions."
)
COMPLETE_MESSAGE = "Episode complete. Build or reset to play again."


class PlayableSession(Protocol):
    """The part of ``ScenarioSession`` playback needs."""

    @property
    def controller(self) -> object | None: ...

    @property
    def done(self) -> bool: ...

    def step_controller(self) -> object: ...


def speed_label(speed: float) -> str:
    return f"{speed:g}x"


@dataclass
class PlaybackState:
    """Inspect/Live City mode, play/pause, speed, and when the next tick is due."""

    mode: PlaybackMode = "inspect"
    playing: bool = False
    speed: float = 1.0
    next_due: float | None = None

    def __post_init__(self) -> None:
        _require_speed(self.speed)

    @property
    def period_seconds(self) -> float:
        """Real seconds between displayed ticks (simulated tick length is unaffected)."""
        return BASE_PERIOD_SECONDS / self.speed

    def set_mode(self, mode: PlaybackMode) -> None:
        """Switch views; leaving Live City pauses. The session is never touched."""
        if mode not in ("inspect", "live"):
            raise ValueError(f"unknown playback mode {mode!r}")
        self.mode = mode
        if mode == "inspect":
            self.pause()

    def play(self, now: float) -> None:
        """Start playing; the first tick is due one period from ``now``."""
        self.playing = True
        self.next_due = now + self.period_seconds

    def pause(self) -> None:
        self.playing = False
        self.next_due = None

    def stop(self) -> None:
        """Stop playback because the session was reset, rebuilt, or completed."""
        self.pause()

    def set_speed(self, speed: float, now: float) -> None:
        """Change only the future cadence; the next tick is due one new period from ``now``."""
        _require_speed(speed)
        self.speed = speed
        if self.playing:
            self.next_due = now + self.period_seconds

    def is_due(self, now: float) -> bool:
        if not self.playing or self.next_due is None:
            return False
        return now >= self.next_due - DUE_TOLERANCE * self.period_seconds

    def mark_advanced(self, now: float) -> None:
        """Schedule the next tick one period after this one (no catch-up)."""
        self.next_due = now + self.period_seconds


def autoplay_blocker(session: PlayableSession) -> str | None:
    """Why this session cannot autoplay, or ``None`` if it can."""
    if session.controller is None:
        return MANUAL_AUTOPLAY_MESSAGE
    if session.done:
        return COMPLETE_MESSAGE
    return None


def start_playback(state: PlaybackState, session: PlayableSession, now: float) -> str | None:
    """Play if the session can autoplay; otherwise stay paused and return why."""
    blocker = autoplay_blocker(session)
    if blocker is not None:
        state.stop()
        return blocker
    state.play(now)
    return None


def advance_if_due(state: PlaybackState, session: PlayableSession, now: float) -> bool:
    """Advance at most one real controller tick if one is due; return whether it did.

    Playback stops (instead of stepping) when the session cannot autoplay, and
    stops right after the tick that completes the episode.
    """
    if not state.playing:
        return False
    if autoplay_blocker(session) is not None:
        state.stop()
        return False
    if not state.is_due(now):
        return False
    session.step_controller()
    state.mark_advanced(now)
    if session.done:
        state.stop()
    return True


def _require_speed(speed: object) -> None:
    if isinstance(speed, bool) or not isinstance(speed, int | float) or not math.isfinite(speed):
        raise ValueError("playback speed must be a finite number")
    if speed not in SPEEDS:
        choices = ", ".join(speed_label(s) for s in SPEEDS)
        raise ValueError(f"playback speed must be one of {choices}")
