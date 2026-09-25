"""Tests for Live City playback scheduling (#56), with injected fake time (no sleeping)."""

from typing import Any

import pytest

from scalerl.dashboard import (
    SPEEDS,
    ManagerSpec,
    PlaybackState,
    ScenarioSession,
    advance_if_due,
    autoplay_blocker,
    benchmark_scenario,
    start_playback,
)
from scalerl.dashboard.playback import COMPLETE_MESSAGE, MANUAL_AUTOPLAY_MESSAGE
from scalerl.environment import ReplicaConfig, SimulatorConfig, TimingConfig

V1_CONFIG = SimulatorConfig(
    timing=TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=3600.0),
    replicas=ReplicaConfig(min_replicas=1, max_replicas=6, initial_replicas=2),
)


class CountingSession(ScenarioSession):
    """A real session that counts controller steps."""

    def __init__(self, *args: Any) -> None:
        self.steps = 0
        super().__init__(*args)

    def step_controller(self) -> dict[str, Any]:
        self.steps += 1
        return super().step_controller()


def make_session(kind: str = "threshold", workload: str = "syn-train-spike") -> CountingSession:
    manager = ManagerSpec(kind, target_replicas=3 if kind == "static" else None)  # type: ignore[arg-type]
    return CountingSession(benchmark_scenario(workload), V1_CONFIG, manager)


def playing(session: ScenarioSession, now: float = 0.0, speed: float = 1.0) -> PlaybackState:
    state = PlaybackState(mode="live", speed=speed)
    assert start_playback(state, session, now) is None
    return state


# --- state machine -----------------------------------------------------------------------


def test_default_is_paused_inspect_at_1x() -> None:
    state = PlaybackState()

    assert (state.mode, state.playing, state.speed, state.next_due) == ("inspect", False, 1.0, None)
    assert SPEEDS == (0.5, 1.0, 2.0, 5.0)


def test_play_sets_playing_and_schedules_the_first_tick_one_period_later() -> None:
    state = PlaybackState(mode="live")

    state.play(now=10.0)

    assert state.playing
    assert state.next_due == pytest.approx(11.0)


@pytest.mark.parametrize(("speed", "period"), [(0.5, 2.0), (1.0, 1.0), (2.0, 0.5), (5.0, 0.2)])
def test_speed_sets_the_real_time_period(speed: float, period: float) -> None:
    state = PlaybackState(speed=speed)
    state.play(now=0.0)

    assert state.period_seconds == pytest.approx(period)
    assert not state.is_due(0.5 * period)
    assert state.is_due(period)


def test_small_timer_jitter_still_counts_as_due() -> None:
    state = PlaybackState()
    state.play(now=0.0)

    assert not state.is_due(0.85)
    assert state.is_due(0.95)  # a timer firing a hair early does not skip a whole period


@pytest.mark.parametrize("speed", [0.0, 3.0, -1.0, float("nan"), float("inf"), True])
def test_unsupported_speeds_are_rejected(speed: object) -> None:
    with pytest.raises(ValueError, match="speed"):
        PlaybackState(speed=speed)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="speed"):
        PlaybackState().set_speed(speed, now=0.0)  # type: ignore[arg-type]


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        PlaybackState().set_mode("replay")  # type: ignore[arg-type]


def test_pause_and_stop_clear_the_schedule() -> None:
    for halt in (PlaybackState.pause, PlaybackState.stop):
        state = PlaybackState()
        state.play(now=0.0)
        halt(state)
        assert (state.playing, state.next_due) == (False, None)
        assert not state.is_due(100.0)


def test_leaving_live_city_pauses() -> None:
    state = PlaybackState(mode="live")
    state.play(now=0.0)

    state.set_mode("inspect")

    assert (state.mode, state.playing) == ("inspect", False)


def test_speed_change_only_reschedules_future_cadence() -> None:
    state = PlaybackState(speed=0.5)
    state.play(now=0.0)  # due at 2.0

    state.set_speed(2.0, now=1.0)

    assert state.playing
    assert state.next_due == pytest.approx(1.5)  # one new period from the change, not now


# --- driving a real session ---------------------------------------------------------------


def test_one_tick_is_due_at_1x_after_one_second() -> None:
    session = make_session()
    state = playing(session)

    assert not advance_if_due(state, session, 0.5)
    assert session.steps == 0
    assert advance_if_due(state, session, 1.0)
    assert (session.steps, session.tick) == (1, 1)


def test_one_event_advances_at_most_one_tick_however_late() -> None:
    session = make_session()
    state = playing(session)

    assert advance_if_due(state, session, 50.0)  # 49 periods late

    assert (session.steps, session.tick) == (1, 1)  # no catch-up burst
    assert state.next_due == pytest.approx(51.0)
    assert not advance_if_due(state, session, 50.0)


def test_rerenders_before_the_next_due_time_never_step() -> None:
    session = make_session()
    state = playing(session)

    for now in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        assert not advance_if_due(state, session, now)

    assert session.steps == 0
    assert session.history == []


def test_paused_playback_never_advances() -> None:
    session = make_session()
    state = playing(session)
    state.pause()

    for now in (1.0, 2.0, 100.0):
        assert not advance_if_due(state, session, now)

    assert session.steps == 0


def test_resume_continues_from_the_current_tick() -> None:
    session = make_session()
    state = playing(session)
    for now in (1.0, 2.0, 3.0):
        advance_if_due(state, session, now)
    state.pause()
    history = session.history
    controller = session.controller

    assert start_playback(state, session, now=10.0) is None
    assert not advance_if_due(state, session, 10.5)
    assert advance_if_due(state, session, 11.0)

    assert session.tick == 4
    assert session.history[:3] == history  # nothing lost or replayed
    assert session.controller is controller


def test_play_pause_step_play_workflow() -> None:
    session = make_session()
    state = playing(session)
    advance_if_due(state, session, 1.0)
    advance_if_due(state, session, 2.0)
    state.pause()

    session.step_controller()  # one deliberate step while paused
    assert session.tick == 3

    start_playback(state, session, now=5.0)
    advance_if_due(state, session, 6.0)

    assert session.tick == len(session.history) == 4
    assert session.steps == 4


def test_speed_change_keeps_the_session_and_never_steps_immediately() -> None:
    session = make_session()
    state = playing(session, speed=0.5)
    advance_if_due(state, session, 2.0)
    history, config, controller = session.history, session.config, session.controller

    state.set_speed(2.0, now=2.1)

    assert not advance_if_due(state, session, 2.1)
    assert (session.history, session.config, session.controller) == (history, config, controller)
    assert advance_if_due(state, session, 2.6)  # the new 0.5 s cadence
    assert session.tick == 2
    assert session.config.timing.control_interval_seconds == 30.0


def test_completion_stops_playback_without_stepping_past_the_end() -> None:
    session = make_session("static")
    for _ in range(session.episode_ticks - 1):
        session.step_controller()
    state = playing(session)

    assert advance_if_due(state, session, 1.0)

    assert session.done
    assert session.tick == session.episode_ticks == 120
    assert not state.playing
    assert not advance_if_due(state, session, 2.0)
    assert start_playback(state, session, now=3.0) == COMPLETE_MESSAGE
    assert (state.playing, session.tick) == (False, 120)


def test_manual_sessions_cannot_autoplay() -> None:
    session = make_session("manual")
    state = PlaybackState(mode="live")

    assert autoplay_blocker(session) == MANUAL_AUTOPLAY_MESSAGE
    assert start_playback(state, session, now=0.0) == MANUAL_AUTOPLAY_MESSAGE
    assert not state.playing

    state.play(now=0.0)  # even if forced into playing, no HOLD is invented
    assert not advance_if_due(state, session, 5.0)
    assert (state.playing, session.tick) == (False, 0)


def test_reset_session_is_not_advanced_by_a_stopped_playback() -> None:
    session = make_session()
    state = playing(session)
    advance_if_due(state, session, 1.0)

    state.stop()
    session.reset()

    assert not advance_if_due(state, session, 2.0)
    assert (session.tick, session.history) == (0, [])


@pytest.mark.parametrize("kind", ["random", "static", "threshold", "predictive"])
def test_every_step_adds_exactly_one_history_row(kind: str) -> None:
    session = make_session(kind, workload="syn-train-bursty")
    state = playing(session, speed=5.0)

    now = 0.0
    while state.playing:
        now += 0.2
        advance_if_due(state, session, now)
        assert len(session.history) == session.tick == session.steps
        assert [row["tick"] for row in session.history] == list(range(session.tick))

    assert session.done and session.tick == 120


@pytest.mark.parametrize("kind", ["threshold", "predictive"])
def test_diagnostics_describe_the_action_behind_the_latest_tick(kind: str) -> None:
    session = make_session(kind, workload="syn-train-bursty")
    state = playing(session)

    for now in range(1, 60):
        advance_if_due(state, session, float(now))
        decision = session.threshold_decision or session.predictive_decision
        assert decision is not None
        # The decision was made before the tick ran and is the action that tick applied.
        assert decision.action == session.history[-1]["requested_action"]
