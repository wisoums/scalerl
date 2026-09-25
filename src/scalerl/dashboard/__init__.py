"""Scenario Lab City View: an interactive view of the ScaleRL simulator.

This package does not import Streamlit; only ``scalerl.dashboard.app`` does.
Launch with ``python -m scalerl.dashboard`` (needs the ``dashboard`` extra).
"""

from scalerl.dashboard.playback import (
    SPEEDS,
    PlaybackMode,
    PlaybackState,
    advance_if_due,
    autoplay_blocker,
    start_playback,
)
from scalerl.dashboard.session import (
    ACTION_LABELS,
    CUSTOM_GENERATORS,
    DEFAULT_AZURE_PATH,
    HELD_OUT_WARNING,
    ManagerKind,
    ManagerSpec,
    Scenario,
    ScenarioInfo,
    ScenarioSession,
    azure_capacity_note,
    benchmark_choices,
    benchmark_scenario,
    custom_scenario,
    icon_row,
)

__all__ = [
    "ACTION_LABELS",
    "CUSTOM_GENERATORS",
    "DEFAULT_AZURE_PATH",
    "HELD_OUT_WARNING",
    "SPEEDS",
    "ManagerKind",
    "ManagerSpec",
    "PlaybackMode",
    "PlaybackState",
    "Scenario",
    "ScenarioInfo",
    "ScenarioSession",
    "advance_if_due",
    "autoplay_blocker",
    "azure_capacity_note",
    "benchmark_choices",
    "benchmark_scenario",
    "custom_scenario",
    "icon_row",
    "start_playback",
]
