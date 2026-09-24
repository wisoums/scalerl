"""Experiment tracking and reproducible run metadata.

``RunSpec``, ``EnvironmentCompatibility``, and ``software_metadata`` work
without MLflow. ``start_tracked_run`` needs the ``mlops`` extra and raises an
``ImportError`` with an install hint otherwise.
"""

from scalerl.mlops.spec import (
    EnvironmentCompatibility,
    RunKind,
    RunSpec,
    SimulatorConfigSource,
    software_metadata,
)
from scalerl.mlops.tracking import INSTALL_HINT, TrackedRun, start_tracked_run

__all__ = [
    "INSTALL_HINT",
    "EnvironmentCompatibility",
    "RunKind",
    "RunSpec",
    "SimulatorConfigSource",
    "TrackedRun",
    "software_metadata",
    "start_tracked_run",
]
