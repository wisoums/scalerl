"""Workload traces that drive ScaleRL simulations."""

from scalerl.workloads.generators import (
    bursty_workload,
    diurnal_workload,
    ramp_workload,
    spike_workload,
    steady_workload,
)
from scalerl.workloads.trace import WorkloadReplay, WorkloadTrace

__all__ = [
    "WorkloadReplay",
    "WorkloadTrace",
    "bursty_workload",
    "diurnal_workload",
    "ramp_workload",
    "spike_workload",
    "steady_workload",
]
