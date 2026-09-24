"""Workload traces that drive ScaleRL simulations."""

from scalerl.workloads.azure import (
    AZURE_FUNCTIONS_2021,
    AzureWindow,
    azure_trace_metadata,
    load_azure_trace,
    load_azure_traces,
)
from scalerl.workloads.generators import (
    bursty_workload,
    diurnal_workload,
    ramp_workload,
    spike_workload,
    steady_workload,
)
from scalerl.workloads.processed import load_processed_trace, save_processed_trace
from scalerl.workloads.trace import WorkloadReplay, WorkloadTrace

__all__ = [
    "AZURE_FUNCTIONS_2021",
    "AzureWindow",
    "WorkloadReplay",
    "WorkloadTrace",
    "azure_trace_metadata",
    "bursty_workload",
    "diurnal_workload",
    "load_azure_trace",
    "load_azure_traces",
    "load_processed_trace",
    "ramp_workload",
    "save_processed_trace",
    "spike_workload",
    "steady_workload",
]
