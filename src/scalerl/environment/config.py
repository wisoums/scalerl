"""Typed, validated configuration for the ScaleRL cloud simulator.

All durations are in seconds. Defaults describe a small, conservative
deployment and are always internally valid.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenConfig(BaseModel):
    """Base for immutable configs that reject unknown fields and non-finite floats."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class TimingConfig(_FrozenConfig):
    """Episode length and controller cadence."""

    control_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Simulated time between two controller decisions.",
    )
    episode_duration_seconds: float = Field(
        default=3600.0,
        gt=0,
        description="Total simulated workload duration of one episode.",
    )

    @model_validator(mode="after")
    def _episode_covers_one_interval(self) -> Self:
        if self.episode_duration_seconds < self.control_interval_seconds:
            raise ValueError("episode_duration_seconds must be at least control_interval_seconds")
        return self


class ReplicaConfig(_FrozenConfig):
    """Replica bounds, lifecycle, capacity, and cost."""

    min_replicas: int = Field(default=1, gt=0, description="Lower bound on replica count.")
    max_replicas: int = Field(default=10, gt=0, description="Upper bound on replica count.")
    initial_replicas: int = Field(default=1, gt=0, description="Replica count at episode start.")
    startup_delay_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Time before a newly requested replica can serve traffic.",
    )
    service_capacity_rps: float = Field(
        default=50.0,
        gt=0,
        description="Requests per second a single ready replica can serve.",
    )
    cost_per_hour: float = Field(
        default=0.10,
        ge=0,
        description="Infrastructure cost of one replica for one hour.",
    )

    @model_validator(mode="after")
    def _check_replica_bounds(self) -> Self:
        if not self.min_replicas <= self.initial_replicas <= self.max_replicas:
            raise ValueError("replica bounds must satisfy min <= initial <= max")
        return self


class SlaConfig(_FrozenConfig):
    """Service-level objective the simulated application is held to."""

    latency_target_seconds: float = Field(
        default=0.5,
        gt=0,
        description="Maximum acceptable request latency.",
    )


class SimulatorConfig(_FrozenConfig):
    """Complete simulator configuration composed from focused sub-configs."""

    timing: TimingConfig = Field(default_factory=TimingConfig)
    replicas: ReplicaConfig = Field(default_factory=ReplicaConfig)
    sla: SlaConfig = Field(default_factory=SlaConfig)
