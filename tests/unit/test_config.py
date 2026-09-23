"""Tests for the typed simulator configuration models."""

from typing import Any

import pytest
from pydantic import ValidationError

from scalerl.environment import ReplicaConfig, SimulatorConfig, SlaConfig, TimingConfig

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def test_defaults_are_documented_and_conservative() -> None:
    config = SimulatorConfig()

    assert config.timing == TimingConfig(
        control_interval_seconds=30.0, episode_duration_seconds=3600.0
    )
    assert config.replicas == ReplicaConfig(
        min_replicas=1,
        max_replicas=10,
        initial_replicas=1,
        startup_delay_seconds=60.0,
        service_capacity_rps=50.0,
        cost_per_hour=0.10,
    )
    assert config.sla == SlaConfig(latency_target_seconds=0.5)

    for model in (TimingConfig, ReplicaConfig, SlaConfig):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} is undocumented"


def test_default_configs_are_internally_valid() -> None:
    config = SimulatorConfig()

    assert SimulatorConfig.model_validate(config.model_dump()) == config


def test_valid_custom_construction() -> None:
    config = SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=15, episode_duration_seconds=7200),
        replicas=ReplicaConfig(
            min_replicas=2,
            max_replicas=20,
            initial_replicas=5,
            startup_delay_seconds=0,
            service_capacity_rps=12.5,
            cost_per_hour=0,
        ),
        sla=SlaConfig(latency_target_seconds=0.25),
    )

    assert config.timing.control_interval_seconds == 15
    assert config.timing.episode_duration_seconds == 7200
    assert config.replicas.initial_replicas == 5
    assert config.replicas.startup_delay_seconds == 0
    assert config.replicas.cost_per_hour == 0
    assert config.sla.latency_target_seconds == 0.25


def test_sub_configs_can_be_overridden_independently() -> None:
    config = SimulatorConfig(sla=SlaConfig(latency_target_seconds=1.0))

    assert config.sla.latency_target_seconds == 1.0
    assert config.timing == TimingConfig()
    assert config.replicas == ReplicaConfig()


def test_nested_dicts_are_accepted() -> None:
    config = SimulatorConfig.model_validate(
        {"replicas": {"min_replicas": 3, "max_replicas": 6, "initial_replicas": 4}}
    )

    assert config.replicas.initial_replicas == 4
    assert config.replicas.service_capacity_rps == ReplicaConfig().service_capacity_rps


# --- replica bounds ---------------------------------------------------------


@pytest.mark.parametrize(
    ("min_replicas", "initial_replicas", "max_replicas"),
    [(1, 1, 1), (3, 3, 5), (3, 5, 5), (2, 3, 4)],
)
def test_replica_bounds_accept_ordered_values(
    min_replicas: int, initial_replicas: int, max_replicas: int
) -> None:
    config = ReplicaConfig(
        min_replicas=min_replicas,
        initial_replicas=initial_replicas,
        max_replicas=max_replicas,
    )

    assert config.min_replicas <= config.initial_replicas <= config.max_replicas


@pytest.mark.parametrize(
    ("min_replicas", "initial_replicas", "max_replicas"),
    [
        (5, 3, 10),  # initial below min
        (1, 11, 10),  # initial above max
        (6, 6, 5),  # min above max
        (4, 2, 3),  # all out of order
    ],
)
def test_replica_bounds_reject_unordered_values(
    min_replicas: int, initial_replicas: int, max_replicas: int
) -> None:
    with pytest.raises(ValidationError, match="min <= initial <= max"):
        ReplicaConfig(
            min_replicas=min_replicas,
            initial_replicas=initial_replicas,
            max_replicas=max_replicas,
        )


@pytest.mark.parametrize("field", ["min_replicas", "max_replicas", "initial_replicas"])
@pytest.mark.parametrize("value", [0, -1])
def test_replica_counts_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match=field):
        ReplicaConfig.model_validate({field: value})


@pytest.mark.parametrize("field", ["min_replicas", "max_replicas", "initial_replicas"])
@pytest.mark.parametrize("value", [2.0, 2.5, "2", True])
def test_replica_counts_must_be_integers(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        ReplicaConfig.model_validate({field: value})


# --- numeric fields ---------------------------------------------------------

POSITIVE_FIELDS = [
    (TimingConfig, "control_interval_seconds"),
    (TimingConfig, "episode_duration_seconds"),
    (ReplicaConfig, "service_capacity_rps"),
    (SlaConfig, "latency_target_seconds"),
]
NON_NEGATIVE_FIELDS = [
    (ReplicaConfig, "startup_delay_seconds"),
    (ReplicaConfig, "cost_per_hour"),
]


@pytest.mark.parametrize(("model", "field"), POSITIVE_FIELDS)
@pytest.mark.parametrize("value", [0, 0.0, -1e-9, -5])
def test_positive_fields_reject_zero_and_negative(
    model: type[Any], field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        model.model_validate({field: value})


@pytest.mark.parametrize(("model", "field"), NON_NEGATIVE_FIELDS)
def test_non_negative_fields_accept_zero(model: type[Any], field: str) -> None:
    assert getattr(model.model_validate({field: 0}), field) == 0


@pytest.mark.parametrize(("model", "field"), NON_NEGATIVE_FIELDS)
@pytest.mark.parametrize("value", [-1e-9, -1])
def test_non_negative_fields_reject_negative(model: type[Any], field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        model.model_validate({field: value})


@pytest.mark.parametrize(("model", "field"), POSITIVE_FIELDS + NON_NEGATIVE_FIELDS)
@pytest.mark.parametrize("value", NON_FINITE)
def test_float_fields_reject_non_finite_values(model: type[Any], field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        model.model_validate({field: value})


@pytest.mark.parametrize(("model", "field"), POSITIVE_FIELDS + NON_NEGATIVE_FIELDS)
@pytest.mark.parametrize("value", ["1.0", True, None])
def test_float_fields_reject_non_numeric_values(
    model: type[Any], field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        model.model_validate({field: value})


def test_integer_values_are_accepted_for_float_fields() -> None:
    timing = TimingConfig(control_interval_seconds=10, episode_duration_seconds=60)

    assert timing.control_interval_seconds == 10.0
    assert timing.episode_duration_seconds == 60.0


# --- timing relationship ----------------------------------------------------


def test_episode_may_equal_one_control_interval() -> None:
    timing = TimingConfig(control_interval_seconds=60, episode_duration_seconds=60)

    assert timing.episode_duration_seconds == timing.control_interval_seconds


def test_episode_shorter_than_control_interval_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least control_interval_seconds"):
        TimingConfig(control_interval_seconds=60, episode_duration_seconds=30)


# --- model hygiene ----------------------------------------------------------


@pytest.mark.parametrize("model", [TimingConfig, ReplicaConfig, SlaConfig, SimulatorConfig])
def test_unknown_fields_are_rejected(model: type[Any]) -> None:
    with pytest.raises(ValidationError, match="unexpected_field"):
        model.model_validate({"unexpected_field": 1})


def test_configs_are_immutable() -> None:
    config = SimulatorConfig()

    with pytest.raises(ValidationError):
        config.replicas.max_replicas = 100  # type: ignore[misc]
    with pytest.raises(ValidationError):
        config.sla = SlaConfig()  # type: ignore[misc]


def test_equal_configs_are_hashable_and_equal() -> None:
    assert SimulatorConfig() == SimulatorConfig()
    assert hash(SimulatorConfig()) == hash(SimulatorConfig())


def test_invalid_nested_config_is_rejected_through_parent() -> None:
    with pytest.raises(ValidationError, match="min <= initial <= max"):
        SimulatorConfig.model_validate(
            {"replicas": {"min_replicas": 4, "initial_replicas": 2, "max_replicas": 8}}
        )


# --- serialization ----------------------------------------------------------


def _custom_config() -> SimulatorConfig:
    return SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=12.5, episode_duration_seconds=900),
        replicas=ReplicaConfig(
            min_replicas=2,
            max_replicas=8,
            initial_replicas=3,
            startup_delay_seconds=45.5,
            service_capacity_rps=75.25,
            cost_per_hour=0.034,
        ),
        sla=SlaConfig(latency_target_seconds=0.15),
    )


@pytest.mark.parametrize("config", [SimulatorConfig(), _custom_config()])
def test_model_dump_round_trip(config: SimulatorConfig) -> None:
    restored = SimulatorConfig.model_validate(config.model_dump())

    assert restored == config
    assert restored.model_dump() == config.model_dump()


@pytest.mark.parametrize("config", [SimulatorConfig(), _custom_config()])
def test_json_round_trip(config: SimulatorConfig) -> None:
    restored = SimulatorConfig.model_validate_json(config.model_dump_json())

    assert restored == config


def test_model_dump_has_stable_nested_shape() -> None:
    assert SimulatorConfig().model_dump() == {
        "timing": {"control_interval_seconds": 30.0, "episode_duration_seconds": 3600.0},
        "replicas": {
            "min_replicas": 1,
            "max_replicas": 10,
            "initial_replicas": 1,
            "startup_delay_seconds": 60.0,
            "service_capacity_rps": 50.0,
            "cost_per_hour": 0.10,
        },
        "sla": {"latency_target_seconds": 0.5},
    }
