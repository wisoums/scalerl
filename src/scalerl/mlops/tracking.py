"""MLflow run tracking for ScaleRL; requires the optional ``mlops`` extra.

MLflow is imported only when a run is started, so the rest of ScaleRL (and
:mod:`scalerl.mlops.spec`) works without it. Tracking goes through an explicit
``MlflowClient``: ``tracking_uri=None`` honors ``MLFLOW_TRACKING_URI`` and
MLflow's defaults, and no global MLflow state is changed.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scalerl.mlops.spec import EnvironmentCompatibility, RunSpec, software_metadata

if TYPE_CHECKING:
    from mlflow import MlflowClient

INSTALL_HINT = 'MLflow tracking requires the mlops extra: pip install -e ".[mlops]"'
ARTIFACT_DIR = "scalerl"
_MAX_PARAMS_PER_BATCH = 100


class TrackedRun:
    """Handle to an active tracked run for logging caller-computed results."""

    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self._client = client
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        """Return the MLflow run ID for referencing this run in reports."""
        return self._run_id

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log one metric value, optionally at ``step`` for time series."""
        self.log_metrics({key: value}, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log caller-computed metrics; values must be finite real numbers.

        Omit metrics that were not computed rather than logging placeholders.
        """
        from mlflow.entities import Metric

        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        timestamp = int(time.time() * 1000)
        entries = []
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"metric {key!r} must be a real number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be finite, got {value!r}")
            entries.append(Metric(key, float(value), timestamp, step or 0))
        self._client.log_batch(self._run_id, metrics=entries)

    def log_artifact_dict(
        self, name: str, payload: Mapping[str, Any], *, artifact_path: str | None = None
    ) -> None:
        """Log ``payload`` as the JSON artifact ``name`` (e.g. ``"evaluation.json"``)."""
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise ValueError(f"artifact name must be a plain .json file name, got {name!r}")
        content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(content)
            self._client.log_artifact(self._run_id, str(path), artifact_path)

    def log_artifact(self, local_path: str | Path, *, artifact_path: str | None = None) -> None:
        """Log an existing file (model, checkpoint, CSV, plot) produced by the caller."""
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact file not found: {path}")
        self._client.log_artifact(self._run_id, str(path), artifact_path)


@contextmanager
def start_tracked_run(
    spec: RunSpec,
    *,
    experiment_name: str = "scalerl",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    git_sha: str | None = None,
) -> Iterator[TrackedRun]:
    """Start a run that records ``spec`` and all lineage metadata automatically.

    The run ends ``FINISHED`` on success, ``KILLED`` on ``KeyboardInterrupt``,
    and ``FAILED`` on any other exception, which is always re-raised.
    """
    client = _client(tracking_uri)
    experiment_id = _experiment_id(client, experiment_name)

    software = software_metadata(git_sha)
    compatibility = EnvironmentCompatibility.from_config(
        spec.simulator_config, spec.benchmark_version
    )
    run = client.create_run(
        experiment_id,
        run_name=run_name or f"{spec.run_kind}-{spec.controller}-{spec.workload_id}",
        tags=_tags(spec, software),
    )
    tracked = TrackedRun(client, run.info.run_id)

    try:
        _log_run_metadata(client, tracked, spec, software, compatibility)
        yield tracked
    except KeyboardInterrupt:
        client.set_terminated(tracked.run_id, "KILLED")
        raise
    except BaseException:
        client.set_terminated(tracked.run_id, "FAILED")
        raise
    client.set_terminated(tracked.run_id, "FINISHED")


def _client(tracking_uri: str | None) -> MlflowClient:
    try:
        from mlflow import MlflowClient
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return MlflowClient(tracking_uri=tracking_uri)


def _experiment_id(client: MlflowClient, name: str) -> str:
    """Return the experiment's ID, creating it if needed.

    Parallel workers may race to create a new experiment; the losers join
    the one the winner created.
    """
    from mlflow.exceptions import MlflowException

    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return experiment.experiment_id


def _tags(spec: RunSpec, software: Mapping[str, Any]) -> dict[str, str]:
    tags = {
        "scalerl.run_kind": spec.run_kind,
        "scalerl.controller": spec.controller,
        "scalerl.benchmark_version": spec.benchmark_version,
        "scalerl.workload_id": spec.workload_id,
        "scalerl.workload_split": spec.workload_split,
        "scalerl.simulator_config_source": spec.simulator_config_source,
        "scalerl.git_sha": software["git_sha"],
        "scalerl.git_dirty": str(software["git_dirty"]).lower(),
        "scalerl.version": software["scalerl_version"],
    }
    if software["git_sha"] != "unknown":
        tags["mlflow.source.git.commit"] = software["git_sha"]
    return tags


def _log_run_metadata(
    client: MlflowClient,
    run: TrackedRun,
    spec: RunSpec,
    software: Mapping[str, Any],
    compatibility: EnvironmentCompatibility,
) -> None:
    from mlflow.entities import Param

    params = _run_params(spec, compatibility)
    entries = [Param(key, value) for key, value in params.items()]
    for start in range(0, len(entries), _MAX_PARAMS_PER_BATCH):
        client.log_batch(run.run_id, params=entries[start : start + _MAX_PARAMS_PER_BATCH])

    spec_json = spec.model_dump(mode="json")
    compatibility_json = compatibility.model_dump(mode="json")
    artifacts = {
        "resolved_run.json": {
            "run_id": run.run_id,
            "run_spec": spec_json,
            "software": software,
            "compatibility": compatibility_json,
        },
        "simulator_config.json": spec_json["simulator_config"],
        "reward_weights.json": spec_json["reward_weights"],
        "software.json": dict(software),
        "compatibility.json": compatibility_json,
    }
    for name, payload in artifacts.items():
        run.log_artifact_dict(name, payload, artifact_path=ARTIFACT_DIR)


def _run_params(spec: RunSpec, compatibility: EnvironmentCompatibility) -> dict[str, str]:
    """Flatten the spec into searchable params; values not provided are omitted."""
    params: dict[str, Any] = {
        "run_kind": spec.run_kind,
        "controller": spec.controller,
        "benchmark_version": spec.benchmark_version,
        "workload_id": spec.workload_id,
        "workload_split": spec.workload_split,
        "simulator_config_source": spec.simulator_config_source,
        "seed": spec.seed,
        "training_steps": spec.training_steps,
        "training_episodes": spec.training_episodes,
        "calibration_workload_ids": list(spec.calibration_workload_ids) or None,
        "calibration_note": spec.calibration_note,
        "evaluation_seeds": list(spec.evaluation_seeds) or None,
    }
    params |= _flatten("sim", spec.simulator_config.model_dump(mode="json"))
    params |= _flatten("reward", spec.reward_weights.model_dump(mode="json"))
    params |= _flatten("hp", spec.hyperparameters)
    params |= _flatten("compat", compatibility.model_dump(mode="json"))
    return {key: _param_value(value) for key, value in params.items() if value is not None}


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, Mapping):
            flat |= _flatten(name, item)
        else:
            flat[name] = item
    return flat


def _param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)
