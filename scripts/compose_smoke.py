"""Compose-network smoke checks, run inside the trainer container.

Invoked by scripts/compose-smoke.sh as
``docker compose run --rm -T trainer python - < scripts/compose_smoke.py``.

1. A real ScaleRL tracked run: the predictive evaluator on one synthetic TRAIN
   workload (no Azure data, no held-out data) through ``MLFLOW_TRACKING_URI``.
2. MLflow: the run finished with params and metrics, and its ``scalerl/``
   metadata artifacts can be listed and downloaded through MLflow's artifact
   proxy (so they round-tripped through the S3-compatible store).
3. Optuna: a tiny throwaway study in ``OPTUNA_STORAGE_URI`` is visible to the
   Optuna Dashboard, then deleted.
4. The Scenario Lab health endpoint answers on the Compose network.
5. The trainer can write the bind-mounted ``outputs/`` (host UID/GID setup).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

import optuna
from mlflow import MlflowClient

from scalerl.evaluation.predictive import evaluate_predictive

WORKLOAD = "syn-train-steady-moderate"
EXPERIMENT = "scalerl-compose-smoke"
DASHBOARD = "http://optuna-dashboard:8080"
UI_HEALTH = "http://scalerl-ui:8501/_stcore/health"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok   {message}")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def mlflow_checks() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    check(tracking_uri == "http://mlflow:5000", f"trainer tracks to {tracking_uri}")
    (result,) = evaluate_predictive(workload_ids=(WORKLOAD,), experiment_name=EXPERIMENT)
    client = MlflowClient(tracking_uri)
    run = client.get_run(result.mlflow_run_id)
    check(run.info.status == "FINISHED", f"run {run.info.run_id} finished")
    check(run.data.tags.get("scalerl.workload_split") == "train", "run is on a TRAIN workload")
    check(
        run.data.params.get("hp.capacity_policy") == "forecast-plus-backlog-v1",
        "params recorded",
    )
    check("sla_violation_rate" in run.data.metrics, "metrics recorded")
    check(
        run.info.artifact_uri.startswith("mlflow-artifacts:"),
        f"artifacts go through the MLflow proxy ({run.info.artifact_uri})",
    )
    names = {Path(item.path).name for item in client.list_artifacts(run.info.run_id, "scalerl")}
    check("resolved_run.json" in names, f"artifacts listed: {sorted(names)}")
    with tempfile.TemporaryDirectory() as directory:
        path = client.download_artifacts(run.info.run_id, "scalerl/resolved_run.json", directory)
        payload = json.loads(Path(path).read_text())
    check(payload["run_id"] == run.info.run_id, "artifact downloaded and matches the run")


def optuna_checks() -> None:
    storage = os.environ["OPTUNA_STORAGE_URI"]
    check(storage.endswith("@postgres:5432/optuna"), "Optuna storage is the PostgreSQL optuna DB")
    name = f"compose-smoke-{uuid.uuid4().hex[:8]}"
    study = optuna.create_study(study_name=name, storage=storage)
    try:
        study.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=0.0))
        studies = json.loads(get(f"{DASHBOARD}/api/studies"))["study_summaries"]
        check(
            any(summary["study_name"] == name for summary in studies),
            f"Optuna Dashboard sees study {name}",
        )
    finally:
        optuna.delete_study(study_name=name, storage=storage)
    print(f"ok   deleted smoke study {name}")


def ui_check() -> None:
    check(get(UI_HEALTH).strip() == b"ok", "Scenario Lab health endpoint")


def outputs_check() -> None:
    probe = Path("outputs") / f".compose-smoke-{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("ok")
    except PermissionError as error:
        raise SystemExit(
            f"FAIL: trainer (uid {os.getuid()}) cannot write outputs/: {error}. "
            "Run scripts/setup-local-stack.sh and match SCALERL_UID/GID to your user."
        ) from error
    probe.unlink()
    check(True, f"trainer (uid {os.getuid()}) can write outputs/")


def main() -> int:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outputs_check()
    mlflow_checks()
    optuna_checks()
    ui_check()
    print("compose smoke (network): all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
