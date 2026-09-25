"""Static checks of the Docker Compose stack (#44); no Docker daemon needed.

The running stack is exercised by scripts/compose-smoke.sh, not by pytest.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).parents[2]
COMPOSE = ROOT / "compose.yaml"
DOCKERFILES = [
    ROOT / "Dockerfile",
    ROOT / "docker" / "mlflow" / "Dockerfile",
    ROOT / "docker" / "optuna-dashboard" / "Dockerfile",
]


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE.read_text())
    return loaded


def services(compose: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = compose["services"]
    return result


def bind(service: dict[str, Any], target: str) -> dict[str, Any]:
    (mount,) = [
        volume
        for volume in service["volumes"]
        if isinstance(volume, dict) and volume["target"] == target
    ]
    result: dict[str, Any] = mount
    return result


def test_stack_has_the_expected_services(compose: dict[str, Any]) -> None:
    assert set(services(compose)) == {
        "scalerl-ui",
        "trainer",
        "mlflow",
        "optuna-dashboard",
        "postgres",
        "garage",
        "garage-init",
    }


def test_every_image_is_pinned() -> None:
    images = re.findall(r"^\s*image:\s*(\S+)", COMPOSE.read_text(), flags=re.MULTILINE)
    froms = [
        line.split()[1]
        for path in DOCKERFILES
        for line in path.read_text().splitlines()
        if line.startswith(("FROM ", "ARG PYTHON_IMAGE="))
    ]
    pinned = [ref.removeprefix("PYTHON_IMAGE=") for ref in images + froms]
    external = [ref for ref in pinned if "${" not in ref and not ref.startswith("scalerl")]
    assert external, "no external images found"
    for ref in external:
        assert ":" in ref or "@sha256:" in ref, ref
        assert not ref.endswith(":latest"), ref


def test_databases_and_artifact_store_are_not_published(compose: dict[str, Any]) -> None:
    for name in ("postgres", "garage", "garage-init", "trainer"):
        assert "ports" not in services(compose)[name], name
    published = {
        name: service["ports"] for name, service in services(compose).items() if "ports" in service
    }
    assert published == {
        # Loopback only: these development UIs are unauthenticated.
        "scalerl-ui": ["127.0.0.1:${SCALERL_UI_PORT:-8501}:8501"],
        "mlflow": ["127.0.0.1:${MLFLOW_UI_PORT:-5000}:5000"],
        "optuna-dashboard": ["127.0.0.1:${OPTUNA_DASHBOARD_PORT:-8080}:8080"],
    }


def test_internal_and_browser_urls_are_distinct(compose: dict[str, Any]) -> None:
    trainer = services(compose)["trainer"]["environment"]
    ui = services(compose)["scalerl-ui"]["environment"]

    assert trainer["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert trainer["OPTUNA_STORAGE_URI"].endswith("@postgres:5432/optuna")
    for env in (trainer, ui):
        assert env["SCALERL_MLFLOW_UI_URL"] == "${SCALERL_MLFLOW_UI_URL:-http://localhost:5000}"
        assert (
            env["SCALERL_OPTUNA_DASHBOARD_URL"]
            == "${SCALERL_OPTUNA_DASHBOARD_URL:-http://localhost:8080}"
        )
    assert "MLFLOW_TRACKING_URI" not in ui  # the Scenario Lab does not need MLflow


def test_mlflow_and_optuna_use_separate_databases(compose: dict[str, Any]) -> None:
    mlflow = services(compose)["mlflow"]["environment"]
    dashboard = services(compose)["optuna-dashboard"]["environment"]
    trainer = services(compose)["trainer"]["environment"]

    assert mlflow["MLFLOW_BACKEND_STORE_URI"].endswith("@postgres:5432/mlflow")
    assert dashboard["OPTUNA_STORAGE_URI"] == trainer["OPTUNA_STORAGE_URI"]
    init = (ROOT / "docker" / "postgres" / "init-databases.sh").read_text()
    assert "CREATE DATABASE mlflow" in init and "CREATE DATABASE optuna" in init


def test_mlflow_proxies_artifacts_to_the_s3_store(compose: dict[str, Any]) -> None:
    mlflow = services(compose)["mlflow"]

    assert "--serve-artifacts" in mlflow["command"]
    assert mlflow["environment"]["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://${MLFLOW_BUCKET:-mlflow}"
    assert mlflow["environment"]["MLFLOW_S3_ENDPOINT_URL"] == "http://garage:3900"
    # Clients never receive S3 credentials.
    for name in ("trainer", "scalerl-ui"):
        env = services(compose)[name]["environment"]
        assert not any(key.startswith(("AWS_", "GARAGE_")) for key in env), name


def test_startup_order_uses_health_and_completion(compose: dict[str, Any]) -> None:
    svc = services(compose)
    assert svc["mlflow"]["depends_on"] == {
        "postgres": {"condition": "service_healthy"},
        "garage-init": {"condition": "service_completed_successfully"},
    }
    assert svc["garage-init"]["depends_on"] == {"garage": {"condition": "service_healthy"}}
    assert svc["optuna-dashboard"]["depends_on"] == {"postgres": {"condition": "service_healthy"}}
    assert "depends_on" not in svc["scalerl-ui"]  # the Scenario Lab starts on its own
    for name in ("scalerl-ui", "mlflow", "optuna-dashboard", "postgres", "garage"):
        assert "healthcheck" in svc[name], name


def test_trainer_is_an_on_demand_utility(compose: dict[str, Any]) -> None:
    trainer = services(compose)["trainer"]

    assert trainer["profiles"] == ["trainer"]
    assert "restart" not in trainer
    assert "sleep" not in " ".join(trainer["command"])
    assert bind(trainer, "/app/outputs") == {
        "type": "bind",
        "source": "./outputs",
        "target": "/app/outputs",
        "bind": {"create_host_path": False},
    }


def test_state_lives_in_named_volumes(compose: dict[str, Any]) -> None:
    assert set(compose["volumes"]) == {"postgres-data", "garage-meta", "garage-data"}
    assert "postgres-data:/var/lib/postgresql" in services(compose)["postgres"]["volumes"]


def test_azure_data_is_mounted_read_only_not_baked(compose: dict[str, Any]) -> None:
    for name in ("scalerl-ui", "trainer"):
        assert bind(services(compose)[name], "/app/data/raw") == {
            "type": "bind",
            "source": "./data/raw",
            "target": "/app/data/raw",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    for path in DOCKERFILES:
        copies = [
            line for line in path.read_text().splitlines() if line.startswith(("COPY", "ADD"))
        ]
        assert copies and not any("data" in line for line in copies), path


def test_build_context_is_an_allow_list() -> None:
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "*"
    allowed = {rule[1:] for rule in rules if rule.startswith("!")}
    assert allowed == {"pyproject.toml", "README.md", "LICENSE", "src/", "docker/"}
    assert not (ROOT / "src" / "data").exists() and not (ROOT / "docker" / "data").exists()


def test_no_secrets_are_hard_coded() -> None:
    compose_text = COMPOSE.read_text()
    for variable in (
        "POSTGRES_PASSWORD",
        "GARAGE_RPC_SECRET",
        "GARAGE_ADMIN_TOKEN",
        "GARAGE_SECRET_ACCESS_KEY",
    ):
        assert re.search(rf"\$\{{{variable}:\?", compose_text), variable
    for path in DOCKERFILES:
        text = path.read_text()
        assert ".env" not in text, path
        assert not re.search(r"^ARG .*(PASSWORD|SECRET|TOKEN)", text, re.MULTILINE), path
    for path in (
        ROOT / "docker" / "garage" / "garage.toml",
        ROOT / "docker" / "postgres" / "init-databases.sh",
    ):
        text = path.read_text()
        assert not re.search(r"^\s*(rpc_secret|admin_token)\s*=", text, re.MULTILINE), path


def test_env_example_defines_every_required_variable() -> None:
    required = set(re.findall(r"\$\{(\w+):\?", COMPOSE.read_text()))
    defined = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert required <= defined, required - defined


def test_env_file_is_ignored_and_example_is_not() -> None:
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert ignored(".env")
    assert not ignored(".env.example")


def test_core_import_does_not_pull_infrastructure_drivers() -> None:
    code = (
        "import sys\n"
        "import scalerl, scalerl.environment, scalerl.controllers, scalerl.workloads\n"
        "import scalerl.mlops, scalerl.tuning, scalerl.dashboard, scalerl.evaluation\n"
        "leaked = {'psycopg', 'boto3', 'botocore', 'sqlalchemy'} & set(sys.modules)\n"
        "assert not leaked, leaked\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_host_bind_directories_ship_with_the_checkout() -> None:
    # Docker must never create them (as root); a clone already has them, owned by the user.
    tracked = subprocess.run(
        ["git", "ls-files", "outputs", "data/raw"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    assert set(tracked) == {"outputs/.gitkeep", "data/raw/.gitkeep"}


def test_setup_script_prepares_directories_and_env(tmp_path: Path) -> None:
    for name in ("scripts/setup-local-stack.sh", ".env.example"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text((ROOT / name).read_text())
    script = tmp_path / "scripts" / "setup-local-stack.sh"
    script.chmod(0o755)

    subprocess.run([str(script)], check=True, capture_output=True)

    assert (tmp_path / "outputs").is_dir() and (tmp_path / "data" / "raw").is_dir()
    env = (tmp_path / ".env").read_text()
    assert f"SCALERL_UID={os.getuid()}\n" in env
    assert f"SCALERL_GID={os.getgid()}\n" in env
    # Re-running keeps an existing .env untouched.
    (tmp_path / ".env").write_text(env + "# mine\n")
    subprocess.run([str(script)], check=True, capture_output=True)
    assert (tmp_path / ".env").read_text().endswith("# mine\n")


def test_runtime_image_builds_with_the_pinned_backend() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    build_lock = (ROOT / "docker" / "scalerl" / "build-requirements.txt").read_text()
    backend = re.search(r'build-backend = "(\w+)', (ROOT / "pyproject.toml").read_text())

    assert backend and backend.group(1).split(".")[0] == "hatchling"
    assert re.search(r"^hatchling==[\d.]+ \\$", build_lock, re.MULTILINE)
    assert "--hash=sha256:" in build_lock
    assert "--require-hashes -r /tmp/build-requirements.txt" in dockerfile
    assert "--no-build-isolation" in dockerfile
    # The only packages pip installs are hash-pinned or the locally built wheel.
    installs = re.findall(r"pip (?:install|wheel)[^\n]*", dockerfile)
    assert installs and all(
        "--require-hashes" in line or "--no-deps" in line for line in installs
    ), installs
