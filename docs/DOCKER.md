# Docker: the full local stack

ScaleRL runs in two modes:

| Mode | What runs | Use it for |
|---|---|---|
| **Lightweight local** | `pip install`, SQLite MLflow/Optuna files | everyday development, tests, quick runs |
| **Full local stack** | Docker Compose: Scenario Lab, trainer, MLflow server, Optuna Dashboard, PostgreSQL, S3-compatible artifact store | reproducible experiments, the portfolio setup, the base for CI (#45) and platform navigation (#59) |

Both use the same code: ScaleRL reads `MLFLOW_TRACKING_URI` and an Optuna storage URI, so only configuration changes.

> **Local development/reproducibility only.** This stack is not a hardened, internet-facing production deployment. It uses example credentials from `.env`, has no TLS or authentication in front of the web UIs, and binds their host ports to `127.0.0.1` only, so they are not reachable from other machines on your network.

## Lightweight local mode (SQLite)

```bash
pip install -e ".[mlops,tuning]"
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m scalerl.tuning.threshold   # Optuna study in sqlite:///outputs/threshold-optuna.db
```

Nothing here needs Docker. See [MLOPS.md](MLOPS.md).

## Full local stack

```bash
scripts/setup-local-stack.sh   # once: creates .env, outputs/, data/raw/
docker compose up --build
```

`scripts/setup-local-stack.sh` creates `.env` from `.env.example` with `SCALERL_UID`/`SCALERL_GID` set to your user, and makes sure `./outputs` and `./data/raw` exist and belong to you. (Both folders already ship with the checkout.) It never overwrites an existing `.env`. `cp .env.example .env` also works on macOS; on Linux, set the UID/GID yourself.

Then open:

| Service | Browser URL |
|---|---|
| **ScaleRL** Scenario Lab / Live City | <http://localhost:8501> |
| **MLflow** UI | <http://localhost:5000> |
| **Optuna Dashboard** | <http://localhost:8080> |

The Scenario Lab starts on its own; it does not wait for MLflow or Optuna, and needs no Azure data.

**Port already in use?** On macOS, **AirPlay Receiver (Control Center) listens on port 5000**. Either turn it off (System Settings → General → AirDrop & Handoff → AirPlay Receiver) or pick other host ports in `.env` and keep the browser URLs in sync:

```bash
MLFLOW_UI_PORT=5001
SCALERL_MLFLOW_UI_URL=http://localhost:5001
SCALERL_UI_PORT=8502   # e.g. while a local `python -m scalerl.dashboard` uses 8501
```

Only host ports change; containers still talk to `http://mlflow:5000`.

### Architecture

```text
                          browser (localhost)
             ┌──────────────────┼───────────────────┐
             ▼                  ▼                   ▼
        scalerl-ui           mlflow          optuna-dashboard
          :8501               :5000               :8080
             │            ┌─────┴──────┐            │
    real ScaleRL package  ▼            ▼            ▼
                      PostgreSQL    Garage       PostgreSQL
                      `mlflow` DB   (S3 API)     `optuna` DB
                                    bucket `mlflow`

trainer (on demand) ── MLFLOW_TRACKING_URI=http://mlflow:5000
                    └─ OPTUNA_STORAGE_URI=postgresql+psycopg://…@postgres:5432/optuna
```

| Service | Role |
|---|---|
| `scalerl-ui` | `python -m scalerl.dashboard` on `0.0.0.0:8501` (ScaleRL runtime image) |
| `trainer` | the same image for evaluation/tuning/training commands; runs on demand and exits |
| `mlflow` | MLflow 3.16.1 tracking server + UI; metadata in PostgreSQL, artifacts proxied to Garage |
| `optuna-dashboard` | the native Optuna Dashboard on the PostgreSQL `optuna` database (it first creates Optuna's tables if no study has run yet, since the dashboard itself never creates them) |
| `postgres` | one PostgreSQL server with two separate databases: `mlflow` and `optuna` |
| `garage` | **S3-compatible artifact store (Garage)** holding MLflow artifacts |
| `garage-init` | one-shot: node layout, S3 access key, and the MLflow bucket (idempotent) |

Startup order uses health checks, not sleeps: `postgres` healthy → `optuna-dashboard`; `garage` healthy → `garage-init` completes → (with `postgres` healthy) `mlflow`.

**Ownership stays as in #54.** Optuna owns studies, trials, search state, and selected parameters. MLflow owns workload runs, metrics, artifacts, and lineage. One trial links to several MLflow run IDs. The two databases never duplicate each other.

### Internal vs browser URLs

Containers reach each other by Compose service name; your browser cannot resolve those names.

| | Inside Compose (containers) | Browser (host) |
|---|---|---|
| MLflow | `MLFLOW_TRACKING_URI=http://mlflow:5000` | `SCALERL_MLFLOW_UI_URL=http://localhost:5000` |
| Optuna | `OPTUNA_STORAGE_URI=postgresql+psycopg://…@postgres:5432/optuna` | `SCALERL_OPTUNA_DASHBOARD_URL=http://localhost:8080` |
| ScaleRL UI | `http://scalerl-ui:8501` | `http://localhost:8501` |

The `SCALERL_*_URL` variables are passed to the Scenario Lab container for the later platform navigation (#59). Never put `http://mlflow:5000` in anything a browser opens.

### Commands

```bash
docker compose up -d --build        # start (detached) and build
docker compose ps                   # status and health
docker compose logs -f mlflow       # logs of one service (omit the name for all)
docker compose restart scalerl-ui   # restart one service
docker compose down                 # stop; keeps all experiment history
docker compose up -d                # start again later
docker compose down -v              # DESTRUCTIVE reset, see below
```

### Trainer

The trainer is not a daemon. Each command runs in a fresh container that exits when it finishes. Its dependencies (`mlflow`, `postgres`) are started automatically:

```bash
# Predictive baseline on one train workload, tracked in the MLflow server
docker compose run --rm trainer \
    python -m scalerl.evaluation.predictive --workload syn-train-ramp-up

# Threshold study: Optuna storage defaults to $OPTUNA_STORAGE_URI (PostgreSQL)
docker compose run --rm trainer \
    python -m scalerl.tuning.threshold --output outputs/threshold-v1.json
```

Runs appear at <http://localhost:5000> and studies at <http://localhost:8080>. Small result files such as `outputs/threshold-v1.json` land in the host's `./outputs` (bind-mounted). Durable experiment artifacts, including future trained models, go to MLflow, so nothing important stays inside an exited container.

The trainer runs as `SCALERL_UID:SCALERL_GID` from `.env`, which the setup script sets to your user, so on Linux it can write `./outputs` and its files belong to you. The `./outputs` and `./data/raw` bind mounts use `create_host_path: false`: if a folder is missing, Compose fails with "bind source path does not exist" instead of silently creating it as root. Run `scripts/setup-local-stack.sh` to fix that; if Docker already created one as root, run `sudo chown -R "$(id -u):$(id -g)" outputs data/raw`.

### Azure data

The raw Azure trace (~291 MB) is **never** part of the Docker build context or any image; `.dockerignore` is an allow-list of `pyproject.toml`, `README.md`, `LICENSE`, `src/`, and `docker/`. At runtime `./data/raw` is bind-mounted **read-only** at `/app/data/raw`, so the default path `data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt` works in the UI and trainer when you have extracted it (see [data/README.md](../data/README.md)). Everything works with an empty `data/raw` (it ships with only a `.gitkeep`); nothing is downloaded automatically. Because the folder belongs to you, you can drop the extracted trace into it at any time.

### Persistence

| Named volume | Holds |
|---|---|
| `postgres-data` | MLflow and Optuna metadata (runs, params, metrics, studies, trials) |
| `garage-meta`, `garage-data` | MLflow artifacts in the `mlflow` bucket |

`docker compose down` keeps these volumes; `docker compose up -d` brings back every run, artifact, and study.

**`docker compose down -v` deletes the volumes: all local MLflow runs, artifacts, and Optuna studies are gone for good.** Use it only to reset the stack. `./outputs` is a host folder and is not touched.

### PostgreSQL

One pinned PostgreSQL server; `docker/postgres/init-databases.sh` creates the `mlflow` and `optuna` databases on first start, both owned by `POSTGRES_USER`. MLflow tables and Optuna tables never share a database. PostgreSQL is not published to the host; only containers on the Compose network reach it.

### S3-compatible artifact store (Garage)

MLflow artifacts live in [Garage](https://garagehq.deuxfleurs.fr/), a maintained, lightweight S3-compatible object store, in the bucket `MLFLOW_BUCKET` (default `mlflow`). The MLflow server runs with `--serve-artifacts` and `MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow`: it **proxies** artifact uploads, listings, and downloads, so ScaleRL clients only need `MLFLOW_TRACKING_URI` and never hold S3 credentials. Only the MLflow server has the S3 key.

`garage-init` configures Garage through its Admin API on every `up`. It assigns the single node a layout, imports the S3 key from `.env`, creates the bucket, and grants the key access. It checks before each step, so re-runs are no-ops. Garage's ports are not published to the host; browse artifacts in the MLflow UI.

Nothing in ScaleRL or MLflow's tracking contract depends on Garage specifically: any S3-compatible store can replace it by changing `MLFLOW_S3_ENDPOINT_URL`, the credentials, and the init step. (Garage replaced MinIO here because MinIO's open-source project was archived and its images are no longer published.)

### Secrets and `.env`

- Run `scripts/setup-local-stack.sh` (or `cp .env.example .env`) before the first launch; `.env` is gitignored and never copied into an image (no `COPY .env`, no secret build `ARG`s).
- The example values are **local development placeholders**, not production-safe secrets. Replace them if you like: `openssl rand -hex 32` for `GARAGE_RPC_SECRET` and `GARAGE_SECRET_ACCESS_KEY`, and `GK` + 24 hex characters for `GARAGE_ACCESS_KEY_ID`. Keep passwords URL-safe, because they appear in database URIs.
- Compose refuses to start with a clear message if a required variable is missing.
- Garage credentials changed after the first start are not re-applied to existing volumes; reset with `docker compose down -v` in that case.

### Smoke test

```bash
scripts/compose-smoke.sh                 # uses the running stack or starts it
SMOKE_BUILD=1 scripts/compose-smoke.sh   # rebuild images first
MLFLOW_UI_PORT=5001 scripts/compose-smoke.sh   # shell variables override .env
```

It first runs `scripts/setup-local-stack.sh`, waits for every health check, and checks the three browser URLs. From the trainer on the Compose network, it then:

- checks the trainer can write the bind-mounted `outputs/` (the Linux UID/GID setup);

- runs a real tracked ScaleRL evaluation: the predictive baseline on `syn-train-steady-moderate`, a synthetic TRAIN workload that needs no Azure data;
- checks the run finished with params and metrics;
- lists and downloads its `scalerl/` metadata artifacts through the MLflow proxy (so they were stored in Garage);
- creates a throwaway Optuna study in PostgreSQL, confirms the Optuna Dashboard lists it, and deletes it;
- checks the Scenario Lab health endpoint.

The tiny MLflow run stays in the `scalerl-compose-smoke` experiment as evidence. It needs no GPU, Azure data, or DQN training; it is meant for CI in #45.

### Pinned versions

| Component | Version |
|---|---|
| Python base image (ScaleRL, MLflow, Optuna Dashboard, garage-init) | `python:3.12.14-slim-trixie` |
| PostgreSQL | `postgres:18.6-trixie` |
| Garage | `dxflrs/garage:v2.4.1` |
| MLflow server | `mlflow==3.16.1` (+ `psycopg[binary]==3.3.6`, `boto3==1.43.102`) |
| Optuna Dashboard | `optuna-dashboard==0.21.0` with `optuna==5.0.0` |
| PyTorch (runtime image) | `torch==2.14.0+cpu` |
| Build backend (ScaleRL wheel) | `hatchling==1.32.4` |

Python dependencies of each image, and the `hatchling` build backend that builds the ScaleRL wheel (`docker/scalerl/build-requirements.txt`, used without build isolation), are fully pinned with hashes in `docker/*/*requirements.txt`, generated by `scripts/compile-docker-requirements.sh` (rerun it after changing dependencies). The project itself keeps its version ranges in `pyproject.toml`; there is no global lock file. All images are multi-architecture (amd64 and arm64), so the same `compose.yaml` works on Apple Silicon and x86_64 Linux without emulation or a GPU.
