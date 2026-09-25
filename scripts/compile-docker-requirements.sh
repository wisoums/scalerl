#!/usr/bin/env bash
# Regenerate the hash-pinned Linux requirements used by the Docker images.
#
# The project itself has no global lock file; these files pin the container
# environments only. Run from the repository root after changing pyproject.toml
# dependencies or docker/*/requirements.in, then rebuild: docker compose build
set -euo pipefail

cd "$(dirname "$0")/.."

compile() {
  uv pip compile "$@" \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --generate-hashes \
    --quiet
}

# ScaleRL runtime (UI + trainer): CPU-only PyTorch so x86_64 does not pull CUDA.
compile pyproject.toml \
  --extra mlops --extra dashboard --extra tuning --extra postgres \
  --torch-backend cpu \
  --custom-compile-command scripts/compile-docker-requirements.sh \
  --output-file docker/scalerl/requirements.txt

# Build backend for the ScaleRL wheel, so image builds never fetch an unpinned hatchling.
compile docker/scalerl/build-requirements.in \
  --custom-compile-command scripts/compile-docker-requirements.sh \
  --output-file docker/scalerl/build-requirements.txt

compile docker/mlflow/requirements.in \
  --custom-compile-command scripts/compile-docker-requirements.sh \
  --output-file docker/mlflow/requirements.txt

compile docker/optuna-dashboard/requirements.in \
  --custom-compile-command scripts/compile-docker-requirements.sh \
  --output-file docker/optuna-dashboard/requirements.txt
