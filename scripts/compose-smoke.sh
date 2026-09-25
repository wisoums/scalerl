#!/usr/bin/env bash
# Smoke-test the local Docker Compose stack (reusable by CI in #45).
#
#   scripts/compose-smoke.sh            # start (if needed) and check the stack
#   SMOKE_BUILD=1 scripts/compose-smoke.sh   # rebuild images first
#   MLFLOW_UI_PORT=5001 scripts/compose-smoke.sh   # when a host port is taken
#
# Needs Docker and a .env (created from .env.example when missing). No Azure
# data, GPU, DQN training, or paid services. Leaves the stack running; stop it
# with `docker compose down` (add -v only to delete all local experiment state).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "No .env found: creating it from .env.example (local development values)."
  cp .env.example .env
fi
# Load .env without overriding variables already set in the shell (same precedence
# as Compose), e.g. SCALERL_UI_PORT=8502 scripts/compose-smoke.sh
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  [[ -n "${!key+x}" ]] || export "$key=$value"
done < .env
mkdir -p outputs data/raw

up_args=(up -d --wait --wait-timeout 300)
[[ "${SMOKE_BUILD:-0}" == "1" ]] && up_args+=(--build)

echo "== starting services and waiting for health checks"
docker compose "${up_args[@]}"
docker compose ps

echo "== browser URLs answer on the host"
check_url() {
  local name=$1 url=$2
  if curl -fsS --max-time 10 -o /dev/null "$url"; then
    echo "ok   $name $url"
  else
    echo "FAIL: $name not reachable at $url" >&2
    exit 1
  fi
}
check_url "ScaleRL UI" "http://localhost:${SCALERL_UI_PORT:-8501}/_stcore/health"
check_url "MLflow" "http://localhost:${MLFLOW_UI_PORT:-5000}/health"
check_url "Optuna Dashboard" "http://localhost:${OPTUNA_DASHBOARD_PORT:-8080}/api/meta"

echo "== ScaleRL run, MLflow artifacts, and Optuna storage from the trainer"
docker compose run --rm -T trainer python - < scripts/compose_smoke.py

echo "compose smoke: passed"
