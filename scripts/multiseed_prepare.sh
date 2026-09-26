#!/usr/bin/env bash
# Canonical controller preparation for multi-seed evaluation (#19). TRAIN/VALIDATION only.
#
#   scripts/multiseed_prepare.sh                          # everything, resumable
#   PYTHON=.venv/bin/python scripts/multiseed_prepare.sh  # a specific interpreter
#
# Tracking: TRACKING_URI, else MLFLOW_TRACKING_URI, else sqlite:///outputs/mlflow.db.
# Run the evaluation against the same tracking URI, since the manifest points at its runs.
#
# 1. Threshold: the exact #13 18-point grid (threshold-sla-first v1) on the synthetic
#    train/validation workloads; resumes the persisted study if interrupted.
# 2. DQN and PPO: the predeclared v1 Optuna studies (dqn-search-v1 / ppo-search-v1,
#    seeded TPE seed 42, 20 trials, 200,000 / 204,800 timesteps per trial) on
#    syn-train-bursty, selected on all three synthetic validation workloads.
# 3. The selected DQN and PPO configurations retrained with training seeds 0-4
#    (every seed kept; no seed is chosen here).
#
# Every step writes a result JSON under outputs/multiseed-v1/prep/ and is skipped when
# that file already exists. Then build the controller manifest with
#   python -m scalerl.evaluation.multiseed manifest --prep-dir outputs/multiseed-v1/prep
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
# The standard MLFLOW_TRACKING_URI wins (e.g. http://mlflow:5000 in the Compose trainer),
# so the manifest and the later evaluation see the same tracking server.
TRACKING="${TRACKING_URI:-${MLFLOW_TRACKING_URI:-sqlite:///outputs/mlflow.db}}"
PREP=outputs/multiseed-v1/prep
TRAIN=syn-train-bursty
VALIDATION=(--validation-workload syn-val-steady-high --validation-workload syn-val-ramp-down --validation-workload syn-val-bursty)
SEEDS=(0 1 2 3 4)
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p "$PREP"

step() {  # step <output file> <command...>
  local output=$1
  shift
  if [[ -f "$output" ]]; then
    echo "== skip (exists): $output"
    return
  fi
  echo "== $(date '+%H:%M:%S') $*"
  "$@"
}

step "$PREP/threshold-v1.json" "$PYTHON" -m scalerl.tuning.threshold \
  --storage "sqlite:///$PREP/threshold-optuna.db" --tracking-uri "$TRACKING" \
  --output "$PREP/threshold-v1.json"

step "$PREP/dqn-tuning-v1.json" "$PYTHON" -m scalerl.tuning.dqn \
  --train-workload "$TRAIN" "${VALIDATION[@]}" --n-trials 20 --timesteps 200000 \
  --storage "sqlite:///$PREP/dqn-optuna.db" --tracking-uri "$TRACKING" \
  --output "$PREP/dqn-tuning-v1.json"

step "$PREP/ppo-tuning-v1.json" "$PYTHON" -m scalerl.tuning.ppo \
  --train-workload "$TRAIN" "${VALIDATION[@]}" --n-trials 20 --timesteps 204800 \
  --storage "sqlite:///$PREP/ppo-optuna.db" --tracking-uri "$TRACKING" \
  --output "$PREP/ppo-tuning-v1.json"

for seed in "${SEEDS[@]}"; do
  step "$PREP/dqn-seed$seed.json" "$PYTHON" -m scalerl.training.dqn \
    --workload "$TRAIN" "${VALIDATION[@]}" --timesteps 200000 --seed "$seed" \
    --hyperparameters "$PREP/dqn-tuning-v1.json" --tracking-uri "$TRACKING" \
    --output "$PREP/dqn-seed$seed.json"
  step "$PREP/ppo-seed$seed.json" "$PYTHON" -m scalerl.training.ppo \
    --workload "$TRAIN" "${VALIDATION[@]}" --timesteps 204800 --seed "$seed" \
    --hyperparameters "$PREP/ppo-tuning-v1.json" --tracking-uri "$TRACKING" \
    --output "$PREP/ppo-seed$seed.json"
done

echo "== $(date '+%H:%M:%S') preparation complete"
