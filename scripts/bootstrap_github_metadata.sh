#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-wisoums/scalerl}"

create_label() {
  local name="$1" color="$2" description="$3"
  gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force >/dev/null
}

ensure_milestone() {
  local title="$1" description="$2"
  local existing
  existing="$(gh api "repos/$REPO/milestones?state=all&per_page=100" --jq ".[] | select(.title == \"$title\") | .number" | head -n1)"
  if [[ -z "$existing" ]]; then
    gh api --method POST "repos/$REPO/milestones" -f title="$title" -f description="$description" >/dev/null
  fi
}

set_issue() {
  local number="$1" milestone="$2"
  shift 2
  local args=(--repo "$REPO" --milestone "$milestone")
  local label
  for label in "$@"; do
    args+=(--add-label "$label")
  done
  gh issue edit "$number" "${args[@]}" >/dev/null
}

# Area labels
create_label "area: environment" "0E8A16" "Simulator and Gymnasium environment"
create_label "area: workloads" "1D76DB" "Traffic generation and workload traces"
create_label "area: baselines" "5319E7" "Non-RL autoscaling controllers"
create_label "area: rl" "D93F0B" "Reinforcement-learning agents and training"
create_label "area: evaluation" "FBCA04" "Benchmarks, metrics, statistics, and experiments"
create_label "area: serving" "0052CC" "Inference API and production-inspired integration"
create_label "area: infra" "006B75" "CI, packaging, Docker, and developer infrastructure"
create_label "area: demo" "BFDADC" "Dashboard and portfolio demo"
create_label "area: docs" "C5DEF5" "Documentation and research reporting"

# Type labels
create_label "type: feature" "A2EEEF" "New implementation work"
create_label "type: experiment" "F9D0C4" "ML or systems experiment"
create_label "type: test" "D4C5F9" "Testing and validation"
create_label "type: docs" "0075CA" "Documentation work"
create_label "type: research" "D876E3" "Research/design investigation"
create_label "type: chore" "EDEDED" "Project maintenance and setup"

# Priority labels
create_label "P0" "B60205" "Critical path"
create_label "P1" "D93F0B" "Important"
create_label "P2" "FBCA04" "Later / polish"

# Milestones
ensure_milestone "M0 — Project Foundation" "Repository structure, CI, documentation, and engineering conventions."
ensure_milestone "M1 — Cloud Simulation Environment" "Deterministic workload, capacity, latency, cost, and Gymnasium environment."
ensure_milestone "M2 — Traditional Baselines" "Static, reactive, and predictive autoscaling controllers."
ensure_milestone "M3 — Deep RL Agents" "DQN, PPO, and reproducible training infrastructure."
ensure_milestone "M4 — Evaluation & Benchmarking" "Held-out workloads, multi-seed evaluation, statistics, and ablations."
ensure_milestone "M5 — Advanced Workloads" "Failures, non-stationarity, and distribution shifts."
ensure_milestone "M6 — Production-Inspired System" "Shadow-mode API, safety fallback, and containerization."
ensure_milestone "M7 — Dashboard & Demo" "Interactive visualization and portfolio demo."
ensure_milestone "M8 — Research Results / v1.0" "Final reproducible benchmark, findings, limitations, and release."

# Issue metadata
set_issue 1  "M0 — Project Foundation"           "type: chore"      "area: infra"       "area: docs"       "P0"
set_issue 2  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "P0"
set_issue 3  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "P0"
set_issue 4  "M1 — Cloud Simulation Environment" "type: feature"    "area: workloads"   "P0"
set_issue 5  "M1 — Cloud Simulation Environment" "type: feature"    "area: workloads"   "P0"
set_issue 6  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "P0"
set_issue 7  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "P0"
set_issue 8  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "P0"
set_issue 9  "M1 — Cloud Simulation Environment" "type: feature"    "area: environment" "area: rl"         "P0"
set_issue 10 "M1 — Cloud Simulation Environment" "type: test"       "area: environment" "P0"
set_issue 11 "M2 — Traditional Baselines"        "type: feature"    "area: baselines"   "P0"
set_issue 12 "M2 — Traditional Baselines"        "type: feature"    "area: baselines"   "P0"
set_issue 13 "M2 — Traditional Baselines"        "type: feature"    "area: baselines"   "P1"
set_issue 14 "M2 — Traditional Baselines"        "type: feature"    "area: baselines"   "P1"
set_issue 15 "M3 — Deep RL Agents"               "type: feature"    "area: rl"          "P0"
set_issue 16 "M3 — Deep RL Agents"               "type: feature"    "area: rl"          "P1"
set_issue 17 "M3 — Deep RL Agents"               "type: feature"    "area: evaluation"  "area: infra"      "P1"
set_issue 18 "M4 — Evaluation & Benchmarking"    "type: experiment" "area: evaluation"  "area: workloads"  "P0"
set_issue 19 "M4 — Evaluation & Benchmarking"    "type: experiment" "area: evaluation"  "P0"
set_issue 20 "M4 — Evaluation & Benchmarking"    "type: experiment" "area: evaluation"  "area: rl"         "P1"
set_issue 21 "M5 — Advanced Workloads"           "type: experiment" "area: workloads"   "area: evaluation" "P1"
set_issue 22 "M6 — Production-Inspired System"   "type: feature"    "area: serving"     "P1"
set_issue 23 "M6 — Production-Inspired System"   "type: feature"    "area: serving"     "area: infra"      "P0"
set_issue 24 "M6 — Production-Inspired System"   "type: chore"      "area: infra"       "P1"
set_issue 25 "M7 — Dashboard & Demo"             "type: feature"    "area: demo"        "P2"
set_issue 26 "M8 — Research Results / v1.0"      "type: research"   "area: evaluation"  "area: docs"       "P1"

echo "GitHub metadata configured for $REPO"
