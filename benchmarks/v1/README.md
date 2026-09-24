# ScaleRL Benchmark v1

- **Version:** v1
- **Frozen on:** 2026-09-24, before any DQN/PPO tuning
- **Manifest:** [`src/scalerl/benchmarks/v1/workloads.json`](../../src/scalerl/benchmarks/v1/workloads.json), shipped as package data and loaded with `scalerl.benchmarks.load_benchmark_manifest()` (also from an installed wheel)

This benchmark freezes **which workloads belong to which split**. It does not fix controller, reward, or simulator hyperparameters.

## Splits

| Split | Use |
|---|---|
| `train` | policy learning |
| `validation` | threshold/predictive/RL hyperparameter tuning and model/checkpoint selection |
| `test` | final held-out reporting only |

The test workloads are **public, not secret**: they are held out from tuning by process. They are used only once all hyperparameters and model choices are frozen, for final multi-seed reporting.

**Forbidden uses of `test` workloads**, directly or through their statistics:

- threshold grid selection;
- predictive-controller hyperparameter selection;
- reward-weight selection;
- DQN or PPO hyperparameter tuning;
- early stopping or model/checkpoint selection;
- simulator capacity calibration (`service_capacity_rps`, replica bounds, startup delay, SLA target).

Code guardrails make accidental use harder, but they are not a security boundary:

- `manifest.train`, `manifest.validation`, and `manifest.tuning` (train + validation) never return test workloads; there is no helper that merges every split;
- each id encodes its split (`syn-train-*`, `azure-val-*`, `syn-test-*`, ...), so moving a workload between splits requires a deliberate rename;
- tests pin exact split membership, every synthetic parameter, and every Azure offset.

## Common episode contract

Every workload uses `duration_seconds = 3600.0` and `control_interval_seconds = 30.0`, i.e. **120 ticks**. Controllers compared directly share one `SimulatorConfig`, so learned policies see compatible observation/action spaces across all workloads.

## Synthetic workloads

Built with the existing generators (`scalerl.workloads`); bursty workloads reproduce exactly from their committed seeds.

| Id | Split | Generator | Parameters | Role |
|---|---|---|---|---|
| `syn-train-steady-moderate` | train | steady | rate 100 | steady-state baseline |
| `syn-train-seasonal` | train | diurnal | baseline 150, amplitude 100, period 3600 s, phase 0 s | one full smooth cycle |
| `syn-train-ramp-up` | train | ramp | 40 → 360 rps | sustained growth |
| `syn-train-spike` | train | spike | 80 → 420 rps from 1500 s for 300 s | abrupt surge |
| `syn-train-bursty` | train | bursty | baseline 90, bursts +250, p 0.08, 90 s, noise σ 10, seed 101 | stochastic bursts |
| `syn-val-steady-high` | validation | steady | rate 250 | higher steady load |
| `syn-val-ramp-down` | validation | ramp | 350 → 60 rps | sustained decline (scale-in) |
| `syn-val-bursty` | validation | bursty | baseline 120, bursts +280, p 0.12, 60 s, noise σ 15, seed 202 | denser, shorter bursts |
| `syn-test-seasonal-shifted` | test | diurnal | baseline 180, amplitude 140, period 3600 s, phase 900 s | larger, phase-shifted cycle |
| `syn-test-spike-hard` | test | spike | 60 → 480 rps from 900 s for 180 s | earlier, sharper surge |

## Azure workloads

Slices of the Azure Functions Invocation Trace 2021 (see [`data/README.md`](../../data/README.md) for source, license, citation, units, and local setup). All are 3600 s windows at 30 s intervals:

| Id | Split | `start_seconds` |
|---|---|---|
| `azure-train-129600` | train | 129600 |
| `azure-train-302400` | train | 302400 |
| `azure-train-475200` | train | 475200 |
| `azure-val-734400` | validation | 734400 |
| `azure-test-993600` | test | 993600 |
| `azure-test-1166400` | test | 1166400 |

**Selection rule:** fixed trace-relative offsets spread across the two-week trace, chosen **before** any RL evaluation and **not** based on controller performance. Offsets are trace-relative seconds (Microsoft modified the timestamps), never calendar times.

**No normalization:** Azure demand is used exactly as loaded (invocation starts per interval ÷ 30 s). Any capacity choice needed to run Azure scenarios is experiment configuration, recorded as metadata; it is never tuned on test workloads.

**Runtime source:** the manifest stores only the logical dataset and offsets, never a local file path. Pass the extracted CSV when building:

```python
from scalerl.benchmarks import build_workload, load_benchmark_manifest

manifest = load_benchmark_manifest()
trace = build_workload(
    manifest.get("azure-train-129600"), azure_csv_path="data/raw/<extracted file>.csv"
)
```

### Local validation and characterization

With the real extracted trace available locally:

```bash
python -m scalerl.benchmarks.validate_azure --azure-csv data/raw/<extracted file>.csv \
    --summary-out benchmarks/v1/azure_characterization.json
```

All six windows are extracted in a single streamed pass over the CSV. This checks that every Azure window has 120 ticks, the 30 s interval, finite non-negative rates, and at least one invocation, and reports mean, max, and standard deviation of RPS plus the coefficient of variation and peak/mean ratio. The optional summary contains only aggregate statistics and provenance, never raw events, and is safe to commit.

**Status: not yet run.** The frozen Azure windows have not been validated against the real dataset, so no Azure characterization is recorded here yet. Benchmark v1 is not final until this validation passes.

These statistics describe the frozen windows; they must not be used to change split membership or tune any setting.

## Changing the benchmark

The meaning of v1 must not change once training or tuning starts. Changing held-out membership, workload parameters, or offsets requires **a new benchmark version** or an explicit, documented correction here, made before any final evaluation. If a frozen Azure window proves unusable (outside the extracted file, empty because of source-data problems, or corrupted), document the reason and correct the benchmark explicitly. Never quietly swap in another window.
