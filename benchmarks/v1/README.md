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
    manifest.get("azure-train-129600"),
    azure_csv_path="data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt",
)
```

### Local validation and characterization

With the real extracted trace available locally:

```bash
python -m scalerl.benchmarks.validate_azure --azure-csv data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt \
    --summary-out benchmarks/v1/azure_characterization.json
```

All six windows are extracted in a single streamed pass over the CSV. This checks that every Azure window has 120 ticks, the 30 s interval, finite non-negative rates, and at least one invocation, and reports mean, max, and standard deviation of RPS plus the coefficient of variation and peak/mean ratio. The optional summary contains only aggregate statistics and provenance, never raw events, and is safe to commit.

**Status: validated on 2026-09-24** against the extracted `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt` (1,980,951 invocations, timestamps spanning 0–1,209,600 s). All six frozen windows lie inside the trace and passed every check. The aggregate summary is committed as [`azure_characterization.json`](azure_characterization.json). The per-window counts were cross-checked against an independent recount with Python's `csv` module, and they matched exactly.

| Id | Split | Invocations | Mean RPS | Max RPS | Std RPS | CV | Peak/mean |
|---|---|---|---|---|---|---|---|
| `azure-train-129600` | train | 8,237 | 2.288 | 4.900 | 0.397 | 0.174 | 2.142 |
| `azure-train-302400` | train | 4,968 | 1.380 | 4.067 | 0.921 | 0.668 | 2.947 |
| `azure-train-475200` | train | 1,636 | 0.454 | 2.167 | 0.307 | 0.676 | 4.768 |
| `azure-val-734400` | validation | 3,333 | 0.926 | 2.467 | 0.741 | 0.801 | 2.664 |
| `azure-test-993600` | test | 4,742 | 1.317 | 3.900 | 0.728 | 0.553 | 2.961 |
| `azure-test-1166400` | test | 5,444 | 1.512 | 3.800 | 0.908 | 0.600 | 2.513 |

Invocations are arrivals (`end_timestamp - duration`) inside the window; rates are invocations per 30 s interval ÷ 30.

These statistics describe the frozen windows; they must not be used to change split membership or tune any setting. In particular, any service-capacity choice for Azure experiments must be justified from train/validation windows only, never from the test rows above.

## Changing the benchmark

The meaning of v1 must not change once training or tuning starts. Changing held-out membership, workload parameters, or offsets requires **a new benchmark version** or an explicit, documented correction here, made before any final evaluation. If a frozen Azure window proves unusable (outside the extracted file, empty because of source-data problems, or corrupted), document the reason and correct the benchmark explicitly. Never quietly swap in another window.
