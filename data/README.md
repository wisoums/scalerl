# ScaleRL Data

ScaleRL supports reproducible synthetic workloads and will also evaluate controllers on selected slices of the Microsoft Azure Functions Invocation Trace 2021.

Raw production traces are **not committed to this repository**.

## Azure Functions Invocation Trace 2021

Official source:

- Dataset repository: https://github.com/Azure/AzurePublicDataset
- Dataset documentation: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md
- Dataset file: `AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar`

The official dataset documentation describes two weeks of Azure Functions invocations beginning on **2021-01-31**.

### Schema, units, and timestamps

Each row is one invocation:

| Column | Meaning | Unit |
|---|---|---|
| `app` | application id (encrypted) | – |
| `func` | function id (encrypted), unique only within an application | – |
| `end_timestamp` | invocation end time | **seconds** |
| `duration` | invocation duration | **seconds** |

Both `end_timestamp` and `duration` are in **seconds**, per the dataset-specific documentation linked above. Ignore any millisecond wording elsewhere in the Azure repository.

Microsoft states that invocation timestamps were **modified** from the production trace. ScaleRL therefore treats them as **trace-relative seconds** only: windows are selected by `start_seconds`, never by calendar dates, weekdays, or times of day.

### License

The dataset is published by Microsoft under a **CC-BY Attribution License**. The dataset license is separate from ScaleRL's MIT license. Anyone downloading or using the Azure trace must follow the Azure dataset's license and attribution requirements.

### Required citation

If the Azure Functions 2021 trace is used in results, cite:

> Yanqi Zhang, Íñigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, Ricardo Bianchini. "Faster and Cheaper Serverless Computing on Harvested Resources." Proceedings of the ACM Symposium on Operating Systems Principles (SOSP), October 2021.

## Local setup (manual)

Downloading and extracting the `.rar` archive is a **manual developer step**; ScaleRL has no download or extraction code, and **CI never downloads the real dataset**.

1. Download `AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar` from the dataset documentation page above.
2. Extract it with any RAR tool (for example `unrar x` or `7z x`).
3. Place the extracted CSV under `data/raw/`.

Resulting layout:

```text
data/
├── README.md
├── raw/          # original/extracted Azure files; never committed
└── processed/    # deterministic cached ScaleRL slices; never committed
```

Both `data/raw/` and `data/processed/` are gitignored.

## Loading a slice

```python
from scalerl.workloads import load_azure_trace

trace = load_azure_trace(
    "data/raw/<extracted file>.csv",
    start_seconds=86_400.0,  # trace-relative, not Unix time
    duration_seconds=3_600.0,  # must be a whole number of control intervals
    control_interval_seconds=30.0,
)
```

The result is the ordinary `WorkloadTrace`, so Azure and synthetic workloads use the same environment, controllers, and evaluation code.

Semantics:

- **Arrivals:** each invocation arrives at `end_timestamp - duration`. Completion times are never used as demand.
- **Window:** half-open `[start_seconds, start_seconds + duration_seconds)`. An arrival exactly on an internal interval boundary belongs to the later interval.
- **Rates:** each interval's arrival count becomes `count / control_interval_seconds` requests per second. Intervals without arrivals are explicit `0.0` samples; the trace always has `duration_seconds / control_interval_seconds` ticks.
- **No reshaping:** demand is never normalized, clipped, or rescaled to fit simulator capacity. Any capacity or scaling choice for an Azure scenario belongs to benchmark configuration (#18) and must be recorded as experiment metadata.
- **Large files:** the CSV is streamed in chunks (`chunk_size`, default 1,000,000 rows) with exact float parsing; results do not depend on chunk size or row order.
- **Validation:** missing files or columns, non-numeric, NaN, or infinite values, and negative durations raise an error instead of being dropped.

## Processed slices

Slices can be cached under `data/processed/` as two plain files:

```python
from scalerl.workloads import azure_trace_metadata, save_processed_trace, load_processed_trace

save_processed_trace(
    trace,
    "data/processed/azure-example.csv",
    azure_trace_metadata(
        source_csv,
        start_seconds=...,
        duration_seconds=...,
        control_interval_seconds=...,
    ),
)
trace, metadata = load_processed_trace("data/processed/azure-example.csv")
```

- `azure-example.csv`: `tick,request_rate` rows (rates written exactly, so reloading is lossless);
- `azure-example.json`: dataset name, source file name, `start_seconds`, `duration_seconds`, `control_interval_seconds`, `tick_count`, loader, ScaleRL version, and `format_version`.

Processed slices are generated locally and never committed.

## Candidate windows

Candidate windows for different traffic shapes must be written as exact values, for example `start_seconds=…, duration_seconds=…, control_interval_seconds=…`, never as vague labels like "Monday" or "day 3 afternoon", because timestamps were modified. None are recorded yet: choosing them requires inspecting the real trace, and freezing any selection into train/validation/test sets belongs to #18.

## Train / validation / test separation

Do not decide the final held-out test set ad hoc during model tuning.

Issue #18 owns the versioned workload manifest that freezes:

- training slices;
- validation/tuning slices;
- held-out test slices.

Held-out traces must not be used for threshold tuning, reward tuning, DQN/PPO hyperparameter selection, or model selection.

## Reproducibility

Processed slices should be reconstructable from:

- source dataset identifier;
- source slice/time range;
- aggregation interval;
- loader version / Git commit (the processed JSON records the ScaleRL version);
- any preprocessing parameters.

CI must use tiny committed fixtures under `tests/`; CI must not download the full Azure dataset.
