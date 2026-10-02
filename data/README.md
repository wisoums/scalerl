# ScaleRL Data

ScaleRL is a student learning project. The data layer is being expanded gradually so model behavior can be tested on more than one narrow traffic source.

Implemented today:
- reproducible synthetic workloads;
- selected slices of the Microsoft Azure Functions Invocation Trace 2021.

Planned benchmark-v2 sources are listed below, but a source is **not considered implemented** until #116 adds a deterministic loader/preparation path and #115 freezes its role in the benchmark.

Raw production traces are **not committed to this repository**.

## Candidate benchmark-v2 data catalog

| Source | Intended role | Status / caution |
| --- | --- | --- |
| [Azure Functions 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) | Additional real serverless invocation patterns; may also contribute execution-time/memory summaries | Planned candidate; loader not yet implemented |
| [Azure Functions Invocation Trace 2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md) | Existing real invocation trace, but benchmark-v2 may sample a broader set of apps/windows | Partially implemented today |
| [Alibaba Cluster Trace — Microservices 2021](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021) | Candidate source of real microservice call-rate/response-time diversity | Planned candidate; semantics must be mapped explicitly |
| [Google ClusterData 2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) | Possible resource/environment variability evidence | **Not an HTTP request trace**; must not be treated as interchangeable with invocation data |
| [SeBS](https://github.com/spcl/serverless-benchmarks) | Possible application/service benchmark for later systems work and calibration | **Benchmark application suite, not a request-trace dataset** |

#115 decides which sources are actually used for TRAIN, VALIDATION, and fresh TEST. #116 owns provenance-aware ingestion.

For every accepted external source, ScaleRL should record:
- official source/version;
- license;
- required citation;
- original schema and units;
- deterministic preparation version;
- source/application identity needed for leakage-safe splitting;
- trace/window fingerprint;
- limitations of how the source maps into ScaleRL.

The goal is not to turn every public systems dataset into request rate. If a source only provides resource usage, service characteristics, or application behavior, that role should remain explicit.

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
3. Place the extracted file under `data/raw/`. The archive contains `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`, which is comma-separated with the header `app,func,end_timestamp,duration` despite its `.txt` extension; pass it to the loader as-is.

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
    "data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt",
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

Each file is replaced atomically, and metadata is serialized before anything is written, so a failed save leaves an existing slice untouched. The JSON records the CSV's SHA-256 (`csv_sha256`); loading rejects a CSV that does not match its sidecar, e.g. after an interrupted save.

Processed slices are generated locally and never committed.

## Benchmark windows

Azure windows are always written as exact values (`start_seconds`, `duration_seconds`, `control_interval_seconds`), never as labels like "Monday", because timestamps were modified. The frozen v1 train/validation/test windows live in [`src/scalerl/benchmarks/v1/workloads.json`](../src/scalerl/benchmarks/v1/workloads.json); see [`benchmarks/v1/README.md`](../benchmarks/v1/README.md) for the selection rule and the local validation command.

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

## Benchmark-v2 split policy

The v1 benchmark windows remain frozen historical evidence.

Benchmark-v2 (#115) will define a **new** train/validation/test split before generalist training begins.

Important rules:
- do not place neighboring windows from the same grouped application into different splits without an explicit leakage-safe rule;
- do not train on the two #46 TEST hours merely to repair their observed failure;
- freeze fresh TEST identities before #118 training;
- keep source/domain labels for sampling and analysis, not as hidden controller hints;
- preserve dataset-specific semantics instead of forcing every source into one misleading schema.

Synthetic workloads remain useful because real datasets may not cover rare but important regimes such as overload, flash crowds, abrupt drops, or controlled regime shifts.
