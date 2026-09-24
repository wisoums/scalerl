# ScaleRL Data

ScaleRL supports reproducible synthetic workloads and will also evaluate controllers on selected slices of the Microsoft Azure Functions Invocation Trace 2021.

Raw production traces are **not committed to this repository**.

## Azure Functions Invocation Trace 2021

Official source:

- Dataset repository: https://github.com/Azure/AzurePublicDataset
- Dataset documentation: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md
- Dataset file: `AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar`

The official dataset documentation describes two weeks of Azure Functions invocations beginning on **2021-01-31**.

### License

The dataset is published by Microsoft under a **CC-BY Attribution License**. The dataset license is separate from ScaleRL's MIT license. Anyone downloading or using the Azure trace must follow the Azure dataset's license and attribution requirements.

### Required citation

If the Azure Functions 2021 trace is used in results, cite:

> Yanqi Zhang, Íñigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, Ricardo Bianchini. "Faster and Cheaper Serverless Computing on Harvested Resources." Proceedings of the ACM Symposium on Operating Systems Principles (SOSP), October 2021.

## Local layout

After downloading and extracting the dataset, use:

```text
data/
├── README.md
├── raw/          # original/extracted Azure files; never committed
└── processed/    # deterministic cached ScaleRL slices; never committed
```

Both `data/raw/` and `data/processed/` are gitignored.

## Loader contract

Issue #43 owns the Azure loader.

The current official 2021 trace is event-level, with fields including:

- `app`
- `func`
- `end_timestamp`
- `duration`

For request-arrival demand, the ScaleRL loader should derive an invocation start time from:

```text
start_timestamp = end_timestamp - duration
```

and aggregate starts into the requested simulator control interval.

The loader must return the existing `WorkloadTrace` type so synthetic and real workloads use the same environment/controller/evaluation code.

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
- loader version / Git commit;
- any preprocessing parameters.

CI must use tiny committed fixtures under `tests/`; CI must not download the full Azure dataset.
