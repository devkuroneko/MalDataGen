# Batches Backward Compatibility Report

Date: 2026-07-10

Scope: final compatibility review for the AppClassNet batches implementation against the original MalDataGen CSV flow.

## Commands Executed

### 1. Existing Tests

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python -m unittest discover -s tests
```

Result: passed.

```text
Ran 69 tests in 0.091s
OK
```

Initial run found three compatibility regressions in isolated test owners:

- `CrossValidation.StratifiedData` assumed `resource_timer` existed.
- `TrTs` and `TsTr` assumed `self.arguments` existed.

Fix applied: added fallbacks so instrumentation is optional when these mixins are used by legacy tests or simple owners.

### 2. Legacy MalDataGen CSV Command

A temporary CSV was created at `/tmp/maldatagen_legacy.csv` with 20 rows, 3 feature columns and a binary label column.

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python main.py \
  -i /tmp/maldatagen_legacy.csv \
  --data_load_label_column -1 \
  --model_type random \
  --number_k_folds 2 \
  --number_samples_per_class 0:4,1:4 \
  -c DecisionTree \
  --output_dir outputs/compat_csv_legacy_2026-07-10_05-52-00 \
  --data_type binary \
  --target_type binary \
  --verbosity 20
```

Result: passed. The run completed both folds and wrote:

```text
outputs/compat_csv_legacy_2026-07-10_05-52-00/EvaluationResults/Results.json
outputs/compat_csv_legacy_2026-07-10_05-52-00/EvaluationResults/metrics.json
```

Compatibility checks from the log:

- `data_format: csv`
- `execution_mode: normal`
- `sample_plan: legacy`
- `number_samples_per_class: {'classes': {0: 4, 1: 4}, 'number_classes': 2}` was accepted at CLI parse time.

Approximate resource usage from `metrics.json`:

- `loading` current memory: `605.602` MB
- `loading` peak memory: `603.855` MB

The absolute value includes TensorFlow/import overhead in this environment; use it as a rough smoke-test reading, not a benchmark.

### 3. AppClassNet Demo, Normal Dry-Run Memory

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python run_appclassnet_top200.py \
  -c sf \
  --execution_mode normal \
  --dry_run_memory
```

Result: passed. The runner stopped after the memory estimate.

Observed memory plan:

- `execution_mode=normal`
- `train_x.npy` shape: `(4347270, 20)`, dtype `float64`, approximate raw feature size `663.34 MiB`
- `train_y.npy` shape: `(4347270,)`, dtype `int16`, approximate raw label size `8.29 MiB`
- global one-hot upper bound: `3.24 GiB`
- default batch one-hot estimate at `batch_size=8192`: `6.25 MiB`

### 4. AppClassNet Demo, Batches Dry-Run Memory

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python run_appclassnet_top200.py \
  -c sf \
  --execution_mode batches \
  --use_mmap \
  --batch_size 1024 \
  --max_samples_per_class 10 \
  --dry_run_memory
```

Result: passed. The runner stopped after the memory estimate.

Observed memory plan:

- `execution_mode=batches`
- `use_mmap=True`
- `batch_size=1024`
- `max_samples_per_class=10`
- raw feature size: `663.34 MiB`
- raw label size: `8.29 MiB`
- global one-hot upper bound: `3.24 GiB`
- batch one-hot estimate at `batch_size=1024`: `800.00 KiB`
- feature batch estimate at `batch_size=1024`: `80.00 KiB`
- batches classifier default: `sgd`
- synthetic format default: `npy_batches`

### 5. Batches Output Directory Dry-Run

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python run_appclassnet_top200.py \
  -c sf \
  --execution_mode batches \
  --dryrun \
  --skip_plots \
  --max_samples_per_class 10 \
  --use_mmap
```

Result: passed.

The generated child command targets:

```text
outputs/appclassnet_top200_demo_batches/...
```

This avoids overwriting the original demo folder:

```text
outputs/appclassnet_top200_demo
```

### 6. Defaults and Loader Imports

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python -c "from Engine.Arguments.ArgumentsDataLoader import DEFAULT_DATA_FORMAT, DEFAULT_EXECUTION_MODE; import run_appclassnet_top200 as r; p=r.build_parser(); a=p.parse_args([]); print(DEFAULT_DATA_FORMAT, DEFAULT_EXECUTION_MODE, a.execution_mode)"
```

Result:

```text
csv normal normal
```

```bash
PYTHONPYCACHEPREFIX=/tmp/maldatagen_pycache .venv/bin/python -c "from Engine.DataIO.CSVLoader import CSVDataProcessor; from Engine.DataIO.XLSLoader import XLSDataProcessor; from Engine.DataIO.NpyXYLoader import NpyXYLoader; print(CSVDataProcessor.__name__, XLSDataProcessor.__name__, NpyXYLoader.__name__)"
```

Result:

```text
CSVDataProcessor XLSDataProcessor NpyXYLoader
```

## Compatibility Findings

- Original CSV loading remains the default: `data_format=csv`.
- AppClassNet runner default execution remains `normal`.
- Existing CSV and XLS loaders still import successfully.
- `number_samples_per_class` still works in the legacy CSV path.
- Existing metric blocks and metric names are preserved:
  - `TR-TS`
  - `TS-TR`
  - `TR-TR`
  - `DistanceMetrics`
  - `EfficiencyMetrics`
  - binary metric names such as `Accuracy`, `Precision`, `Recall`, `F1Score`, `Specificity`, `FalsePositiveRate`, `TrueNegativeRate`, `MeanSquareError`, `MeanAbsoluteError`, `TruePositive`, `FalsePositive`, `TrueNegative`, `FalseNegative`
- New metadata is additive:
  - `BatchClassifier`
  - `ResourceUsage`
- `Results.json` remains written.
- `metrics.json` is additionally written for resource and batch metadata.
- Batches demo output is separated under `outputs/appclassnet_top200_demo_batches`.

## Corrections Made During Review

1. Made resource instrumentation optional for test/legacy owners without `resource_timer`.
2. Made `TrTs` and `TsTr` tolerate owners without `arguments`, preserving isolated evaluator use.
3. Changed AppClassNet batches demo output to `outputs/appclassnet_top200_demo_batches` to avoid overwriting `outputs/appclassnet_top200_demo`.

## Remaining Risks

- The AppClassNet full training path was not executed end-to-end in this review because it is intentionally expensive.
- The requested AppClassNet checks were dry-run memory commands; they validate data discovery, mmap planning and command construction, but not full model convergence.
- TensorFlow emits CUDA factory warnings and Matplotlib cache warnings in this environment. They did not fail tests or smoke runs.
- Peak memory from Linux `resource` and current RSS from `psutil` can differ slightly because they come from different APIs and timing points.
- Batches mode uses batch-friendly classifiers by default, so accuracy may differ from normal mode if normal mode uses the legacy classifier list.

## 48 GB RAM Recommendations

Start batches mode conservatively:

```bash
python3 run_appclassnet_top200.py -c sf \
  --execution_mode batches \
  --use_mmap \
  --batch_size 4096 \
  --eval_batch_size 4096 \
  --generation_batch_size 4096 \
  --max_samples_per_class 500 \
  --batch_classifier sgd
```

Then scale:

1. `--max_samples_per_class 500`
2. `--max_samples_per_class 1000`
3. `--max_samples_per_class 3000`
4. Remove `--max_samples_per_class` only after the previous run fits comfortably.

If RAM pressure remains high, lower all batch sizes in this order:

```text
4096 -> 2048 -> 1024
```

Prefer:

- `--batch_classifier sgd`
- `--batch_classifier passive_aggressive`

Avoid full RandomForest in batches mode. If a tree-based reference is needed, prefer:

```bash
--batch_classifier random_forest_subset --batch_classifier_subset_size 100000
```

and lower `--batch_classifier_subset_size` if memory usage is still high.
