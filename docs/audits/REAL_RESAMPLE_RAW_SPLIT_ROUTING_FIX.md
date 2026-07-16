# Real Resample Raw Split Routing Fix

## Context

The observed failure was:

```text
synthetic_control=real_resample split=test class=0 requires 50 real rows
but only 1 are available.
```

The validated AppClassNet splits have thousands of rows per class, so a count of
one row for class 0 means `real_resample` was sampling from the wrong object.

## Root Cause

`synthesize_data()` received generator/evaluation arrays and passed those arrays
into `_synthesize_control_batches()`. That API accepted generic `x_real_samples`
and `y_real_samples`, so it could be called with transformed evaluation data,
fold dictionaries, aggregate structures, or one-row-per-class data instead of
the raw `DatasetBundle` splits.

For `split_mode=provided`, the control path must not infer the train/test source
from `dictionary_data["x_evaluation_real"]` or any synthetic/evaluation
aggregate. It must use:

- train control: `DatasetBundle.train.X` and `DatasetBundle.train.y`
- test control: `DatasetBundle.test.X` and `DatasetBundle.test.y`

`valid` is not a valid control source.

## Implementation

`main.py` now routes explicit `SplitData` objects into synthetic controls:

```python
_synthesize_control_batches(
    source_data: SplitData,
    split_name: str,
    samples_per_class: int,
    random_state: int,
)
```

The function rejects non-`SplitData` inputs and enforces
`source_data.name == split_name`. Passing `valid` or passing `test` as `train`
raises `SyntheticControlSplitMismatchError`.

Before selection, the control source is audited and validated:

- `X` must be 2D.
- `y` must be 1D.
- `X.shape[0] == y.shape[0]`.
- `X.shape[1] == 20`.
- labels must be integer encoded in `0..199` for the AppClassNet 200-class run.
- all 200 classes must be present.
- one-row-per-class aggregate matrices are rejected with
  `AggregateDataUsedAsRawSamplesError`.

Selection is now strictly raw-row resampling:

```python
class_indices = np.flatnonzero(y == class_id)
rng.choice(class_indices, size=samples_per_class, replace=False)
```

No replacement, quota reduction, centroid repetition, or `valid` fallback is
allowed.

## Manifests

The batch manifests at:

- `synthetic_batches/train/manifest.json`
- `synthetic_batches/test/manifest.json`

now record the control routing fields:

- `control_type`
- `source_split`
- `source_x_path`
- `source_y_path`
- `source_indices`
- `samples_per_class`
- `total_samples`
- `num_classes`
- `feature_count`
- `schema_hash`
- `data_space`
- `transform_id`
- `random_state`

For 50 samples per class and 200 classes, each split writes exactly 10,000 rows.

## Provided Folds

`run_appclassnet_top200.py` now sends `--number_k_folds 1` for batch
`split_mode=provided` subprocesses and prevents campaign `number_k_folds=2` from
overriding that effective fold count. Plot metadata also uses the effective fold
count.

## Tests

Added `tests/test_real_resample_control_routing.py` covering:

- train control uses `DatasetBundle.train`.
- test control uses `DatasetBundle.test`.
- `valid` is rejected.
- centroid/one-row-per-class matrices are rejected.
- 50 per class produces 10,000 rows.
- selected `X` rows remain class-aligned.
- selection is without replacement.
- manifests record `source_split` and source paths.
- `source_indices` stay within the source split bounds.
- train minimum 1388 and test minimum 1539 both allow selecting 50.
- split mismatch raises `SyntheticControlSplitMismatchError`.
- batch `provided` commands use one effective fold.

Validation run:

```text
pytest -q tests/test_real_resample_control_routing.py
pytest -q tests/test_appclassnet_evaluation_mode.py tests/test_data_loader_arguments.py tests/test_synthetic_batch_io.py tests/test_npy_xy_loader.py
python -m py_compile main.py run_appclassnet_top200.py tests/test_real_resample_control_routing.py
```

All selected tests passed.
