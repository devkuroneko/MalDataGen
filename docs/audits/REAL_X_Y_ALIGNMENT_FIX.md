# Real X/Y Alignment Fix

## Summary

The observed failure was caused by an X/y pair from different splits reaching
`SyntheticSanityChecks._stratified_real_subset`.

In incremental batch generation, `_synthesize_data_incremental` could call the
sanity checker with:

- `real_x = self._current_evaluation_source_x`
- `real_y = y_real_samples`

When the incremental split was `train`, `y_real_samples` was the train/fold
label vector, while `_current_evaluation_source_x` was the validation/test
feature matrix. That made indices generated from `real_y` valid for the train
label length but invalid for the evaluation X length. This matches the reported
shape family: `real_x` had 81,832 rows while `real_y` still represented a larger
split with labels beyond index 98,070.

The caller now passes the same aligned pair to the sanity checker:

- `real_x = x_real_samples`
- `real_y = y_real_samples`

No truncation, clipping, or silent index removal was added.

## Validation Contract

`Engine/DataIO/DatasetContracts.py` now provides the central
`validate_xy_alignment(x, y, dataset_name, split=None, fold=None,
source_indices=None)` contract.

It validates:

- X is 2D;
- y is normalized to 1D;
- `X.shape[0] == y.shape[0]`;
- labels are finite;
- errors include split, fold, shapes, source indices, object ids, class count,
  and label min/max.

The function still returns `(X, y)` for compatibility with existing callers.

`AlignedDataset` validates on creation and carries:

- `X`;
- `y`;
- `split_name`;
- `fold_id`;
- `source_indices`;
- `data_space`;
- `transform_id`.

## Flow Changes

### CrossValidation

`Engine/Evaluation/CrossValidation.py` now validates X/y:

- after loading;
- after initial shuffle;
- after each fold train/evaluation shuffle;
- before and after batch stratified selection.

K-fold construction now uses indices against the same shuffled arrays:

- `fold_train_x = shuffled_data[train_index]`;
- `fold_train_y = shuffled_labels[train_index]`;
- `fold_test_x = shuffled_data[val_index]`;
- `fold_test_y = shuffled_labels[val_index]`.

The previous CSV materialization used indices from `shuffled_data` against
`self._data_loaded`, which was corrected.

Each fold records:

- `training_source_indices`;
- `evaluation_source_indices`.

### Synthetic Sanity Checks

`Engine/DataIO/SyntheticSanityChecks.py` now:

- accepts `AlignedDataset`;
- validates real X/y on construction;
- logs fold, split, shapes, object ids, class count, and label min/max;
- validates before and after `_stratified_real_subset`;
- checks `max_selected_index` before NumPy indexing;
- raises contextual `ValueError`/`RuntimeError` instead of leaking a generic
  NumPy `IndexError`.

### Batch Stratified Selection

Batch selection now fails if the `y_path` used to compute indices does not have
the same row count as the split X/y being indexed. This prevents indices from a
full dataset being applied to a previously subsetted fold.

### Main Pipeline

`main.py` now validates X/y before:

- class subset selection;
- model training;
- synthesis;
- incremental synthesis;
- sanity check dataset creation.

The concrete AppClassNet sanity-check bug was fixed in
`_synthesize_data_incremental`: sanity checks now receive the same split pair
that was passed to the incremental synthesizer.

## Test Coverage

Added or extended tests for:

- aligned X/y pass;
- X smaller than y fails before indexing;
- y smaller than X fails before indexing;
- fold X uses fold y;
- subset X/y uses the same indices;
- source indices are recorded for folds and aligned datasets;
- stratified selection preserves alignment;
- shuffle preserves alignment and source index counts;
- mmap and ndarray paths remain covered by existing batch tests;
- 200 classes remain present in batch tests;
- normal and batch contracts remain covered by existing tests;
- TR-TR, TR-TS, and TS-TR evaluation contracts remain covered by existing
  evaluation tests;
- legacy CSV loading remains covered.

Validation run:

```text
pytest -q
149 passed, 18 skipped, 2 warnings, 5 subtests passed
```

## Acceptance Notes

- No out-of-bounds index is allowed to reach NumPy indexing in the sanity subset.
- X/y row count mismatches fail fast with split/fold context.
- Sanity checks receive X/y from the same fold or split.
- No silent truncation, clipping, or index dropping was introduced.
- TR-TR baseline behavior was not changed; evaluation contract tests still pass.
- TR-TS and TS-TR dispatch/evaluation tests still pass.
- The original MalDataGen CSV path remains covered and passing.
