# AppClassNet Top200 TR-TR Baseline

## Objective

This baseline validates the MalDataGen extension for continuous multiclass data,
establishes a real-train to real-test reference for AppClassNet Top200, and keeps
the loading, transformation, sampling, classification, metrics, and artifact
contracts ready for later synthetic evaluations.

The baseline is intentionally independent from synthetic generation. Synthetic
data remains supported elsewhere in the project, but it is not part of the
reported TR-TR numbers.

## TR-TR Definition

TR-TR means:

- Train Real: fit the classifier only with real samples from the provided
  `train` split.
- Test Real: evaluate only on real samples from the provided `test` split.
- No synthetic data is generated, loaded, concatenated, or evaluated.

The provided `valid` split is preserved in the dataset contract. Decision Tree
and Random Forest TR-TR runs do not merge `valid` into `train` and do not use
`valid` as the primary test split.

## File Layout

The `npy_xy` loader expects the AppClassNet Top200 directory to contain:

```text
train_x.npy
train_y.npy
valid_x.npy
valid_y.npy
test_x.npy
test_y.npy
```

By default the runner expects these files under:

```text
Datasets/raw/AppClassNet/top200
```

Override this with `--raw_root`.

## Requirements

- Python 3.10 was used in the validation environment.
- NumPy is required for NPY loading and mmap.
- scikit-learn is required for `DecisionTreeClassifier` and
  `RandomForestClassifier`.
- joblib is required only when `--save_tr_tr_model` is enabled.
- pandas is used for CSV/per-class artifact compatibility.
- `psutil` is recommended for resource reporting.

Memory depends heavily on the classifier:

- Decision Tree integral completed in the validation environment with a traced
  peak around 2.1 GB.
- Random Forest integral can require substantially more memory; do not infer RF
  feasibility from a Decision Tree run.

## Transformation Policy

Use `preserve` for AppClassNet Top200:

```text
--feature_transform preserve
--generator_transform preserve
--classifier_transform preserve
--evaluation_space source
```

AppClassNet Top200 is already in its source numeric space, approximately
`[-0.5, 0.5]`. Applying `MinMaxScaler`, `StandardScaler`, or a second transform
to `[0, 1]` changes the data geometry and makes the baseline inconsistent with
the source dataset. In `preserve` mode, `fit`, `transform`, and
`inverse_transform` do not alter the feature values.

## Tested Commands

The commands below were executed in this repository on 2026-07-21 with the real
AppClassNet Top200 files available under `Datasets/raw/AppClassNet/top200`.

### Diagnostic

This command loads the NPY metadata with mmap, validates shapes, 20 features,
200 classes, labels, ranges, and NaN/Inf status, then exits before training.

```bash
python run_appclassnet_top200.py \
  --pipeline tr_tr \
  --diagnostic_only \
  --source_profile appclassnet_top200 \
  --data_format npy_xy \
  --split_mode provided \
  --execution_mode batches \
  --use_mmap \
  --mmap_npy \
  --feature_transform preserve \
  --generator_transform preserve \
  --classifier_transform preserve \
  --evaluation_space source \
  --raw_root Datasets/raw/AppClassNet/top200 \
  --output_dir results/appclassnet_top200/doc_smoke/diagnostic
```

### Decision Tree Subset

This is a smoke test over all 200 classes, selecting up to 2 train and 2 test
samples per class.

```bash
python run_appclassnet_top200.py \
  --pipeline tr_tr \
  --source_profile appclassnet_top200 \
  --data_format npy_xy \
  --split_mode provided \
  --execution_mode batches \
  --use_mmap \
  --mmap_npy \
  --feature_transform preserve \
  --generator_transform preserve \
  --classifier_transform preserve \
  --evaluation_space source \
  --classifier decision_tree \
  --train_sampling up_to_available \
  --train_samples_per_class 2 \
  --test_sampling up_to_available \
  --test_samples_per_class 2 \
  --eval_batch_size 256 \
  --random_state 20260721 \
  --raw_root Datasets/raw/AppClassNet/top200 \
  --output_dir results/appclassnet_top200/doc_smoke/dt_subset
```

### Decision Tree Integral

This command uses the full provided `train` and `test` splits. It completed in
the validation environment and produced TR-TR metrics with all 200 classes.

```bash
python run_appclassnet_top200.py \
  --pipeline tr_tr \
  --source_profile appclassnet_top200 \
  --data_format npy_xy \
  --split_mode provided \
  --execution_mode batches \
  --use_mmap \
  --mmap_npy \
  --feature_transform preserve \
  --generator_transform preserve \
  --classifier_transform preserve \
  --evaluation_space source \
  --classifier decision_tree \
  --train_sampling all \
  --test_sampling all \
  --eval_batch_size 65536 \
  --random_state 20260721 \
  --raw_root Datasets/raw/AppClassNet/top200 \
  --output_dir results/appclassnet_top200/doc_smoke/dt_integral
```

### Random Forest Subset

This is a smoke test over all 200 classes with 5 trees and `n_jobs=1`.

```bash
python run_appclassnet_top200.py \
  --pipeline tr_tr \
  --source_profile appclassnet_top200 \
  --data_format npy_xy \
  --split_mode provided \
  --execution_mode batches \
  --use_mmap \
  --mmap_npy \
  --feature_transform preserve \
  --generator_transform preserve \
  --classifier_transform preserve \
  --evaluation_space source \
  --classifier random_forest \
  --train_sampling up_to_available \
  --train_samples_per_class 2 \
  --test_sampling up_to_available \
  --test_samples_per_class 2 \
  --rf_n_estimators 5 \
  --rf_n_jobs 1 \
  --eval_batch_size 256 \
  --random_state 20260721 \
  --raw_root Datasets/raw/AppClassNet/top200 \
  --output_dir results/appclassnet_top200/doc_smoke/rf_subset
```

### Random Forest Integral

No Random Forest integral training command is documented as validated in this
version. The validation run estimated that a 100-tree RF integral could require
more memory than was safely available, and the attempted CLI safeguards
`--dry_run_memory` and `--dryrun` still entered the TR-TR pipeline instead of
stopping before fit. Both attempts were interrupted before training.

Use the RF subset command above for smoke validation. Add an explicit memory
guard or run on a machine with a validated memory budget before documenting or
reporting a full RF integral run.

## Sampling Modes

- `all`: use every sample from the selected split. No subset indices are
  materialized.
- `balanced_per_class`: require exactly `samples_per_class` from every class.
  With the default strict policy, the run fails if any class has fewer samples
  than requested.
- `up_to_available`: request `samples_per_class` but select
  `min(requested, available)` for each class.

`classifier subset` means the classifier was trained/evaluated on a selected
real subset, for example `up_to_available:2`. It must not be reported as a full
AppClassNet baseline. `full split` means `train_sampling=all` and
`test_sampling=all` over the provided dataset splits.

## Metrics

TR-TR writes multiclass metrics derived from an accumulated confusion matrix:

- `accuracy`
- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_precision`
- `weighted_recall`
- `weighted_f1`
- per-class `support`, `predicted_count`, `true_positive`, `precision`,
  `recall`, and `f1`

Accuracy is kept for comparability. For imbalanced AppClassNet Top200 results,
`macro_f1` and `balanced_accuracy` should be treated as central metrics.

## Artifacts

Each completed TR-TR run writes a dedicated run directory under:

```text
results/appclassnet_top200/tr_tr/<run_id>/
```

Typical files:

```text
run_config.json
dataset_manifest.json
class_distribution_train.csv
class_distribution_test.csv
metrics.json
per_class_metrics.csv
confusion_matrix.npy
resource_usage.json
environment.json
execution.log
status.json
tr_tr_result.json
manifest.json
model.joblib
```

`model.joblib` is written only when `--save_tr_tr_model` is enabled.

## Reproducing A Run

There is currently no `--run_config` CLI option. To reproduce a run:

1. Open the run directory's `run_config.json`.
2. Check `resolved_arguments`, classifier parameters, transform policy, seed,
   and sample plans.
3. Re-run `run_appclassnet_top200.py` with the same resolved CLI values.
4. Confirm the new run has the same `protocol`, `classifier`,
   `train_sampling`, `test_sampling`, transform policy, and `random_state`.

The original command is also visible in `execution.log` and in the top-level
runner output directory's `RunResults.json`.

## Locating Results

For completed TR-TR runs, inspect:

```text
results/appclassnet_top200/tr_tr/summary.csv
```

Use `run_id`, `classifier`, `seed`, train/test sampling strategy, and metrics to
distinguish full-split runs from subset runs. Do not compare subset metrics as
if they were integral AppClassNet metrics.

## Protocol Separation

- `TR-TR`: real `train` to real `test`; no synthetic data.
- `generate`: real `train` to synthetic batches; no immediate TR-TR result.
- `TR-TS`: train on real, test on synthetic.
- `TS-TR`: train on synthetic, test on real.
- `mixed`: train on real plus synthetic, test on real.

The TR-TR baseline is intentionally independent from synthetic generation and
does not automatically trigger TR-TS, TS-TR, or mixed evaluations.

## Limitations

- Random Forest integral can require too much memory for local execution.
- A subset run is not directly comparable to the full AppClassNet baseline.
- A single seed does not measure variability.
- Accuracy can be dominated by majority classes; use macro-F1 and balanced
  accuracy for imbalance-aware reporting.
- CLI `--dry_run_memory` and `--dryrun` did not stop before TR-TR fit for the RF
  integral attempts during validation, so they should not be treated as safe RF
  memory guards in this version.

## Troubleshooting

- Missing file: confirm all six NPY files exist under `--raw_root`.
- mmap: use both `--use_mmap` and `--mmap_npy` for AppClassNet batches mode.
- `MemoryError`: reduce `--rf_n_estimators`, use subset sampling, keep
  `--rf_n_jobs 1`, or move to a larger-memory machine. Do not silently switch an
  intended integral RF run to subset.
- Insufficient classes: switch from `balanced_per_class` to `up_to_available`,
  or lower `--train_samples_per_class` / `--test_samples_per_class`.
- Invalid labels: labels must be non-negative and compatible with the schema
  classes. AppClassNet Top200 expects labels `0..199`.
- Wrong scale: use `preserve` transforms and verify ranges stay near
  `[-0.5, 0.5]`; do not apply MinMax/Standard scaling for the baseline.
- `not_run`: in TR-TR, synthetic protocols are expected to be `not_run`; the
  `TR-TR` block itself must be `completed`.
- Index out of bounds: regenerate sample plans after changing sampling
  parameters. The planner keys indices by split/config and validates bounds
  before indexing.
