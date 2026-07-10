# AppClassNet and npy_xy Dataset Support

This document describes the optional `npy_xy` input mode added for datasets such as AppClassNet top-200. The original CSV workflow remains the default and continues to work as before.

## CSV vs npy_xy

`csv` is the legacy/default mode. It uses the existing single-file CSV flow through `CSVLoader`, where features and labels are read from one CSV file according to the existing arguments such as `--data_load_path_file_input` and `--data_load_label_column`.

`npy_xy` is an explicit opt-in mode for datasets stored as separate NumPy arrays for X and y. It does not convert data to CSV and is intended for continuous feature matrices with integer label vectors.

Use `npy_xy` only when you pass:

```bash
--data_format npy_xy
```

If `--data_format` is omitted, MalDataGen uses:

```bash
--data_format csv
```

## Expected Files

For provided train/validation/test splits, the expected files are:

```text
train_x.npy
train_y.npy
valid_x.npy
valid_y.npy
test_x.npy
test_y.npy
```

Expected shapes:

```text
X: N x num_features
y: N
```

`X` must be a 2D numeric matrix. `y` must be a 1D integer vector, or safely convertible to 1D. The loader does not materialize a full one-hot label matrix for the dataset.

## AppClassNet Top-200

AppClassNet top-200 is treated as a multiclass dataset, not a binary dataset.

Typical configuration:

```text
num_features: 20
num_classes: 200
target_type: multiclass
feature_type: continuous
split_mode: provided
```

Use `--num_classes 200` even if a small subset does not contain all 200 labels. This preserves the full label domain for conditional generators and metrics.

## Smoke Test

The repository includes a smoke test script that creates small synthetic `.npy` files and validates parsing, loading, schema, and sample planning without needing the real AppClassNet dataset:

```bash
python scripts/smoke_test_npy_xy.py --dry-run
```

Simulate AppClassNet top-200:

```bash
python scripts/smoke_test_npy_xy.py --dry-run --num_classes 200 --samples_per_class 1
```

The script prints the generated temporary file paths, loaded split shapes, inferred schema, sample plan, and a recommended `main.py` command. It only runs the full pipeline if `--run-pipeline` is passed.

To generate reusable synthetic files in a known directory, run:

```bash
RUN_DIR="outputs/maldatagen_npy_xy_demo_$(date +%Y-%m-%d_%H-%M-%S)"

python scripts/smoke_test_npy_xy.py \
  --work_dir "$RUN_DIR/dataset" \
  --num_classes 10 \
  --samples_per_class 2
```

The files created by this command can be used directly by `main.py`.

To run the generated synthetic dataset through the lightweight `random` generator pipeline:

```bash
RUN_DIR="outputs/maldatagen_npy_xy_demo_$(date +%Y-%m-%d_%H-%M-%S)"

python scripts/smoke_test_npy_xy.py \
  --work_dir "$RUN_DIR/dataset" \
  --num_classes 10 \
  --samples_per_class 2 \
  --run-pipeline \
  --model_type random \
  --classifier DecisionTree \
  --output_dir "$RUN_DIR/pipeline"
```

## Minimal npy_xy Command

This is a copy/paste command for synthetic files generated under a dated folder in `outputs/`:

```bash
RUN_DIR="outputs/maldatagen_npy_xy_demo_$(date +%Y-%m-%d_%H-%M-%S)"

python scripts/smoke_test_npy_xy.py \
  --work_dir "$RUN_DIR/dataset" \
  --num_classes 10 \
  --samples_per_class 2

python main.py \
  --data_format npy_xy \
  --split_mode provided \
  --train_x_path "$RUN_DIR/dataset/train_x.npy" \
  --train_y_path "$RUN_DIR/dataset/train_y.npy" \
  --valid_x_path "$RUN_DIR/dataset/valid_x.npy" \
  --valid_y_path "$RUN_DIR/dataset/valid_y.npy" \
  --test_x_path "$RUN_DIR/dataset/test_x.npy" \
  --test_y_path "$RUN_DIR/dataset/test_y.npy" \
  --target_type multiclass \
  --feature_type continuous \
  --num_classes 10 \
  --sample_plan balanced_per_class \
  --samples_per_class 2 \
  --model_type random \
  --classifier DecisionTree \
  --output_dir "$RUN_DIR/pipeline"
```

## Real AppClassNet Example

```bash
python main.py \
  --data_format npy_xy \
  --split_mode provided \
  --train_x_path "Datasets/raw/AppClassNet/top200/train_x.npy" \
  --train_y_path "Datasets/raw/AppClassNet/top200/train_y.npy" \
  --valid_x_path "Datasets/raw/AppClassNet/top200/valid_x.npy" \
  --valid_y_path "Datasets/raw/AppClassNet/top200/valid_y.npy" \
  --test_x_path "Datasets/raw/AppClassNet/top200/test_x.npy" \
  --test_y_path "Datasets/raw/AppClassNet/top200/test_y.npy" \
  --target_type multiclass \
  --feature_type continuous \
  --num_classes 200 \
  --sample_plan balanced_per_class \
  --samples_per_class 1000 \
  --mmap_npy
```

## split_mode=provided

Use:

```bash
--split_mode provided
```

with `npy_xy` when train/valid/test splits already exist on disk. This avoids applying K-fold cross-validation automatically to AppClassNet splits.

Current behavior:

- `train` is used to train the generator.
- `valid` is used for evaluation when present.
- `test` is loaded and reported, but final test evaluation support is still limited in the current TR-TS/TS-TR pipeline.
- If no valid/test split is supplied, unsupported evaluation stages are marked as `not_applicable` instead of `0`.

## sample_plan

The legacy generator contract still uses `number_samples_per_class` internally. The new `SamplePlan` layer translates old and new arguments into that legacy format.

Supported modes:

```text
legacy
class_counts
total_rows
match_train_distribution
balanced_per_class
```

Generate a fixed amount per class:

```bash
--sample_plan balanced_per_class --samples_per_class 1000
```

Generate a total number of rows following the training distribution:

```bash
--sample_plan match_train_distribution --total_synthetic_rows 50000
```

Use explicit class counts through the legacy argument:

```bash
--sample_plan class_counts --number_samples_per_class 0:100,1:100,2:100
```

For `data_format=npy_xy`, generation requires an explicit sampling plan or explicit `--number_samples_per_class`. This prevents accidental generation using the old default counts for a 200-class dataset.

## Backward Compatibility

- `--data_format` defaults to `csv`.
- Existing CSV commands continue to use the original CSV pipeline.
- `CSVLoader` was not removed.
- Existing arguments were not removed.
- `--number_samples_per_class` continues to work.
- The legacy binary metric set is preserved for `target_type=binary` or the old binary CSV flow.
- New `npy_xy` behavior is only enabled when selected explicitly with `--data_format npy_xy`.

## Evaluation Notes

For multiclass datasets, predictive metrics use multiclass-safe metrics:

```text
Accuracy
MacroPrecision
MacroRecall
MacroF1
WeightedPrecision
WeightedRecall
WeightedF1
BalancedAccuracy
```

Skipped or unsupported stages are recorded as:

```text
not_applicable
```

They are not stored as `0.0`, so reports do not confuse skipped evaluations with real poor performance.

## Limitations

- Full AppClassNet top-200 runs can require substantial RAM and runtime.
- Prefer `.npy` memory mapping for large arrays. The loader supports `mmap_mode`; CLI use is controlled with `--mmap_npy`.
- Avoid converting all labels to one-hot for the full dataset. The current flow keeps labels as 1D integers where possible and only encodes labels at model-specific points.
- Start with small subsets or `scripts/smoke_test_npy_xy.py --dry-run` before launching long runs.
- Metrics with 200 classes can be expensive, especially classifier-based TR-TS/TS-TR evaluations.
- Confusion-matrix plotting for 200 classes is not recommended by default; if used, make it an explicit plotting step and expect large figures.
- Final test split evaluation is still limited in the current pipeline. The loader can read `test`, but some evaluation paths may mark final-test-only stages as `not_applicable`.

## Verification

Command run on this repository:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_metrics_target_types \
  tests.test_multiclass_pipeline \
  tests.test_data_loader_arguments \
  tests.test_dataset_contracts \
  tests.test_npy_xy_loader \
  tests.test_cross_validation_data_loading \
  tests.test_sample_planner
```

Result:

```text
Ran 48 tests in 0.054s
OK
```

Known pending work:

- Full end-to-end training on the complete AppClassNet top-200 dataset was not run here.
- Final test split evaluation remains partially supported and may be marked `not_applicable` depending on the evaluation path.
