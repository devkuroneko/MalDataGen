# Dataset Integrity and Class Subsets

## Scope

This audit documents the dataset integrity layer added for the AppClassNet and
`npy_xy` paths. The goal is that `X`, `y`, paths, source indices, labels,
schema, transforms and subset metadata describe the same rows at every stage.

## Implemented Contracts

- `DatasetBundle`, `SplitData`, `DatasetSchema`, `ClassMapping` and
  `SubsetManifest` now live in `Engine/DataIO/DatasetContracts.py`.
- `SplitData` carries `row_count`, `feature_count`, `class_counts`,
  `source_indices`, `data_space`, `transform_id`, `schema_hash`, `dataset_id`,
  `subset_id` and `label_mapping`.
- `DatasetBundle` validates split/schema consistency when constructed.
- Real `npy_xy` loaders mark paths as audit paths, so path existence and stored
  array shape are checked against the in-memory object.

## Mandatory Validations

Implemented checks include:

- `X.shape[0] == y.shape[0]`;
- 2D feature matrices;
- schema feature count agreement;
- AppClassNet audit paths require 20 features;
- integer labels for numeric multiclass domains;
- label range or explicit `class_labels` membership;
- path existence and path shape for loader/materialized artifacts;
- source index length and non-negative bounds;
- split overlap detection when splits share the same physical source;
- consistent `schema_hash`;
- explicit `data_space`.

## Subset Resolution

Subset options are resolved before filtering:

- `selected_original_classes`;
- `original_to_local_mapping`;
- `local_to_original_mapping`;
- `effective_num_classes`;
- deterministic `subset_id`.

In batches mode, class subsets are materialized under:

```text
results/appclassnet_top200/batches/subsets/<subset_id>/
  train_x.npy
  train_y.npy
  valid_x.npy
  valid_y.npy
  test_x.npy
  test_y.npy
  subset_manifest.json
```

The child `main.py` command receives the materialized subset paths and
`--num_classes <K>`. The original subset CLI flag is not forwarded after
materialization, preventing the full dataset paths from remaining active.

For `K=10`, the effective class domain is `0..9`, the VAE/GAN class-count
arguments are overwritten to `10`, and the sample plan is emitted only for
labels `0..9`.

## Routing and Leakage Guards

Canonical routing now treats `test` as the final evaluation split when present:

- TR-TR: train real -> test real;
- TR-TS: train real -> synthetic test;
- TS-TR: synthetic train -> test real;
- TR+TS-TR: train real + synthetic train -> test real.

`valid` remains available as validation metadata and is not used as a
replacement for `test` in final provided-split evaluation.

Additional guards:

- generator fit is rejected if the training split is `test`;
- synthetic sample planning uses train labels when train labels are available;
- `copy` generation uses train rows when an explicit train source is supplied;
- `real_resample` requires the requested source split and refuses `valid`.

## Transform Contract

For `source_profile=appclassnet_top200`, the existing preprocessing policy keeps
`preserve` as default, disables scaler refit by default, and treats preserve
inverse transform as a no-op. `transform_id` and `schema_hash` are propagated
through schemas, splits and synthetic manifests where applicable.

## Verification

Executed:

```text
pytest -q
```

Result:

```text
227 passed, 18 skipped, 2 warnings, 5 subtests passed
```
