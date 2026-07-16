# Data Layout

Last updated: 2026-07-15

## Source Data

- `Datasets/raw/`: local raw datasets. Large and ignored by Git.
- `Datasets/converted/`: local converted datasets. Large and ignored by Git.
- `Datasets/processed/`: processed project datasets. Review before adding broad ignore rules.
- `Datasets/synthetic/`: generated or local synthetic data. Ignored by Git.

No dataset is removed automatically by repository cleanup.

## AppClassNet NPY Layout

Expected separated X/y files:

- `train_x.npy`
- `train_y.npy`
- `valid_x.npy`
- `valid_y.npy`
- `test_x.npy`
- `test_y.npy`

`Engine/DataIO/NpyXYLoader.py` is the canonical loader for this layout.

## Runtime Output Layout

Timestamped runs are written under `outputs/`:

- `DataGenerated/`
- `EvaluationResults/`
- `Logs/`
- `ModelsSaved/`
- `Monitor/`
- `SelectionReports/`
- `Audits/`

Synthetic batches are expected under:

- `DataGenerated/synthetic_batches/train/manifest.json`
- `DataGenerated/synthetic_batches/test/manifest.json`

`results/` may contain historical result indexes, copied outputs or compatibility result trees. Treat it as historical/generated unless a manifest or report states otherwise.

## Manifests

Small manifests required for reproducibility should not be globally ignored. Generated manifests under ignored output directories remain local artifacts; schema/example manifests should stay visible to Git.

## Logs

`logs/` contains local run logs, including seed logs such as `logs/real_resample_seeds/seed_0.log`. Logs are not deleted automatically. Promote any log needed for publication to `docs/audits/` or an explicit archived artifact before cleanup.
