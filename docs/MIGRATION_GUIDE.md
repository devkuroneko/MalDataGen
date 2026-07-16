# Migration Guide

Last updated: 2026-07-15

## Import Migrations

| Legacy import | Replacement | Status |
| --- | --- | --- |
| `Tools.utils` | `Tools.Plot.utils` | deprecated wrapper |
| `Tools.PlotHeatMap` | `Tools.Plot.PlotHeatMap` | deprecated wrapper |

Legacy wrappers emit `DeprecationWarning` and remain available for compatibility.

## CLI Migrations

| Legacy CLI | Replacement |
| --- | --- |
| `--full` | `--run_mode full` |
| `--batch_classifier` | `--eval_classifier` |

Do not remove legacy options silently. Add or keep tests that assert the deprecation warning and canonical output command.

## Data Migration

- Prefer explicit `npy_xy` train/valid/test paths for AppClassNet.
- Keep CSV commands for legacy integration tests and historical experiments.
- Do not move or delete datasets without a separate manual confirmation.

## Artifact Migration

- Treat `outputs/` and `results/` as generated or historical until reviewed against `docs/audits/ARTIFACT_RETENTION_PLAN.md`.
- Promote reproducibility-critical manifests to `docs/audits/` or another tracked location if they must survive output cleanup.

## Code Migration Rules

- New loaders should use `DatasetBundle` and `SplitData`.
- New synthetic writers should use `SyntheticBatchWriter`.
- New evaluation dispatch should use `ExperimentProtocol` and `EvaluationRunner`.
- New runner path metadata should use `RunArtifacts`.
