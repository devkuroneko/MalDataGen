# Repository Cleanup Report

Last updated: 2026-07-15

## Scope

Cleanup was based on current audit reports and focused tests. No datasets, historical results or output trees were deleted.

## Files Removed

None.

## Files Moved

None.

## Files Rewritten As Wrappers

| File | Classification | Canonical replacement | Compatibility action |
| --- | --- | --- | --- |
| `Tools/utils.py` | duplicate exact, wrapper, compatibility legacy | `Tools/Plot/utils.py` | emits `DeprecationWarning`; re-exports `create_directory` |
| `Tools/PlotHeatMap.py` | duplicate exact, wrapper, compatibility legacy | `Tools/Plot/PlotHeatMap.py` | emits `DeprecationWarning`; re-exports canonical symbols |

## Files Added

| File | Purpose |
| --- | --- |
| `Tools/Plot/__init__.py` | makes the canonical plot package importable despite legacy `Tools/Plot.py` |
| `tests/test_legacy_plot_wrappers.py` | compatibility tests for deprecated plot wrappers |
| `docs/ARCHITECTURE.md` | canonical architecture map |
| `docs/EXPERIMENT_PROTOCOLS.md` | evaluation protocol contract |
| `docs/DATA_LAYOUT.md` | dataset and artifact layout |
| `docs/CLI_REFERENCE.md` | CLI entry points and deprecations |
| `docs/MIGRATION_GUIDE.md` | migration path for imports, CLI and artifacts |
| `docs/REPRODUCIBILITY.md` | run reproducibility requirements |
| `docs/audits/ARTIFACT_RETENTION_PLAN.md` | retention categories for data and outputs |

## APIs Deprecated

| API | Substitute | Test coverage |
| --- | --- | --- |
| `import Tools.utils` | `import Tools.Plot.utils` | `tests/test_legacy_plot_wrappers.py` |
| `import Tools.PlotHeatMap` | `import Tools.Plot.PlotHeatMap` | `tests/test_legacy_plot_wrappers.py` |

Existing CLI deprecations remain documented:

- `--full` -> `--run_mode full`
- `--batch_classifier` -> `--eval_classifier`

## Duplicates Consolidated

| Area | Classification | Decision |
| --- | --- | --- |
| `Tools/Plot/utils.py` and `Tools/utils.py` | duplicate exact | `Tools/Plot/utils.py` is canonical; root module is wrapper |
| `Tools/Plot/PlotHeatMap.py` and `Tools/PlotHeatMap.py` | duplicate exact | `Tools/Plot/PlotHeatMap.py` is canonical; root module is wrapper |
| `Tools/Plot.py` and `Tools/Plot/Plot.py` | duplicate semantic | still uncertain; `Tools/Plot/` package is canonical but root file kept for compatibility |
| root plot modules vs `Tools/Plot/*` modules | duplicate semantic | not removed; behavior differs in some root modules |
| Denoising vs Latent diffusion model files | duplicate exact but semantically distinct algorithm families | still uncertain; not consolidated in this pass |
| Wasserstein vs WassersteinGP model files | duplicate exact but semantically distinct algorithm families | still uncertain; not consolidated in this pass |
| generated duplicate outputs under `outputs/` and `results/` | generated/output historical | not deleted; governed by retention plan |
| duplicated generated documentation logos | documentation generated | not deleted; keep until docs publication flow is reviewed |
| empty baseline experiment files | code dead suspected | still uncertain; not removed |

## Canonical Implementations

| Concern | Canonical implementation |
| --- | --- |
| CLI | `main.py`; AppClassNet orchestration in `run_appclassnet_top200.py` |
| DatasetBundle | `Engine/DataIO/DatasetContracts.py` |
| SplitData | `Engine/DataIO/DatasetContracts.py` |
| loaders | `CSVLoader.py`, `NpyXYLoader.py`, `SyntheticBatchIO.py`, `BatchNpyDataset.py` |
| transformations | `Engine/Preprocessing/FeatureTransformManager.py` |
| stratified selection | `Engine/DataIO/StratifiedNpySelection.py`; runner subset helpers for AppClassNet |
| SamplePlan | `DatasetContracts.py` for data contracts; runner `SamplePlan` for command resolution |
| RunArtifacts | `run_appclassnet_top200.py` |
| manifests | `SyntheticBatchIO.py` and runner `command_manifest.json` |
| evaluations | `ExperimentProtocol.py`, `EvaluationRunner.py`, protocol modules |
| metrics | `Engine/Metrics/Metrics.py` and metric-specific modules |
| synthetic controls | `--synthetic_control` runner/runtime option |
| quality audit | `Engine/DataIO/SyntheticQualityAudit.py` |

## Generated Files And Ignore Policy

`.gitignore` was updated to ignore caches, virtual environments, local logs, execution outputs, large result trees, synthetic batches, subset caches and binary data artifacts while keeping small configs and example/schema manifests visible.

## Tests Executed

- `python -m pytest tests/test_legacy_plot_wrappers.py tests/test_plots_npy_support.py -q`
  - Result: 4 passed, 2 warnings.
- `python -m pytest tests/test_legacy_plot_wrappers.py tests/test_plots_npy_support.py tests/test_appclassnet_evaluation_mode.py tests/test_evaluation_integration_spec.py::EvaluationIntegrationSpecTest::test_legacy_csv_command_still_works -q`
  - Result: 90 passed, 1 skipped, 2 warnings, 5 subtests passed.
- `python -m pytest -ra --tb=short`
  - Result: 251 passed, 18 skipped, 9 warnings.

## Risks Remaining

- Root plotting modules still contain semantic duplicates with `Tools/Plot/*`; some root files include newer defensive behavior and were not collapsed.
- Model duplicate files are byte-identical in places but may intentionally represent separate algorithm families.
- `outputs/`, `results/`, `logs/` and datasets were retained for manual review.
