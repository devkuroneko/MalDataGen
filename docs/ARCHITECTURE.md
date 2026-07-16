# MalDataGen Architecture

Last updated: 2026-07-15

## Canonical Runtime Entry Points

- CLI: `main.py` is the canonical model/evaluation runtime CLI.
- AppClassNet orchestration: `run_appclassnet_top200.py` is the canonical campaign runner and command generator.
- Plot package: `Tools/Plot/` is the canonical plotting package. Legacy imports under `Tools/*.py` remain for compatibility where still needed.

## Canonical Data Contracts

- `DatasetBundle`: `Engine/DataIO/DatasetContracts.py`
- `SplitData`: `Engine/DataIO/DatasetContracts.py`
- `DatasetSchema`: `Engine/DataIO/DatasetContracts.py`
- `SamplePlan`: `Engine/DataIO/DatasetContracts.py` for loader/generator contracts; `run_appclassnet_top200.py` has a runner-level `SamplePlan` for CLI resolution.

`SplitData` validates matrix dimensionality, X/y row alignment, source indices and class-count metadata. `DatasetBundle` validates train/valid/test splits against a shared schema.

## Canonical Loaders

- CSV legacy loader: `Engine/DataIO/CSVLoader.py`
- separated NumPy X/y loader: `Engine/DataIO/NpyXYLoader.py`
- synthetic batch reader/writer: `Engine/DataIO/SyntheticBatchIO.py`
- batch dataset iteration: `Engine/DataIO/BatchNpyDataset.py`
- stratified NumPy subset selection: `Engine/DataIO/StratifiedNpySelection.py`

Loaders should return or consume explicit contracts rather than relying on process cwd or implicit global state.

## Canonical Transform Flow

- Feature transforms: `Engine/Preprocessing/FeatureTransformManager.py`
- Transform metadata is carried through `DatasetSchema`, `SplitData`, synthetic manifests and runner command manifests.
- AppClassNet defaults preserve source scale unless a transform is explicitly requested.

## Canonical Evaluation Flow

- Protocol definitions: `Engine/Evaluation/ExperimentProtocol.py`
- Shared evaluator assembly and validation: `Engine/Evaluation/EvaluationRunner.py`
- Legacy facade: `Engine/Evaluation/Evaluation.py`
- Protocol modules: `TrTr.py`, `TrTs.py`, `TsTr.py`, `TrTsTr.py`

Canonical protocol IDs:

- `TR_TR`: train real train, test real test.
- `TR_TS`: train real train, test synthetic test.
- `TS_TR`: train synthetic train, test real test.
- `TR_PLUS_TS_TR`: train real train plus synthetic train, test real test.

## Canonical Artifacts

- Runner command manifest: `command_manifest.json` written by `run_appclassnet_top200.py`.
- Synthetic batch manifest: `DataGenerated/synthetic_batches/<split>/manifest.json`.
- Evaluation results: `EvaluationResults/Results.json`.
- Quality audit: `Engine/DataIO/SyntheticQualityAudit.py`.

`RunArtifacts` in `run_appclassnet_top200.py` is the canonical runner-level record for result paths, manifests and generated-data directories.

## Compatibility Boundaries

- Deprecated wrapper imports emit `DeprecationWarning` and forward to canonical modules.
- Legacy CLI aliases remain accepted where tests cover them, for example `--full` and `--batch_classifier`.
- CSV legacy behavior remains supported and tested.
- Large generated data, datasets and historical results are not deleted by automated cleanup.
