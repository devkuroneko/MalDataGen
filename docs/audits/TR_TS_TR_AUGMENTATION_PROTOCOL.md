# TR+TS-TR Augmentation Protocol Audit

Date: 2026-07-15

## Baseline Before Edits

- Git tree was already dirty before this change set.
- Pre-edit test baseline:
  - Command: `pytest`
  - Result: `207 passed, 18 skipped, 2 warnings`

## Diagnosis

The repository already had strict AppClassNet routing for:

- TR-TR: real train to real test.
- TR-TS: real train to synthetic test in the strict AppClassNet protocol.
- TS-TR: synthetic train to real test.
- `real_resample` and `label_permutation` synthetic controls.

The missing protocol was `TR+TS-TR`: train a classifier on real training rows plus synthetic training rows, then evaluate only on the real test split. This is required for measuring whether synthetic data can improve reduced real training sets without using TEST for model selection.

## Files Changed

- `Engine/Evaluation/EvaluationRunner.py`
- `Engine/Evaluation/Evaluation.py`
- `Engine/Evaluation/TrTsTr.py`
- `main.py`
- `Engine/Arguments/ArgumentsFramework.py`
- `run_appclassnet_top200.py`
- `tests/test_evaluation_runner.py`
- `tests/test_appclassnet_evaluation_mode.py`

## Implementation Notes

- Added evaluation mode `TR+TS-TR` to the shared evaluation contract.
- Added `evaluation_TR_TS_TR` for normal and batches execution.
- In batches mode, real train batches and synthetic train batches are streamed into the selected batch classifier; the real test split is evaluated in batches.
- For provided AppClassNet splits, `TRAIN` is used as real training data and `TEST` as final real evaluation data.
- `VALID` is not used by the new protocol for final evaluation.
- `both` keeps legacy synthetic behavior: TR-TS and TS-TR only.
- `all` now means the full scientific protocol set: TR-TR, TR-TS, TS-TR and TR+TS-TR.
- Added AppClassNet runner pipeline `augmentation` for TR+TS-TR only.
- `augmentation` does not inherit the profile's synthetic-test quota; its default plan generates synthetic train rows only.

## Validation

- Focused tests:
  - Command: `pytest tests/test_evaluation_runner.py tests/test_appclassnet_evaluation_mode.py`
  - Result: `81 passed`
- Full test suite:
  - Command: `pytest`
  - Result: `213 passed, 18 skipped, 2 warnings`

Warnings were unchanged plotting warnings from sklearn/scipy in `tests/test_plots_npy_support.py`.
