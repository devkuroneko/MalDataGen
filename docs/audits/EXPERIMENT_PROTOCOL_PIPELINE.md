# Experiment Protocol Pipeline

## Scope

This audit documents the explicit AppClassNet experimental protocol layer added
to separate real baselines, synthetic substitution, synthetic fidelity,
augmentation, and diagnostic controls.

## Canonical Protocols

| Protocol | Train source | Test source | Purpose |
| --- | --- | --- | --- |
| `TR_TR` | real `TRAIN` | real `TEST` | real baseline |
| `TR_TS` | real `TRAIN` | `synthetic_test` | synthetic fidelity against labels |
| `TS_TR` | `synthetic_train` | real `TEST` | synthetic substitution |
| `TR_PLUS_TS_TR` | real `TRAIN` + `synthetic_train` | real `TEST` | real-data augmentation |

CLI selectors:

```bash
--evaluation_protocol tr_tr
--evaluation_protocol tr_ts
--evaluation_protocol ts_tr
--evaluation_protocol tr_plus_ts_tr
--evaluation_protocol all
```

The legacy `--evaluation_mode` values remain supported when a canonical
`--evaluation_protocol` is not supplied. The legacy `appclassnet_strict`
protocol keeps strict split routing for compatibility.

## Budget Scenarios

The AppClassNet runner exposes `--experiment_budget_scenario`:

| Scenario key | Label |
| --- | --- |
| `r200_r500` | A. R200 -> R500 |
| `s200_r500` | B. S200 -> R500 |
| `r50_r500` | C. R50 -> R500 |
| `r50_s150_r500` | D. R50 + S150 -> R500 |
| `r200_s200_r500` | E. R200 + S200 -> R500 |
| `real_resample_r500` | F. real_resample -> R500 |
| `label_permutation_r500` | G. label_permutation -> R500 |

Individual quota flags still override scenario defaults. The effective values
are written to `results_summary.json` and `experiment_protocol.json`.

Two budgets are recorded separately:

- `generator_training_real_samples_per_class`
- `classifier_training_samples_per_class`

This prevents treating `S200` generated after more real generator training data
as equivalent to a classifier trained on `R200`.

## Controls

`real_resample` is marked as a synthetic control, not a generator result.
`synthetic_train` is sampled from real `TRAIN`; `synthetic_test` is sampled from
real `TEST`; both use selection without replacement and record
`source_indices` in the batch manifest.

`label_permutation` preserves feature rows and permutes labels reproducibly. Its
expected predictive result is near chance.

## Artifacts

Each main pipeline run writes:

- `experiment_protocol.json`
- `results_summary.json`
- legacy `EvaluationResults/Results.json`

The AppClassNet runner also writes the canonical files at the run root beside
`RunResults.json`.

`completed` is assigned only when required classifier metrics are present:
`Accuracy`, `BalancedAccuracy`, `MacroF1`, and `WeightedF1`. Requested protocols
without valid metrics are reported as `failed`; unrequested protocols are
reported as `not_run`.

## Verification

Targeted tests:

```bash
pytest -q tests/test_experiment_protocol_pipeline.py tests/test_evaluation_runner.py tests/test_appclassnet_evaluation_mode.py tests/test_real_resample_control_routing.py
```

Result: `106 passed, 5 subtests passed`.
