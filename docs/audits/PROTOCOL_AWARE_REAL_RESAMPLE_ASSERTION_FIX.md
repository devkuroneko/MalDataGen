# Protocol-Aware real_resample Assertion Fix

## Problem

`synthetic_control=real_resample` validated `TR-TS` and `TS-TR` unconditionally.
For an effective `ts_tr` run, the pipeline correctly skipped `TR-TS` and marked
its classifier metrics as not applicable, but the control assertion still
expected finite `TR-TS` accuracy metrics and failed.

The same execution also carried a legacy `evaluation=['TrAs', 'TsAr']` value
even when the canonical protocol requested only `ts_tr`, creating conflicting
sources for the active protocol.

## Fix

The protocol is now resolved through a single `EvaluationProtocolPlan` with:

- `protocol`
- `run_tr_tr`
- `run_tr_ts`
- `run_ts_tr`
- `run_tr_plus_ts_tr`
- `required_result_keys`

The plan is derived once from canonical protocol arguments or legacy
`evaluation_mode`, and downstream code consumes that plan for evaluation
routing, result initialization, control validation, serialization, and wrapper
reporting.

Legacy evaluation names are derived from the plan. For example, `ts_tr` now
derives `['TsAr']` instead of keeping the old default `['TrAs', 'TsAr']`.

## Validation Rules

For active synthetic result keys, `real_resample` requires:

- `status=completed` when a status field is present;
- finite `Accuracy`;
- finite `BalancedAccuracy`;
- positive `effective_fit_rows` when batch metadata is present.

For inactive synthetic result keys:

- `status` may be `not_applicable`, `not_run`, or absent for legacy files;
- `Accuracy` and `BalancedAccuracy` must remain `null`;
- missing inactive metrics do not fail validation.

Aliases such as `TrAs`, `TsAr`, `tr_ts`, and `ts_tr` are normalized to the
canonical result keys `TR-TS` and `TS-TR`.

## Expected ts_tr Shape

```json
{
  "TR-TS": {
    "DecisionTreeSubset": {
      "1-Fold": {
        "status": "not_applicable",
        "reason": "evaluation_protocol=ts_tr",
        "Accuracy": null,
        "BalancedAccuracy": null
      }
    }
  },
  "TS-TR": {
    "DecisionTreeSubset": {
      "1-Fold": {
        "status": "completed",
        "Accuracy": 0.9,
        "BalancedAccuracy": 0.9
      }
    }
  }
}
```

## Tests

Added focused protocol-aware `real_resample` tests covering inactive protocols,
missing metrics, `NaN`, failed status, canonical key normalization, legacy
evaluation derivation, old `evaluation_mode` compatibility, legacy CSV-shaped
results, and a ten-class `TS-TR` positive-control case.
