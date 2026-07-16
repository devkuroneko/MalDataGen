# Result Serialization and Atomic Write Fix

## Flow Audited

`EvaluationProtocolPlan` is resolved in `Engine/Evaluation/ExperimentProtocol.py`.
The plan is inserted into metric payloads during `Metrics` initialization and
again as result metadata before persistence.

Flow:

1. `EvaluationProtocolPlan`
2. metric/result payload in `Metrics._dictionary_metrics`
3. atomic `EvaluationResults/Results.json`
4. wrapper `write_batches_metrics`
5. atomic `results_summary.json`

Synthetic manifests are produced by `SyntheticBatchWriter.close()` as
`DataGenerated/synthetic_batches/<split>/manifest.json` and are read only after
`Results.json` has been loaded and validated as JSON.

## Root Cause

`Metrics.save_dictionary_to_json()` wrote directly to `Results.json` and caught
all exceptions with a printed message. When an unserializable object such as a
raw `EvaluationProtocolPlan` reached `json.dump`, the final file could be left
partially written. The wrapper then treated JSON read failure as an absent result
and reported `not_run` plus inferred missing artifacts.

## Fix

- `EvaluationProtocolPlan.to_dict()` now returns a JSON-only structure.
- `Engine/DataIO/JsonIO.py` centralizes `to_jsonable`, JSON validation, atomic
  writes, typed read errors, and corrupt-file diagnostics.
- `Results.json`, `metrics.json`, `results_summary.json`,
  `experiment_protocol.json`, `RunResults.json`, `command_manifest.json`,
  `synthetic_quality_audit.json`, and synthetic manifests use atomic writes.
- `Results.json` persistence no longer catches and hides serialization errors.
- Malformed JSON now raises `ResultArtifactCorruptionError` and is not converted
  to `status=not_run`.
- Active protocol summaries require finite core metrics; new batch metadata also
  validates positive `effective_fit_rows`. Requested-but-incomplete protocols
  are `failed`, not `not_run`.
- Inactive synthetic protocols are reported as `not_applicable` with null
  metrics.

## Atomic Write Contract

JSON payloads are converted and validated before opening the final file:

```python
serializable_payload = to_jsonable(payload)
json.dumps(serializable_payload, allow_nan=False)
```

Writes go to `<name>.tmp`, then `flush`, `fsync`, and `os.replace`. On failure,
only the temporary file is removed; an existing final file is preserved.

## Corrupt Artifact Handling

`scripts/diagnose_json_artifacts.py` detects malformed JSON. With
`--move-corrupt`, it moves the file to `*.corrupt`. It never silently repairs or
deletes historical results.

## TS-TR real_resample

For `evaluation_protocol=ts_tr` and `synthetic_control=real_resample`, the
wrapper validates the `TS-TR` result and requires the registered
`synthetic_batches/train/manifest.json`. It does not require a test manifest for
inactive `TR-TS`.

## Control Campaign Routing

`real_resample` and other synthetic controls do not need a trained generator.
The wrapper now exposes a canonical `control` campaign that runs the child
process with `model_type=copy` while recording the wrapper artifact model as
`control`. When `-c sf` or the default demo campaign is combined with
`synthetic_control=real_resample`, the wrapper runs the single `control`
campaign instead of both `variational_demo` and `adversarial_demo`.

Explicit legacy model campaigns such as `-c variational_demo` remain supported
for compatibility.
