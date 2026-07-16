# AppClassNet Demo/Full Execution Mode Refactor

Date: 2026-07-14

## Objective

`run_appclassnet_top200.py` now makes the historical AppClassNet top-200 behaviors explicit:

- `--run_mode demo`
- `--run_mode full`
- `--pipeline {tr_tr,synthetic,all}`

The canonical script name remains `run_appclassnet_top200.py`.

## Compatibility

The legacy entry points are preserved:

```bash
python3 run_appclassnet_top200.py
python3 run_appclassnet_top200.py -c sf
python3 run_appclassnet_top200.py --full -c sf
```

Resolution rules:

1. Explicit `--run_mode` or `--mode` wins.
2. Legacy `--full` resolves to `full` and emits `DeprecationWarning`.
3. An explicit campaign without run mode resolves to `demo`.
4. No run mode and no explicit campaign resolves to `full`.

Examples:

- `run_mode=demo origin=legacy_campaign`
- `run_mode=full origin=default`
- `run_mode=full origin=cli`

## Profiles

Two `ExecutionProfile` dataclasses define the mode defaults.

`DEMO_PROFILE`:

- demo campaigns: `variational_demo`, `adversarial_demo`
- `execution_mode=batches`
- `use_mmap=True`
- `skip_plots=True`
- `save_synthetic_format=npy_batches`
- reduced real and synthetic sample quotas
- reduced VAE/GAN epoch profile matching the demo campaign constants
- output under `outputs/appclassnet_top200/demo/<run_id>/`

`FULL_PROFILE`:

- historical full campaign list
- historical full epoch and quota defaults
- no demo limits
- output under `outputs/appclassnet_top200/full/<run_id>/`

## Precedence

Configurable arguments are resolved as:

```text
CLI > campaign > profile > global default
```

Each resolved parameter records:

- parameter
- cli_value
- campaign_value
- profile_value
- global_default
- effective_value
- origin

The consolidated `RunResults.json` includes this `effective_parameters` block.

## Pipeline Semantics

`pipeline=tr_tr`:

- runs only TR-TR
- routes through the existing real-real baseline path
- marks TR-TS and TS-TR as `not_run`

`pipeline=synthetic`:

- trains/generates synthetic artifacts
- runs TR-TS and TS-TR
- does not request TR-TR

`pipeline=all`:

- passes `--run_tr_tr` to `main.py`
- runs TR-TR, TR-TS and TS-TR in the same child execution for each combination
- requires all three summaries to be `completed` before `RunResults.json` is accepted

The pipeline plan records one preprocessing pass and one synthetic generation pass for `synthetic` and `all`.

## Split Routing

For `split_mode=provided`, the runner continues to pass explicit train, valid and test `.npy` paths to `main.py`.

Evaluation routing remains:

- TR-TR: real train -> real test
- TR-TS: real train -> synthetic test
- TS-TR: synthetic train -> real test

`valid` remains available to the generator/checkpoint/selection path and is not used as the final test split.

## Artifacts

The final consolidated result is written to:

```text
outputs/appclassnet_top200/<demo|full>/<run_id>/RunResults.json
```

Its structure includes:

- run_id
- run_mode
- pipeline
- pipeline_plan
- campaigns
- effective_parameters
- TR-TR
- TR-TS
- TS-TR
- artifacts

Batch-mode metrics are still mirrored to the legacy aggregate path:

```text
results/appclassnet_top200/batches/metrics.json
```

## Verification

Added unit coverage for:

- no-argument full resolution
- `-c sf` demo resolution
- explicit demo/full modes
- CLI mode winning over campaign alias
- deprecated `--full`
- demo/full profile isolation
- pipeline semantics for `tr_tr`, `synthetic`, and `all`
- single preprocessing/synthetic-generation plan counts
- mode-specific output directories
- `RunResults.json` mode recording
- effective parameter origin recording
- legacy full/demo campaign compatibility
- legacy CSV command construction

Manual dry-run verification:

```bash
python3 run_appclassnet_top200.py --run_mode full --pipeline all --dryrun --skip_plots
```

The command returns `0`, logs the full execution plan, and does not start training.
