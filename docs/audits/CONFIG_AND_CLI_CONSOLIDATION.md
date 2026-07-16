# Config And CLI Consolidation

Generated: 2026-07-15

## Scope

This change consolidates AppClassNet runner configuration resolution in `run_appclassnet_top200.py` without changing the public `main.py` parser contract.

The runner now builds one `ResolvedConfig` per generated `main.py` command. The resolved config is composed of:

- `RunConfig`
- `DatasetConfig`
- `GeneratorConfig`
- `EvaluationConfig`
- `ClassifierConfig`
- `TransformConfig`
- `SamplePlan`
- `ResolvedConfig`
- existing `RunArtifacts`

## Precedence

Effective values are resolved with this order:

1. explicit CLI
2. campaign
3. dataset/profile configuration
4. global configuration
5. internal default

Every resolved parameter recorded through the resolver includes:

- `cli_value`
- `campaign_value`
- `profile_value`
- `default_value`
- `effective_value`
- `origin`

`global_default` is kept as a compatibility alias for older audit/tests.

## Before / After

| Concern | Before | After |
| --- | --- | --- |
| Effective config | Spread across profile application, sample resolution, fold resolution and command builders. | One `ResolvedConfig` is created per command and registered in the command manifest. |
| `split_mode=provided` folds | Internal metadata included `effective_number_k_folds`; command builders manually avoided sending it. | `ResolvedConfig.dataset.effective_number_k_folds=1`; only `--number_k_folds 1` is sent. |
| `--effective_number_k_folds` | Guarded by a text check. | Still forbidden and validated before subprocess. |
| `--vae_epochs 30` | Alias could be forwarded as `--vae_epochs`. | Normalized to `--variational_autoencoder_number_epochs 30`; `--vae_epochs` is not sent to `main.py`. |
| `--num_classes_subset 10` | Forwarded as subset argument; effective class domain was implicit in `main.py`. | Manifest records `effective_num_classes=10`; command keeps the subset contract for `main.py`. |
| Synthetic quotas | Train/test/generated values resolved separately in command builders. | `SamplePlan` records train, test, generated, required total and legacy `number_samples_per_class`. |
| Transform config | Batch/normal builders hardcoded `feature_transform=preserve` in places. | `feature_transform`, `generator_transform`, `classifier_transform` and `evaluation_space` are preserved independently. |
| Classifier aliases | Wrapper accepted `--batch_classifier` and also forwarded alias to `main.py`. | Wrapper accepts alias with deprecation warning, normalizes to `--eval_classifier`, and does not send `--batch_classifier`. |
| Classifier metadata | Batch metadata used classifier display and key names. | Metadata includes `requested_classifier`, `effective_classifier` and `effective_fit_rows`. |
| Command validation | Duplicate flags and fold-specific checks. | Duplicate flags, unknown flags, invalid parser types, invalid flag/value pairs and forbidden internal flags are validated before subprocess. |
| Command manifest | Not written before subprocess. | `command_manifest.json` is written under each run output directory before real subprocess execution. |
| Dry run | Command was logged; no manifest contract. | Command is validated and logged canonically; no subprocess and no manifest write. |

## Command Validation

`validate_command_before_subprocess()` now validates `main.py` commands against a parser assembled from the same `Engine.Arguments.*` parser functions used by `main.py`.

It catches:

- repeated flags;
- `--effective_number_k_folds`;
- missing or invalid flag values;
- unknown flags;
- type conversion failures;
- parser choice violations.

## Specific Cases

| Case | Result |
| --- | --- |
| `split_mode=provided` | Sends only `--number_k_folds 1`; records requested and effective folds. |
| `--vae_epochs 30` | Sends `--variational_autoencoder_number_epochs 30`. |
| `--num_classes_subset 10` | Records `effective_num_classes=10` in the resolved config. |
| synthetic train/test | Validates `generated_samples_per_class >= synthetic_train + synthetic_test`. |
| transforms | Keeps all four transform dimensions separate. |
| classifier | Records requested/effective classifier and fit rows; `DecisionTreeSubset` remains named as `DecisionTreeSubset`. |

## Tests Added Or Extended

- CLI precedence and campaign not overwriting explicit values.
- Duplicate command flags.
- Unknown command flags.
- Effective folds for provided split.
- VAE epoch alias normalization.
- Effective class count for class subsets.
- SamplePlan generation.
- Separate transform propagation.
- Legacy CSV command compatibility.
- Existing campaign compatibility.
- Dry run avoiding subprocess and manifest side effects.
- Batch classifier metadata for requested/effective classifier and `effective_fit_rows`.

## Compatibility Notes

- `--batch_classifier` remains accepted by the wrapper as a deprecated alias.
- `--full` remains accepted and still emits its deprecation warning.
- Existing `main.py` options remain valid; the runner now rejects options that do not exist in the destination parser.
- `command_manifest.json` is additive and written only for real subprocess executions.

