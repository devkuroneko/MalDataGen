# CLI Reference

Last updated: 2026-07-15

## Primary CLIs

- `python main.py -h`: canonical runtime CLI.
- `python run_appclassnet_top200.py -h`: AppClassNet campaign runner.
- `python plots.py -h`: plotting CLI.

## AppClassNet Runner

The runner resolves one `ResolvedConfig` per generated `main.py` command. Precedence is:

1. explicit CLI
2. campaign
3. dataset/profile configuration
4. global configuration
5. internal default

The runner writes `command_manifest.json` for real subprocess execution and validates generated commands before execution.

## Common Runtime Options

- `--data_format npy_xy`: load separated NumPy X/y files.
- `--split_mode provided`: use explicit train/valid/test splits.
- `--source_profile appclassnet_top200`: AppClassNet source-scale policy.
- `--evaluation_protocol`: canonical protocol selector or legacy protocol mode.
- `--evaluation_mode`: legacy mode selector.
- `--feature_transform`, `--generator_transform`, `--classifier_transform`, `--evaluation_space`: transform controls.
- `--save_synthetic_format npy_batches`: batch synthetic persistence.
- `--number_samples_per_class`: legacy sample-plan carrier.

## Deprecated Options

- `--full`: deprecated. Use `--run_mode full`.
- `--batch_classifier`: deprecated. Use `--eval_classifier`.

Deprecated options must continue to warn, normalize to the canonical substitute, and remain covered by compatibility tests during the compatibility period.

## Internal Flags

`--effective_number_k_folds` is internal metadata and must not be forwarded to `main.py`. The runner sends `--number_k_folds` with the effective value instead.
