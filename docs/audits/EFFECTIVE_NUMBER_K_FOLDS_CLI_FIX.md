# Effective Number K Folds CLI Fix

## Context

The AppClassNet wrapper used `effective_number_k_folds` as internal metadata for
provided splits, but the generic combination-to-CLI loop forwarded it to
`main.py`:

```text
--number_k_folds 1 --effective_number_k_folds 1
```

`main.py` only exposes `--number_k_folds`, so argparse rejected the subprocess
command with exit code 2.

## Fix

- `effective_number_k_folds` is now internal metadata only.
- `requested_number_k_folds`, `effective_number_k_folds`, `split_mode`,
  `origin`, and `number_k_folds_origin` are kept in the combination metadata
  and logs.
- For `split_mode=provided`, the wrapper resolves
  `effective_number_k_folds=1`.
- For cross-validation, the wrapper preserves the requested fold count.
- The subprocess command passes exactly one fold option:

```text
--number_k_folds <effective_number_k_folds>
```

- `--effective_number_k_folds` is explicitly forbidden in commands targeting
  `main.py`.
- Duplicate command options are detected before `subprocess.run`.

## Regression Coverage

The regression tests cover:

- provided split resolves effective folds to 1;
- cross-validation preserves requested folds;
- batch command contains `--number_k_folds 1`;
- batch command does not contain `--effective_number_k_folds`;
- `--number_k_folds` is not duplicated;
- duplicate command options are rejected before subprocess execution;
- `main.py` accepts the produced real-resample command and exits with code 0;
- legacy CSV command shape remains CSV-based and keeps a single
  `--number_k_folds`.
