# Experiment Protocols

Last updated: 2026-07-15

## Canonical Protocol Selection

Use `--evaluation_protocol` for explicit protocol selection when possible:

- `tr_tr`
- `tr_ts`
- `ts_tr`
- `tr_plus_ts_tr`
- `all`

Legacy selectors are still supported:

- `--evaluation_protocol legacy`
- `--evaluation_protocol appclassnet_strict`
- `--evaluation_mode tr_ts`
- `--evaluation_mode ts_tr`
- `--evaluation_mode both`
- `--evaluation_mode tr_ts_tr`
- `--run_tr_tr`

## Protocol Contracts

| Protocol | Training source | Test source | Canonical module |
| --- | --- | --- | --- |
| TR-TR | real train | real test | `Engine/Evaluation/TrTr.py` |
| TR-TS | real train | synthetic test | `Engine/Evaluation/TrTs.py` |
| TS-TR | synthetic train | real test | `Engine/Evaluation/TsTr.py` |
| TR+TS-TR | real train + synthetic train | real test | `Engine/Evaluation/TrTsTr.py` |

For AppClassNet strict mode, validation and final evaluation must use the test split, not the validation split.

## Synthetic Controls

`--synthetic_control` records control behavior for synthetic evaluation. The default `none` keeps the normal generated data path. Control runs must be reported separately from primary experiment results.

## Reproducibility Requirements

Every run should record:

- CLI command and `command_manifest.json`.
- train/valid/test source paths.
- sample plan and class counts.
- transform configuration and transform history.
- random seed.
- synthetic manifest path.
- evaluation protocol IDs.
- classifier and fit-row metadata.

## Legacy Compatibility

Legacy CSV commands remain valid. Legacy wrapper options should be deprecated with a warning, normalized to the canonical option, and covered by a compatibility test before removal.
