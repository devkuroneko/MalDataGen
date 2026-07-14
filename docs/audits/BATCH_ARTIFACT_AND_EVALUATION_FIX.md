# Batch Artifact and Evaluation Fix

## Auditoria

| Item | Arquivo | Funcao/classe | Linha aprox. |
| --- | --- | --- | --- |
| Criacao de `combination_1_<timestamp>` | `Engine/DataIO/DirectoryManager.py` | `DirectoryManager._resolve_experiment_directory()` | 145-176 |
| Construcao de `results_paths` | `run_appclassnet_top200.py` | `main()` / `results_grouping` | 2528-2610 |
| `write_batches_metrics` | `run_appclassnet_top200.py` | `write_batches_metrics()` | 1416-1465 |
| `validate_requested_evaluations_completed` | `run_appclassnet_top200.py` | `validate_requested_evaluations_completed()` | 1359-1379 |
| Execucao de TR-TS | `main.py` / `Engine/Evaluation/TrTs.py` | `run_synthetic_evaluation_modes()` / `TrTs.evaluation_TR_TS()` | 111-132 / 57-150 |
| Execucao de TS-TR | `main.py` / `Engine/Evaluation/TsTr.py` | `run_synthetic_evaluation_modes()` / `TsTr.evaluation_TS_TR()` | 111-132 / 79-180 |
| `SyntheticBatchWriter` | `Engine/DataIO/SyntheticBatchIO.py` | `SyntheticBatchWriter` | 16-142 |
| `SyntheticBatchReader` | `Engine/DataIO/SyntheticBatchIO.py` | `SyntheticBatchReader` | 145-182 |
| Criacao do `manifest.json` | `Engine/DataIO/SyntheticBatchIO.py` | `SyntheticBatchWriter.close()` | 130-142 |
| Chamada de `plots.py` | `run_appclassnet_top200.py` | `build_plot_command()` / `main()` | 1907-1935 / 2590-2640 |
| Carregamento CSV em `plots.py` | `plots.py` | `load_dataset_processor()` | 115-121 |

## Correcoes aplicadas

- Introduzido `RunArtifacts` em `run_appclassnet_top200.py` para preservar `combination_dir`, `results_json_path`, manifestos `train/test`, `DataGenerated` e `Monitor`.
- O runner agora resolve o diretorio real criado pelo subprocesso (`combination_1_<timestamp>`) com `resolve_actual_run_dir()` e agrega usando esse caminho.
- `write_batches_metrics()` aceita artefatos reais, usa `Results.json` retornado pelo artefato e associa TR-TS ao manifesto `synthetic_batches/test/manifest.json` e TS-TR ao manifesto `synthetic_batches/train/manifest.json`.
- `run_synthetic_evaluation_modes()` usa dois blocos independentes:
  - `TR-TS`: real train contra synthetic test.
  - `TS-TR`: synthetic train contra real test.
- `SyntheticBatchWriter` passou a aceitar `split_name` e `fold_number`; os manifestos incluem `split`, `fold`, `seed`, `model`, `data_space`, `transform_id`, `shape` e `dtype`.
- O modo batches incremental gera dois leitores: `train_reader` e `test_reader`, empacotados em `SyntheticSplitBatchReaders`.
- `plots.py` detecta `.npy` e usa `numpy.load`; CSV antigo continua via `CSVDataProcessor`.
- `plots.py` nao procura `DataOutput_K_fold_*.txt` quando existe `DataGenerated/synthetic_batches`.
- Plots em batches sao executados depois da consolidacao de `metrics.json`; falhas opcionais de plots sao registradas como warning e nao invalidam metricas.

## Testes

Executados:

```bash
python -m py_compile main.py run_appclassnet_top200.py plots.py Engine/DataIO/SyntheticBatchIO.py
pytest -q tests/test_appclassnet_evaluation_mode.py tests/test_plots_npy_support.py tests/test_synthetic_batch_io.py
```

Resultado: `23 passed`.

## Compatibilidade

- O fluxo CSV legado permanece suportado: `SyntheticBatchWriter` manteve o manifesto antigo quando `split_name` nao e usado, e `plots.py` continua carregando CSV por `CSVDataProcessor`.
- O agregador ainda aceita listas antigas de `Results.json`, mas prefere `RunArtifacts` quando disponivel.
