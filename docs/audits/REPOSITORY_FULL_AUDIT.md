# Repository Full Audit

Generated at: 2026-07-15T14:22:17
Repository: `/home/gatopreto/PycharmProjects/MalDataGen`

## Executive Summary

- No functional refactor was performed and no repository files were deleted.
- Audit artifacts were written under `docs/audits/`.
- Current test baseline: `=============================== warnings summary ===============================; ================= 213 passed, 18 skipped, 2 warnings in 19.06s =================`.
- The worktree was already dirty before this audit; modified and untracked files are recorded in `test_baseline.json` and `repository_inventory.json`.

## Repository State

```
## appclassnet
 M Engine/Arguments/Arguments.py
 M Engine/Arguments/ArgumentsDataLoader.py
 M Engine/Arguments/ArgumentsFramework.py
 M Engine/Arguments/__pycache__/Arguments.cpython-310.pyc
 M Engine/Arguments/__pycache__/ArgumentsDataLoader.cpython-310.pyc
 M Engine/Arguments/__pycache__/ArgumentsFramework.cpython-310.pyc
 M Engine/Classifiers/BatchClassifiers.py
 M Engine/DataIO/DatasetContracts.py
 M Engine/DataIO/SyntheticBatchIO.py
 M Engine/Evaluation/CrossValidation.py
 M Engine/Evaluation/Evaluation.py
 M Engine/Evaluation/EvaluationRunner.py
 M Engine/Evaluation/TrTs.py
 M Engine/Evaluation/TsTr.py
 M Engine/Evaluation/__pycache__/CrossValidation.cpython-310.pyc
 M Engine/Evaluation/__pycache__/Evaluation.cpython-310.pyc
 M Engine/Evaluation/__pycache__/TrTs.cpython-310.pyc
 M Engine/Evaluation/__pycache__/TsTr.cpython-310.pyc
 M Engine/Metrics/Metrics.py
 M Engine/Metrics/__pycache__/Metrics.cpython-310.pyc
 M Engine/Models/GenerativeModels.py
 M Engine/Models/__pycache__/GenerativeModels.cpython-310.pyc
 M README.md
 M main.py
 M run_appclassnet_top200.py
 M tests/test_appclassnet_evaluation_mode.py
 M tests/test_evaluation_runner.py
 M tests/test_feature_transform_manager.py
 M tests/test_synthetic_batch_io.py
?? Engine/DataIO/SyntheticQualityAudit.py
?? Engine/Evaluation/TrTsTr.py
?? docs/audits/DEMO_FULL_EXECUTION_MODE_REFACTOR.md
?? docs/audits/EFFECTIVE_NUMBER_K_FOLDS_CLI_FIX.md
?? docs/audits/REAL_RESAMPLE_RAW_SPLIT_ROUTING_FIX.md
?? docs/audits/SYNTHETIC_QUALITY_ROOT_CAUSE.md
?? docs/audits/TR_TS_TR_AUGMENTATION_PROTOCOL.md
?? tests/test_real_resample_control_routing.py
?? tests/test_synthetic_quality_audit.py
```

- Branch: `appclassnet`
- Python: `3.10.12`
- Python executable: `/home/gatopreto/.pyenv/versions/3.10.12/bin/python`
- Pytest: `pytest 9.0.2`
- PYTHONPATH: ``
- OS: `Linux-7.0.11-76070011-generic-x86_64-with-glibc2.35`

## Test Baseline

- `python -m pytest --collect-only -q`: exit `0`, timed_out=`False`
- `python -m pytest -ra --tb=short`: exit `0`, timed_out=`False`
- `python run_appclassnet_top200.py -h`: exit `0`, timed_out=`False`
- `python main.py -h`: exit `0`, timed_out=`False`

Key baseline result: `213 passed, 18 skipped, 2 warnings` from `python -m pytest -ra --tb=short`.
The `main.py -h` command exits 0 but emits TensorFlow/cuDNN/cuBLAS and Matplotlib cache messages before help output.

## Inventory Summary

- arquivos gerados: 8 files, 1505927 bytes
- codigo-fonte: 276 files, 3014737 bytes
- configurações: 5 files, 1132273 bytes
- datasets: 24 files, 23755511768 bytes
- documentação: 93 files, 4345728 bytes
- logs: 10 files, 1714894 bytes
- outputs: 14379 files, 293218475 bytes
- resultados: 16 files, 797728409 bytes
- scripts de execução: 6 files, 80409 bytes
- temporários/legado: 1 files, 1075 bytes
- testes: 23 files, 177585 bytes

Full file-level inventory is in `docs/audits/repository_inventory.json` with path, size, type, likely function, importers, mtime, Git status and source/artifact/duplicate candidacy.

## Exact Duplicates

- code: 11 duplicate hash groups
- dataset: 9 duplicate hash groups
- documentation: 1 duplicate hash groups
- outputs: 3713 duplicate hash groups

- `e6d4ba1fc59f` `code`: Tools/Plot/utils.py, Tools/utils.py
- `1a57d7eafcac` `code`: Tools/Plot/PlotHeatMap.py, Tools/PlotHeatMap.py
- `3f4f7847e263` `documentation`: Docs/Documentation/AdversarialAlgorithm/Icon/Logo.png, Docs/Documentation/AutoencoderAlgorithm/Icon/Logo.png, Docs/Documentation/CallbackDiffusionModel/Icon/Logo.png, Docs/Documentation/CallbackModel/Icon/Logo.png, Docs/Documentation/CallbackResources/Icon/Logo.png, Docs/Documentation/DiffusionAlgorithm/Icon/Logo.png
- `56498b37990f` `outputs`: outputs/out_2026-07-15_14-14-25/Audits/synthetic_quality_audit.json, results/appclassnet_top200/batches/synthetic_quality_audit.json
- `c6a39b51cc40` `code`: Engine/Algorithms/DenoisingDiffusion/GaussianDenoisingDiffusion.py, Engine/Algorithms/LatentDiffusion/GaussianLatentDiffusion.py
- `e3b0c44298fc` `code`: Engine/Classifiers/__init__.py, experiments/classifiers_baseline/decision_tree_baseline.py, experiments/classifiers_baseline/random_forest_baseline.py
- `2fe9360195f8` `code`: Engine/Models/DenoisingDiffusion/VanillaDecoderDiffusion.py, Engine/Models/LatentDiffusion/VanillaDecoderDiffusion.py
- `3c17fd39dc87` `code`: Engine/Models/DenoisingDiffusion/DiffusionAutoencoder.py, Engine/Models/LatentDiffusion/DiffusionAutoencoder.py
- `69e5524c817f` `code`: Engine/Models/DenoisingDiffusion/VariationalAutoencoderModel.py, Engine/Models/LatentDiffusion/VariationalAutoencoderModel.py
- `e887f18fdbe0` `code`: Engine/Models/DenoisingDiffusion/VanillaEncoderDiffusion.py, Engine/Models/LatentDiffusion/VanillaEncoderDiffusion.py
- `2b088e34c307` `code`: Engine/Models/Wasserstein/ModelWassersteinGAN.py, Engine/Models/WassersteinGP/ModelWassersteinGAN.py
- `5266ce585c93` `code`: Engine/Models/Wasserstein/VanillaGenerator.py, Engine/Models/WassersteinGP/VanillaGenerator.py
- `8650fbf5ce5d` `code`: Engine/Models/Wasserstein/VanillaDiscriminator.py, Engine/Models/WassersteinGP/VanillaDiscriminator.py
- `89ec1adb6857` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy, outputs/out_2026-07-15_14-08-34/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy, outputs/out_2026-07-15_14-11-45/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy, outputs/out_2026-07-15_14-12-14/DataGenerated/synthetic_batches/test/class_147/batch_000000.npy
- `4a8365869b11` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy, outputs/out_2026-07-15_14-08-34/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy, outputs/out_2026-07-15_14-11-45/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy, outputs/out_2026-07-15_14-12-14/DataGenerated/synthetic_batches/test/class_058/batch_000000.npy
- `23505ad80657` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy, outputs/out_2026-07-15_14-08-34/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy, outputs/out_2026-07-15_14-11-45/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy, outputs/out_2026-07-15_14-12-14/DataGenerated/synthetic_batches/test/class_194/batch_000000.npy
- `da28870ecc6a` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_039/batch_000000.npy, outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/train/class_039/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_039/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/train/class_039/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_039/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/train/class_039/batch_000000.npy
- `332ed217ebc7` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_059/batch_000000.npy, outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/train/class_059/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_059/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/train/class_059/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_059/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/train/class_059/batch_000000.npy
- `cde6681c9be2` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy, outputs/out_2026-07-15_14-08-34/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy, outputs/out_2026-07-15_14-11-45/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy, outputs/out_2026-07-15_14-12-14/DataGenerated/synthetic_batches/test/class_064/batch_000000.npy
- `72db58612c65` `outputs`: outputs/out_2026-07-15_14-01-50/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy, outputs/out_2026-07-15_14-07-34/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy, outputs/out_2026-07-15_14-08-09/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy, outputs/out_2026-07-15_14-08-34/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy, outputs/out_2026-07-15_14-11-45/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy, outputs/out_2026-07-15_14-12-14/DataGenerated/synthetic_batches/test/class_048/batch_000000.npy
- Additional groups omitted here; see `exact_duplicates.json` for all 3734 groups.

## Semantic Duplicates And Competing Implementations

- AST-identical function duplicate groups: 449.
- Repeated class signatures: 19.
- argparse: 43 candidate functions/methods.
- batch_writer: 79 candidate functions/methods.
- fold_handling: 181 candidate functions/methods.
- manifest: 96 candidate functions/methods.
- npy_loader: 290 candidate functions/methods.
- path_resolution: 159 candidate functions/methods.
- seed_resolution: 103 candidate functions/methods.
- stratified_selection: 246 candidate functions/methods.
- tr_ts_ts_tr: 83 candidate functions/methods.
- transform: 155 candidate functions/methods.
- xy_validation: 1084 candidate functions/methods.

Important competing areas identified:
- NPY loading and X/y validation are split across `NpyXYLoader`, `BatchNpyDataset`, `DatasetContracts`, `CrossValidation`, and runtime guards in `main.py`.
- Batch writing and manifest creation are concentrated in `SyntheticBatchIO`, but audit outputs also write manifests/reports directly in `SyntheticQualityAudit` and evaluation paths.
- TR-TS, TS-TR, TR-TR and TR-TS-TR evaluation logic exists as separate modules plus dispatcher logic in `main.py`.
- Plot implementations exist both at `Tools/` root and under `Tools/Plot/`.

## CLI Argument Matrix

- Wrapper-only options: 30
- Main-only options: 433
- Shared options: 46
- Duplicate option/destination groups: 61 destinations, 65 exact options

Full matrix is in `docs/audits/cli_argument_matrix.json`.

## Architecture Findings

- Import cycles detected by static graph: 0.
- `except Exception` handlers detected: 71, with 35 without rethrow.
- Hardcoded absolute path literals detected: 8.
- Global mutable assignments detected at module level: 0.
- Static `.npy` as CSV/readtxt hits: 0.

## Known Problem Checks

- `effective_number_k_folds`: coberto no worktree atual. Evidence: tests/test_appclassnet_evaluation_mode.py::test_batch_command_sends_only_effective_number_k_folds; tests/test_real_resample_control_routing.py::test_batch_command_for_provided_split_uses_one_effective_fold
- `split_mode_provided_one_fold`: coberto no worktree atual. Evidence: Engine/Arguments/ArgumentsDataLoader.py:106-114; tests/test_appclassnet_evaluation_mode.py::test_provided_split_resolves_effective_folds_to_one
- `num_classes_subset_effective_classes`: coberto por testes de coleta/loader, revisar propagação ponta-a-ponta em runs reais. Evidence: Engine/Arguments/ArgumentsDataLoader.py:273-277; tests/test_multiclass_pipeline.py::test_top_200_sparse_labels_respect_user_num_classes
- `subset_paths_full_dataset`: coberto por teste negativo de alinhamento. Evidence: tests/test_synthetic_sanity_checks.py::test_subset_by_classes_fails_before_masking_when_y_is_from_full_dataset
- `vae_epochs_cli_precedence`: coberto no wrapper. Evidence: run_appclassnet_top200.py:572-602; tests/test_appclassnet_evaluation_mode.py::test_cli_sample_arguments_override_campaign
- `number_samples_per_class_outside_subset`: risco residual: metadado ainda é dicionário multiuso e aparece em muitas camadas. Evidence: main.py:1121-1169; Engine/Models/GenerativeModels.py:4004
- `real_resample_aggregate_data`: coberto por validação/teste. Evidence: main.py:121-123; tests/test_real_resample_control_routing.py::test_centroid_matrix_with_one_row_per_class_is_rejected
- `valid_as_test`: coberto no worktree atual. Evidence: main.py:175-184; tests/test_appclassnet_evaluation_mode.py::test_provided_both_never_routes_valid_to_ts_tr
- `x_y_different_indices`: coberto por validação e testes. Evidence: Engine/Evaluation/CrossValidation.py:220-249; tests/test_real_resample_control_routing.py::test_selected_x_rows_remain_aligned_with_class_labels
- `preserve_inverse_transform`: coberto. Evidence: tests/test_feature_transform_manager.py::test_appclassnet_preserve_inverse_synthetic_batch_is_noop
- `runs_mixed_outputs`: risco residual alto. Evidence: outputs/out_*/...; results/appclassnet_top200/batches/...
- `DecisionTreeSubset_effective_fit_rows`: parcial: procurar registro em outputs; requer validação de artefato real. Evidence: Engine/Classifiers/BatchClassifiers.py; tests/test_synthetic_quality_audit.py
- `SyntheticClassCollapseError_single_metric`: melhorado: auditor usa múltiplas seções, mas exceção ainda condensa em collapsed/reasons. Evidence: Engine/DataIO/SyntheticQualityAudit.py:69-73; Engine/DataIO/SyntheticQualityAudit.py:186
- `decoder_class_conditioning`: risco residual: condicionamento aparece distribuído por modelos. Evidence: Engine/Models/GenerativeModels.py; Engine/Models/Adversarial/VanillaGenerator.py
- `label_mapping_0_199_vs_1_200`: coberto. Evidence: tests/test_multiclass_pipeline.py::test_two_hundred_classes_use_zero_based_labels; tests/test_npy_xy_loader.py::test_can_remap_one_based_labels_to_zero_based_explicitly
- `vae_batch_size_multiple_locations`: risco residual. Evidence: run_appclassnet_top200.py:297; Engine/Arguments/ArgumentsDataLoader.py:292-299; Engine/Arguments/ArgumentsVariationalAutoencoder.py
- `demo_campaign_overrides_cli`: coberto por resolução de precedência. Evidence: run_appclassnet_top200.py:572-602; tests/test_appclassnet_evaluation_mode.py::test_cli_sample_arguments_override_campaign

## Generated Files Audit

- Manifest files: 71
- JSON result/report files: 317
- Data files under generated roots: 14006
- Timestamped run dirs: 35
- Incomplete run dirs by expected-file heuristic: 3
- Missing paths referenced by manifests: 32

## Findings

### Crítico

No critical finding was confirmed by the static audit. The audit did not find Git-tracked files above 10 MiB in the scanned inventory; large datasets and generated artifacts are present in the workspace but are not tracked by Git.

### Alto

#### ALTO-001 - `main.py -h` executa imports pesados e emite logs de TensorFlow/Matplotlib
- Evidence: Baseline de `python main.py -h` tem stderr com TensorFlow/cuDNN/cuBLAS e cache temporário do Matplotlib antes do help.
- Files: main.py:35, Engine/Arguments/Arguments.py:232
- Risk: Help e inspeção de CLI ficam lentos, ruidosos e dependentes de bibliotecas pesadas.
- Behavior affected: CLI, ambiente local e CI.
- Proposed correction: Separar construção do parser de imports de runtime; adiar TensorFlow/Matplotlib até execução real.
- Dependencies: Reorganização de imports e cuidado com compatibilidade do decorator `arguments`.
- Required test: Teste que executa `python main.py -h` e falha se stderr contiver inicialização de runtime pesado.
- Backward compatibility: Média: mudar ordem de imports pode revelar dependências implícitas.

#### ALTO-002 - Saídas são gravadas em `outputs/` e `results/` com responsabilidades sobrepostas
- Evidence: Artefatos equivalentes aparecem em `results/appclassnet_top200/batches/*` e em múltiplos `outputs/out_*/*`.
- Files: Engine/DataIO/SyntheticQualityAudit.py:27, Engine/Evaluation/CrossValidation.py:70, run_appclassnet_top200.py:62
- Risk: Runs diferentes podem ser comparados ou consumidos por engano; manifests globais podem apontar para a última execução.
- Behavior affected: Reprodutibilidade de avaliação e auditoria.
- Proposed correction: Centralizar escrita em `RunArtifacts` e tornar `results/` apenas índice/cache explícito ou remover do caminho de runtime.
- Dependencies: Definição de contrato de diretórios e migração de consumidores existentes.
- Required test: Teste que executa duas runs com timestamps distintos e verifica isolamento completo dos artefatos.
- Backward compatibility: Alta: scripts/documentos podem assumir os caminhos globais atuais.

#### ALTO-003 - Matriz de CLI possui opções exclusivas do wrapper e opções exclusivas do `main.py`
- Evidence: Wrapper-only: 30; main-only: 433; shared: 46.
- Files: run_appclassnet_top200.py, Engine/Arguments/*.py
- Risk: O wrapper pode aceitar opções que não chegam ao destino ou gerar comandos sem cobertura direta no parser final.
- Behavior affected: Execução de campanhas e precedência de parâmetros.
- Proposed correction: Manter uma matriz declarativa de argumentos encaminhados, consumidos localmente e bloqueados.
- Dependencies: Decisão de quais flags pertencem ao wrapper versus `main.py`.
- Required test: Teste parametrizado que valida cada flag gerada pelo wrapper contra o parser de destino.
- Backward compatibility: Média: pode exigir aliases/deprecações.

### Médio

#### MED-001 - Há implementações concorrentes de responsabilidades de plot e ferramentas
- Evidence: Arquivos homônimos ou quase homônimos existem em `Tools/` e `Tools/Plot/`, incluindo `PlotDistanceMetrics.py`, `PlotTrainingCurve.py`, `PlotHeatMap.py`, `PlotConfusionMatrix.py`.
- Files: Tools, Tools/Plot
- Risk: Correções podem ser aplicadas em uma implementação e não na outra.
- Behavior affected: Geração de gráficos e manutenção.
- Proposed correction: Definir implementação canônica e substituir duplicatas por wrappers de compatibilidade.
- Dependencies: Inventariar importadores antes de qualquer migração.
- Required test: Teste de import/execução para cada comando público de plot.
- Backward compatibility: Baixa a média, dependendo de usuários diretos dos módulos antigos.

#### MED-002 - Arquivos de documentação duplicados por convenção de caixa (`Docs/` e `docs/`)
- Evidence: Ambas as árvores existem e contêm documentação gerada/manual.
- Files: Docs, docs
- Risk: Em sistemas case-sensitive funciona, mas em ambientes case-insensitive pode causar confusão e colisões.
- Behavior affected: Documentação e distribuição do pacote.
- Proposed correction: Escolher uma árvore canônica e criar índice de compatibilidade antes de mover conteúdo.
- Dependencies: Checar links relativos e publicação de docs.
- Required test: Teste/link checker ou script que valida links internos.
- Backward compatibility: Média em Windows/macOS case-insensitive.

#### MED-003 - Manipuladores `except Exception` ainda aparecem sem rethrow em pontos de runtime
- Evidence: 35 handlers `except Exception` sem `raise` detectados por AST.
- Files: Engine/Algorithms/Adversarial/AdversarialAlgorithm.py, Engine/Algorithms/DenoisingDiffusion/AlgorithmDenoisingDiffusion.py, Engine/Algorithms/LatentDiffusion/AlgorithmLatentDiffusion.py, Engine/Algorithms/ThirdParty/SDVInterfaceAlgorithm.py, Engine/Callbacks/CallbackResources.py, Engine/DataIO/DatasetContracts.py, Engine/DataIO/DirectoryManager.py, Engine/Evaluation/CrossValidation.py, Engine/Metrics/Distance/ManhattanDistance.py, Engine/Metrics/Metrics.py, Engine/Support/HardwareManager.py, Tools/PlotClassificationMetrics.py, Tools/PlotFidelityMetrics.py, main.py, plots.py, plots_svm.py, run_appclassnet_top200.py, scripts/audit_preprocessing_execution.py, scripts/validate_appclassnet_preprocessing_policy.py
- Risk: Falhas podem virar logs parciais, `sys.exit` ou comportamento degradado sem contexto estruturado.
- Behavior affected: Diagnóstico de falhas em geração, métricas e plots.
- Proposed correction: Trocar por exceções específicas ou reemitir com contexto preservando traceback.
- Dependencies: Auditar intenção de cada handler.
- Required test: Testes negativos por módulo com erro esperado.
- Backward compatibility: Baixa se mensagens públicas forem preservadas.

### Baixo

#### BAIXO-001 - O repositório rastreia `__pycache__`/`.pyc`
- Evidence: Git lista arquivos `Engine/**/__pycache__/*.pyc` como rastreados.
- Files: Engine/Activations/__pycache__/SELU.cpython-310.pyc, Engine/Activations/__pycache__/ELU.cpython-310.pyc, Engine/DataIO/__pycache__/CSVLoader.cpython-310.pyc, Engine/Metrics/Binary/__pycache__/FalsePositiveRate.cpython-310.pyc, Engine/Activations/__pycache__/Softmax.cpython-310.pyc, Engine/Metrics/Binary/__pycache__/TrueNegative.cpython-310.pyc, Engine/Algorithms/VariationalAutoencoder/__pycache__/AlgorithmVariationalAutoencoder.cpython-310.pyc, Engine/Classifiers/Algorithms/__pycache__/StochasticGradientDescent.cpython-310.pyc, Engine/Algorithms/LatentDiffusion/__pycache__/AlgorithmLatentDiffusion.cpython-310.pyc, Engine/Arguments/__pycache__/ArgumentsOptimizer.cpython-310.pyc, Engine/Classifiers/Algorithms/__pycache__/AdaBoost.cpython-310.pyc, Engine/Metrics/Binary/__pycache__/Precision.cpython-310.pyc, Engine/Algorithms/QuantizedVAE/__pycache__/AlgorithmQuantizedVAE.cpython-310.pyc, Engine/Algorithms/Adversarial/__pycache__/AdversarialAlgorithm.cpython-310.pyc, Engine/Arguments/__pycache__/ArgumentsWassersteinGAN.cpython-310.pyc, Engine/Activations/__pycache__/ReLU.cpython-310.pyc, Engine/Models/DiffusionKernel/__pycache__/DiffusionModelUnet.cpython-310.pyc, Engine/Layers/__pycache__/AttentionBlockLayer.cpython-310.pyc, Engine/Classifiers/Algorithms/__pycache__/RandomForest.cpython-310.pyc, Engine/Classifiers/__pycache__/Classifiers.cpython-310.pyc
- Risk: Ruído constante no status e diffs dependentes de versão do Python.
- Behavior affected: Higiene do repositório.
- Proposed correction: Em etapa futura, remover do índice e reforçar `.gitignore`; não apagar nesta auditoria.
- Dependencies: Nenhuma além da política do Git.
- Required test: Verificação de `git ls-files '*__pycache__*'` vazia após correção.
- Backward compatibility: Baixa.

### Informativo

#### INFO-001 - Problemas conhecidos de AppClassNet têm cobertura de regressão no worktree atual
- Evidence: Testes coletados incluem casos para folds efetivos, split provided, subset, real_resample, X/y alignment, label mapping, VAE epochs e argumentos duplicados.
- Files: tests/test_appclassnet_evaluation_mode.py, tests/test_real_resample_control_routing.py, tests/test_feature_transform_manager.py
- Risk: A proteção depende de testes que incluem arquivos modificados/não rastreados no estado atual.
- Behavior affected: Confiança na linha de base.
- Proposed correction: Antes de refatorar, consolidar esses testes e garantir que estejam versionados.
- Dependencies: Decisão sobre inclusão das mudanças atuais no branch.
- Required test: Executar novamente a suíte após qualquer correção funcional.
- Backward compatibility: Nenhum para documentação; médio se os testes não forem integrados ao branch.

#### INFO-002 - Arquivos grandes existem localmente, mas não estão rastreados pelo Git
- Evidence: `Datasets/` ocupa aproximadamente 23 GiB no workspace; `repository_inventory.json` classifica 24 arquivos de dataset, mas `large_tracked_files` ficou vazio na auditoria.
- Files: Datasets/raw/AppClassNet/top200, Datasets/processed, results/appclassnet_top200/batches/preprocessing/scaled_npy
- Risk: Alto consumo local de disco e hashes demorados em auditorias completas.
- Behavior affected: Auditoria, backup local e onboarding.
- Proposed correction: Manter esses caminhos ignorados e documentar como reconstruir ou baixar os dados.
- Dependencies: Política de dados e disponibilidade do dataset original.
- Required test: Verificação de `git ls-files` para garantir que datasets/outputs grandes continuam fora do índice.
- Backward compatibility: Baixa, desde que os caminhos esperados sejam documentados.

## Prioritized Correction Plan

1. Freeze the current test baseline and decide which modified/untracked fixes are part of the branch before any functional work.
2. Stop tracking generated bytecode and define a non-destructive migration plan for large tracked datasets/results.
3. Establish a single artifact contract for run-scoped outputs versus global results caches.
4. Split parser construction from heavy runtime imports so `main.py -h` is cheap and deterministic.
5. Convert the wrapper-to-main CLI forwarding into a tested declarative matrix.
6. Consolidate duplicate plotting/tool modules through compatibility wrappers after importers are verified.
7. Reduce semantic duplication in loaders, X/y validation, manifests, seeds, paths and folds around existing `DatasetBundle` and `RunArtifacts` contracts.

## Acceptance Checklist

- [x] Project tree documented.
- [x] Exact duplicates identified by SHA-256.
- [x] Semantic duplicates identified by AST/roles/calls.
- [x] Parsers and arguments mapped.
- [x] Current tests recorded.
- [x] No repository file deleted.
- [x] Existing failures were not hidden; baseline commands and stderr/stdout are saved in JSON.
- [x] Prioritized correction plan included.
