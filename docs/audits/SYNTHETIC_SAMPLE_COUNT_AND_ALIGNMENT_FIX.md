# Synthetic Sample Count And Alignment Fix

Data: 2026-07-14

## Escopo

Este documento cobre a falha observada em `Engine/DataIO/SyntheticSanityChecks.py::_stratified_real_subset`:

```text
IndexError: index 98070 is out of bounds for axis 0 with size 81832
```

Tambem cobre a inconsistencia de planejamento no AppClassNet top-200, onde a avaliacao podia solicitar `synthetic_train_samples_per_class + synthetic_test_samples_per_class`, mas o subprocesso ainda recebia `number_samples_per_class` fixo em 256 por classe.

## Auditoria Do Fluxo

Pontos localizados:

- `run_synthetic_sanity_checks`: chamado em `main.py` depois de `audit_synthetic_label_generation`, nos caminhos normal, particionado e incremental.
- `SyntheticSanityChecker.__init__`: normaliza `real_y` e agora valida o par real X/y antes de qualquer uso.
- `_train_real_decision_tree_subset`: treina um `DecisionTreeClassifier` sobre subconjunto real estratificado.
- `_stratified_real_subset`: seleciona indices por classe a partir de `real_y` e indexa `real_x`.
- Selecao dos folds: `Engine/Evaluation/CrossValidation.py`, via `StratifiedData`.
- Selecao estratificada em batches: `Engine/DataIO/StratifiedNpySelection.py` cria indices a partir do arquivo `*_y.npy`; `CrossValidation.py` aplica esses indices ao `SplitData`.
- `number_samples_per_class`: campanha AppClassNet em `run_appclassnet_top200.py`; contrato interno em `Engine/DataIO/SamplePlanner.py`; metadados finais em `main.py::_build_generation_metadata`.
- Propagacao de quotas: `run_appclassnet_top200.py::build_main_command` e `build_batch_main_command`.
- Construtor do subprocesso: `run_appclassnet_top200.py`.

## Shapes E Origens

No modo `npy_xy` AppClassNet:

- `train_x` / `train_y`: carregados por `NpyXYLoader` a partir de `train_x.npy` e `train_y.npy`.
- `test_x` / `test_y`: carregados por `NpyXYLoader` a partir de `test_x.npy` e `test_y.npy`.
- `valid_x` / `valid_y`: carregados por `NpyXYLoader` a partir de `valid_x.npy` e `valid_y.npy`.
- Cada `SplitData` valida `X.shape[0] == y.shape[0]`.
- Em batches, `_apply_stratified_split_selection` cria indices a partir do `y_path` do mesmo split e aplica exatamente os mesmos indices a `split.X` e `split.y`.
- O fold `provided` usa `bundle.train` como treino e `bundle.valid` ou `bundle.test` como avaliacao.
- O sanity check agora recebe um `AlignedDataset` do split de avaliacao: `X=self._current_evaluation_source_x` e `y=y_evaluation_real`.

No caso reportado, `real_x` tinha 81.832 linhas, mas os indices vinham de um `real_y` com pelo menos 98.071 posicoes. A causa era a ausencia de um contrato central no ponto de entrada do sanity check: `_stratified_real_subset` criava indices validos para o `real_y` recebido, mas esses indices eram aplicados em outro `real_x`. Agora isso falha antes com `ValueError` informando split/fold/origem.

## Contrato X/Y

Foi adicionado `validate_xy_alignment(x, y, dataset_name)` em `Engine/DataIO/DatasetContracts.py`.

Contrato:

- converte `y` com `reshape(-1)`;
- exige `x.ndim == 2`;
- exige `x.shape[0] == y.shape[0]`;
- exige labels finitas;
- inclui o nome do dataset/split/fold no erro;
- nao trunca arrays;
- nao filtra indices invalidos;
- nao aplica modulo ou clipping.

Tambem foi adicionado `AlignedDataset` para passar `X/y` como uma unidade nos sanity checks.

## Pontos Protegidos

Validacao X/y foi conectada em:

- sanity check antes de `_stratified_real_subset`;
- `_stratified_real_subset` antes de criar indices;
- selecao estratificada de batches antes e depois de aplicar indices;
- folds `provided`;
- `EvaluationRunner.validate` para TR-TR, TR-TS e TS-TR materializados;
- `iter_array_batches` e validacao real em batch classifiers;
- subset real em TS-TR;
- TR-TS antes do treinamento do classificador.

## Correcao Da Origem

Antes, o sanity check podia receber `real_x` trocado para `self._current_evaluation_source_x` e `real_y` ainda vindo como parametro independente. Agora `main.py` cria:

```text
AlignedDataset(
  X=current evaluation source X,
  y=evaluation y,
  split_name="evaluation",
  fold_id=fold+1,
  data_space="source"
)
```

Se o `X` ja corresponde ao fold atual, o `y` do mesmo fold e usado. Se houver selecao por indices, `CrossValidation.py` aplica os mesmos indices a `split.X` e `split.y` antes do fold existir.

## Plano De Geracao Sintetica

O padrao adotado para AppClassNet top-200 em batches e independente:

- sintetic train: usado em TS-TR;
- sintetic test: usado em TR-TS;
- os dois sao gerados como leitores separados quando `execution_mode=batches` e `materialize_synthetic=False`.

O runner agora calcula:

```text
required_generated_per_class =
  synthetic_train_samples_per_class + synthetic_test_samples_per_class
```

Exemplos cobertos por teste:

- 50/50 -> `number_samples_per_class` com 100 por classe;
- 200/200 -> 400 por classe;
- 500/500 -> 1000 por classe.

`main.py::_validate_synthetic_generation_plan` falha antes do treinamento quando o plano interno e menor:

```text
InsufficientSyntheticGenerationPlan: class=... required=... planned=...
```

## Precedencia De Argumentos

A precedencia implementada no runner e:

1. valor explicito da CLI;
2. valor da campanha;
3. default global do parser.

`run_appclassnet_top200.py` marca argumentos explicitos via `annotate_explicit_cli_arguments()` e registra:

```text
requested value
campaign value
effective value
source
```

Com isso, `--train_samples_per_class 1000` permanece `1000` no subprocesso mesmo quando `--synthetic_train_samples_per_class 500` tambem existe.

Tambem foi removida a sobrescrita em `Engine/Arguments/Arguments.py` que fazia:

```text
train_samples_per_class = synthetic_train_samples_per_class
test_samples_per_class = synthetic_test_samples_per_class
```

Agora quotas reais e sinteticas ficam separadas.

## Parametros Duplicados

O runner agora remove o parametro de batch size vindo da campanha antes de aplicar o override de batches, por exemplo:

```text
--variational_autoencoder_batch_size 8192
```

em vez de:

```text
--variational_autoencoder_batch_size 128
--variational_autoencoder_batch_size 8192
```

`deduplicate_command_options()` detecta duplicatas conflitantes antes da execucao e levanta:

```text
DuplicateCommandArgumentConflict
```

Duplicatas identicas sao colapsadas para uma ocorrencia efetiva.

## Fail Fast

Antes de treinar o classificador do sanity check, o log registra:

- `real_x.shape`;
- `real_y.shape`;
- `synthetic_x.shape`;
- `synthetic_y.shape`;
- fold;
- split;
- `max_selected_index` na selecao estratificada real.

Se X/y estiverem desalinhados, a execucao para com `ValueError` antes de qualquer `real_x[indices]`.

## Compatibilidade

O modo CSV legado continua aceitando `number_samples_per_class` explicito e o contrato antigo de `SamplePlanner`. As mudancas novas sao guardrails e propagacao correta de argumentos; nao removem o formato legado.

## Evidencia De Testes

Comando executado:

```bash
pytest -q tests/test_synthetic_sanity_checks.py tests/test_appclassnet_evaluation_mode.py tests/test_cross_validation_data_loading.py tests/test_sample_planner.py
```

Resultado:

```text
44 passed, 5 subtests passed
```

Tambem foi executado:

```bash
python -m py_compile Engine/DataIO/DatasetContracts.py Engine/DataIO/SyntheticSanityChecks.py Engine/Classifiers/BatchClassifiers.py Engine/Evaluation/TrTs.py Engine/Evaluation/TsTr.py Engine/Evaluation/EvaluationRunner.py Engine/Evaluation/CrossValidation.py main.py run_appclassnet_top200.py
```

Resultado: sem erros.
