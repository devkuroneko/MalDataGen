# Provided Split Evaluation Routing Fix

## Escopo

Correção do roteamento das avaliações AppClassNet quando `split_mode=provided`.

Contrato final:

- TR-TR: treino real `train_x/train_y`; teste real `test_x/test_y`.
- TR-TS: treino real `train_x/train_y`; teste `synthetic_test`.
- TS-TR: treino `synthetic_train`; teste real `test_x/test_y`.
- VALID fica reservado para validação interna, monitoramento, seleção de hiperparâmetros, early stopping e checkpoint.

## Rastreamento

### Ponto onde VALID entrava no TS-TR

Antes da correção, `Engine/Evaluation/CrossValidation.py::_build_provided_split_folds()` escolhia:

```python
evaluation_split = bundle.valid or bundle.test
```

Quando `valid` e `test` existiam, o fold `provided` recebia `valid` como `x_evaluation_real/y_evaluation_real`. Depois, `main.py::run_synthetic_evaluation_modes()` passava o mesmo `dictionary_data` genérico para `evaluation_TS_TR()`. Em `Engine/Evaluation/TsTr.py`, TS-TR lia `dictionary_data['x_evaluation_real']` e `dictionary_data['y_evaluation_real']`.

Esse caminho explica a evidência observada: TS-TR reportava as contagens do VALID, incluindo classes 121=495, 150=294, 198=151 e 199=152, em vez do TEST com mínimo por classe 1539.

### Fluxo corrigido

- `SplitData` agora carrega identidade do split: `name`, `x_path`, `y_path`, `dataset_id`, `num_samples`, `class_counts` e `minimum_class_count`.
- `NpyXYLoader` preenche `x_path`, `y_path` e `dataset_id` ao criar `SplitData`.
- `_build_provided_split_folds()` mantém `valid` como split de validação interna do fold, mas também guarda `real_train_split`, `real_valid_split`, `real_test_split` e `dataset_bundle`.
- `run_synthetic_evaluation_modes()` usa `dataset_bundle.train` e `dataset_bundle.test` diretamente quando `split_mode=provided`.
- `EvaluationRunner` registra nos resultados: split names, paths de X/y, dataset_id, shapes, class counts e mínimos por classe.

## Roteamento final

Em `main.py::run_synthetic_evaluation_modes()`:

- TR-TS chama `evaluation_TR_TS(real_train_data=dataset_bundle.train, synthetic_test_data=synthetic_for_tr_ts)`.
- TS-TR chama `evaluation_TS_TR(synthetic_train_data=synthetic_for_ts_tr, real_test_data=dataset_bundle.test)`.

No loop principal, TR-TR com `run_tr_tr=True` chama:

```python
self.evaluation_TR_TR(
    real_train_data=dataset_bundle.train,
    real_test_data=dataset_bundle.test,
)
```

## Guard de identidade

Foi adicionada `EvaluationSplitMismatchError`.

Guards:

- TS-TR exige `real_test_data.name == "test"`.
- TR-TS exige `real_train_data.name == "train"`.
- TR-TR exige `real_train_data.name == "train"` e `real_test_data.name == "test"`.

Erro esperado quando VALID é passado ao TS-TR:

```text
TS-TR requires real split 'test', but received 'valid'.
```

## Política de contagem

Em batches, `--test_samples_per_class` agora é aplicado ao split `test`, não ao `valid`.

Com os dados informados:

- `real_test_data = TEST`
- `requested_test_samples_per_class = 500`
- `real_test_minimum_class_count = 1539`
- `effective_test_samples_per_class = 500`

Portanto `strict` deve aceitar a avaliação. O mínimo do VALID, 151, não interfere no TS-TR.

## Folds

Quando `split_mode=provided`:

- `number_k_folds` efetivo é 1.
- O fold não divide `test`.
- `valid` pode ser usado pelo gerador como validação interna.
- `test` permanece externo e é usado uma única vez na avaliação final.

Em `split_mode=cross_validation` com `npy_xy`, o comportamento anterior continua: os folds são criados somente dentro do `train`, e `valid/test` fornecidos não são concatenados.

## Logs

Foram adicionados logs antes das avaliações:

- TR-TR: `train_split=train`, `test_split=test`, paths de train/test.
- TR-TS: `train_split=train`, `test_split=synthetic_test`, path e mínimo do train.
- TS-TR: `train_split=synthetic_train`, `test_split=test`, `real_test_path`, `real_test_y_path`, `real_test_minimum_class_count`, `requested_test_samples_per_class` e `effective_test_samples_per_class`.

## Testes

Testes adicionados/atualizados:

- `provided` mantém `valid` para avaliação interna, mas não como teste final.
- `provided` com `valid` e `test` guarda `test` como `real_test_split`.
- TR-TS usa `train` e `synthetic_test`.
- TS-TR usa `synthetic_train` e `test`.
- TS-TR não usa `valid` por default.
- Passar `valid` ao TS-TR gera `EvaluationSplitMismatchError`.
- `valid` com baixa cobertura não bloqueia TS-TR strict quando `test` atende a cota.
- Paths e split names são salvos em `EvaluationMetadata`.
- CSV/cross-validation legado continua coberto pelos testes existentes.

Comandos executados:

```bash
python -m py_compile Engine/DataIO/DatasetContracts.py Engine/DataIO/NpyXYLoader.py Engine/Evaluation/CrossValidation.py Engine/Evaluation/EvaluationRunner.py Engine/Evaluation/TrTs.py Engine/Evaluation/TsTr.py Engine/Evaluation/TrTr.py main.py
pytest -q tests/test_dataset_contracts.py tests/test_cross_validation_data_loading.py tests/test_appclassnet_evaluation_mode.py tests/test_evaluation_runner.py
```

Resultado:

```text
67 passed, 5 subtests passed
```

Validação completa:

```bash
pytest -q
```

Resultado:

```text
157 passed, 18 skipped, 2 warnings, 5 subtests passed
```
