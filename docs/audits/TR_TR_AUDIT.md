# TR-TR Audit - AppClassNet Top-200

Data da auditoria: 2026-07-14

Escopo: auditoria exclusiva do TR-TR. Nenhum codigo de execucao foi alterado.

## Conclusao executiva

Existem dois caminhos real-real semanticamente diferentes:

1. `run_appclassnet_top200.py --baseline_real_only`: este e o caminho que corresponde ao baseline validado AppClassNet top-200. Ele usa `train_x/train_y` oficiais para treino, `test_x/test_y` oficiais para teste, amostragem estratificada por classe, `DecisionTreeClassifier` por padrao e preserva a escala original por padrao.
2. `Engine/Evaluation/TrTr.py::TrTr.evaluation_TR_TR()`: este e o TR-TR do pipeline principal. A implementacao treina em `dictionary_data["x_training_real"]` e testa em `dictionary_data["x_evaluation_real"]`, mas a chamada esta comentada em `main.py::SynDataGen.run_experiments()`. Portanto o pipeline sintetico principal nao produz TR-TR atualmente.

O resultado validado esperado, Accuracy proxima de `0.68673` e Macro-F1 proximo de `0.68630`, deve ser associado ao caminho `baseline_real_only`, nao ao TR-TR fold-based do pipeline principal.

## Achados classificados

| Severidade | Achado | Evidencia |
|---|---|---|
| Critico | O TR-TR do pipeline principal existe, mas nao e executado no loop atual. | `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linha ~502: `#self.evaluation_TR_TR(dictionary_data)`. |
| Alto | `baseline_real_only` e o unico caminho auditado que usa explicitamente `train` oficial para treino e `test` oficial para teste. | `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1023-1051. |
| Alto | Em `split_mode=provided`, o pipeline principal usa `valid` como avaliacao quando `valid` e `test` existem; isso nao equivale ao baseline train->test. | `Engine/Evaluation/CrossValidation.py`, funcoes `_build_provided_split_folds()` e `_create_fold()`, linhas ~357-380 e ~329-346. |
| Medio | O baseline aplica `FeatureTransformManager` tambem em TR-TR, mas o default AppClassNet e `classifier_transform=preserve`; transformacoes opcionais sao ajustadas somente no treino. | `run_appclassnet_top200.py`, classe n/a, funcao `run_real_real_baseline()`, linhas ~1053-1069; `Engine/Preprocessing/FeatureTransformManager.py`, classe `FeatureTransformManager`, funcoes `fit()` e `transform()`, linhas ~336-353 e ~404-443. |
| Medio | A funcao TR-TR calcula distancia R-R apos truncar o maior lado para o tamanho do menor; isso nao afeta as metricas preditivas, mas afeta distancia. | `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~109-127. |
| Baixo | A docstring de TR-TR esta semanticamente errada: descreve treino sintetico/teste real, mas o codigo treina e testa em dados reais. | `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~49-60 e ~83-102. |

## Caminho validado: `baseline_real_only`

- Arquivo: `run_appclassnet_top200.py`
- Classe: n/a, funcao de modulo
- Funcao: `run_real_real_baseline()`
- Linhas aproximadas: ~1009-1139
- Classificador default:
  - Arquivo: `run_appclassnet_top200.py`
  - Classe: n/a, funcao de modulo
  - Funcao: `build_baseline_classifier()`
  - Linhas aproximadas: ~770-800
- Argumentos default relevantes:
  - `APPCLASSNET_NUM_CLASSES = 200`: linha ~45
  - `DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS = 1000`: linha ~40
  - `DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS = 500`: linha ~41
  - `--baseline_classifier`, default `decision_tree`: classe n/a, funcao `build_parser()`, linhas ~2122-2125

Pseudocodigo fiel:

```python
train_x_path, train_y_path = validate_raw_split(raw_root, "train")
test_x_path, test_y_path = validate_raw_split(raw_root, "test")

train_x_values = numpy.load(train_x_path, mmap_mode=mmap_mode, allow_pickle=False)
train_y_values = numpy.load(train_y_path, mmap_mode=mmap_mode, allow_pickle=False)
test_x_values = numpy.load(test_x_path, mmap_mode=mmap_mode, allow_pickle=False)
test_y_values = numpy.load(test_y_path, mmap_mode=mmap_mode, allow_pickle=False)

train_indices = select_stratified_indices(train_y_values, 1000, 200, seed=0, split_name="train")
test_indices = select_stratified_indices(test_y_values, 500, 200, seed=1, split_name="test")

train_x, train_y = load_selected_rows(train_x_values, train_y_values, train_indices, seed=2)
test_x, test_y = load_selected_rows(test_x_values, test_y_values, test_indices, seed=3)

classifier_manager.fit(train_x, split_name="train")
train_x = classifier_manager.transform(train_x, split_name="train", input_space="source", output_space="classifier")
test_x = classifier_manager.transform(test_x, split_name="test", input_space="source", output_space="classifier")

classifier = build_baseline_classifier(args)  # default DecisionTreeClassifier(random_state=0)
classifier.fit(train_x, train_y)
predictions = classifier.predict(test_x)

metrics = {
    "Accuracy": accuracy_score(test_y, predictions),
    "MacroF1": f1_score(test_y, predictions, average="macro", zero_division=0),
    "WeightedF1": f1_score(test_y, predictions, average="weighted", zero_division=0),
    "BalancedAccuracy": balanced_accuracy_score(test_y, predictions),
}
```

## Respostas aos pontos investigados

### 1. Se `train_x/train_y` sao realmente usados para treino

Sim no `baseline_real_only`.

- Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1023-1029 carregam `train_x_values` e `train_y_values` do split `train`.
- Evidencia: linhas ~1031-1039 selecionam indices do `train_y_values`.
- Evidencia: linhas ~1050 e ~1073 treinam `classifier.fit(train_x, train_y)`.

No TR-TR do pipeline principal, o treino vem de `dictionary_data["x_training_real"]` e `dictionary_data["y_training_real"]`.

- Evidencia: `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~83-89.

### 2. Se `test_x/test_y` sao realmente usados para teste

Sim no `baseline_real_only`.

- Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1024-1029 carregam o split `test`.
- Evidencia: linhas ~1040-1048 selecionam indices de `test_y_values`.
- Evidencia: linhas ~1051 e ~1075 usam `test_x` para predicao.
- Evidencia: linhas ~1117-1122 calculam metricas contra `test_y`.

No TR-TR do pipeline principal, o teste vem de `dictionary_data["x_evaluation_real"]` e `dictionary_data["y_evaluation_real"]`.

- Evidencia: `Engine/Evaluation/TrTr.py`, linhas ~94-102.

### 3. Se validacao e confundida com teste

No `baseline_real_only`, nao ha evidencia de confusao: ele usa `train` e `test`, nao usa `valid`.

- Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1023-1029.

No pipeline principal com `data_format=npy_xy --split_mode provided`, ha diferenca semantica: se `valid` e `test` existem, a avaliacao usa `valid`, e o `test` e marcado como nao suportado para esse fluxo.

- Evidencia: `Engine/Evaluation/CrossValidation.py`, funcao `_build_provided_split_folds()`, linhas ~357-365 escolhem `bundle.valid or bundle.test` e marcam o test como nao suportado quando ambos existem.
- Evidencia: linhas ~374-380 criam uma unica entrada `1-Fold`.

Classificacao: alto para comparacoes com o baseline, porque `valid` nao e o mesmo conjunto que `test`.

### 4. Se indices de treino aparecem no teste

No `baseline_real_only`, os indices sao selecionados de arrays diferentes: `train_y_values` e `test_y_values`. Nao ha intersecao de indices no mesmo array, pois os splits sao arquivos separados.

- Evidencia: `run_appclassnet_top200.py`, linhas ~1023-1029 carregam arquivos distintos.
- Evidencia: linhas ~1031-1048 chamam `select_stratified_indices()` separadamente para train e test, com seeds diferentes.

Risco residual: o codigo nao valida conteudo duplicado entre arquivos oficiais train/test. Ele presume que os splits externos sao independentes.

No TR-TR do pipeline principal com cross-validation, a independencia dos indices vem do splitter: `train_index` e `val_index` sao produzidos por `StratifiedKFold.split(...)` ou `KFold.split(...)`, e depois usados separadamente para preencher `x_training_real` e `x_evaluation_real`.

- Evidencia: `Engine/Evaluation/CrossValidation.py`, classe/decorator `StratifiedData`, funcao interna `wrapper()`, linhas ~494-508 criam o splitter e iteram `train_index, val_index`.
- Evidencia: `Engine/Evaluation/CrossValidation.py`, classe/decorator `StratifiedData`, funcao interna `wrapper()`, linhas ~540-564 aplicam `train_index` ao treino e `val_index` a avaliacao.

### 5. Se o scaler e ajustado somente no treino

No `baseline_real_only`, sim quando uma transformacao e solicitada; por padrao nao ha escala aplicada.

- Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linha ~1067 chama `classifier_manager.fit(train_x, split_name="train")`.
- Evidencia: linhas ~1068-1069 transformam train e test usando o mesmo manager.
- Evidencia: `Engine/Preprocessing/FeatureTransformManager.py`, classe `FeatureTransformManager`, funcao `transform()`, linhas ~425-430 rejeitam transformar valid/test/synthetic se o scaler nao foi ajustado em `train`.
- Evidencia: `FeatureTransformPolicy.for_profile()`, linhas ~155-168 definem AppClassNet com `classifier_transform="preserve"` e `evaluation_space="source"` por padrao.
- Evidencia: `run_appclassnet_top200.py`, classe n/a, funcao `build_parser()`, linhas ~2094-2103 definem defaults CLI `classifier_transform="preserve"` e `evaluation_space="source"`.

### 6. Se o classificador recebe os dados na escala esperada

No baseline validado, sim se `classifier_transform=preserve`, que e o default AppClassNet.

- Evidencia: `FeatureTransformPolicy.for_profile()`, linhas ~155-168, AppClassNet default `classifier_transform="preserve"`.
- Evidencia: `FeatureTransformManager.fit()`, linhas ~341-345, operacao `preserve` nao cria scaler e preserva a faixa.
- Evidencia: `FeatureTransformManager.transform()`, linhas ~416-423, operacao `preserve` retorna `values` em `input_space`.
- Evidencia: `run_appclassnet_top200.py`, classe n/a, funcao `summarize_feature_matrix()`, linhas ~463-491 registra `global_min`, `global_max` e `scale_guess` no JSON de baseline.

Risco: `run_real_real_baseline()` sempre registra `"data_space": "source"` nas metricas, linha ~1088. Se o usuario passar `--classifier_transform minmax` ou `standard`, o classificador recebe dados transformados, mas o campo `data_space` do JSON continua `source`.

### 7. Se todas as 200 classes estao presentes

No baseline, a selecao estratificada verifica todas as classes de `0` a `199` e falha se alguma classe estiver ausente.

- Evidencia: `run_appclassnet_top200.py`, constante `APPCLASSNET_NUM_CLASSES = 200`, linha ~45.
- Evidencia: `select_stratified_indices()`, linhas ~740-754, itera por `range(num_classes)` e levanta `ValueError` se houver classes ausentes.
- Evidencia: `run_real_real_baseline()`, linhas ~1031-1048 passa `APPCLASSNET_NUM_CLASSES` para train e test.

### 8. Se a amostragem e exatamente estratificada

Sim, desde que cada classe tenha pelo menos a cota solicitada. A funcao seleciona ate `samples_per_class` por classe.

- Evidencia: `select_stratified_indices()`, linhas ~740-750, usa `numpy.flatnonzero(labels_array == class_id)` e `random_generator.choice(..., size=sample_count, replace=False)`.
- Evidencia: linhas ~746-748 registram `short_classes` se uma classe tiver menos amostras que a cota.
- Evidencia: linhas ~1084-1094 salvam cotas solicitadas, contagens por classe e classes curtas no JSON.

Para o baseline validado, a especificacao esperada e exatamente `1000` por classe no treino e `500` por classe no teste. O teste futuro deve falhar se `train_short_classes` ou `test_short_classes` nao estiverem vazios.

### 9. Se a ordem X/y e preservada apos selecao e shuffle

Sim. A selecao aplica os mesmos indices ordenados a X e y, e depois aplica a mesma permutacao aos dois arrays.

- Evidencia: `run_appclassnet_top200.py`, funcao `load_selected_rows()`, linhas ~761-767.
- Detalhe: linhas ~762-765 ordenam os indices e selecionam `x_values[sorted_indices]` e `y_values[sorted_indices]`.
- Detalhe: linhas ~766-767 retornam `x_selected[permutation]` e `y_selected[permutation]` usando a mesma permutacao.

### 10. Se labels continuam entre 0 e 199

No baseline, nao ha chamada explicita a `validate_zero_based_labels()`, mas a selecao estratificada exige a presenca de classes `0..199`; se labels estiverem fora desse conjunto, elas nao entram na selecao por classe. O teste futuro deve validar explicitamente `min(label)==0`, `max(label)==199` e `unique_count==200` em `train_y` e `test_y`.

- Evidencia: `run_appclassnet_top200.py`, funcao `select_stratified_indices()`, linhas ~740-756.
- Evidencia de validador disponivel no projeto: `Engine/DataIO/LabelUtils.py`, funcao `validate_zero_based_labels()`, linhas ~74-95, rejeita labels negativos e labels `>= num_classes` quando `num_classes` e fornecido.

No pipeline principal NPY, os labels sao validados como zero-based.

- Evidencia: `Engine/Evaluation/CrossValidation.py`, funcao `_apply_bundle_to_owner()`, linhas ~289-294.
- Evidencia: funcao `_create_fold()`, linhas ~332-336.

### 11. Se o classificador e recriado a cada execucao

No baseline, sim: `build_baseline_classifier()` retorna uma nova instancia.

- Evidencia: `run_appclassnet_top200.py`, funcao `build_baseline_classifier()`, linhas ~770-800.
- Evidencia: `DecisionTreeClassifier(random_state=0)` e criado na linha ~786.
- Evidencia: `run_real_real_baseline()`, linha ~1071 chama `build_baseline_classifier()` dentro da execucao.

No TR-TR do pipeline principal, `get_trained_classifiers()` chama `classifier_model.get_model(...)` a cada avaliacao/fold.

- Evidencia: `Engine/Classifiers/Classifiers.py`, classe `Classifiers`, funcao `get_trained_classifiers()`, linhas ~132-151.

### 12. Se as metricas sao calculadas sobre todas as predicoes

No baseline, sim: `predictions = classifier.predict(test_x)` e as metricas usam `test_y` inteiro contra `predictions`.

- Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1074-1075 e ~1117-1122.

No TR-TR do pipeline principal, sim para metricas preditivas: a predicao usa todo `x_evaluation_real` e as metricas recebem todo `y_evaluation_real`.

- Evidencia: `Engine/Evaluation/TrTr.py`, funcao `evaluation_TR_TR()`, linhas ~94-102.

Observacao: a distancia R-R trunca o maior lado antes de calcular distancia.

- Evidencia: `Engine/Evaluation/TrTr.py`, linhas ~109-127.

### 13. Se BalancedAccuracy, Macro-F1 e Weighted-F1 estao corretos

No baseline, as metricas usam diretamente scikit-learn:

- `accuracy_score(test_y, predictions)`;
- `f1_score(test_y, predictions, average="macro", zero_division=0)`;
- `f1_score(test_y, predictions, average="weighted", zero_division=0)`;
- `balanced_accuracy_score(test_y, predictions)`.

Evidencia: `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1117-1122.

No pipeline principal multiclass, as metricas tambem usam scikit-learn com averages apropriados:

- Evidencia: `Engine/Metrics/Metrics.py`, classe `Metrics`, funcao `_get_multiclass_metric_values()`, linhas ~355-368.
- Evidencia: `get_task_metrics()`, linhas ~634-671, usa `_get_multiclass_metric_values()` quando a tarefa nao e binaria.

### 14. Se existe diferenca semantica entre `baseline_real_only` e TR-TR do pipeline principal

Sim, ha diferenca semantica importante.

`baseline_real_only`:

- usa split oficial `train` para treino;
- usa split oficial `test` para teste;
- seleciona `1000` amostras por classe no treino e `500` por classe no teste por default;
- usa `DecisionTreeClassifier` por default;
- executa isoladamente e salva `baseline_real_only/metrics.json`;
- ignora `evaluation_mode`, porque executa somente TR-TR.

Evidencia:

- `run_appclassnet_top200.py`, funcao `main()`, linhas ~2250-2274.
- `run_real_real_baseline()`, linhas ~1009-1139.

TR-TR do pipeline principal:

- treina em `dictionary_data["x_training_real"]`;
- testa em `dictionary_data["x_evaluation_real"]`;
- em cross-validation, esses conjuntos sao folds internos;
- em `split_mode=provided`, `x_evaluation_real` e `valid` se existir, senao `test`;
- usa a lista normal de classificadores de `arguments.classifier`;
- nao esta sendo executado no loop principal atual.

Evidencia:

- `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~83-102.
- `Engine/Evaluation/CrossValidation.py`, funcoes `_create_fold()` e `_build_provided_split_folds()`, linhas ~329-346 e ~357-380.
- `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linha ~502.

## Especificacao futura de teste

Nome proposto:

```python
test_appclassnet_tr_tr_golden_baseline
```

Objetivo: garantir que o baseline real-real AppClassNet top-200 continua reproduzindo o comportamento validado e falha quando o pipeline degrada para nivel aleatorio, perde classes, altera escala silenciosamente, mistura treino/teste ou troca o classificador solicitado.

### Arranjo do teste

Entradas:

- `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy` do AppClassNet top-200;
- `num_classes=200`;
- `train_samples_per_class=1000`;
- `test_samples_per_class=500`;
- `baseline_classifier=decision_tree`;
- `classifier_transform=preserve`;
- `source_profile=appclassnet_top200`;
- `use_mmap` opcional, sem alterar resultados esperados.

Execucao esperada:

```python
metrics_path, metrics = run_real_real_baseline(args, raw_root, tmp_output_dir)
```

Assercoes obrigatorias:

```python
assert metrics["mode"] == "baseline_real_only"
assert metrics["classifier"] == "decision_tree"
assert metrics["classifier_params"]["random_state"] == 0
assert metrics["num_classes"] == 200

assert metrics["train_shape"] == [200000, 20]
assert metrics["test_shape"] == [100000, 20]

assert set(metrics["train_class_counts"].keys()) == {str(i) for i in range(200)}
assert set(metrics["test_class_counts"].keys()) == {str(i) for i in range(200)}
assert all(count == 1000 for count in metrics["train_class_counts"].values())
assert all(count == 500 for count in metrics["test_class_counts"].values())
assert metrics["train_short_classes"] == {}
assert metrics["test_short_classes"] == {}

assert metrics["scaler"]["classifier_transform"] == "preserve"
assert metrics["scaler"]["transform_id"] is None
assert metrics["data_space"] == "source"
assert approximately_appclassnet_source_range(metrics["feature_range"]["train"])
assert approximately_appclassnet_source_range(metrics["feature_range"]["test"])

assert metrics["paths"]["train_x"] != metrics["paths"]["test_x"]
assert metrics["paths"]["train_y"] != metrics["paths"]["test_y"]

assert metrics["metrics"]["Accuracy"] > 0.60
assert abs(metrics["metrics"]["Accuracy"] - 0.68673) <= 0.03
assert abs(metrics["metrics"]["MacroF1"] - 0.68630) <= 0.03
assert metrics["metrics"]["Accuracy"] > 0.05
assert metrics["metrics"]["BalancedAccuracy"] > 0.60
assert metrics["metrics"]["WeightedF1"] > 0.60
```

### Condicoes que devem fazer o teste falhar

1. Accuracy cair para proximo de `0.005`.
   - Falha por `metrics["metrics"]["Accuracy"] > 0.05` e pela janela em torno de `0.68673`.
2. Classes desaparecerem.
   - Falha por contagem exata de 200 classes e por cotas `1000/500`.
3. Escalas serem alteradas silenciosamente.
   - Falha por `classifier_transform == "preserve"`, `transform_id is None`, `data_space == "source"` e validacao de faixa aproximada AppClassNet.
4. Treino e teste serem iguais.
   - Falha por caminhos distintos e deve ser reforcada no teste com comparacao de identidade/shape/conteudo quando possivel.
5. Classificador utilizado nao ser o solicitado.
   - Falha por `metrics["classifier"] == "decision_tree"` e parametros do classificador.
   - Reforco recomendado: monkeypatchar `run_appclassnet_top200.build_baseline_classifier()` no teste para capturar `classifier.__class__.__name__` e falhar se nao for `DecisionTreeClassifier`.

### Validacoes adicionais recomendadas dentro do teste

Estas validacoes exigem expor ou reproduzir a selecao dentro do teste, sem alterar o pipeline:

```python
train_y_selected = ...
test_y_selected = ...

assert train_y_selected.min() == 0
assert train_y_selected.max() == 199
assert test_y_selected.min() == 0
assert test_y_selected.max() == 199
assert len(numpy.unique(train_y_selected)) == 200
assert len(numpy.unique(test_y_selected)) == 200

assert not numpy.array_equal(train_x_selected[:1000], test_x_selected[:1000])
assert len(predictions) == len(test_y_selected)
```

Essas validacoes cobrem explicitamente labels `0..199`, independencia basica entre treino/teste e calculo de metricas sobre todas as predicoes.

## Decisao recomendada

Para regressao AppClassNet top-200, tratar `baseline_real_only` como o teste golden TR-TR validado. O TR-TR do pipeline principal deve ser considerado outro fluxo: fold-based ou provided-split-based, atualmente inativo no loop sintetico, e nao deve ser usado para afirmar equivalencia com o resultado train/test oficial ate que seja explicitamente ligado e testado.
