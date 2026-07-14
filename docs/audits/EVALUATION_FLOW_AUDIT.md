# Evaluation Flow Audit - TR-TR, TR-TS, TS-TR

Data da auditoria: 2026-07-14

Escopo: auditoria especifica das implementacoes atuais de TR-TR, TR-TS e TS-TR. Nenhum codigo de execucao foi alterado.

## Sumario

Ha tres realidades diferentes no codigo atual:

1. `main.py` em modo CSV/cross-validation: `1-Fold`, `2-Fold`, etc. sao folds reais criados por `StratifiedKFold` ou `KFold`.
2. `main.py` em `data_format=npy_xy --split_mode provided`: `1-Fold` nao e cross-validation; e o split fornecido `train -> valid` se valid existe, ou `train -> test` se so test existe.
3. `run_appclassnet_top200.py --baseline_real_only`: executa TR-TR separado, usando train/test oficiais e nao grava resultados no mesmo formato fold-based de `main.py`.

Achado mais importante: no pipeline sintetico de `main.py`, TR-TS nao treina no `x_training_real`; ele treina no `x_evaluation_real`. Isso diverge da definicao esperada "treinar classificador em dados reais" quando se assume que "real" significa treino oficial ou fold de treino.

## Classificacao dos achados principais

| Severidade | Achado | Evidencia |
|---|---|---|
| Critico | TR-TS treina o classificador em `x_evaluation_real`, nao em `x_training_real`. Isso pode invalidar a leitura esperada de TR-TS como treino real oficial/fold de treino contra teste sintetico. | `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas ~188-193; modo batches linhas ~96-111. |
| Alto | `1-Fold` em `split_mode=provided` nao e fold real de cross-validation; e apenas o split fornecido `train -> valid/test`. | `Engine/Evaluation/CrossValidation.py`, classe `StratifiedData`, funcao `_build_provided_split_folds()`, linhas ~357-380. |
| Alto | AppClassNet modo normal do runner materializa CSV e usa folds internos; portanto `1-Fold`/`2-Fold` nesse caminho nao representam os splits oficiais train/valid/test. | `run_appclassnet_top200.py`, funcoes `main()` e `build_main_command()`, linhas ~2316-2325 e ~1556-1608; `Engine/Evaluation/CrossValidation.py`, `StratifiedData.wrapper()`, linhas ~465-564. |
| Alto | TS-TR normal prediz em todo `x_evaluation_real`, mas calcula metricas sobre `y_evaluation_real[:total_samples]` e `pred[:total_samples]`; se o sintetico tiver menos linhas que o real, o teste real efetivo e truncado silenciosamente. | `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas ~207-216. |
| Medio | Distancia R-S em TS-TR normal compara sintetico contra `x_training_real`, enquanto a avaliacao preditiva TS-TR testa contra `x_evaluation_real`. | `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas ~226-249. |
| Medio | O mesmo objeto sintetico gerado por fold e reutilizado em TR-TS e TS-TR. Isso e coerente tecnicamente, mas precisa ser declarado na interpretacao das metricas. | `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linhas ~472-500; funcao `run_synthetic_evaluation_modes()`, linhas ~110-128. |
| Medio | TR-TR existe, mas esta comentado no loop sintetico principal; resultados TR-TR de `main.py` nao sao produzidos nesse fluxo. | `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linha ~502; `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~47-127. |
| Medio | Quando `valid` e `test` sao fornecidos em `split_mode=provided`, o pipeline usa `valid` como avaliacao e marca `test` como nao suportado para aquele fluxo. | `Engine/Evaluation/CrossValidation.py`, classe `StratifiedData`, funcao `_build_provided_split_folds()`, linhas ~357-380. |
| Medio | O loader NPY carrega X/y e declara `data_space="source"` sem transformar features; transformacoes ficam no `FeatureTransformManager`. | `Engine/DataIO/NpyXYLoader.py`, classe `NpyXYLoader`, funcoes `load()` e `_load_split()`, linhas ~87-124 e ~146-161; `Engine/Preprocessing/FeatureTransformManager.py`, classe `FeatureTransformPolicy`, funcao `for_profile()`, linhas ~155-168. |
| Baixo | Docstring de TR-TR descreve treino sintetico/teste real, mas a implementacao treina e testa em dados reais. | `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR()`, linhas ~49-90. |
| Baixo | Baseline AppClassNet registra `data_space: source`; se `classifier_transform` for diferente de `preserve`, o classificador opera em outro espaco, embora o default validado continue `preserve/source`. | `run_appclassnet_top200.py`, funcao `run_real_real_baseline()`, linhas ~1053-1088. |

## Origem dos conjuntos usados pelas avaliacoes

### CSV ou NPY com cross-validation

- Arquivo: `Engine/Evaluation/CrossValidation.py`
- Funcao: `StratifiedData.wrapper()`
- Linhas aproximadas: ~465-564

Fluxo:

```python
shuffled_data, shuffled_labels = shuffle(self._data_loaded, self._data_loaded_labels, random_state=42)
splitter = StratifiedKFold(n_splits=number_k_folds, shuffle=True, random_state=42)

for fold, (train_index, val_index) in splitter.split(shuffled_data, labels):
    dictionary_data = {
        "x_training_real": shuffle(shuffled_data[train_index]),
        "y_training_real": shuffle(shuffled_labels[train_index]),
        "x_evaluation_real": shuffle(shuffled_data[val_index]),
        "y_evaluation_real": shuffle(shuffled_labels[val_index]),
    }
```

Conclusao: `1-Fold`, `2-Fold`, etc. sao folds reais de cross-validation somente neste caminho.

### NPY com splits fornecidos

- Arquivo: `Engine/Evaluation/CrossValidation.py`
- Funcoes: `_create_fold()`, `_build_provided_split_folds()`, `StratifiedData.wrapper()`
- Linhas aproximadas: `_create_fold()` ~329-346, `_build_provided_split_folds()` ~357-380, wrapper ~430-435
- Validacao de argumentos: `Engine/Arguments/ArgumentsDataLoader.py`, funcao `validate_data_load_arguments()`, linhas ~96-104, tambem forca `number_k_folds=1` em `split_mode=provided`.

Fluxo:

```python
evaluation_split = bundle.valid or bundle.test
owner.arguments.number_k_folds = 1

dictionary_data = {
    "x_training_real": train.X,
    "y_training_real": train.y,
    "x_evaluation_real": evaluation_split.X,
    "y_evaluation_real": evaluation_split.y,
    "evaluation_split_name": "valid" or "test",
}
```

Se `valid` e `test` existem, `valid` e usado e `test` e marcado como nao suportado.

Conclusao: `1-Fold` neste caminho significa split fornecido, nao fold de cross-validation. Nao ha `2-Fold` nesse modo porque `number_k_folds` e forçado para 1.

### AppClassNet runner

- Arquivo: `run_appclassnet_top200.py`
- Funcoes: `main()`, `build_main_command()`, `build_batch_main_command()`, `run_real_real_baseline()`
- Linhas aproximadas: `main()` ~2235-2405, `build_main_command()` ~1556-1608, `build_batch_main_command()` ~1628-1752, baseline ~1009-1139

Modo `normal`:

```python
preprocess_appclassnet_splits(...)
dataset_path = materialize_appclassnet_csv(split=dataset_split, ...)
main.py --data_load_path_file_input dataset_path
```

Esse caminho usa CSV e cai em cross-validation dentro de `main.py`; nao usa valid/test oficiais na avaliacao sintetica.

Modo `batches`:

```python
main.py \
  --data_format npy_xy \
  --train_x_path train_x.npy \
  --valid_x_path valid_x.npy \
  --test_x_path test_x.npy \
  --split_mode provided \
  --execution_mode batches
```

Esse caminho usa `train` como treino e `valid` como avaliacao; `test` e passado, mas nao usado se `valid` existe.

Loader NPY:

```python
train = _load_split("train", train_x_path, train_y_path)
valid = _load_split("valid", valid_x_path, valid_y_path)
test = _load_split("test", test_x_path, test_y_path)
schema.data_space = "source"
schema.transform_history = []
schema.transform_id = None
```

O loader apenas carrega os arrays e valida labels/formato. Para `source_profile=appclassnet_top200`, a politica default em `FeatureTransformPolicy.for_profile()` e `feature_transform="preserve"`, `classifier_transform="preserve"`, `evaluation_space="source"` e `inverse_transform_synthetic=True`.

Baseline real-real:

```python
train = stratified_sample(raw train, train_samples_per_class)
test = stratified_sample(raw test, test_samples_per_class)
classifier.fit(train)
classifier.predict(test)
```

Esse e o caminho que corresponde ao resultado validado DecisionTree ~0.68673 / Macro-F1 ~0.68630.

## TR-TR atual

### Implementacao em `main.py`

- Arquivo: `Engine/Evaluation/TrTr.py`
- Classe: `TrTr`
- Funcao: `evaluation_TR_TR()`
- Linhas aproximadas: ~47-127
- Chamada no pipeline principal:
  - Arquivo: `main.py`
  - Classe: `SynDataGen`
  - Funcao: `run_experiments()`
  - Linha aproximada: ~502
  - Estado: chamada comentada (`#self.evaluation_TR_TR(dictionary_data)`)

Pseudocodigo fiel:

```python
def evaluation_TR_TR(dictionary_data):
    classifiers = get_trained_classifiers(
        dictionary_data["x_training_real"],
        labels_to_1d_integer(dictionary_data["y_training_real"]),
        numpy.float32,
        get_number_columns(),
    )

    for classifier_name, classifier in classifiers:
        y_pred = classifier.predict(dictionary_data["x_evaluation_real"])
        get_task_metrics(
            dictionary_data["y_evaluation_real"],
            y_pred,
            "TR-TR",
            classifier_name,
            fold_number + 1,
        )

    data_real_a = numpy.array(dictionary_data["x_training_real"])
    data_real_b = numpy.array(dictionary_data["x_evaluation_real"])
    truncate_larger_side_to_match_row_count()
    get_distance_metrics(data_real_a, data_real_b, "R-R", fold_number + 1)
```

Respostas:

1. Treino: `dictionary_data["x_training_real"]`, `dictionary_data["y_training_real"]`.
2. Teste: `dictionary_data["x_evaluation_real"]`, `dictionary_data["y_evaluation_real"]`.
3. Dados: reais nos dois lados.
4. Origem:
   - Cross-validation: folds internos.
   - Provided: train oficial para treino e valid/test oficial para teste.
   - Pipeline sintetico principal: nao executado porque a chamada esta comentada em `main.py` linha ~502.
5. Classificador: lista de `arguments.classifier` via `Classifiers.get_trained_classifiers()` (`Engine/Classifiers/Classifiers.py`, linhas ~132-151). Default geral: RandomForest, KNN, DecisionTree (`ArgumentsFramework.py`, linhas ~67 e ~148-151). Override normal opcional por `--normal_classifier`.
6. Scaler/transformacao:
   - Dentro de TR-TR: nenhum scaler.
   - Dados ja chegam como `dictionary_data`.
   - No fluxo geral, `ModelInputAdapter` nao e usado por TR-TR porque TR-TR nao e chamado no loop atual.
7. `data_space`: assumido `source` para dados reais; TR-TR nao chama `ScaleGuard`.
8. Quantidade por classe:
   - Cross-validation: determinada pelo fold.
   - Provided: todo o split carregado, salvo limitacoes de batches aplicadas antes em `_apply_batch_limits()`.
   - Baseline real-real: ver secao abaixo.
9. Balanceamento:
   - Cross-validation usa `StratifiedKFold` se labels discretos.
   - Provided nao rebalanceia por padrao; batches pode limitar por classe antes de criar o fold.
10. Vazamento treino/teste:
   - Cross-validation: indices independentes por fold.
   - Provided: train e valid/test sao arquivos separados; depende da integridade externa dos splits.
11. Concatenacao/reutilizacao indevida: distancia R-R converte e trunca arrays para mesmo tamanho, linhas ~109-127; nao concatena treino/teste.
12. Mesmo modelo reutilizado entre avaliacoes: nao aplicavel no pipeline atual porque TR-TR nao roda; quando chamado, treina novos classificadores.
13. Mesmo sintetico reutilizado: nao usa sintetico.
14. Classificador reinicializado a cada fold: sim, `get_trained_classifiers()` chama `classifier_model.get_model(...)` para cada avaliacao/fold (`Classifiers.py`, linhas ~132-151).
15. Indices independentes:
   - Cross-validation: sim.
   - Provided: sim se arquivos oficiais forem independentes.

Divergencias:

- Docstring de `TrTr.evaluation_TR_TR()` fala em treino sintetico/teste real nas linhas ~49-53, mas o codigo treina e testa em dados reais.
- Nome TR-TR esta correto em relacao ao codigo.
- No pipeline sintetico principal, TR-TR nao executa apesar de existir bloco de metricas TR-TR.

### TR-TR baseline AppClassNet validado

- Arquivo: `run_appclassnet_top200.py`
- Funcao: `run_real_real_baseline()`
- Linhas aproximadas: ~1009-1139

Pseudocodigo fiel:

```python
train_x_values = numpy.load(train_x.npy)
train_y_values = numpy.load(train_y.npy)
test_x_values = numpy.load(test_x.npy)
test_y_values = numpy.load(test_y.npy)

train_indices = select_stratified_indices(train_y, train_samples_per_class, 200, seed=0)
test_indices = select_stratified_indices(test_y, test_samples_per_class, 200, seed=1)

X_train, y_train = load_selected_rows(train_x_values, train_y_values, train_indices, seed=2)
X_test, y_test = load_selected_rows(test_x_values, test_y_values, test_indices, seed=3)

classifier_manager.fit(X_train)        # default preserve
X_train = classifier_manager.transform(X_train)
X_test = classifier_manager.transform(X_test)

classifier = build_baseline_classifier(args)  # default decision_tree
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
metrics = accuracy/f1/balanced_accuracy
```

Respostas especificas:

1. Treino: subset estratificado do split oficial `train`.
2. Teste: subset estratificado do split oficial `test`.
3. Dados: reais.
4. Origem: splits fornecidos oficiais, nao folds.
5. Classificador: `build_baseline_classifier()`, default `DecisionTreeClassifier(random_state=0)` (`run_appclassnet_top200.py`, linhas ~770-800).
6. Scaler/transformacao: `FeatureTransformManager` stage `classifier`, fit no treino, default `classifier_transform=preserve`; se usuario passar `minmax` ou `standard`, transforma treino/teste com fit somente no treino, linhas ~1053-1069.
7. `data_space`: metrics registram `"data_space": "source"` linha ~1088; se `classifier_transform` nao for preserve, ha inconsistencia nominal porque X usado pelo classificador vira output_space `classifier`, mas o payload ainda registra `source`.
8. Quantidade por classe: `train_samples_per_class` e `test_samples_per_class`; defaults AppClassNet sao 1000 e 500 (`run_appclassnet_top200.py`, linhas ~40-41).
9. Balanceamento: sim, selecao estratificada por classe (`select_stratified_indices`, usado em linhas ~1031-1048).
10. Vazamento: nao ha evidencia de vazamento no codigo; train/test vem de arquivos distintos.
11. Concatenacao/reutilizacao indevida: nao concatena train/test; carrega subsets separados.
12. Mesmo modelo reutilizado entre avaliacoes: nao; baseline executa uma avaliacao.
13. Mesmo sintetico reutilizado: nao ha sintetico.
14. Classificador reinicializado: sim, `build_baseline_classifier()` cria nova instancia.
15. Indices independentes: sim, indices selecionados separadamente com seeds diferentes.

## TR-TS atual

- Arquivo: `Engine/Evaluation/TrTs.py`
- Classe: `TrTs`
- Funcao: `evaluation_TR_TS()`
- Linhas aproximadas: modo batches ~72-136; modo normal ~138-203

### Modo normal

Pseudocodigo fiel:

```python
labels = []
data = []
for label_class, generated_samples in synthetic_data.items():
    labels.extend([label_class] * len(generated_samples))
    data.extend(generated_samples)

synthetic_array = numpy.asarray(data, dtype=numpy.float32)
ScaleGuard.validate_before_evaluation(
    dictionary_data["x_evaluation_real"],
    synthetic_array,
    real_metadata,
    synthetic_metadata,
    context="TR-TS",
)

classifiers = get_trained_classifiers(
    dictionary_data["x_evaluation_real"],
    labels_to_1d_integer(dictionary_data["y_evaluation_real"]),
    numpy.float32,
    get_number_columns(),
)

for classifier in classifiers:
    y_pred = classifier.predict(synthetic_array)
    get_task_metrics(labels, y_pred, "TR-TS", classifier_name, fold_number + 1)
```

Respostas:

1. Treino: `dictionary_data["x_evaluation_real"]`, `dictionary_data["y_evaluation_real"]`.
2. Teste: `synthetic_data` materializado em `synthetic_array`.
3. Dados: treino real, teste sintetico.
4. Origem:
   - Cross-validation: real vem do fold de validacao interno, nao do fold de treino.
   - Provided: real vem de `valid` se existir; se nao existir, de `test`.
5. Classificador: lista normal de `arguments.classifier`, treinada por `get_trained_classifiers()` (`Classifiers.py`, linhas ~132-151). Default geral: RandomForest, KNN, DecisionTree.
6. Scaler/transformacao:
   - Dentro de TR-TS: nenhum scaler.
   - Real e sintetico sao validados por `ScaleGuard`.
   - Transformacao do gerador ocorre antes, em `main.py` via `ModelInputAdapter`.
7. `data_space`: real descrito como `source`; sintetico usa `_current_synthetic_metadata` ou default `source` (`TrTs.py`, linhas ~168-186).
8. Quantidade por classe:
   - Treino real: todas as amostras de `x_evaluation_real`.
   - Teste sintetico: definido por `SamplePlan` criado a partir de `y_evaluation_real` em `main.py` `_build_generation_metadata()` linhas ~560-581.
9. Balanceamento:
   - Real: depende do fold/split.
   - Sintetico: depende de `--sample_plan` ou `number_samples_per_class`.
10. Vazamento:
   - Ha divergencia conceitual: o gerador foi treinado em `x_training_real` (`main.py`, linhas ~465-468), mas o classificador TR-TS e treinado em `x_evaluation_real`.
   - Nao ha vazamento direto train/test do classificador porque ele testa em sintetico, mas o real usado para treinar o classificador tambem foi usado como referencia para definir cotas/labels de geracao.
11. Concatenacao/reutilizacao indevida: concatena todos os sinteticos em listas e depois `numpy.asarray`, linhas ~138-168. Nao concatena real com sintetico.
12. Mesmo modelo reutilizado entre avaliacoes: nao; classificadores TR-TS sao treinados dentro da funcao.
13. Mesmo sintetico reutilizado em TR-TS e TS-TR: sim. `main.py` gera `evaluation_synthetic` uma vez por fold e passa o mesmo objeto para `run_synthetic_evaluation_modes()` (`main.py`, linhas ~472-500).
14. Classificador reinicializado a cada fold: sim, `get_trained_classifiers()` retorna novas instancias treinadas.
15. Indices treino/teste independentes:
   - O classificador usa real de avaliacao e sintetico gerado separadamente.
   - A independencia estatistica do sintetico depende do gerador treinado no fold de treino.

Divergencias:

- Nome TR-TS diz "train real, test synthetic"; isso e verdade em tipo de dado.
- Implementacao real usa o conjunto real de avaliacao (`x_evaluation_real`), nao o conjunto real de treino (`x_training_real`). Evidencia: `TrTs.py`, linhas ~188-193.

### Modo batches

Pseudocodigo fiel:

```python
validate real source metadata vs synthetic manifest metadata

classifier_key = args.eval_classifier  # default decision_tree_subset
train_labels = labels_to_1d_integer(dictionary_data["y_evaluation_real"])
train_batches = iter_array_batches(
    dictionary_data["x_evaluation_real"],
    train_labels,
    args.batch_size,
)
classifier, metadata = train_batch_classifier(classifier_key, train_batches, num_classes, args)

labels, predictions = predict_synthetic_batches(
    classifier,
    synthetic_data,
    max_samples_per_class=args.test_samples_per_class,
)
get_task_metrics(labels, predictions, "TR-TS", classifier_name, fold_number + 1)
```

Respostas especificas:

1. Treino: `x_evaluation_real` em batches.
2. Teste: batches sinteticos do manifesto.
3. Dados: treino real, teste sintetico.
4. Origem: no AppClassNet batches, `x_evaluation_real` e `valid` porque `build_batch_main_command()` passa train/valid/test e `split_mode=provided`; `_build_provided_split_folds()` escolhe `valid`.
5. Classificador: `args.eval_classifier`, default `decision_tree_subset`; opcoes do runner: `decision_tree_subset`, `extra_trees_subset`, `random_forest_light`, `sgd` (`run_appclassnet_top200.py`, linhas ~2046-2050; `BatchClassifiers.py`, linhas ~39-76).
6. Scaler/transformacao: nenhum dentro da avaliacao; manifesto sintetico e real `source` sao validados por `ScaleGuard` (`TrTs.py`, linhas ~73-83).
7. `data_space`: real `source`; sintetico vem de `synthetic_data.manifest["data_space"]`.
8. Quantidade por classe:
   - Treino real: todo `x_evaluation_real` ja possivelmente limitado por `_apply_batch_limits()`.
   - Teste sintetico: pode ser limitado por `test_samples_per_class` em `predict_synthetic_batches()` (`BatchClassifiers.py`, linhas ~254-274).
9. Balanceamento:
   - Subset classifiers coletam subset estratificado por reservatorio em `train_batch_classifier()`.
   - `sgd` usa todos os batches por `partial_fit`.
10. Vazamento: nao ha mistura direta, mas novamente treina no split de avaliacao real (`valid`) e testa em sintetico gerado por gerador treinado no train.
11. Concatenacao/reutilizacao indevida: nao materializa todos os sinteticos; prediz por batch.
12. Mesmo modelo reutilizado entre avaliacoes: nao; `train_batch_classifier()` cria novo classificador.
13. Mesmo sintetico reutilizado em TR-TS e TS-TR: sim, mesmo `SyntheticBatchReader`.
14. Classificador reinicializado a cada fold: sim, `make_batch_classifier()` cria nova instancia por chamada, linhas ~39-76.
15. Indices independentes: valid e sintetico sao fontes diferentes; se `valid` e oficial, independente do train oficial.

## TS-TR atual

- Arquivo: `Engine/Evaluation/TsTr.py`
- Classe: `TsTr`
- Funcao: `evaluation_TS_TR()`
- Linhas aproximadas: modo batches ~96-160; modo normal ~162-249

### Modo normal

Pseudocodigo fiel:

```python
labels = []
data = []
for label_class, generated_samples in synthetic_data.items():
    labels.extend([label_class] * len(generated_samples))
    data.extend(generated_samples)

synthetic_array = numpy.asarray(data, dtype=numpy.float32)
ScaleGuard.validate_before_evaluation(
    dictionary_data["x_evaluation_real"],
    synthetic_array,
    real_metadata,
    synthetic_metadata,
    context="TS-TR",
)

shuffled_data, shuffled_labels = shuffle(data, numpy.array(labels), random_state=42)
classifiers = get_trained_classifiers(shuffled_data, shuffled_labels, numpy.float32, get_number_columns())

for classifier in classifiers:
    y_pred = classifier.predict(dictionary_data["x_evaluation_real"])
    get_task_metrics(
        labels_to_1d_integer(dictionary_data["y_evaluation_real"])[:total_samples],
        numpy.array(y_pred)[:total_samples],
        "TS-TR",
        classifier_name,
        fold_number + 1,
    )

data_real = numpy.array(dictionary_data["x_training_real"])
data_synthetic = numpy.asarray(data)
truncate_larger_side_to_match_row_count()
get_distance_metrics(data_real, data_synthetic, "R-S", fold_number + 1)
```

Respostas:

1. Treino: sinteticos materializados de `synthetic_data`.
2. Teste: `dictionary_data["x_evaluation_real"]`.
3. Dados: treino sintetico, teste real.
4. Origem:
   - Cross-validation: real de teste vem do fold interno de validacao.
   - Provided: real de teste vem de `valid` se existir, senao `test`.
5. Classificador: lista normal `arguments.classifier` via `get_trained_classifiers()`.
6. Scaler/transformacao: nenhum dentro da avaliacao; `ScaleGuard` valida espaco real/sintetico.
7. `data_space`: real `source`; sintetico `_current_synthetic_metadata` ou default `source` (`TsTr.py`, linhas ~181-199).
8. Quantidade por classe:
   - Treino sintetico: sample plan gerado a partir de `y_evaluation_real`.
   - Teste real: todo `x_evaluation_real` para predicao, mas metricas sao truncadas por `total_samples`.
9. Balanceamento: sintetico depende de sample plan; real depende do fold/split.
10. Vazamento:
   - Nao ha treino do classificador em real.
   - O gerador foi treinado em `x_training_real`; teste real e `x_evaluation_real`.
   - Metricas podem truncar teste real, o que nao e vazamento mas e divergencia de avaliacao.
11. Concatenacao/reutilizacao indevida: materializa todos os sinteticos em listas; distancia R-S compara `x_training_real` com sintetico, nao `x_evaluation_real` com sintetico.
12. Mesmo modelo reutilizado entre avaliacoes: nao; classificadores TS-TR sao novos.
13. Mesmo sintético reutilizado em TR-TS e TS-TR: sim, mesmo objeto `evaluation_synthetic`.
14. Classificador reinicializado a cada fold: sim.
15. Indices independentes:
   - Cross-validation: sim entre `x_training_real` usado para gerador e `x_evaluation_real` usado para teste.
   - Provided: sim se train/valid/test oficiais forem independentes.

Divergencias:

- Nome TS-TR esta correto no tipo de dado.
- Implementacao de metricas usa `y_evaluation_real[:total_samples]` e `pred[:total_samples]`, linhas ~212-216. Se `total_samples` sintetico for menor que o tamanho do real de avaliacao, o teste real efetivo e truncado.
- Distancia R-S usa real de treino (`x_training_real`) contra sintetico, linhas ~226-249, enquanto a avaliacao preditiva testa em real de avaliacao.

### Modo batches

Pseudocodigo fiel:

```python
validate real source metadata vs synthetic manifest metadata

classifier_key = args.eval_classifier
train_batches = iter_synthetic_labeled_batches(synthetic_data)
classifier, metadata = train_batch_classifier(
    classifier_key,
    train_batches,
    num_classes_from_real_eval_labels,
    args,
)

evaluation_x, evaluation_y = select_stratified_array_subset(
    dictionary_data["x_evaluation_real"],
    dictionary_data["y_evaluation_real"],
    args.test_samples_per_class,
)
real_labels, predicted_labels = predict_array_batches(
    classifier,
    evaluation_x,
    evaluation_y,
    args.eval_batch_size,
)
get_task_metrics(real_labels, predicted_labels, "TS-TR", classifier_name, fold_number + 1)
mark R-S distance not_applicable
```

Respostas especificas:

1. Treino: batches sinteticos.
2. Teste: `x_evaluation_real`, opcionalmente limitado por `test_samples_per_class`.
3. Dados: treino sintetico, teste real.
4. Origem: AppClassNet batches usa `valid` como teste real quando valid existe.
5. Classificador: `args.eval_classifier`, default `decision_tree_subset`.
6. Scaler/transformacao: nenhum dentro da avaliacao; `ScaleGuard` valida `source` vs manifesto.
7. `data_space`: real `source`; sintetico do manifesto.
8. Quantidade por classe:
   - Treino sintetico: todos os batches ou subset estratificado interno do classificador batch.
   - Teste real: todo `valid` ou cap por `test_samples_per_class` via `_select_stratified_array_subset()` linhas ~56-72.
9. Balanceamento: subset de teste real e estratificado se `test_samples_per_class` for definido; subset classifier de treino tambem e estratificado.
10. Vazamento: sem mistura direta; depende da independencia dos arquivos train/valid.
11. Concatenacao/reutilizacao indevida: nao concatena todos os sinteticos; distancia R-S e explicitamente pulada em batches.
12. Mesmo modelo reutilizado entre avaliacoes: nao.
13. Mesmo sintetico reutilizado em TR-TS e TS-TR: sim.
14. Classificador reinicializado a cada fold: sim.
15. Indices independentes: sim se splits oficiais forem independentes; caps por classe usam selecao separada.

## Reuso de modelos e sinteticos

- Gerador:
  - Arquivo: `main.py`
  - Classe/funcao: `SynDataGen.run_experiments()`
  - Linhas: treino ~432-468, geracao ~470-480, avaliacao ~490-500
  - O gerador e treinado uma vez por fold e gera `evaluation_synthetic` uma vez por fold.

- Sintetico:
  - O mesmo `evaluation_synthetic` e passado para TR-TS e TS-TR via `run_synthetic_evaluation_modes()` (`main.py`, linhas ~110-128 e ~490-500).
  - Portanto sim, o mesmo sintetico e reutilizado nas duas avaliacoes do mesmo fold.

- Classificadores:
  - Normal: `get_trained_classifiers()` treina novos modelos a cada chamada (`Classifiers.py`, linhas ~132-151). TR-TS e TS-TR chamam separadamente; nao reutilizam o mesmo classificador.
  - Batches: `train_batch_classifier()` cria nova instancia por chamada em `make_batch_classifier()` (`BatchClassifiers.py`, linhas ~39-76 e ~170-174).

## Interpretacao de 1-Fold e 2-Fold

- `Metrics.__initialize_dictionary()` cria chaves `1-Fold`, `2-Fold`, etc. conforme `arguments.number_k_folds` (`Engine/Metrics/Metrics.py`, linhas ~202-242).
- Em cross-validation, essas chaves correspondem a folds reais (`CrossValidation.py`, linhas ~494-508).
- Em `split_mode=provided`, `number_k_folds` e forçado para 1 (`CrossValidation.py`, linha ~374). Assim:
  - `1-Fold` = split fornecido train -> valid/test.
  - Nao ha `2-Fold` nesse modo.
- No runner AppClassNet modo normal, `1-Fold` e `2-Fold` sao folds internos sobre CSV materializado, nao train/valid/test oficiais.
- Se o usuario observar `1-Fold` e `2-Fold` em uma execucao AppClassNet normal, eles nao sao repeticoes do experimento; sao folds de cross-validation sobre o dataset CSV escolhido.
- Se o usuario observar somente `1-Fold` em batches/provided, ele nao e fold real; e a unica avaliacao com split fornecido.

## Vazamento, independencia e divergencias por nome

### TR-TR

- Nome vs implementacao: correto, mas nao executado no loop principal.
- Dados efetivos: reais treino/teste.
- Divergencia relevante: docstring incorreta fala em sintetico.
- Vazamento: nao confirmado no codigo.

### TR-TS

- Nome vs implementacao: parcialmente divergente.
- O nome "treinar real" esta correto quanto ao tipo do dado, mas o conjunto real usado e `x_evaluation_real`, nao `x_training_real`.
- Em cross-validation, isso significa treinar classificador no fold de validacao e testar em sintetico.
- Em provided, isso significa treinar classificador no `valid` oficial e testar em sintetico.
- Vazamento: nao e vazamento classico treino/teste porque teste e sintetico, mas diverge da definicao esperada do projeto se TR-TS deveria treinar no real de treino.

### TS-TR

- Nome vs implementacao: correto quanto ao tipo do dado.
- Divergencia: teste real preditivo usa `x_evaluation_real`, mas metricas normais truncam labels/predicoes por `total_samples`; distancia R-S usa `x_training_real`, nao `x_evaluation_real`.
- Vazamento: nao confirmado, mas ha reutilizacao do mesmo sintetico usado no TR-TS.

## Resposta curta por pergunta

| Pergunta | TR-TR em `main.py` | TR-TS normal | TR-TS batches | TS-TR normal | TS-TR batches |
|---|---|---|---|---|---|
| Treino | `x_training_real` | `x_evaluation_real` | `x_evaluation_real` | sintetico | sintetico |
| Teste | `x_evaluation_real` | sintetico | sintetico | `x_evaluation_real` | `x_evaluation_real` |
| Tipo | real->real | real->sintetico | real->sintetico | sintetico->real | sintetico->real |
| Origem | fold train/eval ou train->valid/test | fold eval ou valid/test | valid/test | fold eval ou valid/test | valid/test |
| Classificador | lista normal | lista normal | `eval_classifier` | lista normal | `eval_classifier` |
| Transformacao | nenhuma local | nenhuma local + ScaleGuard | nenhuma local + ScaleGuard | nenhuma local + ScaleGuard | nenhuma local + ScaleGuard |
| `data_space` | assumido source | source validado | source/manifest | source validado | source/manifest |
| Quantidade/classe | fold/split | real eval + sample plan sintetico | real eval + cap sintetico opcional | sample plan sintetico + real eval truncavel | sample plan + cap real opcional |
| Balanceamento | fold estratificado se CV | depende | subset classifier estratificado | depende | subset/cap estratificado |
| Vazamento confirmado | nao | nao classico, mas conjunto real inesperado | nao classico, mas conjunto real inesperado | nao | nao |
| Reuso indevido | TR-TR nao roda | sintetico compartilhado | sintetico compartilhado | sintetico compartilhado | sintetico compartilhado |
| Classificador reinicia | sim | sim | sim | sim | sim |
| Indices independentes | sim em CV/provided externo | real vs sintetico | real vs sintetico | sintetico vs real eval | sintetico vs real eval |

## Recomendacoes pequenas e compativeis

1. Decidir formalmente se TR-TS deve treinar em `x_training_real` ou em `x_evaluation_real`. Se for `x_training_real`, alterar apenas `TrTs.evaluation_TR_TS()` e criar teste de regressao.
2. Corrigir TS-TR normal para calcular metricas sobre todo `x_evaluation_real` ou registrar explicitamente o subset real usado.
3. Separar metricas `R-S` de distancia em TS-TR: hoje a distancia usa real de treino, enquanto a metrica preditiva usa real de avaliacao.
4. Renomear ou anotar `1-Fold` em `split_mode=provided` como `provided-split` no JSON futuro, mantendo compatibilidade com a chave atual.
5. Integrar o baseline real-real AppClassNet como teste/regressao documentado sem misturar com o pipeline sintetico.
