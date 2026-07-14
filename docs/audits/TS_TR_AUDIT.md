# TS-TR Audit - AppClassNet Top-200

Data da auditoria: 2026-07-14

Escopo: auditoria exclusiva de TS-TR. Nenhum codigo de execucao foi alterado.

Definicao esperada: treinar classificador somente com dados sinteticos, testar em dados reais independentes e avaliar utilidade downstream dos sinteticos.

## Conclusao executiva

O TS-TR atual treina o classificador com dados sinteticos e testa em dados reais, mas ha diferencas importantes entre modo normal e batches:

- No modo normal, todos os sinteticos materializados sao usados para treinar, mas as metricas sao calculadas apenas sobre `y_evaluation_real[:total_samples]` e `pred[:total_samples]`. Se o numero de sinteticos for menor que o numero de amostras reais de avaliacao, o teste real efetivo vira um prefixo truncado.
- No modo batches, os sinteticos sao lidos de todos os batches do manifesto. Para classificadores subset, um reservatorio estratificado por classe e usado. Para `partial_fit`, todos os batches sinteticos sao consumidos.
- O classificador batch default em AppClassNet e `decision_tree_subset`, nao `SGDClassifier`.
- Existe risco critico com `model_type="copy"`: a geracao recebe `x_evaluation_real` como `x_real_samples`, e TS-TR depois treina no sintetico copiado e testa no mesmo real de avaliacao.

TS-TR pode ficar no nivel aleatorio por colapso/baixa qualidade dos sinteticos, por classes ausentes, por escala incorreta ou por avaliador inadequado. O codigo atual tem auditorias auxiliares de labels/escala, mas TS-TR nao registra diagnostico completo de colapso das predicoes por classificador.

## Achados classificados

| Severidade | Achado | Evidencia |
|---|---|---|
| Critico | Modo normal trunca as metricas TS-TR para `total_samples` sinteticos, mesmo predizendo em todo `x_evaluation_real`. | `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas ~207-216. |
| Critico | `model_type="copy"` pode gerar sinteticos a partir de `x_evaluation_real`; TS-TR treina nesses sinteticos e testa em `x_evaluation_real`. | `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linhas ~472-480; funcao `synthesize_data()`, linhas ~906-912; `TsTr.evaluation_TS_TR()`, linhas ~200-216. |
| Alto | `synthetic_train_samples_per_class` nao limita universalmente o treino TS-TR: em batches/subset vira quota de reservatorio; em batches/partial_fit todos os batches sao usados; em normal nao e usado diretamente por TS-TR nem pelo `SamplePlanner` atual. | `Engine/Arguments/Arguments.py`, funcao `_configure_classifier_arguments()`, linhas ~128-132; `Engine/DataIO/SamplePlanner.py`, funcao `build_sample_plan_from_args()`, linhas ~27-89; `Engine/Classifiers/BatchClassifiers.py`, funcoes `_subset_quota()`, `_collect_stratified_subset()` e `train_batch_classifier()`, linhas ~105-167 e ~170-237; `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas ~162-216. |
| Alto | Em `split_mode=provided`, o teste real TS-TR usa `valid` quando `valid` e `test` existem; `test` oficial e marcado como nao suportado nesse fluxo. | `Engine/Evaluation/CrossValidation.py`, funcoes `_build_provided_split_folds()` e `_create_fold()`, linhas ~357-380 e ~329-346. |
| Medio | Batches carregam todos os batches do manifesto, mas classificadores subset materializam apenas um subset estratificado; isso e intencional, mas precisa ser registrado na interpretacao. | `Engine/DataIO/SyntheticBatchIO.py`, classe `SyntheticBatchReader`, linhas ~154-171; `Engine/Classifiers/BatchClassifiers.py`, linhas ~114-167 e ~202-214. |
| Medio | O modo normal materializa todos os sinteticos em memoria antes do treino. | `Engine/Evaluation/TsTr.py`, linhas ~162-202. |
| Medio | Real e sintetico sao validados por `ScaleGuard`, mas batches validam metadados do manifesto e nao percorrem todos os valores antes do treino TS-TR. | `Engine/Evaluation/TsTr.py`, linhas ~98-107 e ~181-199; `Engine/Preprocessing/FeatureTransformManager.py`, classe `ScaleGuard`, linhas ~520-555. |
| Medio | Distancia R-S em TS-TR normal usa `x_training_real`, enquanto a avaliacao preditiva testa em `x_evaluation_real`. | `Engine/Evaluation/TsTr.py`, linhas ~226-249. |
| Baixo | O default batch AppClassNet evita SGD: `decision_tree_subset` e o default; SGD existe como opcao. | `run_appclassnet_top200.py`, parser `--eval_classifier`, linhas ~2045-2050; `Engine/Arguments/ArgumentsFramework.py`, linhas ~157-161. |

## Fluxo TS-TR atual

### Normal

- Arquivo: `Engine/Evaluation/TsTr.py`
- Classe: `TsTr`
- Funcao: `evaluation_TS_TR()`
- Linhas aproximadas: ~162-249

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
classifiers = get_trained_classifiers(
    shuffled_data,
    shuffled_labels,
    numpy.float32,
    get_number_columns(),
)

for classifier in classifiers:
    predicted = classifier.predict(dictionary_data["x_evaluation_real"])
    get_task_metrics(
        labels_to_1d_integer(dictionary_data["y_evaluation_real"])[:total_samples],
        numpy.array(predicted)[:total_samples],
        "TS-TR",
        classifier_name,
        fold_number + 1,
    )

data_real_for_distance = numpy.array(dictionary_data["x_training_real"])
data_synthetic_for_distance = numpy.asarray(data, dtype=numpy.float32)
truncate_larger_side_to_same_row_count()
get_distance_metrics(data_real_for_distance, data_synthetic_for_distance, "R-S")
```

### Batches

- Arquivo: `Engine/Evaluation/TsTr.py`
- Classe: `TsTr`
- Funcao: `evaluation_TS_TR()`
- Linhas aproximadas: ~96-160

Pseudocodigo fiel:

```python
ScaleGuard.validate_compatible_metadata(
    ScaleGuard.describe(dictionary_data["x_evaluation_real"], data_space="source"),
    synthetic_manifest_metadata,
    context="TS-TR",
)

train_batches = iter_synthetic_labeled_batches(synthetic_data)
classifier, metadata = train_batch_classifier(
    eval_classifier,
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
    eval_batch_size,
)

get_task_metrics(real_labels, predicted_labels, "TS-TR", classifier_name, fold_number + 1)
mark R-S distance not_applicable
```

## Respostas aos pontos investigados

### 1. Como os sinteticos de treino sao carregados

Modo normal: os sinteticos chegam como dicionario `synthetic_data` e sao materializados em listas `data` e `labels`.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~162-181.

Modo batches: os sinteticos chegam como `SyntheticBatchReader` e sao iterados via `iter_synthetic_labeled_batches()`.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~120-127.
- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, funcao `iter_synthetic_labeled_batches()`, linhas ~86-97.
- Evidencia: `Engine/DataIO/SyntheticBatchIO.py`, classe `SyntheticBatchReader`, linhas ~154-171.

### 2. Se `synthetic_train_samples_per_class` e respeitado

Parcialmente.

- O argumento e copiado para `train_samples_per_class`. Evidencia: `Engine/Arguments/Arguments.py`, classe n/a, funcao `_configure_classifier_arguments()`, linhas ~128-132.
- No runner batches, `synthetic_train_samples_per_class` vira `--train_samples_per_class`. Evidencia: `run_appclassnet_top200.py`, funcao `build_batch_main_command()`, linhas ~1706-1723.
- Em batch classifiers subset, `train_samples_per_class` define `quota_per_class`. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, funcoes `_subset_quota()` e `_collect_stratified_subset()`, linhas ~105-167.
- Em batch classifiers `partial_fit` (`sgd`, `passive_aggressive`, `naive_bayes`, `mlp_small`), a quota nao limita; todos os batches sao usados. Evidencia: `train_batch_classifier()`, linhas ~181-201.
- No modo normal TS-TR, a funcao nao usa `train_samples_per_class`; treina em todos os sinteticos materializados. Evidencia: `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas ~162-202.
- No modo normal, a quantidade gerada vem de `SamplePlanner` por `number_samples_per_class`, `samples_per_class`, `total_synthetic_rows` ou distribuicao dos labels; `build_sample_plan_from_args()` nao le `train_samples_per_class`. Evidencia: `Engine/DataIO/SamplePlanner.py`, classe n/a, funcao `build_sample_plan_from_args()`, linhas ~27-89.

### 3. Se todos os sinteticos sao carregados ou apenas o primeiro batch

Nao e apenas o primeiro batch.

- `SyntheticBatchReader.items()` percorre todas as classes e todos os batches listados no manifesto. Evidencia: `Engine/DataIO/SyntheticBatchIO.py`, linhas ~154-160.
- `iter_synthetic_labeled_batches()` itera o reader inteiro. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~86-97.
- Para subset classifiers, todos os batches sao percorridos para alimentar reservatorios por classe, mas apenas o subset final e materializado para `fit()`. Evidencia: `_collect_stratified_subset()`, linhas ~114-167.
- Para partial fit, todos os batches nao vazios sao enviados para `partial_fit()`. Evidencia: `train_batch_classifier()`, linhas ~181-201.

### 4. Se batches de classes diferentes sao combinados corretamente

Sim, no avaliador batch.

- Cada batch vem acompanhado de `class_label` do manifesto. Evidencia: `SyntheticBatchReader.items()`, linhas ~154-157.
- `iter_synthetic_labeled_batches()` cria `y_batch` preenchido com esse label para todas as linhas do batch. Evidencia: `BatchClassifiers.py`, linhas ~92-97.
- Subset classifiers combinam `x_parts` por `vstack` e `y_parts` por `concatenate`. Evidencia: `_collect_stratified_subset()`, linhas ~146-162.

### 5. Se X sintetico permanece alinhado com y sintetico

Normal: sim por construcao, porque para cada `generated_samples` o codigo adiciona exatamente o mesmo numero de labels.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~170-178.

Batches: sim por construcao, porque cada `x_batch` recebe um `y_batch` de mesmo tamanho.

- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~92-97.
- Ao embaralhar subset, a mesma permutacao e aplicada a X e y. Evidencia: `_collect_stratified_subset()`, linhas ~157-162.

### 6. Se todas as classes 0..199 estao presentes

Antes da avaliacao, a auditoria de labels sinteticos falha se `number_classes=200` e faltar alguma classe observada.

- Evidencia: `Engine/DataIO/SyntheticLabelAudit.py`, classe `SyntheticLabelGenerationAudit`, metodo `_validate_domain()`, linhas ~139-167.
- Normal: chamada em `main.py`, funcao `synthesize_data()`, linhas ~965-974.
- Batches incremental: chamada via `audit.finalize()` em `main.py`, funcao `_synthesize_data_incremental()`, linhas ~1242-1282.

TS-TR em si nao revalida classes 0..199 antes de treinar.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~162-216 e ~120-153.

### 7. Se alguma classe possui zero amostras

A auditoria de labels deve falhar para AppClassNet `number_classes=200` se a classe tiver zero amostras geradas observadas.

- Evidencia: `SyntheticLabelGenerationAudit._validate_domain()`, linhas ~154-162.

No batch classifier subset, uma classe sem linhas simplesmente nao entra em `x_parts/y_parts`; o metadata `train_class_counts` refletiria a ausencia, mas o treino nao falha especificamente por classe ausente.

- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~146-167 e ~217-237.

### 8. Se o classificador aprende apenas uma classe

O codigo atual nao registra, em TS-TR, um diagnostico direto de quantas classes foram previstas no real.

- Predicoes normal: `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linha ~207.
- Predicoes batches: `Engine/Classifiers/BatchClassifiers.py`, classe n/a, funcao `predict_array_batches()`, linhas ~240-251.
- Metricas agregadas sao calculadas em `Metrics.get_task_metrics()`, mas sem salvar distribuicao de predicoes por classe. Evidencia: `Engine/Metrics/Metrics.py`, classe `Metrics`, funcao `get_task_metrics()`, linhas ~634-671.

Se o classificador aprende/preve uma unica classe, Accuracy pode ficar perto de frequencia dessa classe no real; em AppClassNet balanceado isso tende a ficar perto de `0.005`.

### 9. Se existe colapso do gerador

Ha sanity check auxiliar de colapso/sinal de classe, mas TS-TR nao incorpora esse diagnostico por classificador.

- `SyntheticSanityChecker` calcula contagens sinteticas por classe, ranges, predicoes de um DecisionTree real->sintetico e fracao da classe dominante. Evidencia: `Engine/DataIO/SyntheticSanityChecks.py`, linhas ~80-106, ~152-239 e ~277-303.

Um colapso real para uma unica classe deveria aparecer como:

- `synthetic_counts_by_class` com muitas classes ausentes;
- `dominant_predicted_class_fraction` alto;
- `top_predicted_classes` concentrado;
- muitos recalls por classe iguais a zero.

### 10. Se o sintetico esta na mesma escala do teste real

Normal: `ScaleGuard.validate_before_evaluation()` verifica finitude e compatibilidade de `data_space`/`transform_id`.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~181-199.
- Evidencia: `Engine/Preprocessing/FeatureTransformManager.py`, classe `ScaleGuard`, linhas ~520-555.

Batches: valida metadados do manifesto contra real.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~98-107.

Risco: batches nao percorrem todos os valores sinteticos antes do treino TS-TR para validar NaN/inf/range; dependem do manifesto e dos sanity checks.

### 11. Se o classificador aplica scaler proprio

Nao foi encontrada aplicacao de scaler dentro de TS-TR.

- Normal: TS-TR passa `shuffled_data` diretamente para `get_trained_classifiers()`. Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~200-202.
- Batches: `train_batch_classifier()` recebe batches sinteticos diretamente. Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~120-127.
- `DecisionTree.get_model()` converte para `numpy.array(..., dtype=dataset_type)`, mas nao escala. Evidencia: `Engine/Classifiers/Algorithms/DecisionTree.py`, linhas ~95-119.

Transformacoes, quando existem, ocorrem antes via `ModelInputAdapter` e devem retornar sinteticos ao espaco `source`.

- Evidencia: `main.py`, funcao `synthesize_data()`, linhas ~951-963; `FeatureTransformManager.py`, classe `ModelInputAdapter`, linhas ~614-650.

### 12. Se `fit` ou `partial_fit` e usado corretamente

Batches:

- `sgd`, `passive_aggressive`, `naive_bayes`, `mlp_small` usam `partial_fit`. Evidencia: `BatchClassifiers.py`, `PARTIAL_FIT_CLASSIFIERS`, linha ~31, e `train_batch_classifier()`, linhas ~181-201.
- `decision_tree_subset`, `extra_trees_subset`, `random_forest_light` usam `fit` em subset materializado. Evidencia: `SUBSET_CLASSIFIERS`, linha ~32, e `train_batch_classifier()`, linhas ~202-214.

Normal:

- Classificadores normais usam `get_model()`; DecisionTree usa `fit`. Evidencia: `Engine/Classifiers/Classifiers.py`, linhas ~132-151; `DecisionTree.py`, linhas ~109-119.

### 13. Se `classes=np.arange(200)` e informado no primeiro `partial_fit`

Sim, quando o numero de classes configurado e 200.

- Evidencia: `train_batch_classifier()` cria `classes = numpy.arange(int(num_classes), dtype=numpy.int64)` e passa no primeiro `partial_fit()`. `Engine/Classifiers/BatchClassifiers.py`, linhas ~181-196.
- Em TS-TR batches, `num_classes` vem de `self._get_configured_number_classes(dictionary_data['y_evaluation_real'])`. Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~120-125.

### 14. Se o classificador e reinicializado entre folds

Sim.

- Normal: `get_trained_classifiers()` chama `classifier_model.get_model(...)` a cada avaliacao/fold. Evidencia: `Engine/Classifiers/Classifiers.py`, linhas ~132-151.
- Batches: `train_batch_classifier()` chama `make_batch_classifier()` a cada chamada. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~170-174 e ~39-76.

### 15. Se dados reais entram acidentalmente no treinamento

No treinamento do classificador TS-TR, o codigo usa apenas `synthetic_data`.

- Normal: treino em `shuffled_data, shuffled_labels` derivados de `synthetic_data`. Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~162-202.
- Batches: treino em `iter_synthetic_labeled_batches(synthetic_data)`. Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~120-127.

Risco critico indireto: se o gerador `copy` produziu sinteticos a partir de `x_evaluation_real`, os dados reais entram no treino sob forma sintetica/copied.

- Evidencia: `main.py`, `run_experiments()`, linhas ~472-480, e `synthesize_data()`, linhas ~906-912.

### 16. Se o teste real e realmente independente

Depende do caminho.

- Cross-validation: `x_evaluation_real` vem de indices de validacao separados dos indices de treino do gerador. Evidencia: `Engine/Evaluation/CrossValidation.py`, linhas ~494-564.
- Provided: `x_evaluation_real` vem de `valid` se existir; senao `test`. Evidencia: `_build_provided_split_folds()`, linhas ~357-380.
- AppClassNet batches runner passa train/valid/test, mas o pipeline escolhe `valid` como avaliacao e marca `test` como nao suportado. Evidencia: `run_appclassnet_top200.py`, funcao `build_batch_main_command()`, linhas ~1628-1657; `CrossValidation.py`, linhas ~357-365.

Risco: para `copy`, o teste real deixa de ser independente do treino sintetico do classificador.

### 17. Se o classificador escolhido e adequado para o AppClassNet

Para batches AppClassNet, o default e `decision_tree_subset`, que e mais adequado como primeiro indicador do que `SGDClassifier` para este projeto.

- Evidencia: `run_appclassnet_top200.py`, parser `--eval_classifier`, linhas ~2045-2050.
- Evidencia: `Engine/Arguments/ArgumentsFramework.py`, linhas ~157-161 avisam que SGD pode subestimar a qualidade AppClassNet.

Para modo normal, a escolha vem de `arguments.classifier` ou `--normal_classifier`; DecisionTree existe e funciona com subset/materializado.

- Evidencia: `Engine/Classifiers/Classifiers.py`, linhas ~103-118 e ~124-151.
- Evidencia: `Engine/Classifiers/Algorithms/DecisionTree.py`, linhas ~76-119.

### 18. Se `SGDClassifier` esta sendo usado como default indevido

Nao no caminho batches AppClassNet atual.

- Default do runner AppClassNet: `decision_tree_subset`. Evidencia: `run_appclassnet_top200.py`, linhas ~2045-2050.
- Default do framework para `--eval_classifier`: `decision_tree_subset`. Evidencia: `Engine/Arguments/ArgumentsFramework.py`, linhas ~157-161.
- `SGDClassifier` continua disponivel como opcao. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~20-43.

### 19. Se a avaliacao com DecisionTree funciona com subconjunto sintetico materializado

Sim no desenho atual.

- Batches: `decision_tree_subset` cria `DecisionTreeClassifier(random_state=42)` e treina com `fit()` no subset estratificado coletado por reservatorio. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~50-57 e ~202-214.
- Normal: `DecisionTree` cria `DecisionTreeClassifier(...)` e treina com `fit()` nos sinteticos materializados. Evidencia: `Engine/Classifiers/Algorithms/DecisionTree.py`, linhas ~109-119.

Risco no modo normal: se a quantidade sintetica for grande, tudo e materializado em memoria antes do fit.

- Evidencia: `Engine/Evaluation/TsTr.py`, linhas ~162-202.

### 20. Se o TS-TR esta no nivel aleatorio por causa dos dados ou do avaliador

A auditoria estatica nao consegue decidir sem executar, mas aponta como separar causas:

- Se os sinteticos cobrem 200 classes, estao em `source`, sem NaN/inf, com ranges proximos do real, e DecisionTree/ExtraTrees ainda ficam perto de `0.005`, o problema provavelmente e qualidade/condicionamento dos sinteticos.
- Se `SGDClassifier` fica perto de `0.005`, mas `DecisionTreeSubset`/`ExtraTreesSubset` nao, o problema e avaliador inadequado.
- Se poucas classes aparecem em `train_class_counts` ou nas predicoes, o problema e colapso de labels/gerador ou subset.
- Se o modo normal usa poucos sinteticos e trunca o real por `total_samples`, a metrica pode estar medindo um prefixo nao representativo do real.
- Se `model_type="copy"` gera resultados altos, pode ser vazamento por copia do `x_evaluation_real`.

Diagnostico minimo para diferenciar dados versus avaliador:

```python
if classifier in {"decision_tree_subset", "extra_trees_subset"} and accuracy_approx_random:
    inspect synthetic_train_counts_by_class
    inspect synthetic_mean_by_class, synthetic_variance_by_class
    inspect predicted_label_count, prediction_entropy, recall_by_class
elif classifier == "SGDClassifier" and tree_based_classifier_not_random:
    classify_as_evaluator_limited_for_appclassnet
```

## Diagnostico de colapso que deveria ser registrado em TS-TR

O diagnostico deve ser salvo por fold, classificador e modo de execucao, antes/depois de `get_task_metrics()`:

| Campo | Como calcular | Interpretacao |
|---|---|---|
| `predicted_label_count` | `len(unique(predicted_labels))` | Quantas classes o classificador reconheceu no real. |
| `majority_prediction_frequency` | `max(count(predicted_labels)) / len(predicted_labels)` | Detecta colapso para uma classe dominante. |
| `prediction_entropy` | `-sum(p_i * log2(p_i))` sobre a distribuicao de predicoes | Baixa entropia indica colapso. Maximo esperado para 200 classes balanceadas: `log2(200) ~= 7.64`. |
| `never_predicted_classes` | `set(range(200)) - set(predicted_labels)` | Classes nunca reconhecidas no real. |
| `recall_by_class` | `recall_score(y_real, predicted, labels=range(200), average=None, zero_division=0)` | Mostra classes com recall zero. |
| `recognized_class_count` | numero de classes com recall > 0 | Indicador direto de cobertura downstream. |
| `synthetic_mean_by_class` | media de X sintetico por label | Diagnostica classes colapsadas para mesmo centro. |
| `synthetic_variance_by_class` | variancia de X sintetico por label | Variancia zero/baixa indica duplicacao ou colapso. |
| `synthetic_train_counts_by_class` | contagem de y sintetico usado no fit | Detecta classes ausentes antes do treino. |
| `real_test_counts_by_class` | contagem de y real usado no teste efetivo | Detecta truncamento ou cap por classe. |

Pseudocodigo de especificacao:

```python
pred_counts = bincount(predicted_labels, minlength=200)
predicted_label_count = count_nonzero(pred_counts)
majority_prediction_frequency = pred_counts.max() / len(predicted_labels)
prob = pred_counts[pred_counts > 0] / len(predicted_labels)
prediction_entropy = -sum(prob * log2(prob))
never_predicted_classes = [c for c in range(200) if pred_counts[c] == 0]

recall_by_class = recall_score(
    real_labels,
    predicted_labels,
    labels=list(range(200)),
    average=None,
    zero_division=0,
)
recognized_class_count = count_nonzero(recall_by_class > 0)

for class_id in range(200):
    x_c = synthetic_x[synthetic_y == class_id]
    synthetic_mean_by_class[class_id] = mean(x_c, axis=0) if len(x_c) else None
    synthetic_variance_by_class[class_id] = var(x_c, axis=0) if len(x_c) else None
```

## Divergencia entre nome e implementacao

Nome esperado: TS-TR = treinar em sintetico, testar em real.

Implementacao real:

- Tipo dos dados: correto, classificador treina em sintetico e prediz em real.
- Teste real normal: predicao em todo `x_evaluation_real`, mas metrica truncada por `total_samples`.
- Teste real provided: usa `valid` se existe, nao `test` oficial.
- Distancia R-S: compara sintetico contra `x_training_real`, enquanto a metrica preditiva usa `x_evaluation_real`.

## Decisao recomendada

Para AppClassNet top-200, TS-TR deve ser interpretado somente junto com:

- classificador usado (`decision_tree_subset`, `extra_trees_subset`, `random_forest_light`, ou `sgd`);
- contagem sintetica por classe usada no treino;
- contagem real por classe usada no teste;
- diagnostico de colapso das predicoes;
- `data_space`/`transform_id`;
- indicacao explicita de split real usado (`valid` ou `test`).

Antes de qualquer correcao, priorizar um teste de especificacao que compare DecisionTree/ExtraTrees em subset sintetico balanceado e falhe quando o avaliador entrar em nivel aleatorio por colapso de predicoes, classes ausentes ou escala incompatível.
