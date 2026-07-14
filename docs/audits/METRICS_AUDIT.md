# METRICS_AUDIT

Data da auditoria: 2026-07-14

Escopo: auditoria das métricas preditivas usadas em TR-TR, TR-TS e TS-TR para AppClassNet top-200. Nenhum arquivo de código foi modificado.

## Sumário executivo

- **[alto]** As métricas multiclass principais usam `sklearn` corretamente para `average="macro"` e `average="weighted"` com `zero_division=0` (`Engine/Metrics/Metrics.py`, classe `Metrics`, função `_get_multiclass_metric_values`, linhas ~355-368).
- **[alto]** O cálculo não passa `labels=np.arange(200)`. Assim, as médias macro do sklearn são calculadas sobre o conjunto de labels presentes em `y_true` ou `y_pred`, não necessariamente sobre o domínio fixo AppClassNet `0..199`. Se todas as 200 classes estão em `y_true`, isso é equivalente; se classes somem, o resultado pode mascarar ausência de classe.
- **[médio]** `BalancedAccuracy` usa `balanced_accuracy_score(y_true, y_pred)` sem labels explícitas (`Metrics.py`, `_get_multiclass_metric_values`, linha ~367). Para AppClassNet balanceado e com todas as classes presentes, tende a ficar próximo da média dos recalls por classe; pode coincidir com Accuracy quando o suporte por classe é exatamente uniforme.
- **[médio]** Métricas preditivas não aplicáveis são inicializadas como `"not_applicable"`, não como `0` (`Metrics.py`, `__initialize_dictionary`, linhas ~203-248). A agregação pula valores não numéricos via `_numeric_summary` (`linhas ~339-353`). Isso evita médias falsas por avaliações não executadas.
- **[médio]** `EfficiencyMetrics` ainda é inicializado com zeros (`Metrics.py`, linhas ~250-257), mas isso não afeta Accuracy/F1/Precision/Recall.
- **[médio]** Não há regra implementada para marcar `chance_level_suspected` quando `num_classes=200` e Accuracy fica perto de `1/200 ~= 0.005`. Esta auditoria propõe a regra.
- **[médio]** O baseline real-real separado em `run_appclassnet_top200.py::run_real_real_baseline()` calcula `Accuracy`, `MacroF1`, `WeightedF1` e `BalancedAccuracy`, mas não calcula Macro/Weighted Precision/Recall (`run_appclassnet_top200.py`, linhas ~1118-1121). Isso é suficiente para o golden atual informado, mas não cobre toda a lista de métricas do pipeline principal.
- **[baixo]** Testes existentes verificam seleção de métricas multiclass e classes ausentes em `y_pred`, mas não testam domínio fixo `0..199` nem comparação manual por matriz de confusão (`tests/test_metrics_target_types.py`, linhas ~52-75; `tests/test_multiclass_pipeline.py`, linhas ~104-126).

## Onde as métricas são chamadas

### TR-TR

**Arquivo:** `Engine/Evaluation/TrTr.py`  
**Classe:** `TrTr`  
**Função:** `evaluation_TR_TR`

- **Linhas:** ~83-102.
- **y_true:** `labels_to_1d_integer(dictionary_data['y_evaluation_real'], context="TR-TR evaluation labels")`.
- **y_pred:** `classifier_instances.predict(dictionary_data['x_evaluation_real'])`, convertido com `numpy.array`.
- **Chamada:** `get_task_metrics(y_true, y_pred, "TR-TR", classifier_name, fold_number)`.
- **Risco:** **baixo** para shape no caminho normal, porque `labels_to_1d_integer` força vetor 1D; **médio** porque não há domínio explícito `0..199` na métrica.

### TR-TS

**Arquivo:** `Engine/Evaluation/TrTs.py`  
**Classe:** `TrTs`  
**Função:** `evaluation_TR_TS`

- **Modo batches:** linhas ~120-136.
  - **y_true:** labels sintéticos retornados por `predict_synthetic_batches`.
  - **y_pred:** predições do classificador real->sintético.
  - **Chamada:** `get_task_metrics(labels, predictions, "TR-TS", classifier_name, fold_number)`.
- **Modo normal:** linhas ~138-203.
  - **y_true:** lista `labels` criada a partir das chaves/classes sintéticas.
  - **y_pred:** `classifier_instances.predict(synthetic_array)`.
  - **Chamada:** `get_task_metrics(numpy.array(labels), numpy.array(label_predicted), "TR-TS", ...)`.
- **Risco:** **alto** se sintéticos não contêm todas as 200 classes: macro/weighted serão calculados sobre o domínio observado, e não sobre `0..199`.

### TS-TR

**Arquivo:** `Engine/Evaluation/TsTr.py`  
**Classe:** `TsTr`  
**Função:** `evaluation_TS_TR`

- **Modo batches:** linhas ~130-153.
  - **y_true:** `real_labels` produzido por `predict_array_batches`.
  - **y_pred:** `predicted_labels` do classificador sintético->real.
  - **Chamada:** `get_task_metrics(real_labels, predicted_labels, "TS-TR", ...)`.
- **Modo normal:** linhas ~180-216.
  - **y_true:** `labels_to_1d_integer(dictionary_data['y_evaluation_real'])[:total_samples]`.
  - **y_pred:** `numpy.array(label_predicted)[:total_samples]`.
  - **Chamada:** `get_task_metrics(...)`.
- **Risco:** **alto** no modo normal: o slice `[:total_samples]` pode truncar o teste real ao número de sintéticos (`TsTr.py`, linhas ~212-216), então as métricas podem não ser calculadas sobre todo `x_evaluation_real`.

## Implementação central

**Arquivo:** `Engine/Metrics/Metrics.py`  
**Classe:** `Metrics`

### Lista de métricas

**Função:** `_get_classifier_metric_names`  
**Linhas:** ~294-307.

Para tarefas não binárias, o código usa:

- `Accuracy`
- `MacroPrecision`
- `MacroRecall`
- `MacroF1`
- `WeightedPrecision`
- `WeightedRecall`
- `WeightedF1`
- `BalancedAccuracy`

**Classificação:** **baixo**. A lista cobre as métricas solicitadas.

### Conversão de labels

**Função:** `_labels_to_vector`  
**Linhas:** ~309-315.

- Executa `labels = numpy.ravel(numpy.asarray(labels))`.
- Tenta `labels.astype(numpy.int64)`.
- Não valida que floats sejam inteiros.
- Não faz `argmax` para labels one-hot.

**Riscos:**

- **[médio]** Se labels one-hot 2D forem passados, serão achatados, não convertidos por `argmax`. Em geral isso causará mismatch de comprimento e será marcado como `not_applicable`, não como métrica incorreta silenciosa.
- **[médio]** Se labels float não inteiros forem passados diretamente ao `Metrics`, `astype(int64)` trunca. Nas avaliações auditadas, os labels entram por `labels_to_1d_integer` antes de `get_task_metrics` em vários pontos, o que reduz esse risco.

### Cálculo multiclass

**Função:** `_get_multiclass_metric_values`  
**Linhas:** ~355-368.

| Métrica | Implementação atual | Avaliação |
|---|---|---|
| Accuracy | `accuracy_score(real_labels, predict_labels)` | Correta. |
| BalancedAccuracy | `balanced_accuracy_score(real_labels, predict_labels)` | Correta para classes presentes em `y_true`; sem domínio fixo 0..199. |
| MacroPrecision | `precision_score(..., average="macro", zero_division=0)` | Correta; sem `labels=np.arange(200)`. |
| MacroRecall | `recall_score(..., average="macro", zero_division=0)` | Correta; sem `labels=np.arange(200)`. |
| MacroF1 | `f1_score(..., average="macro", zero_division=0)` | Correta; sem `labels=np.arange(200)`. |
| WeightedPrecision | `precision_score(..., average="weighted", zero_division=0)` | Correta; ponderada por suporte em `y_true`. |
| WeightedRecall | `recall_score(..., average="weighted", zero_division=0)` | Correta; em multiclass single-label equivale a Accuracy quando todos os exemplos entram. |
| WeightedF1 | `f1_score(..., average="weighted", zero_division=0)` | Correta; ponderada por suporte em `y_true`. |

**Achado principal:** o cálculo é correto para o domínio observado, mas não força AppClassNet `0..199`.

### Verificação dos 10 itens solicitados

| Item | Implementação atual | Status |
|---|---|---|
| Accuracy | `accuracy_score(real_labels, predict_labels)` em `Metrics._get_multiclass_metric_values()` | Correta para todos os exemplos recebidos. |
| BalancedAccuracy | `balanced_accuracy_score(real_labels, predict_labels)` | Correta para classes presentes em `y_true`; sem domínio fixo `0..199`. |
| MacroPrecision | `precision_score(..., average="macro", zero_division=0)` | Correta para labels observadas; sem `labels=np.arange(200)`. |
| MacroRecall | `recall_score(..., average="macro", zero_division=0)` | Correta para labels observadas; sem `labels=np.arange(200)`. |
| MacroF1 | `f1_score(..., average="macro", zero_division=0)` | Correta para labels observadas; sem `labels=np.arange(200)`. |
| WeightedPrecision | `precision_score(..., average="weighted", zero_division=0)` | Correta, ponderada por suporte em `y_true`; sem domínio fixo. |
| WeightedRecall | `recall_score(..., average="weighted", zero_division=0)` | Correta; em multiclass single-label equivale a Accuracy quando todos os exemplos entram. |
| WeightedF1 | `f1_score(..., average="weighted", zero_division=0)` | Correta, ponderada por suporte em `y_true`. |
| Agregação entre folds | `Metrics.update_mean_std_fold()` percorre `TS-TR`, `TR-TS`, `TR-TR` e folds `*-Fold` | Agrega apenas valores numéricos; não registra `n_valid_folds`. |
| Média e desvio padrão | `_numeric_summary()` usa `numpy.mean` e `numpy.std` com `ddof=0` | Correto como desvio populacional; precisa estar documentado. |

## Classes ausentes e domínio 0..199

### Classes ausentes em `y_pred`

Com `zero_division=0`, classes presentes em `y_true` mas nunca previstas recebem precision/recall/F1 apropriados com zeros onde necessário. Isso é desejável.

**Evidência:** `precision_score`, `recall_score` e `f1_score` usam `zero_division=0` (`Metrics.py`, linhas ~361-366).  
**Classificação:** **baixo**.

### Classes ausentes em `y_true`

Sem `labels=np.arange(200)`, classes ausentes em `y_true` não entram na média macro padrão do sklearn, a menos que apareçam em `y_pred`. Para AppClassNet, isso é perigoso quando uma avaliação perde classes por seleção, geração ou truncamento.

Exemplo auditado com 4 classes esperadas, mas classe 3 ausente:

```text
y_true = [0,0,1,1,2,2]
y_pred = [0,1,1,1,0,2]

sklearn default MacroF1 = 0.655555...
sklearn labels=[0,1,2,3] MacroF1 = 0.491666...
```

**Interpretação:** para AppClassNet, se o domínio esperado é sempre 200 classes, o relatório diagnóstico deveria calcular também métricas com `labels=np.arange(200)`.

**Classificação:** **alto**.

### Ordem das labels

O sklearn ordena labels internamente quando `labels` não é passado. Para médias agregadas isso não altera resultado, mas para validação manual por matriz de confusão é necessário fixar a ordem `0..199`.

**Classificação:** **médio**.

## Shape, one-hot e arredondamento

- **Shape:** `get_task_metrics` não valida explicitamente igualdade de comprimento. Se `sklearn` lançar exceção, o bloco inteiro vira `not_applicable` (`Metrics.py`, linhas ~657-671).  
  **Risco:** **médio**.

- **One-hot:** `_labels_to_vector` usa `ravel`, não `argmax` (`Metrics.py`, linhas ~309-315).  
  **Risco:** **médio** se algum caminho chamar `get_task_metrics` com one-hot; nas avaliações auditadas, isso não parece ocorrer.

- **Arredondamento:** não há `round`/`rint` dentro de `Metrics.py`. A única coerção é `astype(int64)`.  
  **Risco:** **baixo** para predições de classificadores sklearn, que já são labels discretos.

## Inicialização, não aplicável e zeros

### Métricas preditivas

**Arquivo:** `Engine/Metrics/Metrics.py`  
**Função:** `__initialize_dictionary`

- TR-TR, TR-TS e TS-TR inicializam cada fold com `"not_applicable"` (`linhas ~203-236`).
- Summary também inicia com `"not_applicable"` (`linhas ~210-233`).
- `mark_classifier_metrics_not_applicable` e `mark_evaluation_classifiers_not_applicable` mantêm `"not_applicable"` e registram motivo (`linhas ~585-599`).

**Conclusão:** não há geração de métricas zero para avaliações preditivas não executadas.

**Classificação:** **baixo**.

### Folds vazios e execuções parciais

Se uma avaliação não tem dados, os caminhos TR-TS/TS-TR em batches chamam `mark_classifier_metrics_not_applicable()` antes de retornar (`TrTs.py`, classe `TrTs`, função `evaluation_TR_TS`, linhas ~120-130; `TsTr.py`, classe `TsTr`, função `evaluation_TS_TR`, linhas ~145-153). No modo normal, TR-TS também marca `not_applicable` quando não há sintéticos (`TrTs.py`, linhas ~154-165).

Risco residual:

- `update_mean_std_fold()` ignora folds `not_applicable`, mas não grava quantos folds entraram na média.
- Uma execução com 2 folds e só 1 fold válido terá Summary numérico sem `n_valid_folds=1`.

**Classificação:** **médio**.

### Métricas de eficiência

`EfficiencyMetrics` é inicializado com `0` (`Metrics.py`, linhas ~250-257). Isso não afeta as métricas preditivas auditadas, mas pode parecer execução real se usado em relatórios gerais.

**Classificação:** **baixo** para esta auditoria; **médio** para relatórios de eficiência.

## Agregação entre folds

**Arquivo:** `Engine/Metrics/Metrics.py`  
**Função:** `update_mean_std_fold`

- Percorre metodologias `["TS-TR", "TR-TS", "TR-TR"]` (`linha ~528`).
- Para cada classificador, coleta todos os `*-Fold` (`linhas ~531-541`).
- Usa `_numeric_summary(values)` (`linhas ~544-548`).
- `_numeric_summary` tenta converter cada valor para `float`; valores não numéricos como `"not_applicable"` são ignorados (`linhas ~339-353`).
- Se nenhum valor numérico existir, retorna `("not_applicable", "not_applicable")`.

**Conclusão:** médias e desvios são calculados apenas sobre folds executados com valores numéricos.

**Risco:** **médio**. Isso evita zeros falsos, mas a Summary não registra quantos folds foram efetivamente incluídos. Com 2 folds, se só 1 executou, a média parece válida sem `n_valid_folds`.

### Desvio padrão

`numpy.std` é usado com `ddof=0` (`Metrics.py`, `_numeric_summary`, linha ~353). Isso calcula desvio populacional, não amostral.

**Classificação:** **baixo**, desde que documentado.

## BalancedAccuracy vs Accuracy

`BalancedAccuracy` pode duplicar `Accuracy` quando:

- o teste é perfeitamente balanceado por classe;
- todas as classes presentes têm o mesmo suporte;
- a métrica é multiclass single-label.

No golden AppClassNet, o teste validado tem 500 amostras por classe. Portanto, `BalancedAccuracy` próximo de `Accuracy` não prova bug.

**Risco:** **baixo** se o teste está balanceado; **médio** se classes somem e o usuário espera domínio 0..199.

## WeightedF1 vs MacroF1

`WeightedF1` pode ser igual ou muito próximo de `MacroF1` quando o teste é balanceado por classe, porque os pesos por classe são iguais. No golden AppClassNet, 500 amostras por classe tornam isso esperado.

Se as classes sintéticas em TR-TS estiverem balanceadas, `WeightedF1 ~= MacroF1` também pode ser normal. Se o suporte é desbalanceado e as métricas ainda são idênticas, isso deve ser investigado.

**Risco:** **baixo** quando suporte balanceado; **médio** sem relatório de suporte por classe.

## Validação manual por matriz de confusão

Para suportar 200 classes e auditar o sklearn, a validação manual proposta deve usar uma matriz `C` de shape `(200, 200)`, com linhas `y_true` e colunas `y_pred`.

### Fórmulas

```python
labels = np.arange(200)
C = confusion_matrix(y_true, y_pred, labels=labels)

support = C.sum(axis=1)
predicted = C.sum(axis=0)
tp = np.diag(C)
N = C.sum()

accuracy = tp.sum() / N

precision_i = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted != 0)
recall_i = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
f1_i = np.divide(
    2 * precision_i * recall_i,
    precision_i + recall_i,
    out=np.zeros_like(tp, dtype=float),
    where=(precision_i + recall_i) != 0,
)

macro_precision = precision_i.mean()
macro_recall = recall_i.mean()
macro_f1 = f1_i.mean()

weighted_precision = np.average(precision_i, weights=support) if support.sum() else 0.0
weighted_recall = np.average(recall_i, weights=support) if support.sum() else 0.0
weighted_f1 = np.average(f1_i, weights=support) if support.sum() else 0.0

# Para AppClassNet golden, support deve ser >0 para todas as 200 classes.
balanced_accuracy = recall_i.mean()
```

### Comparação com sklearn

Para domínio fixo:

```python
sk_accuracy = accuracy_score(y_true, y_pred)
sk_macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
sk_macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
sk_macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
sk_weighted_precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
sk_weighted_recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
sk_weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
```

Tolerância recomendada:

```text
abs(manual_metric - sklearn_metric) <= 1e-12
```

Para `balanced_accuracy_score`, o sklearn não recebe `labels`; quando todas as 200 classes estão presentes em `y_true`, deve coincidir com `recall_i.mean()` dentro da tolerância. Se alguma classe está ausente em `y_true`, a validação deve marcar a métrica como não comparável ao domínio fixo 200 sem uma regra explícita de tratamento de suporte zero.

### Saídas diagnósticas recomendadas

Para cada avaliação TR-TR, TR-TS e TS-TR em AppClassNet:

```json
{
  "num_classes": 200,
  "labels_domain": [0, 199],
  "y_true_shape": [100000],
  "y_pred_shape": [100000],
  "y_true_classes_present": 200,
  "y_pred_classes_present": 200,
  "missing_y_true_classes": [],
  "missing_y_pred_classes": [],
  "manual_confusion_validation": {
    "sklearn_match": true,
    "tolerance": 1e-12
  }
}
```

Essas chaves não devem substituir métricas legadas; devem ser diagnóstico adicional para detectar perda de domínio, shape errado ou nível aleatório.

## Regra automática proposta: nível aleatório

Adicionar diagnóstico, não reprovação automática:

```python
chance_level_suspected = (
    int(num_classes) == 200
    and accuracy is not None
    and 0.004 <= float(accuracy) <= 0.006
)
```

Quando verdadeiro, salvar:

```json
{
  "chance_level_suspected": true,
  "chance_level_reference": 0.005,
  "reason": "Accuracy near 1/200 for AppClassNet top-200; this does not prove a bug, but should trigger diagnostics."
}
```

Diagnósticos recomendados junto da regra:

- distribuição de `y_true`;
- distribuição de `y_pred`;
- número de classes previstas;
- classe majoritária prevista e percentual;
- classes ausentes em `y_true`;
- classes ausentes em `y_pred`;
- matriz de confusão 200x200 ou resumo por classe;
- `MacroRecall` e `BalancedAccuracy` com `labels=np.arange(200)` quando aplicável.

## Lacunas de teste

- **[alto]** Falta teste de métricas com `num_classes=200` e `labels=np.arange(200)`.
- **[alto]** Falta teste onde uma classe de `0..199` está ausente em `y_true`, para mostrar diferença entre sklearn default e domínio fixo.
- **[médio]** Falta teste manual por matriz de confusão comparando todas as métricas com sklearn.
- **[médio]** Falta teste de `chance_level_suspected` para Accuracy em `[0.004, 0.006]`.
- **[médio]** Falta teste garantindo que Summary registre `n_valid_folds` ou equivalente.

## Conclusão

A implementação atual calcula Accuracy, MacroPrecision, MacroRecall, MacroF1, WeightedPrecision, WeightedRecall, WeightedF1 e BalancedAccuracy de forma correta para o conjunto de labels observado por sklearn. Ela também evita zeros falsos em métricas preditivas não executadas, usando `"not_applicable"` e agregando apenas valores numéricos.

Para AppClassNet top-200, o principal risco é diagnóstico: as métricas não fixam explicitamente o domínio `0..199`. Em execuções onde classes desaparecem de `y_true` ou só aparecem em `y_pred`, as médias macro podem não refletir o contrato de 200 classes. A correção futura deve ser pequena e compatível: adicionar métricas/diagnósticos com `labels=np.arange(num_classes)` e `chance_level_suspected`, sem substituir silenciosamente as métricas legadas.
