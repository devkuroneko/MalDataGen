# TR-TS Audit - AppClassNet Top-200

Data da auditoria: 2026-07-14

Escopo: auditoria exclusiva de TR-TS. Nenhum codigo de execucao foi alterado.

Definicao esperada: treinar classificador com dados reais, testar com dados sinteticos, manter labels sinteticos coerentes com as classes solicitadas e comparar real/sintetico somente no mesmo espaco numerico.

## Conclusao executiva

O TR-TS atual mede a capacidade de um classificador treinado no conjunto real de avaliacao reconhecer os labels atribuidos aos sinteticos. Ele nao treina no `x_training_real`; treina em `x_evaluation_real`.

Isso muda a semantica esperada em AppClassNet:

- Em cross-validation, TR-TS treina no fold de validacao, nao no fold de treino.
- Em `split_mode=provided`, TR-TS treina em `valid` quando `valid` existe, nao em `train`.
- Em batches, ocorre a mesma escolha: o classificador batch e treinado em `x_evaluation_real`.

O pipeline possui guardas importantes de labels e espaco numerico antes da avaliacao, mas TR-TS ainda nao registra um bloco pre-TR-TS completo com range por feature, contagem por classe, classes ausentes/extras, taxa de NaN, taxa fora do range real e distribuicao das predicoes.

## Achados classificados

| Severidade | Achado | Evidencia |
|---|---|---|
| Critico | TR-TS treina no real de avaliacao (`x_evaluation_real`), nao no real de treino (`x_training_real`). | `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas ~96-111 em batches e ~188-193 em normal. |
| Alto | Em AppClassNet `split_mode=provided`, `x_evaluation_real` e `valid` quando `valid` e `test` existem; portanto TR-TS treina no split de validacao. | `Engine/Evaluation/CrossValidation.py`, funcoes `_build_provided_split_folds()` e `_create_fold()`, linhas ~357-380 e ~329-346. |
| Alto | `synthetic_test_samples_per_class` e efetivo em batches via `test_samples_per_class`, mas nao limita o TR-TS normal materializado. | `Engine/Arguments/Arguments.py`, funcao `_configure_classifier_arguments()`, linhas ~128-132; `Engine/Evaluation/TrTs.py`, normal linhas ~146-203 nao usa `test_samples_per_class`; batches linhas ~112-117 usa o cap. |
| Alto | Para `model_type="copy"`, os sinteticos podem vir diretamente do `x_real_samples` passado para geracao, que no loop principal e `x_evaluation_real`; assim o teste sintetico de TR-TS pode conter copias do mesmo conjunto real usado para treinar o classificador TR-TS. | `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linhas ~472-480; funcao `synthesize_data()`, linhas ~906-912. |
| Medio | Labels sinteticos sao auditados antes da avaliacao e falham para dominio incompleto quando `number_classes=200`, mas TR-TS em si confia no objeto sintetico recebido. | `main.py`, funcao `synthesize_data()`, linhas ~965-974; `Engine/DataIO/SyntheticLabelAudit.py`, classe `SyntheticLabelGenerationAudit`, linhas ~139-167 e funcao `audit_synthetic_label_generation()`, linhas ~169-191. |
| Medio | Real e sintetico sao validados por `ScaleGuard`, inclusive NaN/inf no modo normal e `data_space`; porem nao ha criterio TR-TS que falhe por percentual de sinteticos fora do range real. | `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas ~74-83 e ~168-186; `Engine/Preprocessing/FeatureTransformManager.py`, classe `ScaleGuard`, funcoes `describe()`, `validate_finite()`, `validate_compatible_metadata()` e `validate_before_evaluation()`, linhas ~503-555. |
| Medio | Batches fazem cap por classe pegando os primeiros lotes/linhas ate a cota, nao uma amostragem estratificada aleatoria dos sinteticos disponiveis. | `Engine/Classifiers/BatchClassifiers.py`, funcao `predict_synthetic_batches()`, linhas ~254-274. |
| Baixo | O sanity check existente calcula ranges, out-of-range e distribuicao de predicoes, mas e um relatorio auxiliar, nao um gate obrigatorio dentro de TR-TS. | `Engine/DataIO/SyntheticSanityChecks.py`, classe `SyntheticSanityChecker`, linhas ~80-106, ~152-239 e ~277-303. |

## Fluxo TR-TS atual

### Normal

- Arquivo: `Engine/Evaluation/TrTs.py`
- Classe: `TrTs`
- Funcao: `evaluation_TR_TS()`
- Linhas aproximadas: ~138-203

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
    label_predicted = classifier.predict(synthetic_array)
    get_task_metrics(labels, label_predicted, "TR-TS", classifier_name, fold_number + 1)
```

### Batches

- Arquivo: `Engine/Evaluation/TrTs.py`
- Classe: `TrTs`
- Funcao: `evaluation_TR_TS()`
- Linhas aproximadas: ~72-136

Pseudocodigo fiel:

```python
ScaleGuard.validate_compatible_metadata(
    ScaleGuard.describe(dictionary_data["x_evaluation_real"], data_space="source"),
    synthetic_manifest_metadata,
    context="TR-TS",
)

train_labels = labels_to_1d_integer(dictionary_data["y_evaluation_real"])
train_batches = iter_array_batches(dictionary_data["x_evaluation_real"], train_labels, batch_size)

classifier, metadata = train_batch_classifier(
    eval_classifier,
    train_batches,
    configured_number_classes,
    args,
)

labels, predictions, evaluation_time = predict_synthetic_batches(
    classifier,
    synthetic_data,
    max_samples_per_class=args.test_samples_per_class,
)

get_task_metrics(labels, predictions, "TR-TS", classifier_name, fold_number + 1)
```

## Respostas aos pontos investigados

### 1. Qual conjunto real treina o classificador

TR-TS treina em `dictionary_data["x_evaluation_real"]` e `dictionary_data["y_evaluation_real"]`.

- Normal: `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas ~188-193.
- Batches: `Engine/Evaluation/TrTs.py`, linhas ~96-111.

Origem desse conjunto:

- Cross-validation: fold interno de avaliacao criado por `StratifiedKFold`/`KFold`. Evidencia: `Engine/Evaluation/CrossValidation.py`, classe `StratifiedData`, funcao `wrapper()`, linhas ~494-564.
- Provided: `valid` se existe; senao `test`. Evidencia: `_build_provided_split_folds()`, linhas ~357-380.

### 2. Qual conjunto sintetico e usado no teste

O teste usa o `evaluation_synthetic` gerado uma vez por fold em `main.py`.

- Evidencia: `main.py`, funcao `run_experiments()`, linhas ~472-500.
- Normal: TR-TS itera `synthetic_data.items()` e materializa tudo em `synthetic_array`. Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~146-168.
- Batches: TR-TS le o objeto sintetico via `predict_synthetic_batches()` e `iter_synthetic_labeled_batches()`. Evidencia: `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas ~112-117; `Engine/Classifiers/BatchClassifiers.py`, funcoes `iter_synthetic_labeled_batches()` e `predict_synthetic_batches()`, linhas ~86-97 e ~254-274.

### 3. Se os sinteticos sao ineditos ou foram usados no treinamento do gerador

Depende do gerador.

Para os geradores treinados, o treinamento ocorre antes da geracao usando `x_training_for_generator` e `y_training_real`.

- Evidencia: `main.py`, funcao `run_experiments()`, linhas ~432-468.

Os sinteticos avaliados sao gerados depois, em `synthesize_data()`.

- Evidencia: `main.py`, linhas ~472-480 e funcao `synthesize_data()`, linhas ~807-847.

Risco especial: `model_type="copy"` usa `x_real_samples` e `y_real_samples` passados para `synthesize_data()`. No loop principal esses argumentos sao `x_evaluation_for_generator` e `y_evaluation_real`, ou seja, o mesmo split real que treina o classificador TR-TS.

- Evidencia: `main.py`, funcao `run_experiments()`, linhas ~472-480.
- Evidencia: `main.py`, funcao `synthesize_data()`, linhas ~906-912.

Conclusao: para `copy`, os sinteticos podem nao ser ineditos. Para os demais modelos, a auditoria estatica nao prova ineditismo; apenas confirma que a geracao ocorre apos treino do gerador.

### 4. Se `synthetic_test_samples_per_class` e realmente respeitado

Em batches: sim, como cap por classe durante a predicao sintetica.

- Evidencia: `Engine/Arguments/Arguments.py`, funcao `_configure_classifier_arguments()`, linhas ~128-132, copia `synthetic_test_samples_per_class` para `test_samples_per_class`.
- Evidencia: `Engine/Evaluation/TrTs.py`, linha ~116 passa `test_samples_per_class` para `predict_synthetic_batches()`.
- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, funcao `predict_synthetic_batches()`, linhas ~254-274, limita por classe.

Em normal: nao ha evidencia de que seja respeitado por TR-TS. O modo normal materializa todos os sinteticos recebidos e nao consulta `test_samples_per_class`.

- Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~146-203.

O numero de sinteticos gerados no modo normal e controlado por `SamplePlan`/`number_samples_per_class`, nao por `synthetic_test_samples_per_class`.

- Evidencia: `main.py`, funcao `_build_generation_metadata()`, linhas ~560-581.
- Evidencia: `Engine/DataIO/SamplePlanner.py`, funcao `build_sample_plan_from_args()`, linhas ~27-89.

### 5. Se os sinteticos sao selecionados estratificadamente

Normal: nao ha selecao; todos os sinteticos do dicionario sao usados. A estratificacao depende do plano de geracao.

- Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~146-168.

Batches: ha cap por classe, mas nao uma amostragem estratificada aleatoria. O codigo percorre batches e pega as primeiras linhas de cada classe ate atingir a cota.

- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, funcao `predict_synthetic_batches()`, linhas ~254-274.

Plano de geracao:

- `balanced_per_class` cria a mesma cota para cada classe no dominio. Evidencia: `Engine/DataIO/SamplePlanner.py`, linhas ~56-67 e `_class_domain()` linhas ~146-149.
- `total_rows` distribui total de forma balanceada. Evidencia: linhas ~69-74 e ~152-162.
- `match_train_distribution` aloca proporcionalmente aos labels de entrada passados para o plano. Evidencia: linhas ~76-87.

Conclusao: a selecao efetiva de sinteticos para teste nao e uma amostragem estratificada aleatoria em TR-TS. No normal, usa tudo; no batches, usa cap sequencial por classe quando configurado.

### 6. Se todas as 200 classes estao presentes

Para AppClassNet com `number_classes=200`, a auditoria de labels sinteticos falha se faltar alguma classe gerada com contagem positiva.

- Evidencia: `Engine/DataIO/SyntheticLabelAudit.py`, classe `SyntheticLabelGenerationAudit`, metodo `_validate_domain()`, linhas ~139-167.
- Evidencia: chamada em `main.py`, funcao `synthesize_data()`, linhas ~965-974.

Porem TR-TS em si nao revalida classes presentes antes de calcular metricas.

- Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~146-203 e ~112-135.

### 7. Se os labels sinteticos correspondem a classe solicitada na geracao

No modo normal, o label usado na avaliacao e a chave `label_class` do dicionario `synthetic_data`.

- Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~146-154.

No modo batches, `iter_synthetic_labeled_batches()` cria labels a partir do `class_label` do manifesto.

- Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~86-97.

Antes disso, a auditoria de geracao registra `requested_class` e `saved_label` e falha se divergirem.

- Evidencia: `Engine/DataIO/SyntheticLabelAudit.py`, metodo `record()`, linhas ~47-71, e `audit_synthetic_label_generation()`, linhas ~169-191.
- Evidencia batches incremental: `main.py`, funcao `_synthesize_data_incremental()`, linhas ~1252-1268.

### 8. Se o mapping de labels e unico para todos os splits

No loader NPY, o mapping e construido a partir dos labels coletados de train, valid e test juntos.

- Evidencia: `Engine/DataIO/NpyXYLoader.py`, classe `NpyXYLoader`, funcao `load()`, linhas ~87-103.
- Evidencia: `_collect_labels()` e `_build_label_mapping()` sao usados antes de retornar o `DatasetBundle`; `_build_label_mapping()` aparece em linhas ~213-223.

O mapping e guardado em `DatasetBundle.metadata`.

- Evidencia: `Engine/DataIO/NpyXYLoader.py`, linhas ~126-131.

O fold criado valida labels zero-based.

- Evidencia: `Engine/Evaluation/CrossValidation.py`, funcao `_create_fold()`, linhas ~332-336.

### 9. Se real e sintetico estao no mesmo `data_space`

Ha validacao antes de TR-TS.

- Normal: `ScaleGuard.validate_before_evaluation()` e chamado em `Engine/Evaluation/TrTs.py`, linhas ~180-186.
- Batches: `ScaleGuard.validate_compatible_metadata()` e chamado em `Engine/Evaluation/TrTs.py`, linhas ~74-83.
- Guard adicional antes de chamar TR-TS/TS-TR: `main.py`, funcao `run_synthetic_evaluation_modes()`, linhas ~110-114, chama `_guard_current_evaluation_space()`; implementacao em `main.py`, linhas ~616-639.

`ScaleGuard` falha quando `data_space` real e sintetico divergem.

- Evidencia: `Engine/Preprocessing/FeatureTransformManager.py`, classe `ScaleGuard`, funcao `validate_compatible_metadata()`, linhas ~520-541.

### 10. Se `inverse_transform` foi aplicado quando necessario

Para AppClassNet, a politica default e preservar e avaliar em `source`; se o gerador usar transformacao, `inverse_transform_synthetic=True` deve trazer o sintetico de volta para `source`.

- Evidencia default AppClassNet: `Engine/Preprocessing/FeatureTransformManager.py`, classe `FeatureTransformPolicy`, funcao `for_profile()`, linhas ~155-168.
- Evidencia em argumentos: `Engine/Arguments/Arguments.py`, funcao `_normalize_preprocessing_arguments()`, linhas ~159-175, força `inverse_transform_synthetic=True` para AppClassNet quando nao setado.
- Normal: `main.py`, funcao `synthesize_data()`, linhas ~951-963, chama `transform_synthetic_collection_to_source()`.
- Batches incremental: `main.py`, funcao `_synthesize_data_incremental()`, linhas ~1260-1262, chama `inverse_synthetic_batch()`.
- Implementacao: `Engine/Preprocessing/FeatureTransformManager.py`, classe `ModelInputAdapter`, funcoes `inverse_generator_output()`, `transform_synthetic_collection_to_source()` e `inverse_synthetic_batch()`, linhas ~604-645.

Se `inverse_transform_synthetic` nao for aplicado e o sintetico permanecer em `generator`, `ScaleGuard` deve falhar antes de TR-TS.

- Evidencia: `ScaleGuard.validate_compatible_metadata()`, linhas ~528-541.

### 11. Se sinteticos estao em `[0,1]` enquanto reais estao em `[-0.5,0.5]`

O codigo possui mecanismo para detectar espaco incompatível por `data_space` e `transform_id`, mas nao ha uma regra TR-TS especifica que compare `[0,1]` contra `[-0.5,0.5]` e falhe por range.

- Evidencia de metadata: `Engine/Preprocessing/FeatureTransformManager.py`, `ScaleGuard.describe()`, linhas ~503-510.
- Evidencia de mismatch por espaco: `ScaleGuard.validate_compatible_metadata()`, linhas ~528-541.
- Evidencia de ranges auxiliares: `Engine/DataIO/SyntheticSanityChecks.py`, metodo `_scale_report()`, linhas ~224-239.

Conclusao: se o sintetico estiver marcado como `generator` e o real como `source`, TR-TS falha. Se ambos estiverem marcados como `source` por metadado incorreto, a faixa `[0,1]` versus `[-0.5,0.5]` viraria warning/sanity report, nao necessariamente falha TR-TS.

### 12. Se existem NaN, inf, clipping ou arredondamento

NaN/inf:

- Normal: `ScaleGuard.validate_before_evaluation()` chama `validate_finite()` em real e sintetico. Evidencia: `Engine/Preprocessing/FeatureTransformManager.py`, linhas ~513-517 e ~544-549.
- Batches: TR-TS batches valida apenas metadados do manifesto contra real; nao percorre todos os batches para NaN/inf antes de predizer. Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~74-83.

Clipping/arredondamento:

- Nao foi encontrada evidencia de clipping ou arredondamento dentro de TR-TS.
- Ha conversoes para `float32`: normal em `synthetic_array = numpy.asarray(data, dtype=numpy.float32)` (`TrTs.py`, linha ~168); batches em `iter_synthetic_labeled_batches()` (`BatchClassifiers.py`, linhas ~92-97) e `iter_array_batches()` (`BatchClassifiers.py`, linhas ~79-83).

### 13. Se o classificador foi previamente treinado em outro experimento

Nao. TR-TS cria/treina classificador dentro da funcao a cada avaliacao.

- Normal: `get_trained_classifiers()` e chamado dentro de `evaluation_TR_TS()`. Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~188-193.
- `get_trained_classifiers()` chama `classifier_model.get_model(...)` para cada classificador. Evidencia: `Engine/Classifiers/Classifiers.py`, classe `Classifiers`, funcao `get_trained_classifiers()`, linhas ~132-151.
- Batches: `train_batch_classifier()` chama `make_batch_classifier()` a cada execucao. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~170-174 e ~39-76.

Ressalva: as instancias wrapper em `self._dictionary_classifiers` sao reutilizadas como objetos de fabrica, mas o estimador retornado por `get_model(...)` e treinado na chamada atual. Esta auditoria nao encontrou cache de predicoes ou estimador pre-treinado dentro de TR-TS.

### 14. Se ha reutilizacao indevida de predicoes

Nao ha evidencia de reutilizacao de predicoes. As predicoes sao calculadas localmente em cada chamada.

- Normal: `classifier_instances.predict(synthetic_array)` em `Engine/Evaluation/TrTs.py`, linha ~198.
- Batches: `predict_synthetic_batches()` chama `classifier.predict(x_batch)` para cada batch. Evidencia: `Engine/Classifiers/BatchClassifiers.py`, linhas ~254-274.

Ha reutilizacao do mesmo objeto sintetico entre TR-TS e TS-TR no mesmo fold, mas nao das predicoes.

- Evidencia: `main.py`, funcao `run_synthetic_evaluation_modes()`, linhas ~110-128.
- Evidencia: `main.py`, funcao `run_experiments()`, linhas ~472-500.

### 15. Se o pipeline testa no proprio conjunto usado para selecionar parametros

TR-TS treina o classificador no conjunto de avaliacao real. Se esse conjunto for `valid`, ele tambem e, semanticamente, o conjunto normalmente usado para selecao de parametros no workflow AppClassNet.

- Evidencia: `Engine/Evaluation/TrTs.py`, linhas ~188-193 e ~96-111.
- Evidencia: `Engine/Evaluation/CrossValidation.py`, `_build_provided_split_folds()`, linhas ~357-380.

O teste de TR-TS em si e sintetico. Porem, as cotas de geracao sao construidas a partir dos labels reais passados como `y_real_samples`, que no loop principal sao `y_evaluation_real`.

- Evidencia: `main.py`, funcao `run_experiments()`, linhas ~472-480.
- Evidencia: `main.py`, funcao `_build_generation_metadata()`, linhas ~560-581.

Conclusao: nao e vazamento classico de treinar e testar o classificador no mesmo X real, porque o teste e sintetico. Mas a avaliacao usa o split real de avaliacao tanto para treinar o classificador TR-TS quanto para orientar o plano/labels de geracao. Para `copy`, isso pode virar copia direta do mesmo real.

### 16. Se TR-TS esta medindo fidelidade sintetica ou outra propriedade

TR-TS atual mede uma propriedade condicional e operacional: se um classificador treinado no real de avaliacao prediz os labels atribuídos aos sinteticos.

Ele nao mede sozinho:

- fidelidade distribucional completa;
- ineditismo;
- cobertura por classe se o audit anterior for ignorado;
- equivalencia com desempenho real-real train->test;
- qualidade sintetica robusta quando o classificador e fraco ou treinado no split errado.

Os sanity checks auxiliares medem parte da fidelidade por ranges, distancia entre medias de classe e predicoes de um DecisionTree real->sintetico.

- Evidencia: `Engine/DataIO/SyntheticSanityChecks.py`, linhas ~80-106, ~224-239 e ~277-303.

## Verificacoes que deveriam ocorrer imediatamente antes de TR-TS

Estas verificacoes deveriam ser registradas por fold, avaliacao e classificador antes de chamar `get_task_metrics()`. O objetivo e impedir que um resultado TR-TS aparente qualidade sintetica quando, na pratica, os dados estao em espacos diferentes, classes sumiram, labels foram trocados ou o classificador colapsou em uma classe majoritaria.

| Verificacao | Comportamento esperado | Evidencia atual / lacuna |
|---|---|---|
| Range real por feature | Registrar `min/max` de `x_evaluation_real` por feature e range global. | Parcial em `ScaleGuard.describe()` e sanity checks; nao salvo no bloco TR-TS. `FeatureTransformManager.py`, classe `ScaleGuard`, funcao `describe()`, linhas ~503-510; `SyntheticSanityChecks.py`, classe `SyntheticSanityChecker`, funcao `_scale_report()`, linhas ~224-239. |
| Range sintetico por feature | Registrar `min/max` por feature dos sinteticos efetivamente testados, depois de qualquer cap. | Normal calcula stats em `ScaleGuard.validate_before_evaluation()`; batches nao percorre tudo antes de predizer. `TrTs.py`, linhas ~180-186 e ~112-117. |
| `transform_id` | Registrar real e sintetico; exigir match quando `data_space != source`. | `ScaleGuard.validate_compatible_metadata()`, linhas ~520-541. |
| `data_space` | Exigir real e sintetico no mesmo espaco e persistir isso no resultado TR-TS. | Ja existe em `ScaleGuard`, mas nao aparece como precheck TR-TS persistido. |
| Contagem por classe | Registrar contagem real usada para treino e sintetica usada para teste. | Batches registra metadata em `record_batch_classifier_metadata()`; normal nao registra bloco equivalente. `TrTs.py`, linhas ~118-125. |
| Classes ausentes | Falhar ou marcar `not_applicable` se AppClassNet 200 classes nao aparecerem nos sinteticos efetivamente avaliados. | Auditoria de labels falha antes para geracao normal, mas TR-TS nao revalida depois de cap por `test_samples_per_class`. `SyntheticLabelAudit.py`, linhas ~139-167; `BatchClassifiers.py`, linhas ~260-267. |
| Classes extras | Falhar se labels sinteticos fora de `0..199`. | Auditoria de labels cobre antes; TR-TS nao revalida no ponto de avaliacao. |
| Taxa de NaN | Registrar percentual de NaN real e sintetico efetivo. | Normal falha em NaN/inf via `ScaleGuard`; batches nao checa valores de todos os batches antes. `FeatureTransformManager.py`, linhas ~513-517 e ~544-549. |
| Taxa de valores fora do range real | Registrar percentual por feature e global dos sinteticos fora do range real observado. | Sanity check calcula, mas apenas como relatorio auxiliar/warning. `SyntheticSanityChecks.py`, `_scale_report()`, linhas ~224-239. |
| Distribuicao das predicoes | Registrar contagem de predicoes por classe para cada classificador real->sintetico. | Sanity check faz para DecisionTree auxiliar; TR-TS nao registra para cada classificador. |
| Percentual previsto na classe majoritaria | Registrar `max(pred_count)/n`; alertar colapso acima de limiar, por exemplo 90%. | Sanity check registra `dominant_predicted_class_fraction`; TR-TS nao registra para cada classificador. `SyntheticSanityChecks.py`, `_classifier_report()`, linhas ~277-303. |

Pseudocodigo de especificacao futura:

```python
precheck = build_tr_ts_precheck(
    real_x=dictionary_data["x_evaluation_real"],
    real_y=dictionary_data["y_evaluation_real"],
    synthetic_x=synthetic_array_or_batch_stream,
    synthetic_y=synthetic_labels,
    real_metadata=real_metadata,
    synthetic_metadata=synthetic_metadata,
    expected_num_classes=200,
)

assert precheck.real.data_space == precheck.synthetic.data_space
if precheck.real.data_space != "source":
    assert precheck.real.transform_id == precheck.synthetic.transform_id

assert precheck.synthetic.missing_classes == []
assert precheck.synthetic.extra_classes == []
assert precheck.synthetic.nan_rate == 0.0
assert precheck.synthetic.inf_rate == 0.0
record(precheck.synthetic.outside_real_range_percent_by_feature)
record(precheck.prediction_distribution_by_classifier)
record(precheck.majority_prediction_percent_by_classifier)
```

## Divergencia entre nome e implementacao

Nome esperado: TR-TS = treinar classificador em dados reais e testar em dados sinteticos.

Implementacao real:

- O tipo dos dados esta correto: real -> sintetico.
- O conjunto real escolhido e inesperado: `x_evaluation_real`, nao `x_training_real`.
- No AppClassNet provided, isso significa treinar em `valid`.
- O resultado TR-TS mede reconhecibilidade dos sinteticos por um classificador treinado no split de avaliacao, nao fidelidade sintetica geral nem equivalencia com train->test real-real.

## Decisao recomendada

Antes de alterar codigo, decidir formalmente se TR-TS deve treinar em `x_training_real` ou se a semantica atual `x_evaluation_real -> synthetic` deve ser preservada por compatibilidade. Para AppClassNet top-200, a definicao fornecida sugere `x_training_real` como treino real, mantendo `valid`/`test` fora do treino do classificador TR-TS.
