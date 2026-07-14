# Pipeline Limitations And Improvements

Data da consolidacao: 2026-07-14.

Relatorio arquitetural consolidado a partir dos documentos em `docs/audits/`.

Escopo: consolidar limitacoes atuais do MalDataGen para AppClassNet top-200 e propor um roadmap incremental. Nenhuma correcao foi implementada nesta etapa.

Relatorios-base:

- `FULL_PIPELINE_AUDIT.md`
- `EVALUATION_FLOW_AUDIT.md`
- `TR_TR_AUDIT.md`
- `TR_TS_AUDIT.md`
- `TS_TR_AUDIT.md`
- `LABEL_AND_CONDITIONING_AUDIT.md`
- `PREPROCESSING_AND_DATA_SPACE_AUDIT.md`
- `BATCH_AND_MEMORY_AUDIT.md`
- `METRICS_AUDIT.md`
- `SYNTHETIC_GENERATION_AUDIT.md`

## 1. Sumario executivo

O projeto ja tem uma base importante para AppClassNet top-200: loaders NPY preservam `source`, o runner AppClassNet passa `feature_transform=preserve`, o one-hot central suporta `number_classes=200`, os sinteticos tem auditoria de labels e `ScaleGuard` tenta impedir comparacao entre espacos numericos incompatíveis.

Os riscos arquiteturais principais estao na semantica das avaliacoes e na divergencia entre os caminhos `normal`, `batches` e `baseline_real_only`:

1. **TR-TR golden validado nao e o TR-TR do pipeline principal.** O golden AppClassNet usa `run_appclassnet_top200.py::run_real_real_baseline()` com train/test oficiais. O TR-TR em `Engine/Evaluation/TrTr.py::TrTr.evaluation_TR_TR()` existe, mas a chamada esta comentada em `main.py::SynDataGen.run_experiments()` linha aproximada 502.
2. **TR-TS treina no real de avaliacao, nao no real de treino.** `Engine/Evaluation/TrTs.py::TrTs.evaluation_TR_TS()` usa `x_evaluation_real` no modo normal e no modo batches. Em `split_mode=provided`, isso costuma significar `valid`.
3. **TS-TR normal prediz em todo real de avaliacao, mas calcula metricas truncadas por `total_samples` sintetico.** Evidencia em `Engine/Evaluation/TsTr.py::TsTr.evaluation_TS_TR()` linhas aproximadas 207-216.
4. **`1-Fold` nem sempre significa fold de cross-validation.** Em `split_mode=provided`, `1-Fold` e o split fornecido `train -> valid/test`; no runner AppClassNet normal, `1-Fold` e `2-Fold` sao folds internos sobre CSV materializado.
5. **Modo batches reduz memoria, mas nao e streaming end-to-end.** Ha mmap opcional, selecao/materializacao de subsets, classificadores subset que chamam `fit` tradicional e treino gerativo sobre arrays ja materializados.
6. **Defaults legados binarios ainda existem.** Varios geradores ainda tem `number_classes=2`, `sigmoid` e perdas binarias como default. O runner AppClassNet sobrescreve parte disso, mas execucoes diretas podem voltar ao comportamento antigo.
7. **Metricas multiclass usam sklearn corretamente para labels observadas, mas nao fixam explicitamente dominio `0..199`.** Se classes desaparecem, Macro/Weighted podem parecer validas sem refletir o contrato de 200 classes.

Conclusao: antes de mudar modelos ou otimizar memoria, o projeto precisa de uma Fase 0 de testes de regressao que congele o baseline real-real, o dominio de labels, o espaco `source` e o comportamento atual das avaliacoes. Depois, as correcoes devem ser pequenas, compatíveis e orientadas por testes.

## 2. Arquitetura atual

Fluxo geral:

```text
CLI
  -> run_appclassnet_top200.py ou main.py
  -> argumentos e politica de transformacao
  -> carregamento CSV ou NPY XY
  -> DatasetBundle / SplitData / DatasetSchema
  -> selecao estratificada opcional
  -> CrossValidation ou split provided
  -> ModelInputAdapter / FeatureTransformManager
  -> treino gerativo
  -> geracao sintetica
  -> auditorias de labels/escala/sanity
  -> TR-TS e TS-TR no loop sintetico
  -> metricas e Results.json
```

Componentes principais:

- **CLI AppClassNet:** `run_appclassnet_top200.py`, funcoes `build_main_command()`, `build_batch_main_command()`, `run_real_real_baseline()`, `choose_campaigns()`.
- **Entrada CSV legada:** `Engine/DataIO/CSVLoader.py::CSVDataProcessor`.
- **Entrada NPY XY:** `Engine/DataIO/NpyXYLoader.py`, `Engine/DataIO/DatasetContracts.py`.
- **Splits/folds:** `Engine/Evaluation/CrossValidation.py::StratifiedData`.
- **Pre-processamento:** `Engine/Preprocessing/FeatureTransformManager.py`, `ModelInputAdapter`, `ScaleGuard`.
- **Geradores:** `Engine/Models/GenerativeModels.py`, `Engine/Algorithms/*`.
- **Persistencia sintetica:** `CSVDataProcessor.save_csv()` e `Engine/DataIO/SyntheticBatchIO.py::SyntheticBatchWriter`.
- **Avaliacao:** `Engine/Evaluation/TrTr.py`, `TrTs.py`, `TsTr.py`.
- **Metricas:** `Engine/Metrics/Metrics.py`.
- **Classificadores batch:** `Engine/Classifiers/BatchClassifiers.py`.

## 3. Funcionamento atual do TR-TR

Ha dois TR-TR semanticamente diferentes:

1. **Golden AppClassNet:** `run_appclassnet_top200.py::run_real_real_baseline()`.
   - Treino: `train_x/train_y` oficiais.
   - Teste: `test_x/test_y` oficiais.
   - Classificador default: `DecisionTreeClassifier(random_state=0)`.
   - Cotas validadas: 1.000 amostras por classe no treino, 500 por classe no teste.
   - Espaco default: `source`.
   - Resultado esperado: Accuracy ~0.68673, Macro-F1 ~0.68630.

2. **TR-TR do pipeline principal:** `Engine/Evaluation/TrTr.py::TrTr.evaluation_TR_TR()`.
   - Treino: `dictionary_data["x_training_real"]`.
   - Teste: `dictionary_data["x_evaluation_real"]`.
   - Em cross-validation: folds internos.
   - Em `split_mode=provided`: `valid` se existir, senao `test`.
   - Problema: a chamada esta comentada em `main.py::SynDataGen.run_experiments()` linha aproximada 502.

Conclusao: o teste de regressao AppClassNet deve usar `baseline_real_only` ate o TR-TR principal ser reativado, especificado e validado.

## 4. Funcionamento atual do TR-TS

Definicao esperada do projeto: treinar classificador em real e testar em sintetico.

Implementacao atual:

- Arquivo: `Engine/Evaluation/TrTs.py`.
- Classe: `TrTs`.
- Funcao: `evaluation_TR_TS()`.
- Treino atual: `x_evaluation_real`, nao `x_training_real`.
- Teste atual: sintetico gerado no fold.
- Modo normal: materializa sintetico e avalia.
- Modo batches: treina classificador batch em `x_evaluation_real` e prediz sobre batches sinteticos.
- Em `split_mode=provided`: `x_evaluation_real` tende a ser `valid`, porque `CrossValidation._build_provided_split_folds()` usa valid quando valid e test existem.

Impacto: TR-TS mede reconhecibilidade dos sinteticos por um classificador treinado no conjunto real de avaliacao. Isso diverge da leitura esperada "treino real oficial/fold de treino -> teste sintetico".

## 5. Funcionamento atual do TS-TR

Definicao esperada do projeto: treinar classificador em sintetico e testar em real independente.

Implementacao atual:

- Arquivo: `Engine/Evaluation/TsTr.py`.
- Classe: `TsTr`.
- Funcao: `evaluation_TS_TR()`.
- Treino: sintetico.
- Teste: `x_evaluation_real`.
- Modo normal: materializa sinteticos, treina classificador e prediz em todo `x_evaluation_real`, mas calcula metricas com `y_evaluation_real[:total_samples]` e `predictions[:total_samples]`.
- Modo batches: le manifestos sinteticos, usa classificadores batch; `decision_tree_subset` coleta subset estratificado e chama `fit`, enquanto `sgd`, `passive_aggressive`, `naive_bayes` e `mlp_small` usam `partial_fit`.

Impacto: no modo normal, se `total_samples` sintetico for menor que o real de avaliacao, o teste real efetivo e um prefixo truncado. Isso pode produzir resultado aparentemente valido sobre subconjunto nao documentado.

## 6. Diferencas entre comportamento esperado e real

| Tema | Esperado AppClassNet | Real atual | Severidade |
|---|---|---|---|
| TR-TR golden | train oficial -> test oficial | Apenas `baseline_real_only`; TR-TR principal nao roda | Critico |
| TR-TS | treino real de treino -> teste sintetico | treino em `x_evaluation_real` -> teste sintetico | Critico |
| TS-TR | treino sintetico -> todo teste real independente | normal trunca metricas por total sintetico | Critico |
| `1-Fold` | fold real somente quando CV | em provided significa split unico; no normal AppClassNet e CV sobre CSV | Alto |
| AppClassNet normal | usar splits oficiais | materializa CSV e usa cross-validation | Alto |
| AppClassNet batches | streaming equivalente | reduz memoria parcialmente e pode reduzir dataset | Alto |
| Labels | dominio fixo 0..199 | protegido por metadata/runner, mas defaults legados sao 2 | Alto |
| Escala | source preservado | default preserva, mas `--scaler` e `generator_transform` podem mudar espaco | Alto |
| Metricas | dominio 0..199 explicito | sklearn sobre labels observadas | Alto |
| Geracao | 200 condicoes claras | labels salvos sao solicitados, mesmo se gerador ignora condicao | Alto |

## 7. Limitacoes do modo normal

- AppClassNet normal materializa CSV e usa cross-validation sobre o split escolhido, nao os splits oficiais train/valid/test. Evidencia: `run_appclassnet_top200.py::build_main_command()` linhas aproximadas 1556-1608 e fluxo normal descrito em `FULL_PIPELINE_AUDIT.md`.
- Geracao normal mantem `self.data_generated` completo em memoria (`main.py::SynDataGen.synthesize_data()` linhas aproximadas 855-963).
- Salvamento CSV cria listas `labels` e `data` e depois um `DataFrame` completo (`CSVLoader.py::CSVDataProcessor.save_csv()` linhas aproximadas 391-424).
- TR-TS normal nao respeita universalmente `synthetic_test_samples_per_class`.
- TS-TR normal trunca metricas.
- One-hot global pode ser caro para datasets grandes.

## 8. Limitacoes do modo batches

- Nao e streaming end-to-end. `CrossValidation._apply_bundle_to_owner()` e `_create_fold()` podem materializar arrays; selecao por `split.X[indices]` usa indexing avancado.
- `--mmap_npy` so e passado pelo runner AppClassNet quando o usuario usa `--use_mmap` (`run_appclassnet_top200.py::build_batch_main_command()` linhas aproximadas 1697-1698).
- Treino gerativo em batches usa one-hot por batch, mas o X do fold ja esta selecionado/materializado.
- `DecisionTreeSubset`, `ExtraTreesSubset` e `RandomForestLight` nao sao incrementais; coletam subset e chamam `fit`.
- Latent diffusion rejeita `execution_mode=batches` (`GenerativeModels._training_latent_diffusion_model()` linhas aproximadas 1465-1469).
- TR-TR nao tem caminho batch equivalente.
- Batches podem alterar a distribuicao efetiva quando cotas e subsets sao usados.

## 9. Limitacoes dos classificadores

- O golden usa `DecisionTreeClassifier`; isso deve continuar como regressao principal.
- No modo batches, `decision_tree_subset` e default util para memoria, mas nao equivale a DecisionTree treinado em todos os dados.
- `SGDClassifier` existe, mas nao deve ser o unico indicador de qualidade sintetica.
- `classifier_transform` existe na politica, mas as avaliacoes principais treinam diretamente nos arrays recebidos; o argumento pode nao afetar os avaliadores como o usuario espera.
- Predicoes batch acumulam labels/predicoes em listas antes das metricas.

## 10. Limitacoes dos geradores

- Campanha `sf` e ambigua: sem `--full`, roda apenas demos `variational_demo` e `adversarial_demo`; com `--full`, roda a campanha maior.
- Defaults legados em varios modelos ainda sao binarios (`number_classes=2`, `sigmoid`, perdas binarias).
- `data_type != continuous` aciona `numpy.rint` em geradores neurais, destruindo features continuas AppClassNet.
- WGAN/WGAN-GP ignoram learning rates configurados no runner e usam Adam `0.0002` fixo.
- SDV materializa `DataFrame`, usa defaults externos, chama `sample()` descartado e pode omitir classes quando `sample_from_conditions()` nao retorna a classe.
- Difusoes usam clipping `clip_denoised=True`; isso pode reduzir variancia.
- Labels sinteticos sao a classe solicitada, nao uma confirmacao de que a amostra contem sinal daquela classe.

## 11. Problemas de escala

- Default AppClassNet preserva `source`; isso e correto.
- `--scaler minmax|standard` ainda existe e pode alterar AppClassNet explicitamente.
- `generator_transform` pode usar espaco interno `generator`; precisa inverse para `source` antes de avaliacao.
- `ScaleGuard` valida metadata, mas se metadata nao refletir o array fisico, a avaliacao pode aceitar um espaco falso.
- Geradores sigmoid sem inverse podem produzir `[0,1]` contra real `[-0.5,0.5]`.
- `data_type` incorreto pode arredondar features.

## 12. Problemas de labels

- `NpyXYLoader` e `LabelUtils` dao boa base para labels inteiros e one-hot 200.
- Defaults legados `number_classes=2` ainda sao risco em execucoes fora do runner.
- `main._get_configured_number_classes()` usa maximo entre metadata, argumentos e labels; isso protege contra 2, mas mascara inconsistencias.
- Batches sinteticos nao salvam `y.npy`; labels sao reconstruidos por `batches_by_class`.
- `match_train_distribution` pode gerar zero amostras para classes quando `total_rows` e pequeno.
- Label sintetico salvo e declarativo: se gerador ignorar condicionamento, label continua sendo o solicitado.

## 13. Problemas de metricas

- `Accuracy`, Macro/Weighted Precision/Recall/F1 e BalancedAccuracy usam sklearn corretamente para labels observadas.
- Nao ha `labels=np.arange(200)` nas metricas principais.
- `not_applicable` evita zeros falsos em metricas preditivas nao executadas.
- Summary nao registra `n_valid_folds`.
- Nao existe `chance_level_suspected` para Accuracy perto de `0.005`.
- TS-TR normal pode calcular metricas sobre subset truncado.
- BalancedAccuracy pode coincidir com Accuracy em teste balanceado; isso nao e bug, mas precisa suporte por classe no resultado.

## 14. Problemas de memoria

- Modo normal materializa CSV, DataFrame, sinteticos e possivelmente one-hot global.
- Modo batches exige `--use_mmap` para mmap no runner.
- Selecao estratificada por indexing avancado materializa subsets.
- Treino gerativo usa arrays selecionados, nao stream direto do NPY.
- Classificadores subset acumulam reservatorios e materializam `x_subset`.
- Modelos TensorFlow e sessoes so sao limpos no caminho particionado.
- SDV materializa `DataFrame` completo.

## 15. Problemas de compatibilidade

- Caminho CSV legado deve continuar preservado.
- Novos argumentos (`feature_transform`, `generator_transform`, `classifier_transform`, `evaluation_space`, `execution_mode`, `generation_strategy`) nao devem alterar comportamento legado silenciosamente.
- `--scaler` e legado e ainda aceito; sua interacao com AppClassNet deve ser explicitamente documentada.
- `number_k_folds` significa coisas diferentes em CV e provided.
- `sf` sem `--full` nao significa campanha completa.
- Salvar sinteticos em CSV normal e batches NPY tem metadados diferentes.
- `CSVLoader` legado remapeia labels discretas para zero-based e armazena labels internas como `float32`; isso deve continuar compatível, mas precisa ser testado contra labels AppClassNet.
- O help legado de `--scaler` ainda pode induzir o usuario a crer que AppClassNet usa minmax por default, embora o runner atual use `DEFAULT_SCALER="none"`.

## 16. Riscos de data leakage

- `model_type="copy"` pode gerar sinteticos a partir de `x_evaluation_real`; TR-TS pode treinar classificador em `x_evaluation_real` e testar em copias sinteticas do mesmo conjunto.
- TS-TR com `copy` pode treinar em sinteticos copiados de `x_evaluation_real` e testar em `x_evaluation_real`.
- TR-TS usa real de avaliacao para treinar o classificador, o que nao e leakage classico contra sintetico, mas viola a separacao esperada train/valid/test.
- Em `split_mode=provided`, `valid` e usado como avaliacao no pipeline principal; `test` oficial fica fora do fluxo sintetico.
- Reutilizacao do mesmo sintetico em TR-TS e TS-TR e tecnicamente coerente, mas deve ser declarada.

## 17. Riscos de resultados falsamente validos

- Accuracy perto de `0.005` em 200 classes pode passar como numero comum sem diagnostico de chance level.
- Macro/Weighted sem dominio fixo podem mascarar classes ausentes.
- TS-TR normal pode reportar metricas sobre prefixo truncado do real.
- TR-TS pode parecer forte por treinar em valid e testar em sintetico derivado/planejado no mesmo contexto.
- `copy` pode gerar resultado artificialmente alto.
- `data_space` incorreto em metadata pode fazer real e sintetico parecerem compatíveis.
- `decision_tree_subset` pode ser interpretado como DecisionTree completo, mas usa subset.
- `sf` demo pode ser interpretado como campanha completa.
- Summary sem `n_valid_folds` pode parecer estatisticamente completa.
- Baseline real-real calcula o conjunto minimo validado de metricas, mas nao toda a lista multiclass do pipeline principal.
- Metricas de eficiencia ainda iniciam em zero; isso nao afeta metricas preditivas, mas pode ser confundido com medicao real se nao houver status.

## Problemas consolidados e correcoes recomendadas

### P0. TR-TR principal nao executa

- **Severidade:** Critico.
- **Evidencia:** `main.py`, classe `SynDataGen`, funcao `run_experiments()`, linha aproximada 502, chamada `#self.evaluation_TR_TR(dictionary_data)` comentada; `TR_TR_AUDIT.md`.
- **Impacto:** o pipeline sintetico nao produz TR-TR comparavel; regressao AppClassNet depende de `baseline_real_only`.
- **Modulo afetado:** `main.py`, `Engine/Evaluation/TrTr.py`, runner AppClassNet.
- **Correcao recomendada:** manter `baseline_real_only` como golden imediato; depois reativar TR-TR no pipeline com flag/compatibilidade e separar nomes `baseline_real_only` vs `pipeline_tr_tr`.
- **Risco da correcao:** medio; ativar TR-TR pode alterar Results.json legado e tempo de execucao.
- **Testes necessarios:** `test_appclassnet_tr_tr_golden_baseline`; teste de TR-TR pipeline com dataset pequeno; teste de compatibilidade de Results.json.
- **Prioridade:** Fase 0/Fase 2.

### P1. TR-TS treina em `x_evaluation_real`

- **Severidade:** Critico.
- **Evidencia:** `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas aproximadas 96-111 em batches e 188-193 em normal; `EVALUATION_FLOW_AUDIT.md`, `TR_TS_AUDIT.md`.
- **Impacto:** TR-TS nao representa treino real oficial/fold de treino; em provided pode treinar em `valid`.
- **Modulo afetado:** `Engine/Evaluation/TrTs.py`, `Engine/Evaluation/CrossValidation.py`.
- **Correcao recomendada:** decidir contrato. Para AppClassNet, mudar TR-TS para treinar em `x_training_real` e testar em sintetico; preservar comportamento legado via flag se necessario.
- **Risco da correcao:** alto; metricas TR-TS antigas mudarao.
- **Testes necessarios:** teste que inspeciona qual X treina o classificador; teste provided train/valid/test; regressao comparando modo legado se preservado.
- **Prioridade:** Fase 1.

### P2. TS-TR normal trunca teste real

- **Severidade:** Critico.
- **Evidencia:** `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas aproximadas 207-216; `TS_TR_AUDIT.md`, `METRICS_AUDIT.md`.
- **Impacto:** metricas podem ser calculadas sobre prefixo do real, nao todo o conjunto independente.
- **Modulo afetado:** `Engine/Evaluation/TsTr.py`, `Engine/Metrics/Metrics.py`.
- **Correcao recomendada:** calcular metricas sobre todo `x_evaluation_real/y_evaluation_real` ou registrar explicitamente subset real escolhido e selecionar estratificadamente.
- **Risco da correcao:** alto; resultados TS-TR podem mudar bastante.
- **Testes necessarios:** teste com `len(synthetic) < len(real)` garantindo que todas as predicoes reais entram na metrica ou que subset e documentado.
- **Prioridade:** Fase 1.

### P3. AppClassNet normal usa CSV e cross-validation

- **Severidade:** Alto.
- **Evidencia:** `run_appclassnet_top200.py::build_main_command()` linhas aproximadas 1556-1608; `CrossValidation.StratifiedData.wrapper()` linhas aproximadas 465-564; `FULL_PIPELINE_AUDIT.md`.
- **Impacto:** modo normal e batches nao sao semanticamente equivalentes; normal nao usa splits oficiais.
- **Modulo afetado:** `run_appclassnet_top200.py`, `Engine/Evaluation/CrossValidation.py`.
- **Correcao recomendada:** adicionar modo normal AppClassNet com `data_format=npy_xy --split_mode provided`, ou renomear/documentar o modo atual como CV materializado.
- **Risco da correcao:** medio; pode quebrar scripts que esperam CSV.
- **Testes necessarios:** teste CLI dry-run garantindo comandos normal/provided; teste que `1-Fold` em provided usa train->valid/test.
- **Prioridade:** Fase 2.

### P4. `valid` usado como avaliacao quando `test` tambem existe

- **Severidade:** Alto.
- **Evidencia:** `Engine/Evaluation/CrossValidation.py`, classe `StratifiedData`, funcoes `_build_provided_split_folds()` e `_create_fold()`, linhas aproximadas 329-380.
- **Impacto:** pipeline sintetico AppClassNet avalia em valid, nao no test oficial; comparacao com golden train/test fica invalida.
- **Modulo afetado:** `Engine/Evaluation/CrossValidation.py`.
- **Correcao recomendada:** separar `evaluation_split=valid|test`; default compativel atual pode continuar `valid`, mas AppClassNet golden/final deve usar `test` explicitamente.
- **Risco da correcao:** medio; muda interpretacao dos resultados.
- **Testes necessarios:** provided com train/valid/test e selecao explicita de valid/test.
- **Prioridade:** Fase 2.

### P5. Modo batches nao e streaming end-to-end

- **Severidade:** Critico para memoria, alto para semantica.
- **Evidencia:** `CrossValidation._apply_bundle_to_owner()` e `_create_fold()` linhas aproximadas 285-346; `BATCH_AND_MEMORY_AUDIT.md`.
- **Impacto:** usuario pode esperar baixo uso de memoria, mas subsets e folds ainda materializam X.
- **Modulo afetado:** `Engine/Evaluation/CrossValidation.py`, loaders NPY, batch classifiers.
- **Correcao recomendada:** declarar modo batches atual como "memory-reduced"; evoluir para iteradores de split/fold em fases posteriores.
- **Risco da correcao:** alto se reescrever cross-validation; baixo para documentacao/testes.
- **Testes necessarios:** testes de memoria com dataset fake grande/memmap; teste que detecta materializacao indesejada em modo futuro.
- **Prioridade:** Fase 4.

### P6. mmap nao e default em batches

- **Severidade:** Alto.
- **Evidencia:** `run_appclassnet_top200.py::build_batch_main_command()` linhas aproximadas 1697-1698 passa `--mmap_npy` apenas com `--use_mmap`.
- **Impacto:** batches pode carregar NPY em RAM.
- **Modulo afetado:** `run_appclassnet_top200.py`, `NpyXYLoader`, `BatchNpyDataset`.
- **Correcao recomendada:** tornar `--use_mmap` default para AppClassNet batches ou emitir erro/warning forte com estimativa de memoria.
- **Risco da correcao:** baixo-medio; memmap altera performance, mas nao semantica.
- **Testes necessarios:** teste CLI de batch command com mmap default; teste loader com memmap.
- **Prioridade:** Fase 4.

### P7. Defaults legados binarios

- **Severidade:** Critico para AppClassNet fora do runner.
- **Evidencia:** `ArgumentsAutoencoder.py` linha aproximada 49; `ArgumentsVariationalAutoencoder.py` linha 41; `ArgumentsQuantizedVAE.py` linha 42; `ArgumentsWassersteinGAN.py` linha 40; `ArgumentsWassersteinGANGP.py` linha 40.
- **Impacto:** execucoes diretas podem usar `number_classes=2` ou saida sigmoid para 200 classes.
- **Modulo afetado:** `Engine/Arguments/*`, `main.py::_get_configured_number_classes()`, geradores.
- **Correcao recomendada:** nao alterar defaults legados silenciosamente; adicionar validacao AppClassNet que falha se `number_classes != 200` no plano efetivo.
- **Risco da correcao:** baixo se restrito a `source_profile=appclassnet_top200`; alto se mudar defaults globais.
- **Testes necessarios:** teste AppClassNet exige 200; teste legado continua aceitando 2.
- **Prioridade:** Fase 1.

### P8. Labels sinteticos sao declarativos

- **Severidade:** Alto.
- **Evidencia:** algoritmos `get_samples()` gravam `generated_data[label_class] = generated_samples`; `main.py` audita `saved_label=int(label_class)` linhas aproximadas 1262-1268; `LABEL_AND_CONDITIONING_AUDIT.md`.
- **Impacto:** se gerador ignora condicionamento, labels continuam corretos formalmente, mas dados nao contem sinal da classe.
- **Modulo afetado:** `Engine/Algorithms/*`, `SyntheticLabelAudit`, `SyntheticSanityChecks`.
- **Correcao recomendada:** manter labels declarativos, mas adicionar diagnosticos obrigatorios de condicionamento: predicao real->sintetico, entropia, recall por classe, centroides e variancia por classe.
- **Risco da correcao:** baixo; adiciona diagnosticos, nao muda geracao.
- **Testes necessarios:** gerador fake que ignora labels deve disparar warning de colapso.
- **Prioridade:** Fase 3.

### P9. `data_type` nao-continuo arredonda features

- **Severidade:** Alto.
- **Evidencia:** `AdversarialAlgorithm.get_samples()` linhas aproximadas 356-358; `AutoencoderAlgorithm.get_samples()` 306-309; `AlgorithmVariationalAutoencoder.get_samples()` 331-334; difusoes linhas aproximadas 437-448.
- **Impacto:** AppClassNet `[-0.5,0.5]` pode virar inteiros, destruindo escala.
- **Modulo afetado:** algoritmos gerativos e CLI.
- **Correcao recomendada:** para `source_profile=appclassnet_top200`, falhar se `data_type != continuous`.
- **Risco da correcao:** baixo para AppClassNet; nenhum impacto legado se condicionado ao profile.
- **Testes necessarios:** teste CLI/argumentos que rejeita AppClassNet nao-continuo; teste gerador fake nao arredonda em continuous.
- **Prioridade:** Fase 1.

### P10. Espaco numerico depende de metadata correta

- **Severidade:** Alto.
- **Evidencia:** `ScaleGuard.validate_compatible_metadata()` em `FeatureTransformManager.py` linhas aproximadas 520-541; `PREPROCESSING_AND_DATA_SPACE_AUDIT.md`.
- **Impacto:** se array fisico e metadata divergirem, real e sintetico podem ser comparados em escalas diferentes.
- **Modulo afetado:** `FeatureTransformManager`, `ScaleGuard`, `SyntheticBatchWriter`, CSV synthetic metadata.
- **Correcao recomendada:** adicionar validacao amostral de range e `data_space` antes de TR-TS/TS-TR; salvar bloco de precheck por avaliacao.
- **Risco da correcao:** medio; pode falhar execucoes antigas com metadados incompletos.
- **Testes necessarios:** sintetico `[0,1]` marcado source contra real `[-0.5,0.5]` deve disparar diagnostico/falha configuravel.
- **Prioridade:** Fase 1/Fase 2.

### P11. Metricas nao fixam dominio 0..199

- **Severidade:** Alto.
- **Evidencia:** `Engine/Metrics/Metrics.py::_get_multiclass_metric_values()` linhas aproximadas 355-368 usa sklearn sem `labels=np.arange(200)`.
- **Impacto:** classes ausentes podem ser mascaradas em Macro/Weighted.
- **Modulo afetado:** `Engine/Metrics/Metrics.py`.
- **Correcao recomendada:** adicionar metricas/diagnosticos com dominio fixo quando `num_classes` estiver disponivel, sem substituir silenciosamente metricas legadas.
- **Risco da correcao:** baixo-medio; novas chaves em JSON precisam compatibilidade.
- **Testes necessarios:** matriz 200x200 manual; classe ausente em y_true/y_pred; comparacao sklearn com labels explicitas.
- **Prioridade:** Fase 1.

### P12. Falta `chance_level_suspected`

- **Severidade:** Medio-alto.
- **Evidencia:** `METRICS_AUDIT.md` propoe regra; nao ha implementacao atual.
- **Impacto:** Accuracy ~0.005 pode passar sem diagnostico, apesar de ser nivel aleatorio para 200 classes.
- **Modulo afetado:** `Engine/Metrics/Metrics.py`, avaliadores.
- **Correcao recomendada:** adicionar flag diagnostica quando `num_classes=200` e `0.004 <= Accuracy <= 0.006`.
- **Risco da correcao:** baixo; diagnostico nao reprova automaticamente.
- **Testes necessarios:** teste Accuracy no intervalo; teste fora do intervalo.
- **Prioridade:** Fase 1.

### P13. Summary nao registra `n_valid_folds`

- **Severidade:** Medio.
- **Evidencia:** `Metrics.update_mean_std_fold()` agrega apenas valores numericos e ignora `not_applicable`; `METRICS_AUDIT.md`.
- **Impacto:** media pode parecer completa mesmo com folds ausentes.
- **Modulo afetado:** `Engine/Metrics/Metrics.py`.
- **Correcao recomendada:** adicionar `n_valid_folds` e `n_expected_folds` por metrica/classificador.
- **Risco da correcao:** baixo; adiciona campos.
- **Testes necessarios:** uma execucao com fold not_applicable deve ter `n_valid_folds=1`, `n_expected_folds=2`.
- **Prioridade:** Fase 2.

### P14. `copy` pode causar vazamento

- **Severidade:** Critico.
- **Evidencia:** `main.py::SynDataGen.run_experiments()` linhas aproximadas 472-480 passa `x_evaluation_real` para geracao; `synthesize_data()` copy linhas aproximadas 906-912; `TsTr.evaluation_TS_TR()` linhas aproximadas 200-216.
- **Impacto:** TR-TS e TS-TR podem avaliar copias do mesmo real usado em treino/teste.
- **Modulo afetado:** `main.py`, `Engine/Algorithms/Copy/CopyAlgorithm.py`, `TrTs.py`, `TsTr.py`.
- **Correcao recomendada:** bloquear `copy` para AppClassNet synthetic-quality runs ou marcar resultado como leakage-prone; passar `x_training_real` para copy se o contrato exigir treino.
- **Risco da correcao:** medio; copy pode ser baseline legado.
- **Testes necessarios:** copy nao pode usar `x_evaluation_real` em AppClassNet; resultado deve marcar `leakage_risk=true`.
- **Prioridade:** Fase 1.

### P15. Classificadores batch subset podem ser mal interpretados

- **Severidade:** Alto.
- **Evidencia:** `Engine/Classifiers/BatchClassifiers.py::train_batch_classifier()` linhas aproximadas 170-237; `BATCH_AND_MEMORY_AUDIT.md`.
- **Impacto:** `decision_tree_subset` nao e DecisionTree treinado em todos os dados.
- **Modulo afetado:** `BatchClassifiers.py`, runner AppClassNet.
- **Correcao recomendada:** registrar explicitamente `classifier_training_mode=subset|partial_fit|full_fit`, cotas por classe e contagens usadas.
- **Risco da correcao:** baixo.
- **Testes necessarios:** metadata do classificador batch contem contagens por classe e modo.
- **Prioridade:** Fase 2.

### P16. Geracao SDV incompleta/custosa

- **Severidade:** Alto.
- **Evidencia:** `Engine/Algorithms/ThirdParty/SDVInterfaceAlgorithm.py::get_samples()` linhas aproximadas 168-226; chamada `sample()` descartada e classes omitidas se nao retornam.
- **Impacto:** classes podem faltar; memoria/tempo aumentam; features podem ser corrompidas por substituicao de strings por `1`.
- **Modulo afetado:** `SDVInterfaceAlgorithm`.
- **Correcao recomendada:** remover sample descartado, validar condicoes por classe, falhar se classe solicitada faltar, configurar parametros SDV explicitamente.
- **Risco da correcao:** medio; muda comportamento SDV.
- **Testes necessarios:** SDV mock sem classe deve falhar; strings em features devem ser erro ou schema explicito.
- **Prioridade:** Fase 3.

### P17. Parametros WGAN/WGAN-GP do runner nao usados

- **Severidade:** Alto.
- **Evidencia:** `run_appclassnet_top200.py` linhas aproximadas 240-243 e 266-269 configuram `0.001`; `GenerativeModels._training_wasserstein_model()` linhas 2116-2117 e `_training_wasserstein_gp_model()` linhas 2554-2555 usam `0.0002`.
- **Impacto:** reproducibilidade e interpretacao da campanha ficam incorretas.
- **Modulo afetado:** `run_appclassnet_top200.py`, `Engine/Models/GenerativeModels.py`.
- **Correcao recomendada:** usar argumentos configurados no treino ou remover parametros do runner se nao suportados.
- **Risco da correcao:** medio; resultados de WGAN mudarao.
- **Testes necessarios:** teste monkeypatch/spy de Adam recebendo learning rate configurado.
- **Prioridade:** Fase 3.

### P18. Sanity checks de qualidade sintetica ainda nao sao gates uniformes

- **Severidade:** Medio-alto.
- **Evidencia:** `SyntheticSanityChecks.py::SyntheticSanityChecker.run()` linhas aproximadas 80-106, `_scale_report()` 224-239, `_classifier_report()` 277-303.
- **Impacto:** colapso, duplicacao, baixa variancia e distancia distributiva podem nao bloquear interpretacao de TR-TS/TS-TR.
- **Modulo afetado:** `SyntheticSanityChecks.py`, avaliadores, Results.json.
- **Correcao recomendada:** adicionar precheck obrigatorio por avaliacao com ranges, classes, NaN/inf, out-of-range, predicao majoritaria, entropia, variancia e centroides.
- **Risco da correcao:** medio; pode aumentar tempo.
- **Testes necessarios:** sintetico constante, classe ausente, escala errada, predicao colapsada.
- **Prioridade:** Fase 3.

### P19. `sf` ambiguo

- **Severidade:** Medio-alto.
- **Evidencia:** `run_appclassnet_top200.py::choose_campaigns()` linhas aproximadas 360-368; sem `--full` retorna demos, com `--full` retorna `DEFAULT_CAMPAIGN`.
- **Impacto:** resultados de demo podem ser confundidos com campanha completa.
- **Modulo afetado:** runner AppClassNet, logs/resultados.
- **Correcao recomendada:** registrar `campaign_resolved`, `full`, lista de modelos e `demo=true|false` em todos os resultados.
- **Risco da correcao:** baixo.
- **Testes necessarios:** dry-run `-c sf` e `-c sf --full` verificando metadata.
- **Prioridade:** Fase 2.

### P20. Estrategias `per_class`/`grouped_classes` alteram semantica

- **Severidade:** Medio.
- **Evidencia:** `main.py::_synthesize_data_partitioned()` linhas aproximadas 1067-1148 treina gerador por classes/grupos e limpa sessao.
- **Impacto:** reduz memoria, mas cada modelo ve subset de classes; qualidade e comparabilidade mudam.
- **Modulo afetado:** `main.py`, geradores.
- **Correcao recomendada:** registrar estrategia, classes por unidade, linhas de treino por unidade e tratar resultados separadamente.
- **Risco da correcao:** baixo para metadata; alto se tentar unificar semantica.
- **Testes necessarios:** grouped_classes registra unidades e contagens; per_class falha se faltar classe.
- **Prioridade:** Fase 3/Fase 4.

### P21. Quotas sintéticas de treino/teste nao têm semantica uniforme

- **Severidade:** Alto.
- **Evidencia:** `TR_TS_AUDIT.md` mostra que `synthetic_test_samples_per_class` limita batches via `test_samples_per_class`, mas nao limita TR-TS normal materializado (`Engine/Evaluation/TrTs.py`, normal linhas aproximadas 146-203; batches ~112-117). `TS_TR_AUDIT.md` mostra que `synthetic_train_samples_per_class` vira quota de reservatorio em alguns classificadores batch, mas partial-fit usa todos os batches e o normal nao aplica diretamente a quota (`BatchClassifiers.py` ~105-237; `TsTr.py` ~162-216).
- **Impacto:** resultados normal/batches podem usar quantidades sintéticas diferentes sem que o nome da avaliacao revele isso.
- **Modulo afetado:** `Engine/Arguments/Arguments.py`, `Engine/Evaluation/TrTs.py`, `Engine/Evaluation/TsTr.py`, `Engine/Classifiers/BatchClassifiers.py`, `SamplePlanner.py`.
- **Correcao recomendada:** definir contrato unico para `synthetic_train_samples_per_class` e `synthetic_test_samples_per_class`, registrar contagens efetivas por classe em todas as avaliacoes e preservar comportamento legado atras de flag se necessario.
- **Risco da correcao:** alto; muda a quantidade efetiva de dados usados nas metricas.
- **Testes necessarios:** TR-TS normal e batches com cota pequena; TS-TR normal e batches com cota pequena; verificar contagem por classe usada em fit/predict.
- **Prioridade:** Fase 1/Fase 2.

### P22. `classifier_transform` nao e aplicado de forma uniforme aos avaliadores

- **Severidade:** Medio-alto.
- **Evidencia:** `PREPROCESSING_AND_DATA_SPACE_AUDIT.md` aponta que `classifier_transform` existe em `FeatureTransformPolicy` e `main.py::_build_model_input_adapter()` (~602-614), mas TR-TR/TR-TS/TS-TR treinam classificadores diretamente nos arrays recebidos. O baseline real-real aplica `classifier_transform` se solicitado (`run_appclassnet_top200.py::run_real_real_baseline()`, ~1053-1069), mas registra `"data_space": "source"` mesmo apos transformacao (~1077-1100).
- **Impacto:** o usuario pode acreditar que o classificador avaliativo recebeu minmax/standard quando o pipeline principal nao aplicou; no baseline, metadata pode declarar `source` mesmo quando o classificador operou em outro espaco.
- **Modulo afetado:** `main.py`, `FeatureTransformManager.py`, `run_appclassnet_top200.py`, avaliadores TR-TR/TR-TS/TS-TR.
- **Correcao recomendada:** separar `evaluation_input_space` e `classifier_training_space`, aplicar transformacao explicitamente ou rejeitar `classifier_transform` nos avaliadores enquanto nao suportado.
- **Risco da correcao:** medio; pode alterar metricas de quem usa transformacao explicita.
- **Testes necessarios:** spy do transform manager garantindo fit apenas no treino e transform em test/sintetico; baseline com `classifier_transform=minmax` deve registrar metadata correta.
- **Prioridade:** Fase 2.

### P23. `feature_transform=preserve` ainda pode duplicar I/O no runner

- **Severidade:** Medio.
- **Evidencia:** `FULL_PIPELINE_AUDIT.md` registra que `run_appclassnet_top200.py::preprocess_appclassnet_splits()` ainda cria NPY pre-processado mesmo quando a politica e preserve, linhas aproximadas 850-1006.
- **Impacto:** tempo e disco desnecessarios para AppClassNet, especialmente em execucoes repetidas ou datasets maiores.
- **Modulo afetado:** `run_appclassnet_top200.py`.
- **Correcao recomendada:** quando `feature_transform=preserve` e dtype/shape ja sao aceitaveis, apontar diretamente para os NPY originais ou criar hardlink/symlink/copy lazy documentada.
- **Risco da correcao:** medio; paths esperados por scripts e caches podem mudar.
- **Testes necessarios:** dry-run preserve nao cria novo array; modo transformado ainda salva NPY e scaler; compatibilidade com caminhos existentes.
- **Prioridade:** Fase 4.

### P24. Labels CSV legados sao remapeados e armazenados como float

- **Severidade:** Medio.
- **Evidencia:** `LABEL_AND_CONDITIONING_AUDIT.md` e `PREPROCESSING_AND_DATA_SPACE_AUDIT.md` apontam `CSVLoader._encode_discrete_labels()` (~303-317) e `_process_label_column()` (~296): labels discretas sao ordenadas, remapeadas para zero-based e armazenadas como `float32`; `LabelUtils.labels_to_1d_integer()` (~14-34) converte depois para `int64`.
- **Impacto:** caminho CSV continua funcional, mas labels originais podem nao corresponder diretamente ao dominio interno; floats nao-inteiros poderiam ser truncados se chegarem diretamente ao `Metrics`.
- **Modulo afetado:** `Engine/DataIO/CSVLoader.py`, `Engine/DataIO/LabelUtils.py`, `Engine/Metrics/Metrics.py`.
- **Correcao recomendada:** manter comportamento legado, mas documentar mapping no output e adicionar validação explícita de labels inteiras antes de metricas/one-hot.
- **Risco da correcao:** baixo se for validação; alto se mudar remapeamento legado.
- **Testes necessarios:** comando antigo CSV com labels nao zero-based; leitura/salvamento preserva mapping; labels fracionarias devem falhar claramente.
- **Prioridade:** Fase 0/Fase 5.

### P25. Manifesto batch pressupoe um unico label por batch sintetico

- **Severidade:** Medio.
- **Evidencia:** `LABEL_AND_CONDITIONING_AUDIT.md` aponta que batches sinteticos nao salvam `y.npy`; `SyntheticBatchWriter.write_batch()` registra o batch sob `class_label` (~71-116), e `SyntheticBatchReader.items()`/`BatchClassifiers.iter_synthetic_labeled_batches()` reconstroem y pela chave do manifesto (~154-160 e ~86-97).
- **Impacto:** se um batch fisico contiver varias classes, todas as linhas seriam rotuladas como a classe do manifesto; o contrato atual so representa batches mono-classe.
- **Modulo afetado:** `Engine/DataIO/SyntheticBatchIO.py`, `Engine/Classifiers/BatchClassifiers.py`, geracao incremental.
- **Correcao recomendada:** manter mono-classe como contrato e validar isso no writer; se batches multi-classe forem necessários, salvar `y` por batch.
- **Risco da correcao:** baixo para validação; medio se adicionar novo formato.
- **Testes necessarios:** writer rejeita batch multi-classe quando y fornecido; reader reconstrói labels por manifesto; contrato documentado no manifest.
- **Prioridade:** Fase 2/Fase 4.

### P26. Reprodutibilidade de geradores neurais nao esta centralizada

- **Severidade:** Medio-alto.
- **Evidencia:** `SYNTHETIC_GENERATION_AUDIT.md` registra uso de `numpy.random.normal`, `numpy.random.choice` e `tensorflow.random.normal` nos algoritmos sem seed local/global evidente; seeds aparecem em selecao/manifesto, nao no treino/amostragem neural.
- **Impacto:** resultados de TR-TS/TS-TR podem variar entre execucoes com o mesmo comando, dificultando regressao e comparacao de geradores.
- **Modulo afetado:** `Engine/Algorithms/*`, `Engine/Models/GenerativeModels.py`, `run_appclassnet_top200.py`.
- **Correcao recomendada:** introduzir seed de experimento propagada para NumPy, TensorFlow e geradores por fold/modelo, sem mudar defaults legados quando seed nao for fornecida.
- **Risco da correcao:** medio; pode alterar distribuicao de resultados e performance.
- **Testes necessarios:** mesma seed gera hashes/metricas iguais em gerador controlado; seeds diferentes mudam amostras; manifesto registra seed efetiva.
- **Prioridade:** Fase 3.

### P27. Clipping das difusoes pode mascarar problema de escala e variancia

- **Severidade:** Medio.
- **Evidencia:** `PREPROCESSING_AND_DATA_SPACE_AUDIT.md` e `SYNTHETIC_GENERATION_AUDIT.md` apontam `clip_by_value`/`clip_denoised=True` em `GaussianLatentDiffusion.py` e `GaussianDenoisingDiffusion.py` (~401-407), com campanhas AppClassNet configurando `clip_min=-0.5` e `clip_max=0.5`.
- **Impacto:** manter o range pode ser desejado, mas tambem pode esconder saida fora de escala e reduzir variancia sintética.
- **Modulo afetado:** difusoes, campanhas AppClassNet, sanity checks.
- **Correcao recomendada:** registrar percentual de valores clipados e estatisticas antes/depois do clipping; nao tratar range correto como prova de qualidade.
- **Risco da correcao:** baixo se diagnostico; medio se mudar clipping.
- **Testes necessarios:** difusao fake com valores fora do range registra taxa de clipping; variancia muito baixa dispara warning.
- **Prioridade:** Fase 3.

### P28. Baseline real-real nao calcula toda a lista multiclass principal

- **Severidade:** Medio.
- **Evidencia:** `METRICS_AUDIT.md` aponta que `run_appclassnet_top200.py::run_real_real_baseline()` calcula Accuracy, MacroF1, WeightedF1 e BalancedAccuracy (~1118-1121), mas nao Macro/Weighted Precision/Recall.
- **Impacto:** o golden validado cobre o essencial informado, mas nao valida todas as metricas usadas no pipeline principal.
- **Modulo afetado:** `run_appclassnet_top200.py`, `Engine/Metrics/Metrics.py`.
- **Correcao recomendada:** adicionar as metricas faltantes ao baseline como campos novos, preservando os campos existentes.
- **Risco da correcao:** baixo; adiciona chaves.
- **Testes necessarios:** baseline controlado calcula as oito metricas multiclass; comparacao manual por matriz de confusao.
- **Prioridade:** Fase 0/Fase 2.

### P29. Metricas de eficiencia inicializadas com zero podem confundir status

- **Severidade:** Baixo-medio.
- **Evidencia:** `METRICS_AUDIT.md` aponta que metricas preditivas puladas usam `"not_applicable"`, mas `EfficiencyMetrics` ainda inicia em zeros (`Metrics.py`, ~250-257).
- **Impacto:** zeros de eficiencia podem parecer medicao real quando a etapa nao executou, embora nao afetem Accuracy/F1/Precision/Recall.
- **Modulo afetado:** `Engine/Metrics/Metrics.py`, Results.json.
- **Correcao recomendada:** adicionar status para eficiencia ou usar `not_applicable` quando a medicao nao foi executada.
- **Risco da correcao:** baixo-medio; consumidores antigos podem esperar numero.
- **Testes necessarios:** avaliacao pulada nao grava eficiencia como zero sem status; compatibilidade de JSON.
- **Prioridade:** Fase 2.

### P30. Ajuda/documentacao de `--scaler` esta desatualizada para AppClassNet

- **Severidade:** Medio.
- **Evidencia:** `FULL_PIPELINE_AUDIT.md` aponta que o help de `--scaler` em `ArgumentsDataLoader.py::add_argument_data_load()` (~218-221) ainda afirma default AppClassNet minmax, mas `run_appclassnet_top200.py` define `DEFAULT_SCALER="none"` (~42) e o perfil AppClassNet resolve `preserve`.
- **Impacto:** risco operacional de usuarios ativarem ou esperarem normalizacao errada, prejudicando comparabilidade com o baseline source.
- **Modulo afetado:** `Engine/Arguments/ArgumentsDataLoader.py`, docs/README/runner help.
- **Correcao recomendada:** corrigir help e documentacao sem alterar comportamento; mencionar que `--scaler` e legado e que AppClassNet default e `none/preserve`.
- **Risco da correcao:** baixo.
- **Testes necessarios:** teste de help/parser ou snapshot de defaults; teste AppClassNet dry-run mostra `feature_transform=preserve`.
- **Prioridade:** Fase 0/Fase 5.

## Roadmap

### Fase 0 — testes de regressao

Objetivo: congelar comportamento conhecido antes de corrigir.

1. Implementar especificacao `test_appclassnet_tr_tr_golden_baseline` como teste/manual gate quando dados reais estiverem disponiveis.
2. Adicionar fixtures pequenas para TR-TR, TR-TS e TS-TR validando quais conjuntos sao usados.
3. Adicionar testes de labels `0..199`, label `199`, label `200` rejeitado e classe ausente.
4. Adicionar teste de `data_space=source` e `feature_transform=preserve` para AppClassNet.
5. Adicionar teste de metricas multiclass com dominio fixo 200 e matriz de confusao manual.
6. Adicionar teste de `not_applicable` sem zeros preditivos.
7. Adicionar snapshots de CLI/dry-run para `sf`, `sf --full`, normal e batches.
8. Adicionar testes de comando legado CSV, help/default de `--scaler` e baseline com todas as metricas multiclass.

### Fase 1 — correcoes criticas

Objetivo: impedir resultados semanticamente falsos.

1. Corrigir ou versionar TR-TS para treinar em `x_training_real`.
2. Corrigir TS-TR normal para nao truncar silenciosamente o teste real.
3. Bloquear/diagnosticar `copy` como leakage-prone em AppClassNet.
4. Validar `number_classes=200` e `data_type=continuous` para AppClassNet.
5. Adicionar diagnostico `chance_level_suspected`.
6. Adicionar metricas com dominio fixo `labels=np.arange(200)` como novas chaves.
7. Adicionar precheck de `data_space`, `transform_id` e range antes de TR-TS/TS-TR.
8. Uniformizar ou registrar quotas efetivas de `synthetic_train_samples_per_class` e `synthetic_test_samples_per_class`.

### Fase 2 — consistencia dos modulos de avaliacao

Objetivo: tornar TR-TR, TR-TS e TS-TR comparaveis e auditaveis.

1. Separar explicitamente `cross_validation_fold` de `provided_split`.
2. Adicionar `evaluation_split=valid|test` para provided.
3. Reativar TR-TR principal de forma opt-in ou registrar `baseline_real_only` no mesmo esquema de resultados.
4. Registrar `n_valid_folds`, `n_expected_folds`, contagens por classe e subset efetivo.
5. Registrar `classifier_training_mode` para batch classifiers.
6. Registrar metadata resolvida de campanha `sf`.
7. Padronizar blocos de precheck para TR-TR, TR-TS e TS-TR.
8. Corrigir metadata/status de `classifier_transform`, metricas de eficiencia e baseline real-real.

### Fase 3 — qualidade da geracao

Objetivo: diagnosticar se sinteticos sao uteis, condicionais e diversos.

1. Tornar sanity checks por classe e por feature obrigatorios para AppClassNet.
2. Adicionar entropia de predicoes, classes nunca previstas, recall por classe e classe majoritaria.
3. Adicionar medias, variancias, quantis, KS/Wasserstein e distancias entre centroides.
4. Adicionar duplicacao e nearest-neighbor distance.
5. Corrigir WGAN/WGAN-GP para usar parametros configurados.
6. Corrigir SDV para nao descartar samples, validar classes e controlar parametros.
7. Registrar claramente estrategias `single_conditional`, `per_class` e `grouped_classes`.
8. Tratar demos como smoke tests, nao como medida de qualidade sintetica.
9. Registrar seed efetiva dos geradores neurais e diagnosticos de clipping/variancia.

### Fase 4 — eficiencia de memoria

Objetivo: aproximar batches de streaming real sem mudar semantica sem registro.

1. Tornar mmap default ou fortemente recomendado em AppClassNet batches.
2. Evitar `np.asarray`/indexing avancado que materializa memmap quando possivel.
3. Criar iteradores de splits provided para treino/avaliacao sem materializar tudo.
4. Fazer TR-TR ter caminho batch equivalente ou documentar ausencia.
5. Reduzir acumulacao de labels/predicoes em listas, usando acumuladores de matriz de confusao.
6. Fazer geradores que ainda geram classe inteira respeitarem `generation_batch_size`.
7. Limpar sessoes TensorFlow em mais caminhos quando modelos sao descartados.
8. Eliminar I/O duplicado quando `feature_transform=preserve` e validar o contrato mono-classe dos batches sinteticos.

### Fase 5 — generalizacao arquitetural

Objetivo: generalizar MalDataGen sem quebrar comportamento original.

1. Formalizar contratos `DatasetBundle`, `SplitData`, `DatasetSchema`, `SamplePlan`, `DataSpace` e `EvaluationProtocol`.
2. Separar avaliacao em protocolos nomeados: `real_real_train_test`, `real_to_synthetic`, `synthetic_to_real`, `cross_validation`.
3. Criar adaptadores de dataset por perfil (`legacy_csv`, `appclassnet_top200`) sem ifs espalhados.
4. Tornar transformacoes declarativas, com fit split, transform id e inverse obrigatorios.
5. Versionar Results.json para incluir metadata nova sem quebrar consumidores antigos.
6. Separar qualidade sintetica de utilidade downstream: C2ST/distribuicao vs TS-TR.
7. Manter defaults legados no perfil legado e defaults AppClassNet no perfil AppClassNet.
8. Documentar formalmente o caminho CSV legado, incluindo remapeamento de labels e compatibilidade de `--scaler`.

## Encerramento

O caminho mais seguro e nao tentar "arrumar tudo" de uma vez. Primeiro, congelar o comportamento validado e expor diagnosticos que hoje ficam implicitos. Depois, corrigir TR-TS, TS-TR e os gates de escala/labels. So entao faz sentido comparar geradores e otimizar memoria, porque nesse ponto as metricas passarao a representar o experimento pretendido.
