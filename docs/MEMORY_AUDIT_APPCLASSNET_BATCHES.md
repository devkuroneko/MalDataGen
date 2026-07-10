# Auditoria de Memoria: AppClassNet top-200 e modo batches

## Escopo

Comando auditado:

```bash
python3 run_appclassnet_top200.py -c sf
```

O alias `sf` seleciona as campanhas demo `variational_demo` e `adversarial_demo`, ambas com `number_k_folds=2`. O runner tambem suporta campanhas maiores com `DEFAULT_K_FOLDS=5`; as estimativas abaixo usam `K` para numero de folds e `N` para numero de linhas do dataset materializado.

Nenhuma alteracao comportamental foi feita nesta auditoria.

## Mapa do fluxo atual

1. `run_appclassnet_top200.py` escolhe campanhas e prepara o CSV.
   - `choose_campaigns(["sf"])` retorna `["variational_demo", "adversarial_demo"]`.
   - `materialize_appclassnet_csv(...)` valida `train_x.npy/train_y.npy` por split.
   - `numpy.load(..., mmap_mode="r")` carrega os `.npy` como memmap no runner.
   - O runner escreve um CSV em chunks com `pandas.DataFrame(chunk)` e `to_csv`.
   - Por padrao `--dataset_split=train`; `--dataset_split=all` escreve `train`, `valid` e `test` sequencialmente no mesmo CSV, sem concatenar todos os splits em RAM no runner.

2. O runner chama `main.py` via subprocesso para cada campanha/combinacao.
   - O comando gerado passa `--data_load_path_file_input <csv>`, `--data_type continuous`, `--output_dir ...`, classificadores e parametros do modelo.
   - O modo atual nao passa `--data_format npy_xy`; portanto o pipeline principal usa CSV, nao os `.npy` diretamente.

3. `main.py` instancia `SynDataGen` e executa `run_experiments()`.
   - Decoradores aplicados: `import_metrics`, `import_classifiers`, `StratifiedData`.
   - `StratifiedData` chama `load_dataset_from_args`.
   - Com `data_format=csv`, `CSVDataProcessor.load_csv()` roda o caminho legado.

4. `CSVDataProcessor` carrega e converte tudo para arrays.
   - `pandas.read_csv(...)` carrega o CSV completo em RAM.
   - Labels sao extraidas para `numpy.array(..., dtype=float32)`.
   - Labels discretas sao remapeadas para indices zero-based com uma lista Python intermediaria.
   - Features viram `data_file.values.astype(numpy.float32)`.

5. `StratifiedData` cria todos os folds antes do primeiro treino.
   - `shuffle(self._data_loaded, self._data_loaded_labels)` cria copias embaralhadas completas.
   - `StratifiedKFold.split(...)` gera indices.
   - Para cada fold:
     - salva CSV de treino e avaliacao com `numpy.column_stack` + `pandas.DataFrame`;
     - cria `training_shuffled_data`, `training_shuffled_labels`, `evaluation_shuffled_data`, `evaluation_shuffled_labels`;
     - guarda esses arrays em `self.list_folds`.
   - Como todos os folds ficam em `list_folds`, a memoria cresce aproximadamente com `K` vezes o dataset.

6. Para cada fold, o modelo generativo e treinado.
   - `train_model(...)` recebe arrays completos do fold.
   - Modelos condicionais chamam `one_hot_encode_labels(...)`, criando matriz `(n_amostras_fold, 200)` em `float32`.

7. Dados sinteticos sao gerados e mantidos em memoria.
   - `synthesize_data(...)` atribui `self.data_generated = algorithm.get_samples(...)`.
   - Os algoritmos retornam `dict[label] = ndarray/list de amostras`.
   - Se `save_data=True`, `CSVDataProcessor.save_csv(...)` achata o dicionario em listas Python `labels` e `data`, cria `pandas.DataFrame` e grava arquivo.

8. Avaliacoes treinam classificadores e calculam metricas.
   - `TS-TR`: transforma sinteticos em listas `labels, data`, embaralha, treina classificadores no sintetico e prediz no real.
   - `TR-TS`: transforma sinteticos em listas `labels, data`, treina classificadores em dados reais e prediz no sintetico.
   - Metricas guardam escalares no JSON, mas as etapas de avaliacao criam arrays de labels/predicoes e, para distancia, copias completas de `data_real` e `data_synthetic`.

## Pontos auditados

### Onde os `.npy` sao carregados

- `run_appclassnet_top200.py:478-479`: `numpy.load(x_path, mmap_mode="r")` e `numpy.load(y_path, mmap_mode="r")` durante a preparacao do CSV.
- `Engine/DataIO/NpyXYLoader.py:142-152`: loader alternativo `npy_xy`, com `numpy.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)`.
- `Engine/Evaluation/CrossValidation.py:87-88`: para `npy_xy`, `mmap_mode` so vira `"r"` quando `arguments.mmap_npy` e verdadeiro; caso contrario usa `None` e carrega em RAM.

No comando auditado, o fluxo principal nao usa `NpyXYLoader`; ele usa `.npy` apenas para gerar CSV e depois le o CSV completo.

### Se `np.load` usa mmap ou RAM

- Runner: usa mmap sempre na preparacao (`mmap_mode="r"`).
- `NpyXYLoader`: suporta mmap, mas o default efetivo em `CrossValidation._mmap_mode_for_npy()` e `None` se `--mmap_npy` nao for passado.
- Se `NpyXYLoader` falha ao mapear, tenta novamente sem mmap, carregando tudo em RAM.

### Onde X/y viram pandas DataFrame

- `run_appclassnet_top200.py:434-437`: cada chunk `.npy` vira `pandas.DataFrame` para escrever CSV.
- `Engine/Evaluation/CrossValidation.py:62-73`: folds de treino/avaliacao sao salvos com `numpy.column_stack` e `pandas.DataFrame`.
- `Engine/DataIO/CSVLoader.py:410-411`: sinteticos gerados viram DataFrame para salvar.
- `Engine/Algorithms/ThirdParty/SDVInterfaceAlgorithm.py:123`: SDV recebe `pandas.DataFrame(x_real_samples, ...)`.

### Onde y vira one-hot

- `Engine/DataIO/LabelUtils.py:98-103`: `one_hot_encode_labels` cria `zeros((n, num_classes), dtype=float32)`.
- Chamadas principais em `Engine/Models/GenerativeModels.py`:
  - adversarial: linhas em torno de `269-271`;
  - autoencoder: em torno de `589-590`;
  - latent diffusion: em torno de `1378-1425`;
  - wasserstein: em torno de `2015-2017`;
  - wasserstein_gp: em torno de `2449-2451`;
  - variational: em torno de `2852`;
  - denoising_diffusion: em torno de `3246-3248`.
- Algoritmos tambem fazem one-hot ao gerar amostras por classe, por exemplo `AlgorithmWassersteinGAN.py`, `AlgorithmVariationalAutoencoder.py`, `AlgorithmDenoisingDiffusion.py` e outros.

Para AppClassNet top-200, one-hot custa `n * 200 * 4 bytes`, ou cerca de `800N bytes`; isso e 10x maior que X com 20 features `float32`.

### Onde train/valid/test sao concatenados

- Runner: somente quando `--dataset_split=all`, `iter_splits("all")` percorre `train`, `valid`, `test` e escreve todos no mesmo CSV. E uma concatenacao em disco, em chunks, nao uma concatenacao completa em memoria.
- `NpyXYLoader` carrega splits separados em `DatasetBundle`; com `split_mode=cross_validation`, o pipeline usa apenas `train` e avisa que `valid/test` nao sao concatenados nem usados.
- Com `split_mode=provided`, o pipeline cria um fold com `train` e usa `valid` ou `test` como avaliacao; se ambos existem, `test` e marcado como nao aplicavel.

### Onde sinteticos sao acumulados em memoria

- `main.py:612-681`: `self.data_generated = ...get_samples(...)` guarda o dicionario inteiro de sinteticos por fold.
- `CSVLoader.save_csv`: `labels.extend(...)` e `data.extend(...)` duplicam os sinteticos em listas Python antes do DataFrame.
- `TsTr.evaluation_TS_TR`: `labels, data = [], []`; depois `data.extend(generated_samples)`, `shuffle(data, numpy.array(labels))` e `numpy.array(data)` para distancia.
- `TrTs.evaluation_TR_TS`: mesmo padrao de listas para labels/dados e predicao em todos os sinteticos.

### Onde metricas guardam arrays grandes

- O dicionario de metricas guarda escalares e resumos, nao matrizes grandes.
- As alocacoes grandes estao nos argumentos temporarios das metricas:
  - `TsTr.py:98-102`: `labels_to_1d_integer(...)` e `numpy.array(label_predicted)`.
  - `TrTs.py:111-112`: `numpy.array(labels)` e `numpy.array(label_predicted)`.
  - `TsTr.py:112-135`: `numpy.array(dictionary_data['x_training_real'])` e `numpy.array(data)` para distancia.
  - `TrTr.py`: cria copias de treino e avaliacao reais para distancia, embora `evaluation_TR_TR` esteja comentado no fluxo atual.
- Distancias iteram sobre arrays e nao guardam matriz `N x N`, mas exigem que os dois conjuntos completos estejam residentes.

### Onde classificadores sao treinados

- `Engine/Classifiers/Classifiers.py:120-131`: `get_trained_classifiers(...)` itera pelos classificadores configurados.
- `RandomForest.py:97-117`: converte `X` e `y` com `numpy.array(..., dtype=...)` e chama `RandomForestClassifier.fit`.
- `DecisionTree.py:98-118`: converte e chama `DecisionTreeClassifier.fit`.
- O mesmo padrao existe em SVM, KNN, NaiveBayes, GradientBoosting e SGD.

No runner AppClassNet, `DEFAULT_CLASSIFIER` inclui `RandomForest SupportVectorMachine KNN DecisionTree NaiveBayes GradientBoosting StochasticGradientDescent`. SVM e KNN sao particularmente ruins para datasets grandes; SVM kernelizado tende a memoria/tempo superlinear, e KNN guarda o treino.

## Estimativa aproximada de memoria

Definicoes:

- `N`: linhas no CSV materializado.
- `F = 20`: features.
- `C = 200`: classes.
- `K`: folds (`2` para `-c sf`; `5` para campanhas padrao).
- `X32`: matriz `N x 20 float32` = `80N bytes`.
- `y32`: vetor/coluna `N float32` = `4N bytes`.
- `onehot`: `N x 200 float32` = `800N bytes`.

Exemplo de escala:

| N | X32 | y32 | one-hot 200 classes |
|---:|---:|---:|---:|
| 10M | ~0.75 GiB | ~0.04 GiB | ~7.45 GiB |
| 50M | ~3.73 GiB | ~0.19 GiB | ~37.25 GiB |
| 100M | ~7.45 GiB | ~0.37 GiB | ~74.51 GiB |

Etapas:

1. Preparacao `.npy -> CSV` no runner:
   - X/y memmap: baixo RSS, dependente de pagina ativa.
   - DataFrame por chunk: `chunk_size * 21` colunas; com `chunk_size=100_000`, ordem de dezenas de MB.
   - Com `--prepare_sampling balanced`, `select_balanced_indices` cria arrays de indices; pode crescer ate `max_samples * 8 bytes` mais arrays temporarios por classe.

2. Leitura CSV em `main.py`:
   - `pandas.read_csv`: tende a `N x 21` em `float64/int64`, cerca de `168N bytes`, mais indice/overhead.
   - `replace/dropna` pode criar blocos/copias internas.
   - Labels: `4N bytes`.
   - Features finais: `80N bytes`.
   - Durante `_store_final_data`, DataFrame e ndarray coexistem.

3. Estratificacao:
   - Dataset base: `X32 + y32 ~= 84N bytes`.
   - `shuffle(...)`: mais `~84N bytes`.
   - Por fold guardado em `list_folds`: treino + avaliacao somam aproximadamente outro `~84N bytes`.
   - Total aproximado so para arrays de dados antes de treinar: `base + shuffled + K * folds = (K + 2) * 84N bytes`, sem contar pandas.
   - Para `K=2`: `~336N bytes`; para `K=5`: `~588N bytes`.

4. Treino generativo:
   - One-hot do fold de treino: ate `(K-1)/K * 800N bytes`.
   - Para `K=2`, um fold de treino custa `~400N bytes`; para `N=100M`, isso sozinho passa de 37 GiB.
   - TensorFlow/Keras ainda mantem tensores, buffers, batches e pesos.

5. Sintese e avaliacao:
   - Sinteticos planejados no runner: `200 classes * 256 amostras = 51.200 amostras` por fold quando `number_samples_per_class` e respeitado literalmente; isso e pequeno.
   - Se o plano for proporcional ao split de avaliacao, o custo vira `S * 80 bytes` para features, mais listas Python e DataFrames.
   - `TS-TR` e `TR-TS` duplicam sinteticos em listas e arrays.
   - Classificadores fazem copia `numpy.array` do treino recebido; RandomForest/DecisionTree ainda alocam suas estruturas internas.

## Pontos de explosao de memoria

1. CSV completo em pandas.
   - A preparacao usa memmap, mas o subprocesso perde esse beneficio e le o CSV inteiro.

2. Conversao DataFrame -> ndarray mantendo ambos vivos.
   - `pandas.read_csv` + `data_file.values.astype(float32)` duplicam a matriz completa em um momento critico.

3. Folds materializados antecipadamente.
   - `list_folds` guarda todos os folds; com K folds, praticamente K copias do dataset completo ficam residentes.

4. One-hot `N x 200`.
   - Com 200 classes, one-hot e maior que X por fator 10.
   - Em datasets grandes, essa matriz sozinha pode consumir dezenas de GiB.

5. Salvamento de folds.
   - `_save_data_to_csv` usa `numpy.column_stack`, criando matriz `X+y` completa para treino e avaliacao, depois DataFrame.
   - Esse passo roda para cada fold antes do treino.

6. Avaliacao por listas Python.
   - `data.extend(generated_samples)` transforma ndarrays em listas de linhas, com overhead alto.
   - Depois `shuffle`, `numpy.array(data)` e `DataFrame` recriam copias.

7. Classificadores.
   - Cada classificador converte entrada com `numpy.array`, mesmo se ja for `float32`.
   - RandomForest/DecisionTree constroem arvores em memoria.
   - SVM kernelizado e KNN nao sao bons candidatos para modo batch real; SGD/NaiveBayes/alguns modelos com `partial_fit` sao candidatos melhores.

## Mudancas possiveis sem quebrar o modo normal

1. Adicionar um modo opt-in no runner, por exemplo `--execution_mode normal|batches`, mantendo `normal` como default.
2. No modo batches, chamar `main.py` com `--data_format npy_xy --mmap_npy --train_x_path ... --train_y_path ...` em vez de materializar CSV, preservando o caminho atual para `normal`.
3. Criar um iterador de folds lazy que gere um fold por vez e libere arrays antes do proximo fold, sem alterar o contrato legado em modo normal.
4. Evitar salvar CSV de treino/avaliacao por fold em modo batches, ou salvar por chunk quando `save_intermediate_folds=True`.
5. Trocar `numpy.array(..., dtype=...)` por `numpy.asarray(..., dtype=...)` nos caminhos batch, para evitar copias quando dtype ja coincide.
6. Substituir one-hot completo por:
   - one-hot por batch;
   - sparse/categorical integer labels quando o modelo aceitar;
   - `tf.data.Dataset` que monta labels condicionais em batches.
7. Avaliar metricas em streaming:
   - preditores com `predict` por chunk;
   - metricas de classificacao acumuladas por contadores/confusion matrix;
   - distancias acumuladas por soma incremental.
8. Para comparacao inicial, restringir classificadores do modo batches a modelos com suporte incremental ou memoria previsivel: `StochasticGradientDescent`, `NaiveBayes` com `partial_fit`, possivelmente arvores apenas em amostras limitadas.

## Proposta de arquitetura: modo normal vs modo batches

### Modo normal

Manter o comportamento atual:

- `.npy -> CSV` via runner;
- `main.py` le CSV completo;
- folds materializados em `list_folds`;
- modelos e avaliadores recebem arrays completos;
- todos os classificadores atuais continuam disponiveis.

Esse modo serve como baseline de compatibilidade e comparacao de metricas.

### Modo batches

Fluxo proposto:

1. Runner
   - Novo argumento `--execution_mode batches`.
   - Em batches, nao gerar CSV por default.
   - Passar caminhos `.npy` para `main.py`:
     - `--data_format npy_xy`;
     - `--mmap_npy`;
     - `--train_x_path`, `--train_y_path`;
     - opcionalmente `--valid_x_path`, `--valid_y_path`, `--test_x_path`, `--test_y_path`;
     - `--split_mode cross_validation|provided`.

2. Loader
   - Usar `NpyXYLoader` com memmap.
   - Preservar X como `np.memmap` quando dtype ja for compativel.
   - Converter labels para vetor inteiro compacto (`int32` ou `int64`) uma vez.

3. Folds lazy
   - Gerar indices dos folds, mas nao guardar matrizes de todos os folds.
   - Para cada fold:
     - criar views/chunks ou batches por indices;
     - treinar;
     - avaliar;
     - descartar referencias.
   - Quando indices aleatorios forem grandes, salvar indices em `np.memmap` temporario ou gerar por split provido.

4. Treino generativo
   - Introduzir `BatchDataset`/`tf.data.Dataset` que produz `(X_batch, y_onehot_batch)`.
   - One-hot so dentro do batch.
   - Modelos atuais podem continuar aceitando arrays no modo normal; batch mode usa adaptadores.

5. Sintese
   - Trocar `get_samples(...) -> dict completo` por interface opcional `iter_samples(...)`.
   - Consumidores podem:
     - salvar chunks;
     - treinar classificadores incrementais;
     - calcular metricas streaming.
   - Para compatibilidade, manter `get_samples` no modo normal.

6. Avaliacao
   - `TS-TR`: treinar classificador incremental com chunks sinteticos.
   - `TR-TS`: treinar classificador incremental/limitado com real em chunks e predizer sinteticos em chunks.
   - Metricas de classificacao: acumular matriz de confusao por chunk.
   - Distancias: acumular somas/medias por chunk, sem converter listas inteiras para arrays.

7. Classificadores
   - Criar camada de capacidades:
     - `supports_batch_fit`;
     - `supports_partial_fit`;
     - `requires_full_matrix`.
   - Em batches, bloquear ou amostrar classificadores que exigem matriz completa, com mensagem explicita.

## Plano incremental de implementacao

1. Adicionar flags sem mudar default:
   - `--execution_mode normal|batches`;
   - `--batch_size`;
   - `--batch_classifiers`;
   - `--save_intermediate_folds`.

2. Fazer o runner montar comando `main.py` para `npy_xy` no modo batches.
   - Manter materializacao CSV no modo normal.

3. Criar gerador lazy de folds.
   - Primeiro passo: nao guardar todos os folds em `list_folds`; processar fold a fold.
   - Preservar caminho atual atras do modo normal.

4. Remover salvamento obrigatorio de folds no modo batches.
   - Se necessario, salvar por chunks.

5. Introduzir batches para one-hot.
   - Comecar por modelos Keras que ja aceitam `tf.data.Dataset`.
   - Validar `variational_demo` e `adversarial_demo` primeiro.

6. Adaptar avaliacao streaming.
   - Primeiro implementar predicao por chunk e acumulacao de metricas.
   - Depois adaptar distancias.

7. Classificadores em batch.
   - Comecar com `StochasticGradientDescent` e `NaiveBayes`.
   - Para RandomForest/DecisionTree, manter modo full ou usar amostragem configurada, pois scikit-learn nao oferece `partial_fit` para esses estimadores.

8. Medicao e comparacao.
   - Logar RSS antes/depois de cada etapa.
   - Emitir relatorio por fold com pico aproximado de memoria, tempo e metricas.

## Testes necessarios

1. Compatibilidade do modo normal.
   - `run_appclassnet_top200.py -c sf --dryrun` gera o mesmo comando de hoje quando `execution_mode` nao e informado.
   - Preparacao CSV continua usando `mmap_mode="r"` e chunks.

2. Loader `npy_xy`.
   - Com `--mmap_npy`, `X` permanece `np.memmap` quando possivel.
   - Sem `--mmap_npy`, comportamento atual permanece.
   - Labels top-200 zero-based e one-based sao validados/remapeados conforme configuracao.

3. Folds lazy.
   - Mesmo `random_state=42` gera indices equivalentes ao modo normal para datasets pequenos.
   - Apenas um fold fica residente por vez.
   - `split_mode=provided` usa `valid` ou `test` conforme contrato atual.

4. One-hot por batch.
   - Shapes `(batch, 200)` corretos.
   - Ultimo batch menor funciona.
   - Resultado numerico em dataset pequeno bate com one-hot completo.

5. Sintese streaming.
   - `get_samples` legado continua retornando dicionario no modo normal.
   - `iter_samples` produz mesma contagem por classe no modo batches.
   - Salvamento por chunk gera arquivo com numero correto de linhas e labels.

6. Avaliacao streaming.
   - Accuracy, macro/weighted precision/recall/F1 e balanced accuracy batem com avaliacao completa em dataset pequeno.
   - Predicao por chunk nao altera ordem/contagem de labels.
   - Distancias streaming batem com implementacao atual em amostra pequena.

7. Classificadores.
   - Modo batches rejeita ou amostra classificadores full-matrix com erro controlado.
   - `StochasticGradientDescent`/`NaiveBayes` incremental recebem `classes=np.arange(200)`.
   - RandomForest/DecisionTree continuam funcionando no modo normal.

8. Regressao de memoria.
   - Teste de smoke com `.npy` sintetico grande o suficiente para detectar duplicacao de folds.
   - Validar que pico de RSS no modo batches nao cresce linearmente com `K`.

