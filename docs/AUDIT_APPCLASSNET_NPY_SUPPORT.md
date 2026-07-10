# Auditoria para suporte nativo a AppClassNet top-200 em `.npy`

Data da auditoria: 2026-07-09

## Objetivo

Auditar o fluxo atual do MalDataGen antes de qualquer alteracao de codigo para adicionar suporte ao AppClassNet top-200, que possui arquivos separados `X/y` em `.npy` para `train`, `valid` e `test`.

Restricoes consideradas:

- O fluxo original com CSV unico contendo features e label deve continuar funcionando exatamente como antes.
- Argumentos antigos nao devem ser removidos.
- Comportamento padrao nao deve mudar.
- Exemplos existentes nao devem quebrar.
- Esta auditoria nao altera arquivos de codigo.

## Estrutura relevante do projeto

Arquivos principais:

- `main.py`: classe `SynDataGen`, orquestracao do pipeline, treino, geracao, avaliacao e salvamento.
- `Engine/Arguments/Arguments.py`: agrega todos os parsers e chama `parse_args()`.
- `Engine/Arguments/ArgumentsFramework.py`: argumentos gerais, como `--model_type`, `--number_k_folds`, `--number_samples_per_class`, `--classifier`.
- `Engine/Arguments/ArgumentsDataLoader.py`: argumentos de entrada CSV, como `--data_load_path_file_input` e `--data_load_label_column`.
- `Engine/DataIO/CSVLoader.py`: carregamento CSV, separacao de features/labels, codificacao de labels discretos e salvamento de sinteticos.
- `Engine/Evaluation/CrossValidation.py`: decorator `StratifiedData`, carregamento via `@autoload`, inferencia de classes e montagem de folds.
- `Engine/Models/GenerativeModels.py`: instancia e treina os modelos gerativos.
- `Engine/Algorithms/*/Algorithm*.py`: implementacoes de treino/amostragem por modelo; em geral recebem `number_samples_per_class`.
- `Engine/Evaluation/TrTs.py`: avaliacao train-real/test-synthetic.
- `Engine/Evaluation/TsTr.py`: avaliacao train-synthetic/test-real e metricas de distancia real/sintetico.
- `Engine/Evaluation/TrTr.py`: avaliacao train-real/test-real, atualmente comentada em `main.py`.
- `Engine/Metrics/Metrics.py`: inicializacao e calculo de metricas.
- `Engine/Classifiers/Classifiers.py`: catalogo e treino de classificadores para avaliacao.
- `run_appclassnet_top200.py`: runner externo existente que converte AppClassNet `.npy` para CSV materializado e chama `main.py`.
- `Scripts/converter_npy_to_csv.py`: conversor simples e antigo de `.npy` para CSVs separados.

## Fluxo atual do pipeline

### 1. Parsing de argumentos

Entrada:

- `main.py` instancia `SynDataGen()`.
- `SynDataGen.__init__` usa o decorator `@arguments`.
- `Engine/Arguments/Arguments.py:134-182` inicializa `Arguments`, registra parsers e executa `parse_args()`.

Pontos relevantes:

- `Engine/Arguments/ArgumentsFramework.py:95-96` define `--number_samples_per_class` com default `"1:256,2:256"`.
- `Engine/Arguments/ArgumentsFramework.py:76-88` converte esse argumento em `{"classes": ..., "number_classes": ...}`.
- `Engine/Arguments/ArgumentsFramework.py:138-154` define `--model_type` e seu default atual `adversarial`.
- `Engine/Arguments/ArgumentsDataLoader.py:45-72` define os argumentos CSV e `--data_type`.
- `Engine/Arguments/ArgumentsDataLoader.py:38` define o caminho CSV default como `Datasets/converted/train_x.csv`.
- `Engine/Arguments/ArgumentsDataLoader.py:41` define `DEFAULT_DATA_TYPE = 'binary'`.

Implicacao para AppClassNet:

- Hoje nao ha argumento nativo para `train_x.npy`, `train_y.npy`, `valid_x.npy`, `valid_y.npy`, `test_x.npy`, `test_y.npy`.
- O unico ponto de entrada de dados no core e `--data_load_path_file_input`, assumindo um CSV unico.

### 2. Carregamento CSV

Entrada:

- `main.py:331-333` aplica decorators em `run_experiments`: `@import_metrics`, `@import_classifiers`, `@StratifiedData`.
- `Engine/Evaluation/CrossValidation.py:98-99` aplica `@autoload` no wrapper de cross-validation.
- `Engine/DataIO/CSVLoader.py:496-504` em `autoload` reinicializa `CSVDataProcessor` e chama `self.load_csv()`.

Fluxo no loader:

- `Engine/DataIO/CSVLoader.py:196-203` chama `pandas.read_csv(self._data_load_path_file_input)`.
- `Engine/DataIO/CSVLoader.py:223-230` troca infinitos por NaN e remove linhas com NaN.
- `Engine/DataIO/CSVLoader.py:183-187` aplica sample limit, filtro de colunas, separacao de label e limite de colunas.
- `Engine/DataIO/CSVLoader.py:330-335` armazena features em `self._data_loaded` como `float32`.

Assuncao atual:

- Entrada e sempre um CSV unico.
- Nao existe contrato para arrays `.npy` separados.

### 3. Separacao de features e labels

Local principal:

- `Engine/DataIO/CSVLoader.py:287-300`.

Comportamento:

- Se `--data_load_label_column == -1`, `Engine/DataIO/CSVLoader.py:290-292` usa a ultima coluna do CSV como label.
- `Engine/DataIO/CSVLoader.py:294-295` extrai a coluna de label para `self._data_loaded_labels`.
- `Engine/DataIO/CSVLoader.py:296` chama `_encode_discrete_labels()`.
- `Engine/DataIO/CSVLoader.py:297` remove a coluna de label das features.

Assuncao atual:

- Labels estao dentro do mesmo CSV.
- Por default, label esta na ultima coluna.
- Labels sao carregados como `float32`.

Implicacao para AppClassNet:

- AppClassNet ja vem com `X` e `y` separados; anexar `y` no CSV e apenas um workaround.
- O suporte nativo deve preencher os mesmos atributos (`_data_loaded`, `_data_loaded_labels`, headers e metadados) sem depender de coluna de label.

### 4. Inferencia de classes

Locais principais:

- `Engine/DataIO/CSVLoader.py:302-316`.
- `Engine/Evaluation/CrossValidation.py:70-77`.
- `Engine/Evaluation/CrossValidation.py:117-129`.

Comportamento:

- `_encode_discrete_labels()` verifica se labels sao discretos por `numpy.allclose(labels, numpy.rint(labels))`.
- Se forem discretos, cria mapping zero-based e sobrescreve `self._data_loaded_labels`.
- `_build_class_metadata(labels)` usa `numpy.unique(flat_labels.astype(int), return_counts=True)` e retorna `classes` e `number_classes`.
- O decorator `StratifiedData` sobrescreve `self.arguments.number_samples_per_class` com metadados inferidos do CSV/fold.

Assuncao atual:

- Classe e inferida a partir dos labels carregados.
- Labels discretos podem ser convertidos para inteiro.
- O objeto `number_samples_per_class` e esperado pelos modelos.

Implicacao para AppClassNet:

- Para top-200, a inferencia deve preservar exatamente o dominio de 200 classes, mesmo quando um split/fold nao contem todas as classes.
- A codificacao zero-based e compatível se os labels AppClassNet forem `0..199`. Se forem `1..200`, o loader atual recodifica para `0..199`; a saida salva usa `_decode_label()`.

### 5. Cross-validation

Local principal:

- `Engine/Evaluation/CrossValidation.py:79-230`.

Comportamento:

- `@StratifiedData` carrega dados via CSV antes de rodar `run_experiments`.
- `Engine/Evaluation/CrossValidation.py:114-115` embaralha `self._data_loaded` e `self._data_loaded_labels`.
- `Engine/Evaluation/CrossValidation.py:131-135` usa `StratifiedKFold` para labels discretos.
- `Engine/Evaluation/CrossValidation.py:136-140` usa `KFold` para labels nao discretos.
- `Engine/Evaluation/CrossValidation.py:171-187` salva CSVs auxiliares de treino/avaliacao por fold.
- `Engine/Evaluation/CrossValidation.py:200-213` monta `self.list_folds` com:
  - `x_training_real`
  - `y_training_real`
  - `x_evaluation_real`
  - `y_evaluation_real`
  - campos sinteticos inicialmente `None`

Observacao arquitetural:

- O restante do pipeline usa `self.list_folds`, nao sabe diretamente se os dados vieram de CSV.
- Este e o melhor ponto de extensao para AppClassNet: um novo produtor de folds pode popular o mesmo contrato sem mudar `run_experiments`.

Risco:

- Se o suporte `.npy` alterar `StratifiedData` diretamente, pode quebrar o fluxo CSV.
- Melhor adicionar uma selecao opt-in de backend/fold provider, mantendo o caminho atual como default.

### 6. Treino dos modelos gerativos

Local principal:

- `main.py:386-389` chama `self.train_model(...)`.
- `main.py:489-581` implementa `train_model`.
- `Engine/Models/GenerativeModels.py` implementa os metodos especificos por modelo.

Comportamento:

- `main.py:531-538` prepara labels com `_prepare_labels_for_conditional_generation()` e chama `self.training_model(...)`.
- `main.py:432-438` transforma labels discretos em `int`; labels nao discretos viram pseudo-classe zero.
- Modelos condicionais usam `to_categorical(..., num_classes=self._number_samples_per_class["number_classes"])`, por exemplo:
  - adversarial: `Engine/Models/GenerativeModels.py:262-265`;
  - autoencoder: `Engine/Models/GenerativeModels.py:582-585`;
  - wasserstein: `Engine/Models/GenerativeModels.py:2008-2012`;
  - wasserstein_gp: `Engine/Models/GenerativeModels.py:2442-2446`;
  - variational, quantized e diffusion tambem possuem chamadas equivalentes em `GenerativeModels.py`.

Assuncao atual:

- Labels usados no treino sao inteiros e compativeis com one-hot.
- `self._number_samples_per_class["number_classes"]` existe.

Implicacao para AppClassNet:

- Top-200 exige `number_classes=200` quando o modelo usa one-hot.
- Deve haver garantia de que labels estao no intervalo esperado por `to_categorical`.
- O suporte nativo a `.npy` deve preencher `_number_samples_per_class` antes de `import_models/training_model`.

### 7. Geracao por classe

Local principal:

- `main.py:584-694` implementa `synthesize_data`.
- `main.py:468-487` cria metadados de geracao.
- `Engine/Algorithms/*/get_samples()` gera dados por classe.

Comportamento:

- `main.py:591` chama `_build_generation_metadata(y_real_samples)`.
- O resultado tem formato `{"classes": {class_id: count}, "number_classes": N, "data_type": ...}`.
- `main.py:594-680` despacha para `get_samples(number_samples_per_class)` do algoritmo selecionado.
- Exemplo no autoencoder: `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py:283-312` itera `for label_class, number_instances in number_samples_per_class["classes"].items()`.
- Exemplo no adversarial: `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py:313-360` tambem gera por classe e usa one-hot.
- Exemplo no SMOTE: `Engine/Algorithms/SMOTE/AlgorithmSMOTE.py:281-324` usa `class_sample_map["classes"]`.

Assuncao atual:

- Toda geracao e class-condicional.
- `number_samples_per_class` e obrigatorio no formato com `classes`.
- A saida sintetica e um dicionario `{label_class: generated_samples}`.

Implicacao para AppClassNet:

- Top-200 se encaixa no contrato class-condicional, desde que `number_classes=200` e counts por classe estejam corretos.
- O fluxo atual gera sinteticos na distribuicao do fold de avaliacao, porque `_build_generation_metadata()` usa `y_evaluation_real` passado por `synthesize_data`.

### 8. Avaliacao TR-TS, TS-TR e equivalentes

Locais:

- `main.py:407-408` chama `evaluation_TR_TS` e `evaluation_TS_TR`.
- `main.py:410` deixa `evaluation_TR_TR` comentado.
- `Engine/Evaluation/TrTs.py:58-100`.
- `Engine/Evaluation/TsTr.py:58-121`.
- `Engine/Evaluation/TrTr.py` existe, mas nao roda por default.

TR-TS:

- `Engine/Evaluation/TrTs.py:72-81` reconstrói arrays `labels` e `data` a partir do dicionario sintetico por classe.
- `Engine/Evaluation/TrTs.py:87-90` treina classificadores em `x_evaluation_real`/`y_evaluation_real`.
- `Engine/Evaluation/TrTs.py:93-100` prediz sobre dados sinteticos e calcula metricas.

TS-TR:

- `Engine/Evaluation/TsTr.py:72-81` reconstrói arrays sinteticos.
- `Engine/Evaluation/TsTr.py:83-98` treina classificadores em sinteticos e avalia em `x_evaluation_real`.
- `Engine/Evaluation/TsTr.py:103-121` calcula metricas de distancia entre `x_training_real` e sinteticos.

Metricas:

- `Engine/Metrics/Metrics.py:135-150` inicializa metricas binarias.
- `Engine/Metrics/Metrics.py:263-274` usa metricas macro/weighted para `data_type != binary`.
- `Engine/Metrics/Metrics.py:159-163` remove Hamming/Jaccard para `continuous`.

Assuncao atual:

- Avaliacao principal e classificatoria.
- Labels discretos sao esperados para classificadores.
- Dados sinteticos chegam como dicionario por classe.

Implicacao para AppClassNet:

- Top-200 e classificacao multiclasse; deve usar `--data_type multiclass` para semantica de metricas, ou `continuous` se o objetivo principal for preservar features e remover Hamming/Jaccard. O runner atual usa `continuous` para evitar arredondamento.
- Ha tensao entre "features continuas" e "target multiclasse": `data_type` atual mistura semanticas de feature e target.

### 9. Salvamento dos dados sinteticos

Local:

- `main.py:690-692` chama `self.save_data_generated()` quando `--save_data True`.
- `main.py:697-708` usa decorator `@autosave`.
- `Engine/DataIO/CSVLoader.py:390-431` implementa `save_csv`.

Comportamento:

- `CSVLoader.save_csv()` percorre `generated_data.items()`.
- Reconstroi `labels` repetindo a chave da classe.
- Usa `_decode_label()` para restaurar label original quando houve encoding discreto.
- Cria DataFrame com `self._data_loaded_header[:-1]` para features e `self._data_loaded_header[-1]` para label.
- Salva em `DataOutput_K_fold_{fold_number}_{generator_name}.txt`.

Assuncao atual:

- Existe header de CSV original.
- A ultima coluna do header e a coluna de label.
- Sinteticos estao agrupados por classe.

Implicacao para AppClassNet:

- Suporte `.npy` nativo deve criar headers sinteticos equivalentes (`f0..fN`, `label`) para manter `save_csv()` funcionando.
- Se o modo AppClassNet usar nomes de colunas diferentes, precisa preservar compatibilidade com os plots e CSV output.

## Onde o codigo assume os pontos solicitados

### CSV unico

- `Engine/Arguments/ArgumentsDataLoader.py:45-46`: apenas um `--data_load_path_file_input`.
- `Engine/DataIO/CSVLoader.py:196-203`: `pandas.read_csv()` em um unico arquivo.
- `Engine/DataIO/CSVLoader.py:287-300`: label extraido do mesmo DataFrame.
- `Engine/Evaluation/CrossValidation.py:98-99`: `@autoload` sempre carrega via CSV antes do split.
- `run_appclassnet_top200.py:441-533`: workaround atual materializa `.npy` separados em um CSV unico.

### Label na ultima coluna

- `Engine/Arguments/ArgumentsDataLoader.py:33`: `DEFAULT_DATA_LOAD_LABEL_COLUMN = -1`.
- `Engine/DataIO/CSVLoader.py:290-292`: se `-1`, usa `data_file.columns[-1]`.
- `Engine/DataIO/CSVLoader.py:333-334`: header final e `features + [label]`.
- `Engine/DataIO/CSVLoader.py:410-411`: salvamento assume `header[:-1]` como features e `header[-1]` como label.

### Classificacao binaria

- `Engine/Arguments/ArgumentsDataLoader.py:41`: `DEFAULT_DATA_TYPE = 'binary'`.
- `Engine/Arguments/ArgumentsFramework.py:95`: default `number_samples_per_class` tem duas classes (`1:256,2:256`).
- `Engine/Arguments/ArgumentsAdversarial.py:38,48-49`: `Sigmoid` e `binary_crossentropy`.
- `Engine/Arguments/ArgumentsVariationalAutoencoder.py:41-49`: 2 classes, `binary_crossentropy`, `sigmoid`.
- `Engine/Arguments/ArgumentsWassersteinGAN.py:40-46`: 2 classes, `binary_crossentropy`, `sigmoid`.
- `Engine/Arguments/ArgumentsWassersteinGANGP.py:40-46`: 2 classes, `binary_crossentropy`, `sigmoid`.
- `Engine/Metrics/Metrics.py:135-150`: metricas binarias continuam sendo o conjunto default.

### Labels one-hot

- `Engine/Models/GenerativeModels.py:262-265`: adversarial.
- `Engine/Models/GenerativeModels.py:582-585`: autoencoder.
- `Engine/Models/GenerativeModels.py:2008-2012`: wasserstein.
- `Engine/Models/GenerativeModels.py:2442-2446`: wasserstein_gp.
- `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py:283-304`: one-hot na geracao.
- `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py:335-352`: one-hot na geracao.
- Ha ocorrencias equivalentes em quantized, variational e diffusion.

### `number_classes=2`

- `Engine/Arguments/ArgumentsAutoencoder.py:49`.
- `Engine/Arguments/ArgumentsVariationalAutoencoder.py:41`.
- `Engine/Arguments/ArgumentsWassersteinGAN.py:40`.
- `Engine/Arguments/ArgumentsWassersteinGANGP.py:40`.
- `main.py:450-460` tambem consulta argumentos de `*_number_classes` como candidatos ao dominio de classes.

### `number_samples_per_class` obrigatorio

- `Engine/Arguments/ArgumentsFramework.py:95-96`: argumento geral default.
- `Engine/Evaluation/CrossValidation.py:117-129`: sobrescrito com metadados inferidos.
- `main.py:468-487`: cria metadados de geracao no mesmo formato.
- `main.py:591-680`: todos os algoritmos recebem esse objeto.
- `Engine/Models/*/Vanilla*` e `DiffusionModelUnet` frequentemente validam que `number_samples_per_class` possui `number_classes`.

## Riscos de quebra de compatibilidade

1. Mudar o loader default de CSV para NPY quebraria usuarios existentes. O suporte `.npy` precisa ser opt-in.
2. Alterar `--data_load_path_file_input` ou `--data_load_label_column` quebraria scripts e exemplos.
3. Trocar o decorator `@StratifiedData` sem preservar `@autoload` pode mudar a ordem de inicializacao de `CSVDataProcessor`, `Metrics`, `Classifiers` e `GenerativeModels`.
4. Alterar o formato de `self.list_folds` quebraria `run_experiments`, avaliacao e modelos.
5. Alterar o formato de `self.data_generated` quebraria avaliacao e `save_csv`.
6. Mudar defaults como `data_type='binary'`, `model_type='adversarial'`, `number_samples_per_class='1:256,2:256'` ou `number_k_folds=5` mudaria comportamento historico.
7. Inferir sempre 200 classes em qualquer dataset quebraria datasets CSV pequenos; isso deve ser feito apenas no modo AppClassNet/NPY.
8. Usar splits `train/valid/test` diretamente pode mudar a semantica atual de K-fold. Deve ser uma opcao nova, nao substituicao do K-fold CSV.
9. AppClassNet top-200 pode ter classes ausentes em subsets limitados; `to_categorical` exige labels dentro do dominio definido.
10. O runner atual usa `data_type=continuous` para evitar arredondamento de features; mudar para `multiclass` sem outra flag poderia voltar a arredondar dados sinteticos em alguns algoritmos.

## Plano incremental de modificacao

### Fase 0: preservar contrato atual

Nao alterar comportamento default. O comando historico `python main.py` deve continuar:

- lendo `--data_load_path_file_input` como CSV;
- usando `--data_load_label_column=-1`;
- rodando `StratifiedKFold`;
- salvando sinteticos pelo `CSVLoader.save_csv()`.

### Fase 1: introduzir selecao opt-in de fonte de dados

Adicionar novos argumentos sem remover antigos:

- `--data_source {csv,npy}` com default `csv`;
- `--npy_train_x`, `--npy_train_y`;
- `--npy_valid_x`, `--npy_valid_y`;
- `--npy_test_x`, `--npy_test_y`;
- `--npy_split_mode {kfold,holdout}` ou similar;
- `--npy_feature_prefix f` e `--npy_label_name label` opcionais;
- `--npy_number_classes` opcional para fixar `200` no AppClassNet.

Compatibilidade:

- Se `data_source=csv`, ignorar todos os argumentos `.npy`.
- Defaults antigos permanecem iguais.

### Fase 2: criar um loader NPY paralelo ao CSV

Criar algo como `Engine/DataIO/NPYLoader.py` ou `AppClassNetNPYLoader.py` com responsabilidades equivalentes ao `CSVDataProcessor`:

- carregar `X` e `y` com `numpy.load(..., mmap_mode='r')` quando apropriado;
- validar `X.ndim == 2`, `y.ndim in {1,2}`, `len(X) == len(y)`;
- converter `X` para `float32` apenas se o modo atual esperar isso;
- normalizar shape de `y` para `(n, 1)`;
- preencher `_data_loaded`, `_data_loaded_labels`, `_data_loaded_header`, `_data_original_header`;
- reutilizar ou extrair a logica de encoding discreto para evitar duplicacao.

Compatibilidade:

- Nao mudar `CSVDataProcessor`.
- Nao mudar o contrato dos atributos consumidos pelo restante do pipeline.

### Fase 3: separar produtor de folds

Adicionar uma camada de fold provider:

- `CSVKFoldProvider`: comportamento atual, incluindo `@autoload`.
- `NPYKFoldProvider`: carrega `train_x/train_y` e aplica K-fold no treino, se desejado.
- `NPYHoldoutProvider`: usa `train` para treino e `valid` ou `test` como avaliacao, sem K-fold.

Contrato de saida deve continuar:

```python
{
    "x_training_real": ...,
    "y_training_real": ...,
    "x_evaluation_real": ...,
    "y_evaluation_real": ...,
    "x_training_synthetic": None,
    "y_training_synthetic": None,
    "x_evaluation_synthetic": None,
    "y_evaluation_synthetic": None,
}
```

Compatibilidade:

- `main.py:361-408` nao deveria precisar saber se o fold veio de CSV ou NPY.

### Fase 4: garantir dominio de classes para top-200

No modo `data_source=npy` com `--npy_number_classes 200`:

- preservar `number_classes=200` mesmo se o fold/subset nao tiver todas as classes;
- validar labels antes de `to_categorical`;
- emitir erro claro se labels estiverem fora de `[0, number_classes - 1]`;
- se labels forem `1..200`, permitir mapeamento zero-based configuravel e decodificacao no salvamento.

Compatibilidade:

- Para CSV, continuar inferindo classes como hoje.

### Fase 5: resolver a tensao `continuous` vs `multiclass`

Hoje `data_type` mistura:

- tipo das features para pos-processamento (`continuous` evita `numpy.rint`);
- tipo da tarefa para metricas (`binary` vs metricas multiclasse).

Plano seguro:

- Nao mudar `--data_type` agora.
- Adicionar uma flag opt-in futura como `--feature_value_type {binary,continuous}` ou `--round_generated_data`.
- Para AppClassNet, recomendar `--data_type continuous` enquanto nao houver separacao melhor, pois preserva features.

### Fase 6: adaptar runner AppClassNet para modo nativo

Depois do suporte core:

- `run_appclassnet_top200.py` pode escolher entre:
  - modo legado: materializa CSV e chama `main.py`;
  - modo nativo: passa paths `.npy` ao `main.py`.
- O modo legado pode permanecer como fallback.

## Testes necessarios

### Testes de compatibilidade CSV original

1. `python main.py` com defaults deve continuar chamando o fluxo CSV.
2. `--data_load_path_file_input <csv>` com label na ultima coluna deve carregar, separar labels e criar folds.
3. `--data_load_label_column <coluna>` deve continuar funcionando.
4. CSV com labels `1,2` deve continuar sendo recodificado para `0,1` internamente e decodificado no salvamento.
5. `--number_k_folds 2` deve criar exatamente dois folds no formato atual.
6. `--save_data True` deve continuar salvando `DataOutput_K_fold_*_<model>.txt` com as colunas originais.
7. `--data_type binary` deve manter metricas binarias.
8. `--data_type continuous` deve continuar evitando `numpy.rint()` nos algoritmos ja adaptados.
9. Scripts existentes como `run_experiments.py` e comandos documentados no README devem continuar parseando sem novos argumentos obrigatorios.
10. Nenhum novo argumento `.npy` deve ser necessario para o modo CSV.

### Testes unitarios para loader NPY

1. Carregar `X.npy` 2D e `y.npy` 1D com shapes compativeis.
2. Rejeitar `X` nao 2D.
3. Rejeitar `y` com tamanho diferente de `X`.
4. Converter `y` para shape `(n, 1)`.
5. Criar headers `f0..fN,label`.
6. Inferir classes corretamente para labels `0..199`.
7. Preservar `number_classes=200` quando configurado, mesmo se subset tiver menos classes.
8. Falhar com mensagem clara se label estiver fora do dominio configurado.

### Testes de fold provider NPY

1. Modo `kfold`: criar `list_folds` com o mesmo contrato do CSV.
2. Modo `holdout`: usar `train` como treino e `valid`/`test` como avaliacao.
3. Garantir que os folds possuem arrays `float32` para features e labels no formato esperado.
4. Garantir que `self._number_samples_per_class` e `arguments.number_samples_per_class` estao preenchidos antes do treino.

### Testes de integracao AppClassNet top-200

1. Rodar um subset pequeno balanceado com `--data_source npy --npy_number_classes 200`.
2. Treinar pelo menos um modelo simples/rapido (`copy`, `smote` ou demo de `variational`) com `number_k_folds=2`.
3. Verificar que `to_categorical` recebe labels validos.
4. Verificar que a saida sintetica contem labels decodificados corretamente.
5. Verificar que `Results.json` contem chaves esperadas para `TS-TR`, `TR-TS`, `DistanceMetrics` e `EfficiencyMetrics`.

## Recomendacao arquitetural final

O suporte AppClassNet `.npy` deve entrar como uma nova fonte de dados e uma nova estrategia de montagem de folds, nao como alteracao do `CSVDataProcessor`. O ponto de integracao mais seguro e antes de `run_experiments` consumir `self.list_folds`: se o modo novo produzir exatamente o mesmo contrato de folds e metadados, o treino, geracao, avaliacao e salvamento podem ser reaproveitados com menor risco.

Prioridade de preservacao:

1. Manter `data_source=csv` como default.
2. Manter todos os argumentos antigos e seus defaults.
3. Manter o formato de `self.list_folds`.
4. Manter o formato de `self.data_generated`.
5. Manter `save_csv()` funcionando com headers sinteticos quando a origem for NPY.

O runner atual `run_appclassnet_top200.py` prova que AppClassNet pode ser encaixado no pipeline ao materializar CSV. A evolucao nativa deve remover essa materializacao como requisito, mas sem retirar o caminho CSV legado.
