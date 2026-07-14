# Full Pipeline Audit - MalDataGen AppClassNet top-200

Data da auditoria: 2026-07-13

Escopo: varredura do repositorio MalDataGen para entender o pipeline atual, com foco em AppClassNet top-200, TR-TR, TR-TS e TS-TR. Esta etapa nao altera codigo de execucao; a unica escrita e este relatorio.

Referencia de regressao ja validada para AppClassNet top-200:

- `DecisionTreeClassifier`.
- 1.000 amostras por classe no treino.
- 500 amostras por classe no teste.
- 200 classes, labels esperadas entre `0` e `199`.
- Accuracy aproximada: `0.68673`.
- Macro-F1 aproximado: `0.68630`.
- Duracao aproximada: 4 segundos.
- Nivel aleatorio em 200 classes: `1/200 = 0.005`.

## Sumario executivo

O projeto hoje tem dois caminhos principais:

1. Caminho original: `main.py` carrega CSV unico, monta folds via cross-validation e executa treinamento gerativo + avaliacoes sinteticas.
2. Caminho AppClassNet: `run_appclassnet_top200.py` orquestra NPY `X/y` separados. Em `normal`, materializa CSV e chama `main.py` no fluxo legado. Em `batches`, chama `main.py` com `data_format=npy_xy`, `split_mode=provided`, mmap opcional e sintenticos em batches.

O contrato AppClassNet esta parcialmente adequado: loaders carregam sem escalar, labels multiclass sao validadas como zero-based, `FeatureTransformManager` centraliza scalers e `ScaleGuard` bloqueia comparacoes real/sintetico em espacos diferentes. O default de AppClassNet no runner e `source/preserve`, sem `MinMaxScaler` ou `StandardScaler`.

Achados principais:

- **Critico: nenhum achado critico confirmado nesta auditoria.**
- **Alto:** `run_appclassnet_top200.py` modo `normal` nao usa os splits oficiais train/valid/test para TR-TS/TS-TR; ele materializa um CSV do split escolhido e `main.py` faz cross-validation sobre esse CSV. Evidencia: `run_appclassnet_top200.py`, funcoes `materialize_appclassnet_csv()` e `build_main_command()`, linhas ~1451-1625; `Engine/Evaluation/CrossValidation.py`, `StratifiedData.wrapper()`, linhas ~443-582.
- **Alto:** `split_mode=provided` cria um unico fold com `valid` como avaliacao quando valid e test existem, e marca `test` como nao suportado no pipeline TR-TS/TS-TR. Evidencia: `Engine/Evaluation/CrossValidation.py`, `_build_provided_split_folds()`, linhas ~357-380.
- **Alto:** no TS-TR normal, as metricas preditivas sao truncadas por `total_samples` sintetico, podendo avaliar apenas prefixo do split real. Evidencia: `Engine/Evaluation/TsTr.py`, `evaluation_TS_TR()`, linhas ~212-216.
- **Alto:** geracao normal materializa todos os sinteticos em memoria como `dict[class_id] -> ndarray/list`; `generation_batch_size` so e respeitado nos caminhos incrementais de `execution_mode=batches`. Evidencia: `main.py`, `SynDataGen.synthesize_data()`, linhas ~855-1002; `_synthesize_data_incremental()`, linhas ~1208-1305.
- **Medio:** `run_appclassnet_top200.py` sempre cria NPY pre-processado em `preprocess_appclassnet_splits()`, mesmo quando `feature_transform=preserve`, duplicando I/O e espaco em disco sem mudar escala. Evidencia: linhas ~850-1006.
- **Medio:** TR-TR nao roda no fluxo sintetico padrao de `main.py`; a chamada esta comentada, e o TR-TR validado vive no caminho separado `--baseline_real_only`. Evidencia: `main.py`, `SynDataGen.run_experiments()`, linha ~502; `run_appclassnet_top200.py`, `run_real_real_baseline()`, linhas ~1009-1139.
- **Medio:** o help de `--scaler` em `ArgumentsDataLoader.py` ainda afirma que o runner AppClassNet defaulta para minmax, mas o runner atual define `DEFAULT_SCALER="none"`. Evidencia: `Engine/Arguments/ArgumentsDataLoader.py`, `add_argument_data_load()`, linhas ~218-221; `run_appclassnet_top200.py`, constante linha ~42.
- **Baixo:** metricas de eficiencia ainda inicializam zeros, mas metricas preditivas/distancia inicializam `not_applicable` e avaliacoes puladas sao marcadas explicitamente. Evidencia: `Engine/Metrics/Metrics.py`, `__initialize_dictionary()`, linhas ~203-290.

## Diagrama textual do fluxo completo

```text
CLI
  -> main.py
     -> Engine/Arguments/Arguments.py
        -> add_argument_framework()
        -> add_argument_data_load()
        -> validate_data_load_arguments()
     -> Engine/Evaluation/CrossValidation.py::StratifiedData
        -> CSV: CSVDataProcessor.load_csv()
        -> NPY: NpyXYLoader.load() -> DatasetBundle/SplitData/DatasetSchema
        -> batch limits / selecao estratificada / mmap
        -> cross_validation folds OU split_mode=provided fold unico
     -> main.py::SynDataGen.run_experiments()
        -> FeatureTransformPolicy.for_profile()
        -> ModelInputAdapter.fit_generator() somente no treino
        -> ModelInputAdapter.transform_generator_input()
     -> Engine/Models/GenerativeModels.py::training_model()
        -> algoritmo gerativo selecionado
        -> labels zero-based -> one-hot
     -> main.py::SynDataGen.synthesize_data()
        -> get_samples()
        -> inverse_synthetic_batch/collection se configurado
        -> SyntheticBatchWriter ou CSV legado
     -> run_synthetic_evaluation_modes()
        -> TR-TS
        -> TS-TR
        -> ScaleGuard valida mesmo data_space
     -> Metrics.get_task_metrics()/get_distance_metrics()
     -> Metrics.save_dictionary_to_json()

CLI AppClassNet
  -> run_appclassnet_top200.py
     -> diagnostic_only OU baseline_real_only OU campanha sintetica
     -> preprocess_appclassnet_splits()
     -> normal: materialize_appclassnet_csv() -> main.py CSV/cross_validation
     -> batches: main.py npy_xy/split_mode=provided
     -> Results.json / metrics.json / manifestos
```

## 1. Entrada e preparacao dos dados

### CLI geral

- Arquivo: `Engine/Arguments/Arguments.py`
- Classe/funcao: `Arguments.__init__()`, linhas ~208-253.
- Entradas: argumentos de framework, loader, modelos, classificadores e preprocessing.
- Saidas: `self.arguments` validado e diretorios criados.
- Formatos: apenas configuracao.
- Shapes esperados: ainda nao aplicavel.
- Copias em memoria: nenhuma nesta etapa.
- Mudancas de escala: `_normalize_preprocessing_arguments()` so traduz aliases; nao aplica scaler.
- Risco: **baixo**. Mantem `data_format=csv`, `split_mode=cross_validation`, `source_profile=legacy_csv` por default.

- Arquivo: `Engine/Arguments/ArgumentsFramework.py`
- Classe/funcao: `add_argument_framework()`, linhas ~124-262.
- Entradas: `--number_samples_per_class`, `--sample_plan`, `--classifier`, `--eval_classifier`, `--evaluation_mode`, `--normal_classifier`, `--generation_strategy`, `--number_k_folds`, `--model_type`.
- Saidas: parser argparse.
- Formatos: `number_samples_per_class` vira `{"classes": {...}, "number_classes": ...}`.
- Shapes: nao aplicavel.
- Copias em memoria: nenhuma.
- Mudancas de escala: nenhuma.
- Risco: **baixo**. Nao remove argumentos antigos e preserva classificadores legados por default.

- Arquivo: `Engine/Arguments/ArgumentsDataLoader.py`
- Classe/funcao: `add_argument_data_load()`, linhas ~148-302; `validate_data_load_arguments()`, linhas ~69-146.
- Entradas CSV: `--data_load_path_file_input`, `--data_load_label_column`, limites de linhas/colunas.
- Entradas NPY: `--data_format npy_xy`, paths train/valid/test, `--split_mode`, `--target_type`, `--feature_type`, `--num_classes`, `--mmap_npy`.
- Saidas: argumentos validados.
- Shapes esperados AppClassNet: `X=(n,20)`, `y=(n,)`, `num_classes=200`.
- Copias em memoria: nenhuma direta.
- Mudancas de escala: nenhuma; `--scaler`, `--feature_transform`, `--generator_transform`, `--classifier_transform` apenas configuram politicas.
- Risco: **medio**. Help de `--scaler` esta desatualizado sobre AppClassNet minmax, linhas ~218-221.

### CLI AppClassNet

- Arquivo: `run_appclassnet_top200.py`
- Classe/funcao: `build_parser()`, linhas ~1912-2200; `main()`, linhas ~2203-2443.
- Entradas: `--raw_root`, `--converted_root`, `--execution_mode normal|batches`, `--use_mmap`, `--baseline_real_only`, `--diagnostic_only`, `--scaler`, `--feature_transform`, `--generator_transform`, `--classifier_transform`, `--evaluation_space`, quotas por classe.
- Saidas: comandos para `main.py`, CSV preparado, NPY pre-processado, `metrics.json`, `Results.json`, manifestos.
- Formatos: NPY separado `train_x.npy/train_y.npy/valid_x.npy/valid_y.npy/test_x.npy/test_y.npy`; CSV `f0..f19,label`.
- Shapes esperados: 20 features, 200 classes.
- Copias em memoria: em `normal`, pandas carrega CSV completo no subprocesso; em `batches`, mmap e selecoes por classe reduzem carga, mas subsets podem materializar arrays.
- Mudancas de escala: por default `source/preserve`; se `--scaler minmax|standard` ou transform explicito, ajusta somente no treino.
- Risco: **alto**. Modo normal converte um split para CSV e cai no contrato legado de folds, nao nos splits oficiais.

### CSV loader

- Arquivo: `Engine/DataIO/CSVLoader.py`
- Classe/funcao: `CSVDataProcessor.load_csv()`, linhas ~164-195.
- Entradas: CSV unico.
- Saidas: `_data_loaded` em `float32`, `_data_loaded_labels`, headers.
- Formatos: DataFrame pandas; label default ultima coluna quando `data_load_label_column == -1`.
- Shapes: X `(n,m)`, y `(n,1)` apos encoding.
- Copias em memoria: `pandas.read_csv()` carrega tudo; `data_file.values.astype(numpy.float32)` copia features, linha ~334; labels viram `float32`, linha ~296.
- Mudancas de escala: nenhuma; `_normalize_data()` esta desativado, linhas ~372-376.
- Risco: **medio**. `_encode_discrete_labels()` remapeia labels discretas para zero-based automaticamente, linhas ~303-317. Isso e util para modelos condicionais, mas altera labels internas do CSV legado.

### NPY loader, DatasetBundle, SplitData, DatasetSchema

- Arquivo: `Engine/DataIO/NpyXYLoader.py`
- Classe/funcao: `NpyXYLoader.load()`, linhas ~87-139; `_load_split()`, linhas ~146-161.
- Entradas: paths NPY, `mmap_mode`, `dtype`, `target_type`, `feature_type`, `num_classes`, `source_profile`.
- Saidas: `DatasetBundle(train, valid, test, schema, metadata)`.
- Formatos: `SplitData.X` 2D ndarray/memmap; `SplitData.y` 1D inteiro.
- Shapes AppClassNet: `X=(n,20)`, `y=(n,)`.
- Copias em memoria: `numpy.load(..., mmap_mode="r")` evita carregar X inteiro; `_collect_labels()` concatena labels dos splits, linhas ~276-281.
- Mudancas de escala: nenhuma. Para `source_profile=appclassnet_top200`, registra range `(-0.5,0.5)`, `already_normalized=True`, `data_space="source"`, linhas ~107-124.
- Risco: **baixo** para escala; **medio** para memoria de labels concatenados.

- Arquivo: `Engine/DataIO/DatasetContracts.py`
- Classes: `DatasetSchema`, linhas ~43-110; `SplitData`, linhas ~113-160; `SamplePlan`, linhas ~162-208; `DatasetBundle`, linhas ~210-282.
- Entradas: arrays e metadata.
- Saidas: contratos validados.
- Formatos: dataclasses internas.
- Shapes: X 2D; y 1D ou espremivel.
- Copias em memoria: `SplitData.__post_init__()` usa `numpy.asarray(self.X)`, linha ~125; normalmente preserva memmap, mas converte wrappers.
- Mudancas de escala: nenhuma.
- Risco: **baixo**. `DatasetBundle._validate_split()` rejeita labels fora de `[0,num_classes-1]`, linhas ~251-273.

### SamplePlan

- Arquivo: `Engine/DataIO/SamplePlanner.py`
- Classe/funcao: `build_sample_plan_from_args()`, linhas ~27-89; `sample_plan_to_legacy_metadata()`, linhas ~92-103.
- Entradas: argumentos, labels treino, `number_classes`, `data_type`.
- Saidas: `SamplePlan` e metadata legada `{"classes": ..., "number_classes": ..., "data_type": ...}`.
- Formatos: labels 1D inteiras; contagens por classe.
- Shapes: opera sobre y `(n,)`.
- Copias em memoria: converte labels para vetor inteiro; contagens pequenas por classe.
- Mudancas de escala: nenhuma.
- Risco: **baixo**. Para `npy_xy`, exige plano explicito se nao houver `number_samples_per_class`, linhas ~42-47.

### Selecao estratificada e mmap

- Arquivo: `Engine/DataIO/StratifiedNpySelection.py`
- Funcao: `select_stratified_indices_from_npy()`, linhas ~17-96.
- Entradas: `y_path`, `samples_per_class`, `num_classes`, seed, mmap.
- Saidas: indices `int64` embaralhados e relatorio global.
- Formatos: y NPY 1D.
- Shapes: ate `samples_per_class * num_classes`.
- Copias em memoria: reservatorios por classe e vetor final de indices; nao carrega X.
- Mudancas de escala: nenhuma.
- Risco: **baixo**. Escaneia y inteiro e evita pegar apenas blocos iniciais em dados ordenados por classe.

- Arquivo: `Engine/DataIO/BatchNpyDataset.py`
- Classe/funcao: `BatchNpyDataset.__init__()`, linhas ~36-70; `iter_batches()`, linhas ~195-207.
- Entradas: `x_path`, `y_path`, `batch_size`, `mmap_mode`, limites por amostra/classe.
- Saidas: batches `(x_batch, y_batch)`.
- Formatos: NPY X/y.
- Shapes: X batch `(b,20)`, y batch `(b,)`.
- Copias em memoria: indexacao avancada por indices materializa batch; `astype(copy=False)` evita copia quando dtype ja bate.
- Mudancas de escala: nenhuma.
- Risco: **medio**. Selecao estratificada materializa indices proporcionais ao subset selecionado.

### Modo normal vs batches

- Arquivo: `Engine/Evaluation/CrossValidation.py`
- Classe/funcao: `load_dataset_from_args()`, linhas ~247-283; `_apply_bundle_to_owner()`, linhas ~285-320; `StratifiedData.wrapper()`, linhas ~408-590.
- Entradas normal: CSV completo.
- Entradas batches: `DatasetBundle` NPY e `split_mode`.
- Saidas: `self.list_folds` no contrato legado.
- Formatos:
  - Cross-validation: varios dicts com `x_training_real`, `y_training_real`, `x_evaluation_real`, `y_evaluation_real`.
  - Provided: um dict/fold unico.
- Shapes: train/evaluation conforme split ou fold; AppClassNet 20 colunas.
- Copias em memoria: `_apply_bundle_to_owner()` converte `bundle.train.X` para `float32`, linhas ~289 e ~312; `_apply_stratified_split_selection()` materializa `split.X[indices]`, linhas ~202-203.
- Mudancas de escala: nenhuma.
- Risco: **alto**. Em `provided`, `valid` tem prioridade sobre `test`; se ambos existem, `test` e marcado como nao suportado, linhas ~357-365.

## 2. Pre-processamento e espacos numericos

### Politica e scalers

- Arquivo: `Engine/Preprocessing/FeatureTransformManager.py`
- Classe/funcao: `FeatureTransformPolicy.for_profile()`, linhas ~146-206.
- Entradas: `source_profile`, `feature_transform`, `generator_transform`, `classifier_transform`, `evaluation_space`.
- Saidas: politica de transformacao.
- Formatos: dataclass.
- Shapes: nao aplicavel.
- Copias em memoria: nenhuma.
- Mudancas de escala: AppClassNet defaulta para `preserve`, `evaluation_space="source"`, expected range `[-0.5,0.5]`, `inverse_transform_synthetic=True`, linhas ~155-168.
- Risco: **baixo**. `auto` resolve para `preserve` em AppClassNet, linhas ~218-221.

- Arquivo: `Engine/Preprocessing/FeatureTransformManager.py`
- Classe/funcao: `FeatureTransformManager.fit()`, linhas ~336-354; `partial_fit_batches()`, linhas ~356-402; `transform()`, linhas ~404-443; `inverse_transform()`, linhas ~449-453.
- Entradas: X treino ou batches.
- Saidas: X transformado, `transform_id`, `transform_history`, `data_space`.
- Formatos: ndarray `float32`.
- Shapes: preserva `(n,m)`.
- Copias em memoria: `numpy.asarray(values, dtype=float32)` pode copiar; scaler transforma em nova matriz.
- Mudancas de escala: `minmax` para `[0,1]`, `standard` para padronizado; `preserve` retorna input em `source`.
- Risco: **baixo**. Scaler e ajustado no treino; valid/test usam scaler ja treinado.

- Arquivo: `Engine/Preprocessing/FeatureTransformManager.py`
- Classe/funcao: `ScaleGuard.validate_compatible_metadata()`, linhas ~520-541; `validate_before_evaluation()`, linhas ~543-555.
- Entradas: metadata real/sintetico e arrays.
- Saidas: erro ou validacao.
- Formatos: dicts com `data_space`, `transform_id`, `transform_history`.
- Shapes: arrays 2D.
- Copias em memoria: calculo de stats em chunks internos via `numpy.asarray`.
- Mudancas de escala: nenhuma.
- Risco: **baixo**. Bloqueia comparacao quando espacos ou `transform_id` divergem.

### ModelInputAdapter

- Arquivo: `Engine/Preprocessing/FeatureTransformManager.py`
- Classe/funcao: `ModelInputAdapter`, linhas ~579-650.
- Entradas: politica, X real de treino/avaliacao, sinteticos do gerador.
- Saidas: entrada do gerador, sinteticos em `source` ou `generator`.
- Formatos: ndarray ou dict de arrays.
- Shapes: preserva colunas.
- Copias em memoria: inverse transform cria array; dict sintetico pode ser refeito por classe.
- Mudancas de escala: se gerador usa `minmax/standard`, inverse pode retornar a `source`.
- Risco: **medio**. `transform_generator_input()` e chamado para treino e avaliacao; manager foi ajustado no treino, mas o atributo `generator_input_metadata` fica com a ultima chamada.

### AppClassNet preprocessing externo

- Arquivo: `run_appclassnet_top200.py`
- Classe/funcao: `normalize_preprocessing_arguments()`, linhas ~585-617; `preprocess_appclassnet_splits()`, linhas ~850-1006.
- Entradas: raw NPY e politica.
- Saidas: `scaled_npy/{train,valid,test}_x.npy`, y copiado, `scaler.joblib`, `preprocessing_manifest.json`.
- Formatos: NPY `float32`.
- Shapes: iguais aos splits originais.
- Copias em memoria/disco: escreve novo X NPY para cada split; y e salvo via `numpy.save`.
- Mudancas de escala: fit no train, transform aplicado em train/valid/test; com `preserve`, transforma identidade e ainda grava novo NPY.
- Risco: **medio** por I/O duplicado; **baixo** para vazamento de scaler, pois fit e train-only.

## 3. Treinamento gerativo

### Algoritmos suportados

- Arquivo: `Engine/Arguments/ArgumentsFramework.py`
- Funcao: `add_argument_framework()`, linhas ~244-260.
- Modelos aceitos: `smote`, `random`, `adversarial`, `latent_diffusion`, `denoising_diffusion`, `wasserstein`, `wasserstein_gp`, `variational`, `autoencoder`, `quantized`, `diffusion_kernel`, `copy`, `copula`, `ctgan`, `tvae`.
- Risco: **medio**. `diffusion_kernel` passa sem implementacao efetiva em `training_model()`, linhas ~4092-4098.

- Arquivo: `Engine/Models/GenerativeModels.py`
- Classe/funcao: `GenerativeModels.training_model()`, linhas ~3984-4106.
- Entradas: `arguments`, `input_shape`, `x_real_samples`, `y_real_samples`, `monitor_path`, `k_fold`.
- Saidas: instancia treinada no atributo do algoritmo selecionado.
- Formatos: X `float32`, y inteiro zero-based; muitos modelos usam one-hot condicional.
- Shapes AppClassNet: X `(n,20)`, y `(n,)`, one-hot `(n,200)` ou dataset batch-wise.
- Copias em memoria: one-hot global em modo normal; em batches, `_batch_one_hot_dataset()` gera por batch.
- Mudancas de escala: recebe X ja adaptado pelo `ModelInputAdapter`.
- Risco: **alto** em modo normal para memoria one-hot de datasets grandes.

### Condicionamento por classe e one-hot

- Arquivo: `Engine/DataIO/LabelUtils.py`
- Funcoes: `validate_zero_based_labels()`, linhas ~74-95; `one_hot_encode_labels()`, linhas ~98-103; `to_one_hot_batch()`, linhas ~106-128.
- Entradas: labels.
- Saidas: labels inteiras ou matriz one-hot.
- Formatos: y zero-based; one-hot `float32`.
- Shapes AppClassNet: `(n,200)` ou `(batch,200)`.
- Copias em memoria: aloca matriz one-hot completa/batch.
- Mudancas de escala: nenhuma.
- Risco: **baixo** para validade de labels; **medio** para memoria.

- Arquivo: `Engine/Models/GenerativeModels.py`
- Funcoes: `_fit_features_and_one_hot()`, linhas ~156-174; `_fit_autoencode_with_one_hot()`, linhas ~177-195.
- Entradas: modelo, X, y, metadata de classes.
- Saidas: `model.fit(...)`.
- Formatos: normal usa one-hot global; batches usa `tensorflow.data.Dataset`.
- Shapes: normal `(n,200)`, batches `(None,200)`.
- Copias em memoria: one-hot global no normal.
- Mudancas de escala: nenhuma.
- Risco: **medio**.

### Geracao por classe e batches

- Arquivo: `main.py`
- Classe/funcao: `SynDataGen.synthesize_data()`, linhas ~807-1002.
- Entradas: X/y de avaliacao, X/y treino, metadata de amostragem.
- Saidas: `self.data_generated`.
- Formatos:
  - Normal: dict `{class_id: generated_samples}`.
  - Batches incremental: `SyntheticBatchReader`.
- Shapes: sinteticos por classe `(k,20)`.
- Copias em memoria: normal materializa todos os sinteticos; `numpy.vstack` aparece antes de validacao de escala em `_guard_current_evaluation_space()`, linhas ~621-625.
- Mudancas de escala: chama `transform_synthetic_collection_to_source()` quando adapter existe, linhas ~951-963.
- Risco: **alto** para memoria no modo normal.

- Arquivo: `main.py`
- Classe/funcao: `_synthesize_data_incremental()`, linhas ~1208-1305; `_synthesize_data_partitioned()`, linhas ~1004-1177.
- Entradas: plano por classe, gerador treinado ou unidades por classe/grupo.
- Saidas: `SyntheticBatchReader` com manifest.
- Formatos: `npy_batches`, `csv_batches` ou `single_npy`.
- Shapes: batches `(b,20)`.
- Copias em memoria: apenas batch corrente, exceto subset de treino por classes em `_subset_by_classes()`, linhas ~679-683.
- Mudancas de escala: `inverse_synthetic_batch()` antes de gravar, linhas ~1121-1123 e ~1260-1262.
- Risco: **medio**. `per_class/grouped_classes` treina subgeradores e pode ser caro; subset por mask materializa `unit_x`.

- Arquivo: `Engine/DataIO/SyntheticBatchIO.py`
- Classes: `SyntheticBatchWriter`, linhas ~16-131; `SyntheticBatchReader`, linhas ~134-171.
- Entradas: batches sinteticos por classe.
- Saidas: arquivos NPY/CSV e `manifest.json`.
- Formatos: manifest inclui `data_space`, `transform_id`, `transform_history`.
- Shapes: valida `(n,num_features)`.
- Copias em memoria: `numpy.save` escreve batch; `single_npy` usa memmap.
- Mudancas de escala: nenhuma; apenas registra o espaco recebido.
- Risco: **baixo**.

## 4. Avaliacao

### Orquestracao

- Arquivo: `main.py`
- Funcao: `run_synthetic_evaluation_modes()`, linhas ~110-128.
- Entradas: fold real e sinteticos.
- Saidas: chama TR-TS e/ou TS-TR ou marca `not_applicable`.
- Formatos: dict fold; synthetic dict ou reader.
- Shapes: depende do modo.
- Copias em memoria: chama guarda de escala para dict normal.
- Mudancas de escala: valida antes de avaliar.
- Risco: **baixo**. Nao cria metricas zero para avaliacao pulada.

### TR-TR

- Arquivo: `Engine/Evaluation/TrTr.py`
- Classe/funcao: `TrTr.evaluation_TR_TR()`, linhas ~47-127.
- Entradas: `x_training_real`, `y_training_real`, `x_evaluation_real`, `y_evaluation_real`.
- Saidas: metricas `TR-TR` e distancia `R-R`.
- Formatos: arrays real-real.
- Shapes: treino `(n_train,m)`, avaliacao `(n_eval,m)`.
- Copias em memoria: `numpy.array()` para distancia, linhas ~109 e ~113.
- Mudancas de escala: nenhuma local.
- Risco: **medio**. Funcao existe, mas nao e chamada no fluxo sintetico padrao de `main.py`; chamada comentada na linha ~502.

- Arquivo: `run_appclassnet_top200.py`
- Funcao: `run_real_real_baseline()`, linhas ~1009-1139.
- Entradas: raw NPY train/test, quotas por classe.
- Saidas: `metrics.json`.
- Formatos: X/y NPY; classificador sklearn.
- Shapes de regressao: treino `(200000,20)` para 1000/classe; teste `(100000,20)` para 500/classe.
- Copias em memoria: `load_selected_rows()` materializa subsets, linhas ~761-767.
- Mudancas de escala: por default `classifier_transform=preserve`; scaler opcional ajustado no treino, linhas ~1053-1069.
- Risco: **baixo** para o teste validado. Este e o melhor teste de regressao atual.

### TR-TS

- Arquivo: `Engine/Evaluation/TrTs.py`
- Classe/funcao: `TrTs.evaluation_TR_TS()`, linhas ~54-208.
- Definicao efetiva: treina classificador em real e testa em sintetico.
- Entradas normal: `dictionary_data` + synthetic dict.
- Entradas batches: `SyntheticBatchReader`.
- Saidas: metricas `TR-TS`.
- Formatos: normal materializa `synthetic_array`; batches prediz por batches.
- Shapes: real treino usado pela funcao e `x_evaluation_real`, nao `x_training_real`, linhas ~188-193.
- Copias em memoria: normal cria listas `labels`, `data` e `numpy.asarray(data)`, linhas ~138-168.
- Mudancas de escala: `ScaleGuard.validate_before_evaluation()` normal, linhas ~180-186; manifest validation batches, linhas ~72-83.
- Risco: **medio**. Nome conceitual diz "train real"; implementacao normal treina em `x_evaluation_real`, nao `x_training_real`. Isso deve ser decidido antes de alterar, pois pode ser legado.

### TS-TR

- Arquivo: `Engine/Evaluation/TsTr.py`
- Classe/funcao: `TsTr.evaluation_TS_TR()`, linhas ~77-249.
- Definicao efetiva: treina classificador em sintetico e testa em real.
- Entradas normal: synthetic dict e split real.
- Entradas batches: `SyntheticBatchReader`.
- Saidas: metricas `TS-TR` e distancia `R-S` no normal.
- Formatos: normal materializa todos sinteticos; batches treina/prediz por batches.
- Shapes: sintetico `(n_synth,20)`, real avaliacao `(n_eval,20)`.
- Copias em memoria: normal cria listas e `numpy.asarray(data)`, linhas ~162-181 e ~230.
- Mudancas de escala: `ScaleGuard.validate_before_evaluation()`, linhas ~193-199; batches valida manifest, linhas ~97-107.
- Risco: **alto**. Normal trunca `y_evaluation_real` e `label_predicted` por `total_samples`, linhas ~212-216, o que pode reduzir a avaliacao real silenciosamente.

### Cross-validation e provided splits

- Arquivo: `Engine/Evaluation/CrossValidation.py`
- Classe/funcao: `StratifiedData.wrapper()`, linhas ~408-590.
- Entradas: dados carregados.
- Saidas: `list_folds`.
- Formatos:
  - `cross_validation`: `StratifiedKFold`/`KFold`.
  - `provided`: fold unico train -> valid/test.
- Shapes: folds conforme split.
- Copias em memoria: shuffle completo no cross-validation, linhas ~465-467; slices de treino/valid, linhas ~540-563.
- Mudancas de escala: nenhuma.
- Risco: **alto** para confundir folds com splits oficiais. O codigo avisa que `provided` seta `number_k_folds=1`, `ArgumentsDataLoader.py`, linhas ~96-103.

### Classificadores

- Arquivo: `Engine/Classifiers/Classifiers.py`
- Classe/funcao: `Classifiers.__init__()`, linhas ~101-130; `get_trained_classifiers()`, linhas ~132-151.
- Entradas: lista `arguments.classifier`.
- Saidas: classificadores treinados.
- Formatos: sklearn wrappers internos.
- Shapes: X `(n,m)`, y `(n,)`.
- Copias em memoria: subset variants fatiam prefixo, linhas ~153-168.
- Mudancas de escala: nenhuma.
- Risco: **medio**. `DecisionTreeSubset`/`RandomForestSubset` usam prefixo, nao amostragem estratificada.

- Arquivo: `Engine/Classifiers/BatchClassifiers.py`
- Funcoes: `make_batch_classifier()`, linhas ~39-76; `train_batch_classifier()`, linhas ~170-237.
- Entradas: batches X/y, `eval_classifier`.
- Saidas: classificador sklearn e metadata.
- Formatos: partial_fit (`sgd`, `passive_aggressive`, `naive_bayes`, `mlp_small`) ou subset (`decision_tree_subset`, `extra_trees_subset`, `random_forest_light`).
- Shapes: batches `(b,20)`.
- Copias em memoria: subset classifiers coletam reservatorio estratificado e fazem `vstack`, linhas ~114-167.
- Mudancas de escala: nenhuma.
- Risco: **medio**. `SGDClassifier` esta disponivel, mas nao e o unico indicador; default AppClassNet e `decision_tree_subset`, `run_appclassnet_top200.py`, linhas ~2045-2050.

### Metricas e salvamento

- Arquivo: `Engine/Metrics/Metrics.py`
- Classe/funcao: `Metrics.__initialize_dictionary()`, linhas ~186-292; `get_task_metrics()`, linhas ~634-671; `update_mean_std_fold()`, linhas ~506-567; `save_dictionary_to_json()`, linhas ~426-444.
- Entradas: labels reais/preditas; folds.
- Saidas: `Results.json` e `metrics.json`.
- Formatos:
  - Binary: metricas legadas.
  - Multiclass: Accuracy, Macro/Weighted Precision/Recall/F1, BalancedAccuracy.
- Shapes: labels 1D.
- Copias em memoria: conversoes via `numpy.ravel/asarray`.
- Mudancas de escala: nenhuma.
- Risco: **baixo** para preditivas puladas (`not_applicable`); **baixo/medio** para eficiencia inicializada com zero por desenho, linhas ~250-257.

- Arquivo: `run_appclassnet_top200.py`
- Funcoes: `write_baseline_batches_metrics()`, linhas ~1293-1317; `write_batches_metrics()`, linhas ~1319-1345.
- Entradas: `Results.json` e manifestos.
- Saidas: `results/appclassnet_top200/batches/metrics.json`.
- Formatos: resumo TR-TR/TR-TS/TS-TR com `status`.
- Shapes: nao aplicavel.
- Copias em memoria: sumariza ate `max_batches=200`, linhas ~1204-1253.
- Mudancas de escala: reporta `data_space`.
- Risco: **baixo**. Avaliacoes nao executadas ficam `not_run`, nao zero.

## 5. Compatibilidade

### Caminho original CSV

- Arquivos: `main.py`, `Engine/Arguments/ArgumentsDataLoader.py`, `Engine/DataIO/CSVLoader.py`, `Engine/Evaluation/CrossValidation.py`.
- Contrato: `data_format=csv`, CSV unico, cross-validation, classificadores legados, `source_profile=legacy_csv`, transforms `preserve`.
- Compatibilidade: **mantida** por defaults.
- Risco: **medio** apenas pela codificacao zero-based automatica de labels discretas no CSV loader, que ja esta presente no estado atual.

### Caminho AppClassNet

- Arquivo: `run_appclassnet_top200.py`.
- Contrato:
  - Dados raw: NPY X/y por split.
  - 20 features.
  - 200 classes.
  - Labels `[0,199]`.
  - Default sem scaler: `DEFAULT_SCALER="none"`, `feature_transform="preserve"`, `classifier_transform="preserve"`, `evaluation_space="source"`.
- Compatibilidade: **parcial**. O baseline real-real usa train/test corretamente; o fluxo sintetico batches usa `provided`, mas avalia valid antes de test; o normal usa CSV/cross-validation.
- Risco: **alto** para comparar resultados entre normal, batches e baseline.

### Diferencas normal vs batches

- Normal:
  - Converte AppClassNet para CSV.
  - `main.py` carrega CSV completo.
  - Usa cross-validation.
  - Sinteticos em memoria.
  - Multiplos classificadores legados por default.
- Batches:
  - Usa NPY direto.
  - `split_mode=provided`.
  - Mmap opcional via `--use_mmap`/`--mmap_npy`.
  - Sinteticos em manifest/batches por default.
  - Um classificador de avaliacao selecionado por `--eval_classifier`.
- Risco: **alto**. As metricas nao sao diretamente comparaveis se split, classificador, quantidade por classe e espaco numerico nao forem alinhados.

### Argumentos novos que alteram comportamento antigo

- `--data_format npy_xy`: opt-in; nao altera CSV default. Risco **baixo**.
- `--split_mode provided`: seta `number_k_folds=1` e usa split fornecido. Risco **medio** por nao usar test quando valid existe.
- `--source_profile appclassnet_top200`: muda defaults de politica quando usado. Risco **baixo** no runner; opt-in no `main.py`.
- `--feature_transform`, `--generator_transform`, `--classifier_transform`: podem escalar dados se explicitados. Risco **medio**.
- `--execution_mode batches`: altera fluxo de avaliacao, geracao e classificadores. Risco **alto**.
- `--normal_classifier`: override opt-in da lista legada. Risco **baixo**.
- `--eval_classifier`: default AppClassNet `decision_tree_subset`; evita SGD como unico indicador. Risco **baixo**.

## Matriz de riscos

| Severidade | Achado | Evidencia | Impacto |
|---|---|---|---|
| Alto | Modo normal AppClassNet usa CSV + cross-validation, nao splits oficiais | `run_appclassnet_top200.py` ~1451-1625; `CrossValidation.py` ~443-582 | Regressao real-real train/test nao valida esse caminho |
| Alto | `provided` ignora test quando valid existe | `CrossValidation.py` ~357-365 | Avaliacoes sinteticas podem usar valid, nao test |
| Alto | TS-TR normal trunca avaliacao por total sintetico | `TsTr.py` ~212-216 | Pode mascarar desempenho em subconjunto |
| Alto | Sinteticos normal em memoria | `main.py` ~855-1002 | Risco de RAM alto no top-200 |
| Medio | Preprocess preserve ainda grava NPY novo | `run_appclassnet_top200.py` ~850-1006 | I/O e disco desnecessarios |
| Medio | TR-TR ausente no fluxo sintetico padrao | `main.py` ~502; runner ~1009-1139 | TR-TR validado fica separado |
| Medio | Help `--scaler` desatualizado | `ArgumentsDataLoader.py` ~218-221; runner ~42 | Confusao operacional |
| Medio | TR-TS normal treina em `x_evaluation_real` | `TrTs.py` ~188-193 | Semantica TR-TS deve ser revisada antes de mexer |
| Baixo | AppClassNet default sem scaler esta implementado | `FeatureTransformPolicy` ~155-168; runner ~2070-2104 | Baixo risco, contrato correto |
| Baixo | Metricas puladas usam `not_applicable/not_run` | `Metrics.py` ~568-608; runner ~1256-1345 | Evita zeros falsos |

## Conclusoes para proximas etapas

1. Usar `run_appclassnet_top200.py --baseline_real_only --baseline_classifier decision_tree --train_samples_per_class 1000 --test_samples_per_class 500` como teste de regressao TR-TR real-real.
2. Antes de comparar TR-TS/TS-TR com o baseline, decidir explicitamente se a avaliacao sintetica deve usar `valid` ou `test`. O codigo atual de `split_mode=provided` usa `valid` quando ambos existem.
3. Qualquer correcao deve ser pequena e testavel: primeiro alinhar split/evaluation, depois corrigir TS-TR truncado, depois reduzir I/O do `preserve`.
4. Nao aplicar normalizacao automatica ao AppClassNet. Qualquer transform deve continuar fitando apenas no treino e registrando `data_space`, `transform_id` e `transform_history`.
