# PREPROCESSING_AND_DATA_SPACE_AUDIT

Data da auditoria: 2026-07-14

Escopo: auditoria exclusiva do pré-processamento e dos espaços numéricos do AppClassNet top-200 no MalDataGen. Nenhum arquivo de código foi modificado.

## Sumário executivo

- **[crítico]** O caminho principal AppClassNet em `run_appclassnet_top200.py` força `--feature_transform preserve` nos comandos normal e batches (`build_main_command`, linhas ~1588-1598; `build_batch_main_command`, linhas ~1683-1694). Portanto, no fluxo principal, os NPY já normalizados em aproximadamente `[-0.5, 0.5]` entram como `source`.
- **[alto]** Ainda existe compatibilidade com `--scaler minmax|standard`: `Engine/Arguments/Arguments.py::_normalize_preprocessing_arguments` mapeia `--scaler` para `feature_transform` quando `--feature_transform` não foi fornecido (`linhas ~151-174`), e `run_appclassnet_top200.py::normalize_preprocessing_arguments` repete a lógica (`linhas ~585-603`). Isso não é default, mas é um caminho explícito que pode transformar AppClassNet.
- **[alto]** O gerador pode operar em espaço interno `generator` com `MinMaxScaler` ou `StandardScaler` via `--generator_transform`. A intenção é inverter sintéticos para `source`; `ModelInputAdapter.inverse_synthetic_batch` e `transform_synthetic_collection_to_source` fazem isso (`FeatureTransformManager.py`, linhas ~614-645). Se `inverse_transform_synthetic` for desativado ou metadata estiver incorreta, TR-TS/TS-TR podem comparar espaços diferentes; `ScaleGuard` deve bloquear por `data_space`.
- **[médio]** O baseline real-real separado aplica `classifier_transform` se solicitado (`run_appclassnet_top200.py::run_real_real_baseline`, linhas ~1053-1069), mas registra `"data_space": "source"` mesmo após transformar `train_x`/`test_x` para `"classifier"` (`linhas ~1077-1100`). Com default `preserve`, não muda escala; com `classifier_transform=minmax|standard`, o relatório fica semanticamente ambíguo.
- **[médio]** `classifier_transform` existe na política principal (`main.py::_build_model_input_adapter`, linhas ~602-614), mas as avaliações TR-TR/TR-TS/TS-TR não chamam `classifier_manager`; elas treinam classificadores diretamente nos arrays recebidos. No pipeline principal, `classifier_transform` não parece efetivamente aplicado aos avaliadores.
- **[baixo]** Loaders AppClassNet não normalizam features. `NpyXYLoader._load_split` apenas carrega `numpy.load`, aplica `astype` somente se `dtype` foi solicitado e valida labels (`NpyXYLoader.py`, linhas ~146-161). O schema marca `data_space="source"` e `already_normalized=True` para AppClassNet (`linhas ~107-124`).
- **[baixo]** Há muitas conversões `astype`/`numpy.asarray(..., dtype=float32)` em loaders, transform manager, avaliação e classificadores. Elas podem copiar memória e reduzir precisão, mas não mudam escala.
- **[médio]** Difusões aplicam `clip_by_value` no `x_recon` quando `clip_denoised=True`; as campanhas AppClassNet configuram `clip_min=-0.5`/`clip_max=0.5`. Isso pode ser desejável para manter o range source, mas é uma transformação do output sintético do gerador.

## Matriz de espaço de dados

| Etapa | Real train | Real test | Synthetic train | Synthetic test |
|---|---|---|---|---|
| Espaço esperado | `source`, aproximadamente `[-0.5,0.5]` | `source`, aproximadamente `[-0.5,0.5]` | `source`, após inverse quando gerador usa espaço interno | `source`, após inverse quando gerador usa espaço interno |
| Espaço atual default AppClassNet | `source` | `source` | `source` se `generator_transform=preserve` ou inverse habilitado | `source` se `generator_transform=preserve` ou inverse habilitado |
| Transformação aplicada default | `numpy.load`; possível `astype` opcional; sem scaler | `numpy.load`; possível `astype` opcional; sem scaler | saída do gerador preservada ou invertida para `source`; `astype(float32)` em batches | saída do gerador preservada ou invertida para `source`; `astype(float32)` em materialização/avaliação |
| Caminho transformado explícito | `feature_transform`/`generator_transform`/`classifier_transform` podem aplicar minmax/standard, fit no treino | transform com scaler ajustado no treino | pode ficar em `generator` se inverse desativado | pode ficar em `generator` se inverse desativado |
| Persistência | loader não persiste scaler; pré-processamento materializado pode salvar `scaler.joblib` | reusa scaler de treino se materializado | batches salvam `data_space`/`transform_id` no manifesto | normal salva metadata em memória; batches salvam manifesto |
| Guarda atual | `ScaleGuard.describe(..., data_space="source")` | `ScaleGuard.describe(..., data_space="source")` | manifest/metadata `data_space` e `transform_id` | manifest/metadata `data_space` e `transform_id` |

## Inventário das ocorrências localizadas

| Ocorrência buscada | Localização | Tipo | Efeito sobre AppClassNet |
|---|---|---|---|
| `MinMaxScaler` | `Engine/Preprocessing/FeatureTransformManager.py`, imports linhas ~24-25 e `_make_scaler()` linhas ~278-285 | scaler central | Só aplica se `feature_transform`, `generator_transform` ou `classifier_transform` resolver para `minmax`; output `[0,1]`; fit esperado em train. |
| `StandardScaler` | `Engine/Preprocessing/FeatureTransformManager.py`, imports linhas ~24-25 e `_make_scaler()` linhas ~278-285 | scaler central | Só aplica se transform resolver para `standard`; output sem range fixo; fit esperado em train. |
| `fit` | `FeatureTransformManager.fit()` linhas ~336-354; `partial_fit_batches()` linhas ~356-402; baseline `run_real_real_baseline()` linhas ~1066-1067 | ajuste de scaler/modelo | Scaler de pré-processamento é ajustado no treino. `model.fit` de redes/classificadores é treinamento de modelo, não pré-processamento. |
| `fit_transform` | `FeatureTransformManager.fit_transform_train()` linhas ~445-447 | helper | Chama `fit(train)` e depois `transform(train)`; não apareceu como caminho principal AppClassNet. |
| `transform` | `FeatureTransformManager.transform()` linhas ~404-443; `main.py::run_experiments()` linhas ~432-441; runner materializado linhas ~906-922 | aplicação de scaler | Aplica preserve/minmax/standard conforme stage; bloqueia valid/test/synthetic se o scaler não foi ajustado em train. |
| `inverse_transform` | `FeatureTransformManager.inverse_transform()` linhas ~449-453; `ModelInputAdapter.inverse_generator_output()` linhas ~604-612; `inverse_synthetic_batch()` linhas ~640-645 | retorno para source | Volta sintético do espaço `generator` para `source` quando `inverse_transform_synthetic=True`. |
| `astype` | `NpyXYLoader._load_split()` linhas ~150-151; `FeatureTransformManager.transform()` linha ~412; avaliações/classificadores em vários pontos | dtype/cópia | Não muda escala; pode reduzir para `float32` ou labels para inteiros. |
| `clip`/`clip_by_value` | `GaussianLatentDiffusion.py` linhas ~401-407; `GaussianDenoisingDiffusion.py` linhas ~401-407; `AlgorithmWassersteinGAN.py` linhas ~163-166; `HellingerDistance.py` linha ~100 | clipping de modelo/métrica | Difusões podem limitar output sintético; Wasserstein clipa pesos do discriminador; Hellinger clipa distribuição para métrica, não dados avaliados. |
| `round`/`rint` | `AutoencoderAlgorithm.py` linhas ~306-309; `AdversarialAlgorithm.py` linhas ~356-358; VAE/WGAN/QuantizedVAE/Diffusion equivalentes | discretização de sintéticos | Só deve ocorrer se `data_type != "continuous"`; destruiria o range AppClassNet se ativado por engano. |
| `sigmoid`/`tanh` | defaults em `Engine/Arguments/*` e modelos geradores; runner AppClassNet sobrescreve para `linear` em campanhas principais | ativação neural | `sigmoid` tende a `[0,1]`; `tanh` tende a `[-1,1]`; se usado com source `[-0.5,0.5]`, precisa de política de espaço coerente. |
| normalização manual | `CSVLoader._normalize_data()` linhas ~372-376; `XLSLoader._normalize_data()` linhas ~255-261 | desabilitada | Logs informam que normalização CSV/XLS está desabilitada; features preservadas. |
| divisão por máximo | `SyntheticSanityChecks.py` linhas ~225-292; `FeatureTransformManager._array_stats()` linha ~108; `QuantizedVAE.train_step()` linha ~214 | diagnóstico/loss | Frações e estatísticas, ou loss normalizada por variância; não transforma arrays de entrada/saída do AppClassNet. |
| soma/subtração de constantes | `NpyXYLoader._handle_multiclass_label_base()` linhas ~190-211; difusão e losses em modelos | labels/modelo | Subtração `-1` é remapeamento opcional de labels 1-based, não features; difusão usa constantes internamente no processo generativo. |

## Ocorrências e comportamento das transformações

### 1. Loader NPY AppClassNet

**Arquivo:** `Engine/DataIO/NpyXYLoader.py`  
**Classe:** `NpyXYLoader`  
**Funções:** `load`, `_load_split`, `_load_array`, `_source_feature_range`

- **Linhas:** schema em ~107-124; `_source_feature_range` em ~141-144; `_load_split` em ~146-161; `_load_array` em ~163-170.
- **Entradas:** caminhos `train_x.npy/train_y.npy`, `valid_x.npy/valid_y.npy`, `test_x.npy/test_y.npy`; X 2D; y separado.
- **Saídas:** `DatasetBundle` com `SplitData(X, y, name)` e `DatasetSchema`.
- **Formato/shape esperado:** X `(n, 20)`, y `(n,)` ou equivalente normalizado para 1D; classes 0..199.
- **Transformação de escala:** nenhuma.
- **Dtype:** se `self.dtype` não for `None`, executa `x_values.astype(self.dtype, copy=False)` (`_load_split`, linhas ~150-151). Caso contrário preserva dtype do NPY.
- **Input range:** esperado AppClassNet `[-0.5,0.5]`.
- **Output range:** igual ao input.
- **Persistência/reuso:** não persiste scaler; schema registra `source_feature_range=(-0.5,0.5)`, `current_feature_range=(-0.5,0.5)`, `data_space="source"`, `transform_id=None`.
- **Inverse transform:** não existe no loader.
- **Estágio:** loader.
- **Risco:** **baixo**. O loader cumpre a regra de apenas carregar dados; a única cópia possível vem de `astype` opcional ou do modo sem mmap.

### 2. Loader CSV legado

**Arquivo:** `Engine/DataIO/CSVLoader.py`  
**Classe:** `CSVLoader`  
**Funções:** `_process_label_column`, `_encode_discrete_labels`, `save_csv`

- **Linhas:** docstring de normalização desabilitada em ~80-84; labels em ~290-317; salvamento em ~419-435.
- **Entradas:** CSV único com coluna de label.
- **Saídas:** features preservadas e labels separados.
- **Transformação de escala:** nenhuma em features.
- **Dtype:** labels viram `numpy.float32` (`_process_label_column`, linha ~296); labels discretos são remapeados para índices zero-based e armazenados como `float32` (`_encode_discrete_labels`, linhas ~303-317).
- **Input/output range:** features preservadas.
- **Persistência/reuso:** ao salvar sintético, escreve `.space.json` com metadata de `data_space`, `transform_id` e `transform_history` (`save_csv`, linhas ~426-435).
- **Inverse transform:** não há reversão de normalização; log diz explicitamente que CSV preserva valores (`linha ~419`).
- **Estágio:** loader/salvamento legado.
- **Risco:** **baixo** para AppClassNet, pois o fluxo NPY não depende deste loader; **médio** para compatibilidade se alguém interpretar remapeamento de labels CSV como transformação de features.

### 2b. Loader XLS legado

**Arquivo:** `Engine/DataIO/XLSLoader.py`  
**Classe:** `XLSLoader`  
**Funções:** `_normalize_data`, `save_xls`

- **Linhas:** `_normalize_data` em ~255-261; salvamento em ~243.
- **Transformação de escala:** nenhuma; o método apenas registra que a normalização XLS está desabilitada.
- **Input/output range:** features preservadas.
- **Persistência/reuso:** não persiste scaler.
- **Inverse transform:** não existe para features.
- **Estágio:** loader/salvamento legado.
- **Risco:** **baixo** para AppClassNet, que usa NPY, mas importante para a regra de não alterar comportamento original.

### 3. Política central de transformação

**Arquivo:** `Engine/Preprocessing/FeatureTransformManager.py`  
**Classes:** `FeatureTransformPolicy`, `FeatureTransformManager`, `ScaleGuard`, `ModelInputAdapter`

#### Defaults por perfil

- **Função:** `FeatureTransformPolicy.for_profile`
- **Linhas:** ~146-206.
- **AppClassNet default:** `feature_transform="preserve"`, `generator_transform="preserve"`, `classifier_transform="preserve"`, `evaluation_space="source"`, `expected_source_min=-0.5`, `expected_source_max=0.5`, `allow_refit=False`, `allow_double_transform=False`, `inverse_transform_synthetic=True` (`linhas ~155-168`).
- **Auto:** `resolve_transform` retorna `preserve` para AppClassNet quando o transform é `auto` (`linhas ~208-222`).
- **Risco:** **baixo** no default; **alto** se um wrapper altera `generator_transform=auto` para `minmax` por causa de sigmoid, como ocorre no runner.

#### MinMaxScaler e StandardScaler

- **Função:** `FeatureTransformManager._make_scaler`
- **Linhas:** ~278-285.
- **Comportamento:** `preserve` retorna `None`; `minmax` cria `MinMaxScaler(feature_range=(0,1))`; `standard` cria `StandardScaler()`.
- **Input range:** valores recebidos, no AppClassNet esperado `[-0.5,0.5]`.
- **Output range:** minmax `[0,1]`; standard média aproximada 0 e variância 1, sem range fixo.
- **Persistência/reuso:** `save/load` existem no manager; transform_id é hash do scaler e metadados (`_build_transform_id`, linhas ~287-301).
- **Inverse transform:** `inverse_transform` usa `scaler.inverse_transform` para operação não-preserve (`linhas ~449-453`).
- **Estágio:** feature/generator/classifier, dependendo do `stage`.
- **Risco:** **alto** se usado sem inverse antes de avaliação; **médio** para classifier, pois pode ser correto para alguns classificadores, mas não deve alterar a regressão DecisionTree AppClassNet por default.

#### Fit normal

- **Função:** `FeatureTransformManager.fit`
- **Linhas:** ~336-354.
- **Quando é chamada:** `ModelInputAdapter.fit_generator` no pipeline principal (`main.py`, linhas ~432-433); baseline real-real (`run_appclassnet_top200.py`, linhas ~1066-1067); scripts/testes; pré-processamento materializado usa `partial_fit_batches`.
- **Split:** o chamador passa `split_name="train"` no fluxo AppClassNet encontrado.
- **Input range:** esperado `[-0.5,0.5]` para AppClassNet source.
- **Output range:** `preserve` mantém input; `minmax` marca `[0,1]`; `standard` marca `None`.
- **Persistida:** pode ser salvo por `save`; no runner de pré-processamento é salvo como `scaler.joblib`.
- **Reutilizada:** sim, por `transform` em train/valid/test.
- **Inverse:** sim para minmax/standard.
- **Cópias em memória:** `numpy.asarray(train_x, dtype=float32)` antes de `scaler.fit` pode materializar cópia.
- **Risco:** **baixo** no default; **alto** se `fit` for chamado fora do treino. O código bloqueia refit quando `allow_refit=False`.

#### Fit-transform helper

- **Função:** `FeatureTransformManager.fit_transform_train`
- **Linhas:** ~445-447.
- **Quando é chamada:** não foi localizada no caminho principal AppClassNet; é um helper disponível.
- **Split:** sempre chama `fit(... split_name="train")`.
- **Input/output range:** igual ao transform configurado (`preserve`, `minmax` ou `standard`).
- **Persistida/reutilizada:** usa o manager corrente; só persiste se `save()` for chamado depois.
- **Inverse:** herdado de `inverse_transform`.
- **Estágio:** feature/generator/classifier conforme o manager.
- **Risco:** **baixo** enquanto não for usado fora de train.

#### Fit em batches

- **Função:** `FeatureTransformManager.partial_fit_batches`
- **Linhas:** ~356-402.
- **Quando é chamada:** `run_appclassnet_top200.py::preprocess_appclassnet_splits` (`linhas ~884-886`).
- **Split:** train.
- **Comportamento preserve:** percorre batches apenas para min/max de diagnóstico; não cria scaler.
- **Comportamento minmax/standard:** faz `partial_fit` quando disponível; para minmax refaz `fit` sobre `vstack([feature_min, feature_max])` (`linha ~398`).
- **Input/output range:** preserve mantém; minmax `[0,1]`; standard sem range fixo.
- **Cópias em memória:** `numpy.asarray(..., dtype=float32)` por batch; para minmax, `numpy.vstack([feature_min, feature_max])`.
- **Risco:** **médio**. A política de fit em train está correta, mas se `feature_transform` for ligado explicitamente, cria NPY materializado em outro espaço.

#### Transform

- **Função:** `FeatureTransformManager.transform`
- **Linhas:** ~404-443.
- **Quando é chamada:** runner de pré-processamento (`_write_transformed_split`, linhas ~822-847), pipeline principal para entrada do gerador (`main.py`, linhas ~434-441), baseline real-real (`run_appclassnet_top200.py`, linhas ~1068-1069), testes/scripts.
- **Split:** train, valid, test ou synthetic, conforme chamador.
- **Guardas:** se `split_name` é `valid|validation|test|synthetic`, exige `fit_split=="train"` (`linhas ~425-430`); `_check_double_transform` bloqueia transform duplicado quando `allow_double_transform=False` (`linhas ~309-321`).
- **Dtype:** sempre `numpy.asarray(values, dtype=float32)` (`linha ~412`).
- **Input/output range:** preserve igual; minmax `[0,1]`; standard padronizado.
- **Persistida:** registra `transform_history` e `transform_id` para transform não-preserve (`linhas ~431-442`).
- **Inverse:** via `inverse_transform`.
- **Estágio:** feature/generator/classifier.
- **Risco:** **baixo** no default; **alto** se metadata for perdida entre geração e avaliação.

#### Inverse transform

- **Função:** `FeatureTransformManager.inverse_transform`
- **Linhas:** ~449-453.
- **Quando é chamada:** `ModelInputAdapter.inverse_generator_output`, `inverse_synthetic_batch`.
- **Split:** sintético gerado.
- **Input range:** espaço `generator` se minmax/standard; source se preserve.
- **Output range:** volta ao range source esperado quando scaler foi ajustado no treino.
- **Persistida/reutilizada:** usa o mesmo scaler do gerador.
- **Risco:** **crítico** se sintético gerado em `[0,1]` for salvo/testado sem inverse contra real `[-0.5,0.5]`.

### 4. Pipeline principal normal

**Arquivo:** `main.py`  
**Classe:** `SynDataGen`  
**Funções:** `run_experiments`, `_build_model_input_adapter`, `_guard_current_evaluation_space`, `synthesize_data`

- **Linhas:** construção/uso do adapter em ~432-448; policy em ~602-614; guarda em ~616-639; inverse/materialização em ~951-963.
- **Entrada:** `dictionary_data['x_training_real']`, `dictionary_data['x_evaluation_real']`.
- **Treino do gerador:** `fit_generator` ajusta scaler do gerador somente em `x_training_real` (`linhas ~432-433`).
- **Transformação de entrada do gerador:** train e valid/evaluation passam por `transform_generator_input` (`linhas ~434-441`).
- **Metadata real:** evaluation real é descrito como `source` (`linhas ~442-448`).
- **Geração:** `synthesize_data` recebe `x_evaluation_for_generator` e `x_training_for_generator`.
- **Inverse do sintético normal:** `transform_synthetic_collection_to_source` é chamado após geração (`linhas ~951-954`); metadata fica `source` se preserve ou inverse habilitado (`linhas ~955-963`).
- **Guarda antes de avaliação:** materializa sintéticos com `numpy.vstack` e valida finitude + compatibilidade de metadata (`_guard_current_evaluation_space`, linhas ~621-639).
- **Cópias em memória:** `numpy.vstack` em todos os sintéticos materializados; `numpy.asarray(..., dtype=float32)` por classe.
- **Risco:** **alto** para grandes volumes por cópia total; **baixo** para espaço default; **crítico** se `_current_synthetic_metadata` não refletir arrays reais.

### 5. Pipeline batches e salvamento sintético

**Arquivo:** `main.py`  
**Classe:** `SynDataGen`  
**Funções:** `_synthesize_data_partitioned`, `_synthesize_data_incremental`

- **Linhas:** inverse em batch particionado ~1120-1123; writer incremental ~1216-1238; inverse em batch incremental ~1259-1262.
- **Entrada:** batches gerados por classe.
- **Transformação:** `inverse_synthetic_batch` antes de `writer.write_batch` quando adapter existe.
- **Manifesto:** `SyntheticBatchWriter` recebe `data_space=synthetic_space_after_generation()`, `transform_id` somente quando espaço não é source, e `transform_history` do gerador (`linhas ~1216-1238`).
- **Output range:** default source; com `generator_transform=minmax|standard` e inverse true, source; com inverse false, generator.
- **Cópias em memória:** por batch, não `vstack` global.
- **Risco:** **médio**. O modo batches usa a mesma política de espaço, mas TR-TS/TS-TR validam metadata do manifest, não recalculam range completo antes da avaliação.

### 6. Pré-processamento materializado AppClassNet

**Arquivo:** `run_appclassnet_top200.py`  
**Funções:** `normalize_preprocessing_arguments`, `_write_transformed_split`, `preprocess_appclassnet_splits`

- **Linhas:** normalização de argumentos ~585-617; escrita por split ~822-847; pré-processamento ~850-1006.
- **Entrada:** raw root AppClassNet com splits NPY.
- **Fit:** `FeatureTransformManager.partial_fit_batches` em train (`linhas ~884-886`).
- **Transform:** cada split train/valid/test passa por `manager.transform(... split_name=split_name, input_space="source", output_space="transformed")` (`linhas ~906-922`).
- **Persistência:** salva `scaler.joblib`, `preprocessing_stats.json`, `preprocessing_manifest.json` (`linhas ~888-904`, ~940-1005).
- **Input range:** raw AppClassNet esperado `[-0.5,0.5]`.
- **Output range:** preserve mantém; minmax `[0,1]`; standard sem range fixo.
- **Manifesto:** registra `original_ranges`, `current_ranges`, `transformations`, `transform_id`, `fit_split=train`, `generator_input_space`, `synthetic_output_space`, `evaluation_space`, `classifier_input_space` (`linhas ~948-1004`).
- **Cópias em memória:** `numpy.lib.format.open_memmap` reduz memória no output; cada chunk transformado pode copiar para `float32`.
- **Risco:** **alto** se um diretório `scaled_npy` transformado for reutilizado como se fosse `source`; **baixo** no comando principal atual, que passa `--feature_transform preserve`.

### 7. Argumentos CLI relevantes

**Arquivo:** `Engine/Arguments/ArgumentsDataLoader.py`  
**Função:** parser de argumentos

- **Linhas:** `--scaler` ~218-221; `--source_profile` ~223-225; `--feature_transform` ~227-229; `--generator_transform` ~231-233; `--classifier_transform` ~235-237; `--evaluation_space` ~239-241; `--allow_double_transform` ~243-244; `--allow_scaler_refit` ~246-247; `--inverse_transform_synthetic` ~249-250.
- **Observação:** o help de `--scaler` diz que o runner AppClassNet "defaults to minmax" (`linhas ~218-221`), mas o código atual do runner força `--feature_transform preserve` nos comandos principais. Isso é divergência de documentação.
- **Risco:** **médio**. Usuários podem acreditar que minmax é default. A regra do projeto exige não aplicar normalização automática ao AppClassNet; o código principal respeita, o texto do help não.

**Arquivo:** `Engine/Arguments/Arguments.py`  
**Função:** `_normalize_preprocessing_arguments`

- **Linhas:** ~151-176.
- **Comportamento:** se `--scaler` foi fornecido, emite warning; para AppClassNet, `scaler=none` e ausência de `--feature_transform` vira `preserve`; `scaler=minmax|standard` e ausência de `--feature_transform` vira o transform correspondente; `inverse_transform_synthetic` é forçado para true se false.
- **Risco:** **alto** apenas no uso explícito de `--scaler minmax|standard`.

**Arquivo:** `run_appclassnet_top200.py`  
**Função:** `normalize_preprocessing_arguments`

- **Linhas:** ~585-617.
- **Comportamento adicional:** se campanhas selecionadas usam sigmoid e `generator_transform=="auto"`, altera para `minmax` (`linhas ~598-603`).
- **Risco:** **alto**. A alteração é explícita no runner, mas muda o espaço interno do gerador. O sintético precisa ser invertido antes da avaliação.

### 8. Baseline real-real

**Arquivo:** `run_appclassnet_top200.py`  
**Função:** `run_real_real_baseline`

- **Linhas:** seleção até ~1050-1051; classifier policy/manager ~1053-1069; métricas/metadata ~1077-1120.
- **Entrada:** `train_x/train_y` e `test_x/test_y` selecionados estratificadamente.
- **Fit scaler:** `classifier_manager.fit(train_x, split_name="train")`.
- **Transform:** train e test passam por `classifier_manager.transform(... output_space="classifier")`.
- **Default:** `classifier_transform=preserve`, portanto sem mudança de escala.
- **Se explícito:** minmax/standard é ajustado somente em train e aplicado em test.
- **Input/output range:** preserve `[-0.5,0.5]`; minmax `[0,1]`; standard padronizado.
- **Persistência:** não salva scaler, mas métricas incluem `transform_id`, `data_min/data_max` ou `mean/scale` se existirem.
- **Problema:** `"data_space": "source"` é gravado mesmo que `classifier_transform` não seja preserve (`linha ~1088`).
- **Risco:** **médio**. Não afeta o golden baseline default; pode confundir auditoria quando classifier transform for usado.

### 9. Avaliação TR-TS e TS-TR

**Arquivos:** `Engine/Evaluation/TrTs.py`, `Engine/Evaluation/TsTr.py`  
**Classes:** `TrTs`, `TsTr`

- **TR-TS batches:** valida metadata real source vs manifest sintético (`TrTs.py`, linhas ~72-83).
- **TR-TS normal:** converte sintéticos para `float32`, valida `ScaleGuard.validate_before_evaluation`, treina classificador em `x_evaluation_real` (`linhas ~168-190`).
- **TS-TR batches:** valida metadata real source vs manifest sintético (`TsTr.py`, linhas ~96-107).
- **TS-TR normal:** converte sintéticos para `float32`, valida `ScaleGuard.validate_before_evaluation`, embaralha e treina em sintético (`linhas ~181-202`).
- **Transformação de escala:** nenhuma própria de avaliação.
- **Dtype/cópias:** `numpy.asarray(data, dtype=float32)` e shuffle podem materializar cópias.
- **Risco:** **baixo** para bloqueio de espaços incompatíveis no normal; **médio** em batches por depender do manifest e não de range recalculado completo.

### 10. Classificadores

**Arquivo:** `Engine/Classifiers/Algorithms/DecisionTree.py`  
**Classe:** `DecisionTree`  
**Função:** `training_model`

- **Linhas:** ~95-119.
- **Transformação:** `numpy.array(x_samples_training, dtype=dataset_type)` e `numpy.array(y_samples_training, dtype=dataset_type)`; depois `DecisionTreeClassifier.fit`.
- **Scaler:** nenhum.
- **Risco:** **baixo**. Preserva escala exceto dtype; y pode virar `float32`, mas sklearn aceita labels numéricas.

**Arquivo:** `Engine/Classifiers/BatchClassifiers.py`  
**Funções:** `iter_array_batches`, `iter_synthetic_labeled_batches`, `train_batch_classifier`

- **Linhas:** batch dtype ~79-97; subset/materialização ~150-158; partial_fit ~170-198; fit normal ~202-214.
- **Transformação:** batches de X para `float32`, labels para `int64`; cria labels sintéticos por `numpy.full`.
- **Scaler:** nenhum; `SGDClassifier` não é envolvido por `StandardScaler`.
- **Risco:** **médio** para classificadores sensíveis à escala em modo incremental; **baixo** para DecisionTree/RandomForest.

### 11. Transformações específicas do gerador

**Arquivos principais:** `Engine/Algorithms/*`, `Engine/Models/*`, `Engine/Arguments/*`

- **Rounding:** vários algoritmos arredondam sintéticos com `numpy.rint` quando `number_samples_per_class.get("data_type") != "continuous"`. Exemplos: `AutoencoderAlgorithm.get_samples` (`AutoencoderAlgorithm.py`, linhas ~306-309), `AdversarialAlgorithm.get_samples` (`AdversarialAlgorithm.py`, linhas ~356-358), `AlgorithmWassersteinGAN.get_samples` (`AlgorithmWassersteinGAN.py`, linhas ~220-223), além de VAE/QuantizedVAE/LatentDiffusion/DenoisingDiffusion em linhas equivalentes.
- **AppClassNet esperado:** `feature_type/data_type` contínuo. Nesse caso, o rounding não deve ocorrer.
- **Risco:** **crítico** se AppClassNet for executado com `data_type` não-contínuo: valores `[-0.5,0.5]` podem virar inteiros, destruindo escala.

- **Sigmoid/tanh/linear:** defaults legados de vários modelos usam sigmoid, por exemplo `ArgumentsAutoencoder.py` (`DEFAULT_AUTOENCODER_LAST_ACTIVATION_LAYER="sigmoid"`, linha ~53), `ArgumentsVariationalAutoencoder.py` (`linha ~46`), `ArgumentsQuantizedVAE.py` (`linha ~47`), `ArgumentsWassersteinGAN.py` (`linha ~46`) e `ArgumentsWassersteinGANGP.py` (`linha ~46`). O runner AppClassNet sobrescreve várias campanhas para `linear` (`run_appclassnet_top200.py`, linhas ~154, ~174, ~216, ~237, ~263).
- **Regra do runner:** se campanha selecionada ainda usa sigmoid e `generator_transform=auto`, o runner muda para `minmax` (`run_appclassnet_top200.py`, linhas ~598-603).
- **Input/output range:** sigmoid tende a `[0,1]`; tanh tende a `[-1,1]`; linear sem limite.
- **Risco:** **alto**. Um gerador com sigmoid em dados source `[-0.5,0.5]` pode produzir `[0,1]`; o caminho correto é usar espaço interno minmax e inverse para source antes de avaliação.

- **Clipping em difusão:** `GaussianLatentDiffusion.p_mean_variance()` e `GaussianDenoisingDiffusion.p_mean_variance()` aplicam `tensorflow.clip_by_value(x_recon, self._clip_min, self._clip_max)` quando `clip_denoised=True` (`GaussianLatentDiffusion.py`, linhas ~401-407; `GaussianDenoisingDiffusion.py`, linhas ~401-407). O runner AppClassNet configura `latent_diffusion_gaussian_clip_min=-0.5`, `latent_diffusion_gaussian_clip_max=0.5`, `denoising_diffusion_gaussian_clip_min=-0.5` e `denoising_diffusion_gaussian_clip_max=0.5` nas campanhas (`run_appclassnet_top200.py`, linhas ~293-294 e ~325-326).
- **Input/output range:** clipping limita o output reconstruído ao intervalo configurado, não ajustado por treino. Para AppClassNet isso só é semanticamente correto se o intervalo for `[-0.5,0.5]`.
- **Persistência/reuso:** parâmetros do modelo/argumentos, não scaler persistido.
- **Inverse transform:** independente do `FeatureTransformManager`; se o gerador também estiver em espaço `generator`, ainda precisa da política de inverse para comparação em `source`.
- **Risco:** **médio** quando configurado para `[-0.5,0.5]`; **alto** se ficar em defaults incompatíveis com o source real.

- **Clipping de pesos Wasserstein:** `AlgorithmWassersteinGAN.train_step()` usa `tensorflow.clip_by_value(weight, -self._clip_value, self._clip_value)` nos pesos do discriminador (`AlgorithmWassersteinGAN.py`, linhas ~163-166). Isso não transforma X real, X sintético nem espaço de avaliação.
- **Risco:** **baixo** para pré-processamento; é restrição interna do treinamento WGAN.

- **Normalização por variância em loss:** `AlgorithmQuantizedVAE.train_step()` divide o MSE por `self._train_variance` (`AlgorithmQuantizedVAE.py`, linha ~214). Isso normaliza a loss, não os dados salvos ou avaliados.
- **Risco:** **baixo** para data_space; pode afetar treinamento, mas não é scaler de entrada.

### 12. Sanity checks de sintéticos

**Arquivo:** `Engine/DataIO/SyntheticSanityChecks.py`  
**Classe:** `SyntheticSanityCheckRunner`  
**Funções:** `_real_feature_and_class_stats`, `_synthetic_stats_and_predictions`, `_scale_report`

- **Linhas:** real stats ~108-131; sintético stats ~152-207; scale report ~224-239.
- **Transformação:** não transforma escala; converte para `float64` para estatística e saneia NaN para classificador quando necessário.
- **Verificações:** range real por feature, range sintético por feature, out-of-range por feature, classes ausentes/extras, distribuição de predições.
- **Risco:** **baixo**. É diagnóstico, não loader/gerador/classificador.

## Investigações solicitadas

### 1. Normalização duplicada

**Achado:** não ocorre no default AppClassNet.  
**Evidência:** `FeatureTransformPolicy.for_profile` define `allow_double_transform=False` para AppClassNet (`FeatureTransformManager.py`, linhas ~155-168); `_check_double_transform` bloqueia operação duplicada por `transform_history`/`transform_id` (`linhas ~309-321`).  
**Risco:** **médio** se arrays já materializados em `scaled_npy` forem reintroduzidos como raw source sem manifest.

### 2. Fit em valid ou test

**Achado:** não há evidência no caminho AppClassNet principal.  
**Evidência:** `main.py::run_experiments` chama `fit_generator` em `x_training_real` (`linhas ~432-433`); `transform_generator_input` em valid (`linhas ~438-441`); `FeatureTransformManager.transform` rejeita valid/test/synthetic se `fit_split!="train"` (`FeatureTransformManager.py`, linhas ~425-430`). Runner materializado ajusta apenas train (`run_appclassnet_top200.py`, linhas ~884-886).  
**Risco:** **baixo**.

### 3. Fit em sintético

**Achado:** scaler de pré-processamento não é ajustado em sintético. O classificador TS-TR é treinado em sintético por definição, mas isso não é scaler.  
**Evidência:** `ModelInputAdapter` só expõe `fit_generator(train_x)` (`FeatureTransformManager.py`, linhas ~589-591); TS-TR normal valida espaço e treina classificador em sintético (`TsTr.py`, linhas ~181-202).  
**Risco:** **baixo** para scaler; **médio** para classificador incremental se sintético estiver fora de escala.

### 4. Real em source e sintético em transformed/generator

**Achado:** o código tenta bloquear.  
**Evidência:** `ScaleGuard.validate_compatible_metadata` falha se `real_space != synthetic_space` (`FeatureTransformManager.py`, linhas ~520-536`) e compara `transform_id` se espaço não-source (`linhas ~537-541`). TR-TS/TS-TR chamam validação em normal e batches (`TrTs.py`, linhas ~72-83 e ~168-186; `TsTr.py`, linhas ~96-107 e ~181-199).  
**Risco:** **crítico** se metadata do sintético não corresponder ao array físico; **médio** em batches porque a validação inicial usa manifest.

### 5. Modo normal e batches usando políticas diferentes

**Achado:** as políticas vêm dos mesmos argumentos; a diferença é de materialização/validação.  
**Evidência:** comandos normal e batches passam `--feature_transform preserve`, `--generator_transform`, `--classifier_transform`, `--evaluation_space` (`run_appclassnet_top200.py`, linhas ~1588-1607 e ~1683-1752). Normal faz `vstack` e valida valores sintéticos (`main.py`, linhas ~621-639); batches escreve manifest e valida metadata (`main.py`, linhas ~1216-1238; `TrTs.py`, linhas ~72-83; `TsTr.py`, linhas ~96-107).  
**Risco:** **médio** por diferença na profundidade de validação.

### 6. Gerador salvando output interno diretamente

**Achado:** no default não. Com `generator_transform` não-preserve e inverse desativado, sim, por configuração explícita: o writer marca `data_space="generator"` e `transform_id`.  
**Evidência:** `ModelInputAdapter.synthetic_space_after_generation` retorna `source` se preserve ou inverse, senão `generator` (`FeatureTransformManager.py`, linhas ~647-650); writer usa esse valor (`main.py`, linhas ~1224-1233).  
**Risco:** **alto** para consumo externo; **crítico** se a avaliação ignorar metadata.

### 7. Classificador aplicando transformação adicional

**Achado:** baseline real-real pode aplicar `classifier_transform`; avaliações principais não parecem aplicar `classifier_manager`.  
**Evidência:** baseline chama `classifier_manager.fit/transform` (`run_appclassnet_top200.py`, linhas ~1066-1069). No pipeline principal, `ModelInputAdapter` cria `classifier_manager` (`FeatureTransformManager.py`, linhas ~582-585), mas `main.py` só usa métodos de gerador (`linhas ~432-441`) e TR-TS/TS-TR treinam diretamente nos arrays (`TrTs.py`, linhas ~188-190; `TsTr.py`, linhas ~200-202`).  
**Risco:** **médio**. O argumento `--classifier_transform` pode não afetar o avaliador principal, gerando divergência entre expectativa e implementação.

### 8. Manifesto inconsistente com arrays reais

**Achado:** o manifesto registra intenção e ranges; não há prova estática de inconsistência no default.  
**Evidência:** pré-processamento salva `preprocessing_manifest.json` com ranges e transform_id (`run_appclassnet_top200.py`, linhas ~948-1005); writer sintético registra `data_space`/`transform_id` (`main.py`, linhas ~1216-1238).  
**Risco:** **médio**. Falta validação amostral obrigatória do arquivo físico contra o manifesto em todos os pontos de consumo.

### 9. Uso do argumento legado `--scaler`

**Achado:** suportado e potencialmente transformador.  
**Evidência:** `ArgumentsDataLoader.py` define `--scaler` (`linhas ~218-221`); `Arguments.py::_normalize_preprocessing_arguments` converte scaler minmax/standard para `feature_transform` se `--feature_transform` não foi passado (`linhas ~162-171`); runner repete (`run_appclassnet_top200.py`, linhas ~592-596).  
**Risco:** **alto** se usado em AppClassNet sem entendimento de espaço; **baixo** no default.

### 10. Transformações automáticas ativadas por defaults

**Achado:** scaler automático não é default para AppClassNet; `auto` resolve para `preserve` no `FeatureTransformPolicy`. O runner, porém, altera `generator_transform=auto` para `minmax` quando a campanha usa sigmoid.  
**Evidência:** `resolve_transform` retorna preserve (`FeatureTransformManager.py`, linhas ~218-222); regra sigmoid no runner (`run_appclassnet_top200.py`, linhas ~598-603).  
**Risco:** **alto** para geradores sigmoid; **baixo** para campanhas AppClassNet com last activation `linear`.

## Diagrama textual do pré-processamento e espaço

```text
CLI
  -> ArgumentsDataLoader define scaler/source_profile/feature_transform/generator_transform/classifier_transform/evaluation_space
  -> Arguments.py e run_appclassnet_top200.py normalizam argumentos
     -> AppClassNet default: preserve/source/inverse synthetic true
     -> uso explicito: scaler minmax|standard pode virar feature_transform
     -> campanhas sigmoid + generator_transform auto podem virar generator_transform minmax

Carregamento
  -> NpyXYLoader._load_array: numpy.load, opcional mmap
  -> NpyXYLoader._load_split: opcional astype(dtype), valida X 2D e labels
  -> DatasetSchema: data_space source, range esperado [-0.5,0.5], transform_id None

Selecao
  -> selecao estratificada no runner/batches preserva pares X/y
  -> sem scaler

Pre-processamento central
  -> FeatureTransformPolicy.for_profile
  -> FeatureTransformManager.fit ou partial_fit_batches somente no train
  -> transform em train/valid/test/synthetic com mesmo transform_id
  -> preserve: sem escala
  -> minmax: [0,1]
  -> standard: padronizado

Treinamento do gerador
  -> main.py fit_generator(train source)
  -> transform_generator_input(train/valid) para espaco generator se configurado
  -> modelo treina no espaco recebido

Geracao
  -> algoritmos geram arrays por classe
  -> se data_type != continuous, alguns algoritmos aplicam numpy.rint
  -> ModelInputAdapter inverse-transforma sintéticos para source quando necessario
  -> normal: dict materializado; batches: SyntheticBatchWriter + manifest

Avaliacao
  -> ScaleGuard valida real source vs synthetic metadata
  -> TR-TS/TS-TR nao aplicam scaler proprio
  -> batch mode valida manifest; normal mode tambem valida array materializado

Metricas e salvamento
  -> métricas recebem arrays no espaço que passou pelo guard
  -> resultados/manifestos registram data_space/transform_id quando disponível
```

## Conclusão

No estado atual do código, o AppClassNet top-200 não recebe `MinMaxScaler` ou `StandardScaler` por default no caminho principal. Os loaders NPY preservam `source`, o runner principal força `feature_transform=preserve`, e o `FeatureTransformPolicy` de AppClassNet também preserva por padrão.

Os riscos reais estão nos caminhos explícitos ou semânticos:

- `--scaler minmax|standard` ainda pode ativar transformação via compatibilidade legada.
- `generator_transform=auto` pode virar `minmax` para campanhas sigmoid.
- sintéticos gerados em espaço interno dependem de `inverse_transform_synthetic` e metadata correta.
- batches confiam mais em manifest do que em validação completa de range físico.
- `classifier_transform` é aplicado no baseline separado, mas não está claramente integrado nas avaliações principais.
- rounding por `numpy.rint` destrói AppClassNet se `data_type` não for `continuous`.

Para regressão AppClassNet, a configuração segura continua sendo: `source_profile=appclassnet_top200`, `feature_transform=preserve`, `generator_transform=preserve` ou inverse obrigatório, `classifier_transform=preserve`, `evaluation_space=source`, `feature_type=continuous`, `num_classes=200`.
