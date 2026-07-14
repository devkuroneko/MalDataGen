# BATCH_AND_MEMORY_AUDIT

Data da auditoria: 2026-07-14

Escopo: auditoria do modo `execution_mode=batches` para AppClassNet top-200. O objetivo é avaliar se o modo batches reduz memória sem alterar a semântica do experimento. Nenhum arquivo de código foi modificado.

## Sumário executivo

- **[crítico]** O modo batches não é streaming end-to-end. O loader NPY pode usar `mmap`, mas `CrossValidation._apply_bundle_to_owner` converte `bundle.train.X` para `numpy.asarray(..., dtype=float32)` e `_create_fold` converte train/evaluation para arrays (`Engine/Evaluation/CrossValidation.py`, funções `_apply_bundle_to_owner` e `_create_fold`, linhas ~285-320 e ~329-346). Se há seleção estratificada, `split.X[indices]` usa indexing avançado e materializa o subconjunto (`linhas ~202-203`).
- **[alto]** `mmap_mode="r"` no loader principal só ocorre quando `--mmap_npy` chega ao `main.py`; no runner AppClassNet, `build_batch_main_command` só passa `--mmap_npy` se o usuário informou `--use_mmap` (`run_appclassnet_top200.py`, função `build_batch_main_command`, linhas ~1697-1698). O runner usa mmap automaticamente em algumas leituras próprias de diagnóstico/preprocessamento quando `execution_mode=batches` (`linhas ~630 e ~874`), mas isso não implica `--mmap_npy` no subprocesso `main.py`.
- **[alto]** O treinamento gerativo em batches ainda treina sobre arrays já selecionados/materializados. O ganho real está em one-hot por batch (`Engine/Models/GenerativeModels.py`, `_batch_one_hot_dataset`, `_fit_features_and_one_hot`, `_fit_autoencode_with_one_hot`, linhas ~109-195), não em ler X diretamente do NPY durante o fit.
- **[alto]** Os classificadores batch se dividem em dois regimes: `SGDClassifier`, `PassiveAggressiveClassifier`, `GaussianNB` e `MLPClassifierSmall` usam `partial_fit` de verdade; `DecisionTreeSubset`, `ExtraTreesSubset` e `RandomForestLight` percorrem batches para coletar um subconjunto estratificado em memória e depois chamam `fit` tradicional (`Engine/Classifiers/BatchClassifiers.py`, `train_batch_classifier`, linhas ~170-237).
- **[médio]** A geração sintética batch é incremental no salvamento: gera lotes por classe e grava via `SyntheticBatchWriter.write_batch` (`main.py`, `_synthesize_data_incremental`, linhas ~1208-1281; `SyntheticBatchIO.py`, linhas ~71-131). Porém cada chamada `generator.get_samples(batch_plan)` ainda materializa o batch completo em memória.
- **[médio]** TR-TS/TS-TR em batches evitam materializar todo sintético, mas acumulam todos os labels e predições em listas antes de converter para arrays (`Engine/Classifiers/BatchClassifiers.py`, `predict_array_batches` e `predict_synthetic_batches`, linhas ~240-274).
- **[médio]** TR-TR não possui caminho batch equivalente: `TrTr.evaluation_TR_TR` treina/prediz sobre arrays do fold e ainda copia arrays para métricas de distância com `numpy.array` (`Engine/Evaluation/TrTr.py`, linhas ~83-127).
- **[baixo]** O último batch não é descartado nos iteradores principais; os loops usam `min(start + batch_size, total)` em `BatchNpyDataset`, `iter_array_batches` e geração (`BatchNpyDataset.py`, linhas ~209-218; `BatchClassifiers.py`, linhas ~79-83; `main.py`, linhas ~1252-1259).

## Fluxos distintos

| Etapa | Implementação atual | Incremental de verdade? | Observação |
|---|---|---:|---|
| Carregamento em batches | `NpyXYLoader` com `numpy.load(..., mmap_mode)`; `BatchNpyDataset` existe como utilitário/teste | Parcial | No pipeline principal depende de `--mmap_npy`; `BatchNpyDataset` não aparece integrado ao `main.py`/`CrossValidation.py` atual. |
| Pré-processamento em batches | `FeatureTransformManager.partial_fit_batches` e `_write_transformed_split` por chunk | Sim para fit/transform por chunk | Escreve NPY transformado via `open_memmap`; não é o fluxo principal preserve. |
| Seleção de subconjunto através de batches | reservoir sampling em y e `split.X[indices]` | Não | Reduz o dataset, mas materializa o subconjunto selecionado. |
| Treinamento gerativo | `model.fit(dataset)` com one-hot por batch | Parcial | One-hot é batch-wise; X já está em array materializado. |
| Geração em batches | loop por classe e `generation_batch_size`; `SyntheticBatchWriter` | Sim para salvamento | Cada batch gerado existe inteiro na memória antes de gravar. |
| Avaliação em batches TR-TS/TS-TR | iteradores de real/sintético + batch classifier | Parcial | Classificador pode ser `partial_fit` ou subset+fit; labels/predições acumulam em memória. |
| Avaliação TR-TR | caminho normal | Não | Usa arrays do fold e distância R-R materializada. |

## Evidências por etapa

### 1. Carregamento NPY e mmap

**Arquivo:** `Engine/DataIO/NpyXYLoader.py`  
**Classe:** `NpyXYLoader`  
**Funções:** `_load_array`, `_load_split`

- **Evidência:** `_load_array` usa `numpy.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)` (`NpyXYLoader.py`, linhas ~163-170). `_load_split` só aplica `x_values.astype(self.dtype, copy=False)` se `dtype` foi passado (`linhas ~146-161`).
- **mmap:** possível, mas controlado pelo chamador.
- **Risco:** **médio**. O loader suporta mmap, mas não garante mmap sozinho.

**Arquivo:** `Engine/Evaluation/CrossValidation.py`  
**Funções:** `_mmap_mode_for_npy`, `load_dataset_from_args`

- **Evidência:** `_mmap_mode_for_npy` retorna `'r'` somente se `arguments.mmap_npy` for true (`linhas ~232-233`). `load_dataset_from_args` passa esse valor ao `NpyXYLoader` (`linhas ~267-281`).
- **Conclusão:** no `main.py`, `execution_mode=batches` não basta para mmap; precisa `--mmap_npy`.
- **Risco:** **alto**.

**Arquivo:** `run_appclassnet_top200.py`  
**Funções:** `build_batch_main_command`, `main`

- **Evidência:** `build_batch_main_command` adiciona `--mmap_npy` somente quando `parsed_arguments.use_mmap` é true (`linhas ~1697-1698`). O runner avisa quando `execution_mode=batches` está sem `--use_mmap` (`linhas ~2278-2279`).
- **Conclusão:** o runner reconhece o risco, mas não força mmap.
- **Risco:** **alto**.

### 2. `np.asarray`, `astype` e materialização

**Arquivo:** `Engine/Evaluation/CrossValidation.py`  
**Função:** `_apply_bundle_to_owner`

- **Evidência:** `owner._data_loaded = numpy.asarray(bundle.train.X, dtype=numpy.float32)` antes da seleção (`linhas ~285-290`) e novamente após `_apply_batch_limits` (`linhas ~311-320`).
- **Memória:** se `bundle.train.X` já for `float32` memmap, `numpy.asarray` pode preservar uma view/base sem copiar todos os bytes; se o dtype for diferente, copia o array inteiro. Como AppClassNet é esperado em float32, o risco prático depende dos arquivos reais.
- **Risco:** **alto**, porque o código não garante ausência de cópia.

**Arquivo:** `Engine/Evaluation/CrossValidation.py`  
**Função:** `_apply_stratified_split_selection`

- **Evidência:** depois de selecionar índices, executa `split.X = numpy.asarray(split.X[indices], dtype=numpy.float32)` e `split.y = numpy.asarray(split.y[indices])` (`linhas ~202-203`).
- **Memória:** `split.X[indices]` é indexing avançado; copia o subconjunto selecionado. Para 200 classes * 1000 amostras * 20 features float32, isso é ~16 MB para X, mais temporários.
- **Risco:** **alto**. Reduz dataset, mas altera o regime para “subconjunto materializado”.

**Arquivo:** `Engine/DataIO/BatchNpyDataset.py`  
**Classe:** `BatchNpyDataset`  
**Funções:** `_load_array`, `_iter_contiguous_batches`, `_iter_indexed_batches`, `_format_batch`

- **Evidência:** `_load_array` usa `numpy.load(..., mmap_mode=self.mmap_mode)` (`linhas ~86-90`); `_iter_contiguous_batches` fatia ranges contíguos (`linhas ~209-212`); `_iter_indexed_batches` usa `self._x[batch_indices]` (`linhas ~214-218`); `_format_batch` aplica `astype` se `dtype` foi solicitado (`linhas ~246-249`).
- **Integração atual:** busca por `BatchNpyDataset` mostra uso em testes e documentação, mas não no caminho `main.py`/`CrossValidation.py`; o pipeline AppClassNet atual usa `NpyXYLoader` e depois seleção/materialização de splits.
- **Conclusão:** contíguo é mais amigável a mmap; indexed/shuffled materializa o batch.
- **Risco:** **médio**.

### 3. Indexing avançado, seleção e shuffle

**Arquivo:** `Engine/DataIO/StratifiedNpySelection.py`  
**Função:** `select_stratified_indices_from_npy`

- **Evidência:** carrega labels com `numpy.load(..., mmap_mode=mmap_mode)` e faz reservoir sampling por classe (`linhas ~17-96`). Ao final concatena índices selecionados com `numpy.concatenate(selected_parts)` e embaralha (`linhas ~58-69`).
- **Memória:** proporcional ao número de índices selecionados, não a X. Para 200.000 linhas selecionadas, vetor `int64` ~1.6 MB.
- **Semântica:** seleção é estratificada por classe e varre o y completo, evitando viés de arquivos ordenados.
- **Risco:** **baixo** para distribuição por classe; **médio** por ordem final aleatória diferente do caminho normal.

**Arquivo:** `Engine/DataIO/BatchNpyDataset.py`  
**Função:** `_iter_shuffled_batches`

- **Evidência:** se há `_indices`, copia todos os índices selecionados e embaralha (`linhas ~223-226`). Se não há seleção e `num_samples <= 1_000_000`, cria permutação global (`linhas ~229-231`). Acima disso, usa shuffle local por chunk (`linhas ~234-244`).
- **Semântica:** shuffle local por chunk não é shuffle global; a própria docstring registra que linhas não cruzam fronteiras de chunk (`linhas ~25-30`).
- **Risco:** **médio**. Pode introduzir viés de ordem se o arquivo estiver ordenado por classe e não houver seleção estratificada global.

### 4. DataFrame e CSV

**Arquivo:** `Engine/Evaluation/CrossValidation.py`  
**Função:** `_save_data_to_csv`

- **Evidência:** concatena `data` e `labels` com `numpy.column_stack`, cria `pandas.DataFrame` e salva CSV (`linhas ~207-218`).
- **Modo batches:** no caminho de KFold legado, se `execution_mode=="batches"`, o código pula a materialização CSV intermediária (`linhas ~521-538`).
- **Risco:** **baixo** em batches; **alto** no modo normal/CSV, onde DataFrame materializa todo fold.

### 5. Train/valid/test e concatenação

**Arquivo:** `Engine/Evaluation/CrossValidation.py`  
**Funções:** `_build_provided_split_folds`, wrapper `StratifiedData`

- **Evidência:** `split_mode=provided` cria um fold com `bundle.train` e `bundle.valid or bundle.test` (`linhas ~357-380`). Se valid e test existem, marca test como não aplicável ao pipeline atual (`linhas ~362-365`).
- **Concatenação:** não concatena train/valid/test no modo provided. No modo `cross_validation`, para `npy_xy`, avisa que usa apenas train para K-fold e não concatena valid/test (`linhas ~437-441`).
- **Risco:** **baixo** para concatenação indevida; **médio** para semântica, porque AppClassNet com valid+test não avalia test final no pipeline principal atual.

### 6. One-hot global vs one-hot batch

**Arquivo:** `Engine/DataIO/LabelUtils.py`  
**Funções:** `one_hot_encode_labels`, `to_one_hot_batch`

- **Evidência:** `one_hot_encode_labels` aloca matriz `(n, num_classes)` float32 completa (`linhas ~98-103`). `to_one_hot_batch` aloca só `(batch, num_classes)` (`linhas ~106-128`).
- **Risco:** **crítico** se caminho normal for usado em AppClassNet grande; **baixo** no helper batch.

**Arquivo:** `Engine/Models/GenerativeModels.py`  
**Funções:** `_batch_one_hot_dataset`, `_fit_features_and_one_hot`, `_fit_autoencode_with_one_hot`

- **Evidência:** em modo batches, `_fit_features_and_one_hot` e `_fit_autoencode_with_one_hot` criam `tensorflow.data.Dataset.from_generator` com one-hot por batch (`linhas ~123-166 e ~177-195`). Fora de batches, chamam `_labels_to_one_hot` global (`linhas ~168-174 e ~189-195`).
- **Limite:** `_one_hot_batch_generator` começa com `x_values = numpy.asarray(x_real_samples, dtype=numpy.float32)` (`linhas ~109-120`), então o X já precisa estar disponível como array.
- **Risco:** **médio**. Resolve one-hot global, mas não streaming de X direto do NPY.

### 7. Treinamento gerativo em batches

**Arquivo:** `main.py`  
**Classe:** `SynDataGen`  
**Função:** `run_experiments`

- **Evidência:** antes do treino, `x_training_real` passa por `ModelInputAdapter.transform_generator_input`; o gerador treina com `self.train_model(x_training_for_generator, y_training_real, ...)` (`linhas ~432-468` em auditorias anteriores).
- **Conclusão:** o fit do gerador usa o array do fold, não um iterador NPY.
- **Risco:** **alto** para memória se o fold/subconjunto for grande.

**Arquivo:** `Engine/Models/GenerativeModels.py`  
**Funções:** `_training_adversarial_modelo`, `_training_autoencoder_model`, `_training_quantized_VAE_model`, `_training_variational_autoencoder_model`, `_training_wasserstein_model`, `_training_wasserstein_gp_model`

- **Evidência:** modelos suportados chamam `_fit_features_and_one_hot` ou `_fit_autoencode_with_one_hot` (`linhas ~362-372, ~687-697, ~1018-1028, ~2977-2987, ~2130-2140, ~2569-2579`).
- **Modelos bloqueados em batches:** `latent_diffusion` rejeita batches por depender de one-hot global (`linhas ~1465-1469`); `random_noise` e `smote` também rejeitam (`linhas ~4039-4062`).
- **Risco:** **médio**. Há proteção contra alguns modelos incompatíveis; ainda há materialização de X.

### 8. Geração em batches

**Arquivo:** `main.py`  
**Funções:** `synthesize_data`, `_synthesize_data_incremental`, `_synthesize_data_partitioned`

- **Evidência:** se `execution_mode=="batches"` e `materialize_synthetic=False`, chama `_synthesize_data_incremental` (`linhas ~838-847`). Se `materialize_synthetic=True`, registra warning e segue geração materializada (`linhas ~849-853`).
- **Incremental:** `_synthesize_data_incremental` itera classes e subranges com `generation_batch_size`; chama `generator.get_samples(batch_plan)[label_class]`, aplica inverse se necessário, grava com `writer.write_batch` e registra auditoria (`linhas ~1208-1281`).
- **Particionado:** `_synthesize_data_partitioned` treina geradores por classe/grupo e grava batches; após cada unidade chama `_release_current_generator` (`linhas ~1004-1177`).
- **Risco:** **médio**. Sintéticos não são acumulados globalmente, mas cada batch é materializado; `audit.record` converte o batch para `numpy.asarray` e calcula stats (`SyntheticLabelAudit.py`, linhas ~47-71).

**Arquivo:** `Engine/DataIO/SyntheticBatchIO.py`  
**Classes:** `SyntheticBatchWriter`, `SyntheticBatchReader`

- **Evidência:** `write_batch` salva cada batch como `.npy`, `.csv` ou fatia em `single_npy` (`linhas ~71-117`). `SyntheticBatchReader._load_batch` usa `numpy.load(..., mmap_mode="r")` para `.npy` (`linhas ~162-170`).
- **Risco:** **baixo** para `npy_batches`; **médio** para `csv_batches` porque `numpy.loadtxt` materializa o CSV; **médio** para `single_npy` por arquivo grande, embora use `open_memmap` na escrita.

### 9. Avaliação em batches

**Arquivo:** `Engine/Evaluation/TrTs.py`  
**Classe:** `TrTs`  
**Função:** `evaluation_TR_TS`

- **Evidência:** em batches, valida metadata, treina classificador em batches de real evaluation, e prediz em batches sintéticos (`linhas ~72-136`).
- **Treino usado:** `dictionary_data['x_evaluation_real']`, não `x_training_real`, conforme auditoria TR-TS anterior.
- **Risco:** **médio** para semântica TR-TS; **baixo** para não materializar sintético.

**Arquivo:** `Engine/Evaluation/TsTr.py`  
**Classe:** `TsTr`  
**Função:** `evaluation_TS_TR`

- **Evidência:** em batches, treina em `iter_synthetic_labeled_batches(synthetic_data)`, seleciona subconjunto estratificado do real evaluation e prediz por batches (`linhas ~96-160`).
- **Materialização:** `_select_stratified_array_subset` seleciona arrays reais; `predict_array_batches` acumula labels/predictions.
- **Risco:** **médio**.

**Arquivo:** `Engine/Evaluation/TrTr.py`  
**Classe:** `TrTr`  
**Função:** `evaluation_TR_TR`

- **Evidência:** não há branch batches; treina com `dictionary_data['x_training_real']`, prediz `dictionary_data['x_evaluation_real']`, depois cria `numpy.array` de train/evaluation para distância R-R (`linhas ~83-127`).
- **Risco:** **alto** para memória e semântica de batches, porque TR-TR não é incremental.

### 10. Classificadores batch

**Arquivo:** `Engine/Classifiers/BatchClassifiers.py`  
**Funções:** `make_batch_classifier`, `train_batch_classifier`, `_collect_stratified_subset`

- **Parcialmente incrementais:** `sgd`, `passive_aggressive`, `naive_bayes`, `mlp_small` (`PARTIAL_FIT_CLASSIFIERS`, linha ~31) usam `partial_fit` e informam `classes=np.arange(num_classes)` no primeiro batch (`linhas ~181-198`).
- **Subconjunto + fit:** `decision_tree_subset`, `extra_trees_subset`, `random_forest_light` coletam reservoir por classe, fazem `numpy.vstack`/`numpy.concatenate`, embaralham e chamam `classifier.fit` (`linhas ~114-167 e ~202-214`).
- **Risco:** **alto** se o usuário espera DecisionTree realmente incremental; **médio** por usar subset para reduzir memória.

### 11. Limpeza de modelos e sessões

**Arquivo:** `main.py`  
**Classe:** `SynDataGen`  
**Função:** `_release_current_generator`

- **Evidência:** zera referências do gerador ativo por `model_type`, chama `tensorflow.keras.backend.clear_session()` e `gc.collect()` (`linhas ~685-710`).
- **Quando é chamada:** no caminho de geração particionada, depois de cada unidade (`main.py`, `_synthesize_data_partitioned`, linha ~1148).
- **Limite:** no caminho incremental de gerador único, não há chamada explícita logo após finalizar geração; modelos permanecem até fim do fold/objeto.
- **Risco:** **médio**. Particionado limpa melhor; single generator pode manter memória TensorFlow.

## Respostas objetivas aos 20 pontos

1. **Se `np.load` usa `mmap_mode="r"`:** **parcial**. `SyntheticBatchReader` usa mmap para `.npy` (`SyntheticBatchIO.py`, ~162-168). `NpyXYLoader` usa o mmap recebido (`NpyXYLoader.py`, ~163-170), mas `CrossValidation._mmap_mode_for_npy` só ativa com `--mmap_npy` (`CrossValidation.py`, ~232-233). O runner usa mmap em diagnósticos/preprocessamento quando `execution_mode=batches` (`run_appclassnet_top200.py`, ~630 e ~874), mas só propaga `--mmap_npy` ao `main.py` com `--use_mmap` (`~1697-1698`). `BatchNpyDataset` também suporta mmap, mas não está no caminho principal atual.
2. **Se algum `np.asarray` materializa o memmap completo:** **possível**. `_apply_bundle_to_owner` chama `numpy.asarray(bundle.train.X, dtype=float32)` (`CrossValidation.py`, ~285-320). Se dtype divergir, copia inteiro; se já for float32, tende a não copiar integralmente.
3. **Se `astype` cria cópia integral:** **possível**. `NpyXYLoader._load_split` usa `astype` se dtype foi solicitado (`NpyXYLoader.py`, ~150-151); `BatchNpyDataset._format_batch` idem por batch (`BatchNpyDataset.py`, ~246-249); labels sempre podem copiar para int64 (`LabelUtils.py`, ~14-34).
4. **Se indexing avançado copia grandes matrizes:** **sim**. `split.X[indices]` em `_apply_stratified_split_selection` copia o subconjunto selecionado (`CrossValidation.py`, ~202-203).
5. **Se DataFrame materializa todos os dados:** **sim no caminho CSV/normal**, não no branch batches de KFold. `_save_data_to_csv` cria `DataFrame` completo (`CrossValidation.py`, ~207-218), mas batches pula CSV intermediário (`~521-538`).
6. **Se train/valid/test são concatenados:** **não encontrado**. Provided split usa train e valid/test sem concatenação (`CrossValidation.py`, ~357-380); KFold em `npy_xy` usa apenas train e avisa que valid/test não são usados (`~437-441`).
7. **Se one-hot global é criado:** **não no helper batch dos modelos suportados**. Batch usa `to_one_hot_batch`; normal usa `one_hot_encode_labels` global (`GenerativeModels.py`, ~156-195; `LabelUtils.py`, ~98-128). Alguns modelos são bloqueados em batches por risco de one-hot global.
8. **Se sintéticos são acumulados em listas:** **no caminho normal/materializado, sim**; no incremental, não globalmente. `synthesize_data` normal mantém `self.data_generated` dict completo (`main.py`, ~855-963). Incremental grava por batch (`~1208-1281`).
9. **Se `np.concatenate` é chamado repetidamente:** **não em loop crítico de geração incremental**. Há `numpy.concatenate` para índices selecionados (`StratifiedNpySelection.py`, ~66-68) e para subset classifier (`BatchClassifiers.py`, ~157-158). Algoritmos gerativos podem concatenar chunks internos por chamada de batch.
10. **Se todos os batches são mantidos na memória:** **não para sintéticos incremental**; **sim para o subconjunto final de classificadores subset**. `SyntheticBatchWriter` grava e mantém manifest leve; `_collect_stratified_subset` mantém reservatórios e depois `x_subset`.
11. **Se modelos e sessões anteriores são liberados:** **só no particionado**. `_release_current_generator` limpa referências, Keras session e GC (`main.py`, ~685-710; chamado em ~1148).
12. **Se existe `gc.collect` ou equivalente:** **sim**, em `_release_current_generator` (`main.py`, ~709-710).
13. **Se TensorFlow/PyTorch mantém modelos anteriores:** **TensorFlow pode manter no caminho single generator**; particionado chama `clear_session`. Não há evidência de PyTorch nesse caminho.
14. **Se batches são realmente usados no treinamento:** **sim para one-hot e classificadores `partial_fit`**; **não para streaming de X do NPY ao gerador**. `_one_hot_batch_generator()` começa convertendo `x_real_samples` com `numpy.asarray(..., dtype=float32)` (`GenerativeModels.py`, ~109-120).
15. **Se modo batches apenas seleciona subconjunto e chama fit tradicional:** **sim para DecisionTree/ExtraTrees/RandomForest batch classifiers**; **parcial para gerador**, que usa arrays selecionados e minibatches internos.
16. **Se apenas o primeiro batch é utilizado:** **não encontrado**. Loops iteram todos os batches; `partial_fit` trata primeiro batch apenas para passar `classes`.
17. **Se a ordem dos batches causa perda de classes:** **não diretamente**. Seleção estratificada por reservoir cobre classes observadas; porém shuffle local por chunk pode preservar vieses de ordem quando não há seleção.
18. **Se shuffle por chunk produz viés:** **possível**. `BatchNpyDataset` documenta que chunk shuffle não move linhas entre chunks (`linhas ~25-30 e ~234-244`).
19. **Se o último batch é descartado:** **não**. Os loops usam `min(...)` para incluir resto.
20. **Se batches alteram distribuição por classe:** **sim quando quotas são aplicadas**. `_apply_batch_limits` seleciona `samples_per_class` por split (`CrossValidation.py`, ~103-146), e isso reduz/altera o dataset efetivo para quotas estratificadas.

## Semântica versus redução de memória

| Mecanismo | Reduz memória? | Altera semântica do experimento? | Evidência |
|---|---:|---:|---|
| `--mmap_npy` no `main.py` | Sim | Não deveria | `CrossValidation._mmap_mode_for_npy()` retorna `r` só com `arguments.mmap_npy`, linhas ~232-233. |
| Seleção `train_samples_per_class`/`test_samples_per_class` | Sim | Sim, por protocolo explícito | `_apply_batch_limits()` aplica seleção por split, linhas ~103-146. |
| `max_samples_per_class`/`max_train_samples` | Sim | Sim | `_batch_samples_per_class()` converte limites em quota por classe, linhas ~123-146. |
| One-hot por batch | Sim | Não deveria | `GenerativeModels._batch_one_hot_dataset()`, linhas ~123-154. |
| `decision_tree_subset` | Sim | Sim | `_collect_stratified_subset()` materializa reservoir e `train_batch_classifier()` chama `classifier.fit`, linhas ~114-214. |
| `sgd`/`passive_aggressive`/`naive_bayes`/`mlp_small` | Sim | Pode alterar avaliador | `partial_fit` usa classes globais no primeiro batch, linhas ~181-198; não é DecisionTree golden. |
| Sintético `npy_batches` | Sim | Não deveria se metadata/ordem de labels estiver correta | `SyntheticBatchWriter.write_batch()` grava por classe/batch, linhas ~71-116. |
| `--materialize_synthetic` em batches | Não | Não deveria, mas aumenta memória | `main.py` avisa alto uso quando `materialize_synthetic=True`, linhas ~849-853. |

## Estimativas de memória

Assumindo AppClassNet top-200, 20 features `float32`, labels `int64` após validação, treino golden com 1.000 amostras/classe e teste com 500 amostras/classe:

| Item | Fórmula | Exemplo golden | Observação |
|---|---:|---:|---|
| X real treino | `200 * 1000 * 20 * 4` bytes | ~15.26 MiB | Se memmap sem cópia, fica majoritariamente fora de RAM; se selecionado/indexado, materializa. |
| y real treino | `200 * 1000 * 8` bytes | ~1.53 MiB | `validate_zero_based_labels` retorna int64. |
| X real teste | `200 * 500 * 20 * 4` bytes | ~7.63 MiB | Evaluation fold materializado. |
| y real teste | `200 * 500 * 8` bytes | ~0.76 MiB | Labels/predições também consomem similar. |
| One-hot treino global | `200000 * 200 * 4` bytes | ~152.59 MiB | Evitado pelos helpers batch dos modelos suportados. |
| One-hot por batch 8192 | `8192 * 200 * 4` bytes | ~6.25 MiB | Por batch; além de X batch. |
| X batch 8192 | `8192 * 20 * 4` bytes | ~0.625 MiB | Pequeno para AppClassNet. |
| Sintético treino 1000/classe | `200000 * 20 * 4` bytes | ~15.26 MiB | Incremental evita manter tudo; materializado mantém dict completo. |
| Sintético teste 500/classe | `100000 * 20 * 4` bytes | ~7.63 MiB | TR-TS normal materializa; batch lê por arquivo/lote. |
| Labels/predições 200k | `2 * 200000 * 8` bytes | ~3.05 MiB | Batch prediction acumula labels e predictions em listas e depois arrays. |
| Matriz de confusão 200x200 | `200 * 200 * 8` bytes | ~0.31 MiB | Usada no sanity check; pequena. |
| Stats por classe | `200 * 20 * 8 * alguns arrays` | <1 MiB | `SyntheticSanityChecks` mantém counts/sums/means/min/max. |
| Cópia temporária por `vstack` subset | próximo de X subset | ~15.26 MiB para 200k | `_collect_stratified_subset` cria `x_parts` e `x_subset`. |
| DecisionTreeClassifier | dependente de nós | variável | Pode crescer bem mais que X em classes complexas; não incremental. |
| RandomForest/ExtraTrees | `n_estimators * árvores` | alto | `n_jobs=-1` pode aumentar memória por paralelismo. |
| Modelo gerativo TensorFlow | pesos + ativações + otimizador | dominante | Depende da arquitetura; provavelmente maior que X tabular. |

Para o dataset bruto completo, use:

```text
X_bytes = n_rows * 20 * 4
y_bytes_int64 = n_rows * 8
one_hot_bytes = n_rows * 200 * 4
```

Exemplo por 1.000.000 linhas: X ~76.29 MiB, y int64 ~7.63 MiB, one-hot global ~762.94 MiB.

## Classificação final de incrementalidade

### Verdadeiramente incrementais

- **Pré-processamento por chunk:** `FeatureTransformManager.partial_fit_batches` e `_write_transformed_split` no runner (`run_appclassnet_top200.py`, linhas ~822-922).
- **One-hot por batch:** `to_one_hot_batch` + `tensorflow.data.Dataset.from_generator` (`LabelUtils.py`, ~106-128; `GenerativeModels.py`, ~109-195).
- **Classificadores com `partial_fit`:** `sgd`, `passive_aggressive`, `naive_bayes`, `mlp_small` (`BatchClassifiers.py`, ~31 e ~181-198).
- **Geração sintética incremental:** `_synthesize_data_incremental` + `SyntheticBatchWriter` (`main.py`, ~1208-1281; `SyntheticBatchIO.py`, ~71-131).
- **Leitura sintética `.npy` por batch:** `SyntheticBatchReader._load_batch` com mmap (`SyntheticBatchIO.py`, ~162-170).

### Reduzem dataset, mas não são incrementais

- **Seleção estratificada por classe:** varre y por mmap, mas depois materializa `split.X[indices]` (`CrossValidation.py`, ~148-205).
- **DecisionTree/ExtraTrees/RandomForest batch classifiers:** coletam subset por reservoir e chamam `fit` tradicional (`BatchClassifiers.py`, ~114-167 e ~202-214).
- **Geração particionada:** reduz classes por gerador e limpa sessão entre unidades, mas cada unidade treina com `unit_x` materializado (`main.py`, ~1067-1148).

### Materializados

- **Fold principal provided:** `_create_fold` monta arrays train/evaluation (`CrossValidation.py`, ~329-346).
- **TR-TR:** usa arrays completos e copia para distância R-R (`TrTr.py`, ~83-127).
- **Predições finais:** `predict_array_batches`/`predict_synthetic_batches` acumulam labels e predictions (`BatchClassifiers.py`, ~240-274).
- **Modo `--materialize_synthetic`:** força dict sintético completo em memória (`main.py`, ~849-963).

## Riscos para semântica AppClassNet

- **[crítico]** Se o objetivo é manter a semântica do experimento completo, `max_samples_per_class`, `train_samples_per_class`, `test_samples_per_class` e `batch_classifier_subset_size` precisam ser tratados como parte explícita do protocolo, porque batches pode reduzir o dataset antes do treino/avaliação.
- **[alto]** O golden baseline real-real com `DecisionTreeClassifier`, 1000/classe train e 500/classe test deve ser validado no caminho batch separadamente, porque TR-TR do pipeline principal não usa `BatchClassifiers.py` e ainda materializa arrays.
- **[alto]** O default `decision_tree_subset` em batches não é um DecisionTree treinado incrementalmente em todos os dados; é uma árvore treinada em subconjunto estratificado coletado de batches.
- **[médio]** `--use_mmap` deveria ser considerado obrigatório para o runner AppClassNet em `execution_mode=batches` se a meta é reduzir memória. Hoje é apenas recomendado e gera warning.
- **[médio]** `csv_batches` reduz memória de escrita, mas leitura por `numpy.loadtxt` materializa cada CSV batch e é mais custosa que `.npy`.
- **[médio]** A sanidade de sintéticos é incremental, mas calcula predições por batch e matriz de confusão; memória é pequena para 200 classes, mas tempo pode crescer com todos os sintéticos.

## Conclusão

O modo batches atual reduz memória em partes importantes, especialmente one-hot, geração sintética e avaliação TR-TS/TS-TR com sintéticos em disco. Ele não preserva uma semântica “mesmo experimento, apenas streaming” de ponta a ponta: há seleção/materialização de subconjuntos, treinamento gerativo com arrays já carregados, TR-TR sem branch batch, e classificadores batch que em alguns casos apenas coletam subset e chamam `fit`.

Para AppClassNet top-200, a leitura correta do modo batches é: **modo de redução de memória por mmap + seleção estratificada + one-hot batch-wise + sintéticos persistidos por batch**, não um pipeline totalmente incremental sobre todos os splits originais.
