# Label And Conditioning Audit - AppClassNet Top-200

Data da auditoria: 2026-07-14

Escopo: fluxo de labels, classes e condicionamento por classe. Nenhum codigo de execucao foi alterado.

## Conclusao executiva

O caminho AppClassNet `npy_xy` tem uma base correta para 200 classes: o loader junta labels de train/valid/test para inferir metadata, valida labels inteiros, preserva mapping unico e expõe `num_classes=200` quando o argumento e passado. O one-hot centralizado usa `num_classes=number_samples_per_class["number_classes"]`, portanto pode criar vetores de 200 posicoes e falha se label `199` estiver fora do dominio configurado.

Os principais riscos encontrados sao:

- Ainda existem defaults legados `number_classes=2` em argumentos de varios modelos. O runner AppClassNet sobrescreve esses parametros para 200, mas execucoes diretas de `main.py` podem depender da inferencia/metadata para escapar desses defaults.
- O `CSVLoader` converte labels discretos para `float32` apos codificacao zero-based; depois os helpers convertem para `int64`. Isso nao arredonda labels fracionarios, mas e uma conversao float->int permitida quando os valores sao inteiros.
- Nao ha uso direto de `to_categorical` no fluxo auditado; o projeto usa helpers proprios em `LabelUtils.py` para criar one-hot.
- Labels sinteticos sao definidos pela classe solicitada/chave do dicionario/manifesto, nao por um output de classe do gerador. Isso e coerente para geracao condicional, mas significa que o label salvo e uma declaracao do plano de geracao.
- Batches sinteticos nao salvam um `y.npy` separado; o label e reconstruido pela chave `batches_by_class` do manifesto. Um unico batch fisico com multiplas classes nao e representavel nesse contrato sem labels separados.
- `match_train_distribution` pode atribuir zero amostras a classes quando `total_rows` e pequeno. Para AppClassNet top-200, `balanced_per_class` e o caminho mais seguro para garantir todas as classes.

## Achados classificados

| Severidade | Achado | Evidencia |
|---|---|---|
| Critico | Defaults legados de varios modelos ainda sao `number_classes=2`; AppClassNet precisa sobrescrever ou inferir corretamente 200. | `Engine/Arguments/ArgumentsAutoencoder.py`, constante `DEFAULT_AUTOENCODER_NUMBER_CLASSES`, linha ~49; `ArgumentsVariationalAutoencoder.py`, linha ~41; `ArgumentsQuantizedVAE.py`, linha ~42; `ArgumentsWassersteinGAN.py`, linha ~40; `ArgumentsWassersteinGANGP.py`, linha ~40. |
| Alto | O runner AppClassNet sobrescreve modelos principais para 200, mas isso depende do uso do runner/campanha. | `run_appclassnet_top200.py`, campanhas `autoencoder`, `variational`, `quantized`, `wasserstein`, `wasserstein_gp`, linhas ~141-260. |
| Alto | `number_classes` em `main.py` e o maximo entre metadata, argumentos de modelos e labels observados; isso protege contra defaults 2 quando schema/metadata tem 200, mas tambem mascara inconsistencias de configuracao. | `main.py`, classe `SynDataGen`, funcoes `_get_configured_number_classes()` e `_build_generation_metadata()`, linhas ~532-581. |
| Alto | Labels sinteticos sao salvos como a classe solicitada, nao inferidos pelo output do gerador. Se o gerador ignorar condicionamento, o label ainda sera o solicitado. | Algoritmos `get_samples()` usam `generated_data[label_class] = generated_samples`, por exemplo `AutoencoderAlgorithm.py` linhas ~284-312; `AdversarialAlgorithm.py` linhas ~335-361; `main.py` grava/audita `saved_label=int(label_class)`, linhas ~1262-1268. |
| Medio | `CSVLoader` codifica labels discretos zero-based e armazena como `float32`; helpers posteriores exigem valores inteiros e convertem para `int64`. | `Engine/DataIO/CSVLoader.py`, funcao `_encode_discrete_labels()`, linhas ~303-317; `Engine/DataIO/LabelUtils.py`, funcao `labels_to_1d_integer()`, linhas ~14-34. |
| Medio | `SynDataGen._prepare_labels_for_conditional_generation()` tambem faz `labels.astype(int)` antes do plano de geracao; no caminho NPY isso vem depois da validacao de labels inteiros, mas em caminhos legados e uma conversao direta. | `main.py`, classe `SynDataGen`, funcao `_prepare_labels_for_conditional_generation()`, linhas ~524-527. |
| Medio | `match_train_distribution` pode gerar contagem zero para classes quando `total_rows` e menor/insuficiente; isso pode perder classes na geracao. | `Engine/DataIO/SamplePlanner.py`, funcao `_allocate_proportional()`, linhas ~165-182. |
| Medio | Em batches, labels sinteticos nao existem como arquivo y separado; a leitura posterior reconstroi labels pelo manifesto. | `Engine/DataIO/SyntheticBatchIO.py`, `SyntheticBatchWriter.write_batch()` linhas ~71-116; `SyntheticBatchReader.items()` linhas ~154-160; `BatchClassifiers.iter_synthetic_labeled_batches()` linhas ~86-97. |
| Medio | O contrato de batches assume que cada batch pertence a uma unica classe. Se um arquivo de batch contiver amostras de multiplas classes, todas seriam rotuladas pela chave `class_label` do manifesto. | `Engine/DataIO/SyntheticBatchIO.py`, classe `SyntheticBatchReader`, funcao `items()`, linhas ~154-160; `Engine/Classifiers/BatchClassifiers.py`, funcao `iter_synthetic_labeled_batches()`, linhas ~86-97. |
| Baixo | `RandomNoiseAlgorithm.fit()` usa `argmax` para converter one-hot em classe interna; nos algoritmos condicionais principais, argmax nao define labels sinteticos salvos. | `Engine/Algorithms/RandomNoise/AlgorithmRandomNoise.py`, funcao `fit()`, linhas ~123-136. |
| Baixo | Arredondamento (`numpy.rint`) aparece nos valores gerados quando `data_type != "continuous"`, nao nos labels. | Exemplos: `AutoencoderAlgorithm.py` linhas ~306-310; `AdversarialAlgorithm.py` linhas ~356-358; `AlgorithmQuantizedVAE.py` linhas ~279-281. |
| Baixo | Nao foi encontrado uso de `to_categorical`; o one-hot e manual e validado por `LabelUtils`. | Busca por `to_categorical` em `Engine`, `main.py` e `run_appclassnet_top200.py`; `Engine/DataIO/LabelUtils.py`, funcoes `one_hot_encode_labels()` e `to_one_hot_batch()`, linhas ~98-128. |

## Fluxo de labels e classes

### Entrada NPY

- Arquivo: `Engine/DataIO/NpyXYLoader.py`
- Classe: `NpyXYLoader`
- Funcoes: `load()`, `_load_split()`, `_validate_multiclass_labels()`, `_handle_multiclass_label_base()`, `_get_class_metadata()`
- Linhas aproximadas: ~87-139, ~146-188, ~190-223, ~248-281

Comportamento:

```python
train = load_split("train")
valid = load_split("valid")
test = load_split("test")
original_labels = collect_labels(train, valid, test)
label_mapping = build_label_mapping(original_labels)
if target_type == "multiclass":
    handle_1_based_remap(train, valid, test)
class_labels, num_classes = get_class_metadata(train, valid, test)
schema = DatasetSchema(num_classes=num_classes, class_labels=class_labels, data_space="source")
```

Evidencias:

- O mapping e criado a partir de labels coletados de todos os splits, nao separadamente por split: `NpyXYLoader.load()`, linhas ~97-103.
- Labels 1-based so sao remapeados se `remap_labels_to_zero_based=True`: `NpyXYLoader._handle_multiclass_label_base()`, linhas ~190-211.
- Para labels 1-based sem remap, `class_labels` vira `tuple(range(1, num_classes + 1))`: `_get_class_metadata()`, linhas ~256-270.
- Para zero-based, `max_label >= num_classes` falha: `_get_class_metadata()`, linhas ~271-274.

### Entrada CSV legada

- Arquivo: `Engine/DataIO/CSVLoader.py`
- Classe: `CSVDataProcessor`
- Funcoes: `_process_label_column()`, `_encode_discrete_labels()`, `_decode_label()`, `save_csv()`
- Linhas aproximadas: ~288-317, ~378-405

Comportamento:

- Labels discretos sao mapeados para indices zero-based pela ordem `sorted(numpy.unique(labels))`.
- O mapping inverso e usado ao salvar CSV sintetico.
- O label interno vira `float32` apos codificacao, mas os helpers de label exigem inteiro antes de one-hot/avaliacao.

Risco:

- Ordem de classes no CSV depende de `sorted(unique_labels)`. Isso e estavel, mas pode diferir de uma ordem semantica externa se labels originais nao forem numericos simples.

### Metadata para geracao

- Arquivo: `main.py`
- Classe: `SynDataGen`
- Funcoes: `_prepare_labels_for_conditional_generation()`, `_get_configured_number_classes()`, `_build_generation_metadata()`
- Linhas aproximadas: ~524-581

Comportamento:

```python
labels = ravel(y_real_samples).astype(int)
number_classes = max(
    metadata["number_classes"],
    arguments.*_number_classes,
    max(labels) + 1,
    unique(labels).shape[0],
)
sample_plan = build_sample_plan_from_args(arguments, labels, number_classes)
generation_metadata = {
    "classes": {label: count},
    "number_classes": number_classes,
}
```

Conclusao:

- Para AppClassNet correto, `number_classes` deve ficar 200 por schema/argumentos.
- Classe `199` nao e perdida se `number_classes=200`, porque o one-hot aceita indices `0..199`.
- Se os labels observados forem `1..200` e nao houver remap, `max(labels)+1` pode empurrar `number_classes` para 201 em algumas execucoes diretas; o loader tambem registra `class_labels=1..num_classes` nesse caso.

### SamplePlan

- Arquivo: `Engine/DataIO/SamplePlanner.py`
- Funcoes: `build_sample_plan_from_args()`, `_class_domain()`, `_allocate_balanced_total()`, `_allocate_proportional()`, `sample_plan_to_legacy_metadata()`
- Linhas aproximadas: ~27-103, ~146-182

Comportamento:

- `balanced_per_class`: cria `{0: n, ..., 199: n}` quando `number_classes=200`.
- `total_rows`: distribui linhas por `range(number_classes)`, podendo alocar `base` e remainder.
- `match_train_distribution`: usa apenas labels presentes em `train_counts`, e contagens podem virar zero quando `total_rows` e pequeno.
- `legacy` com NPY sem plano explicito falha.

Risco:

- Classes com poucos exemplos nao sao ignoradas por `balanced_per_class`, mas podem receber zero em `match_train_distribution` se o total sintetico for pequeno.

## One-hot e condicionamento

### Helpers centrais

- Arquivo: `Engine/DataIO/LabelUtils.py`
- Funcoes: `labels_to_1d_integer()`, `validate_zero_based_labels()`, `one_hot_encode_labels()`, `to_one_hot_batch()`
- Linhas aproximadas: ~14-34, ~74-128

Garantias:

- Labels precisam ser 1D e inteiros.
- Labels negativos falham.
- Se `num_classes=200`, label `200` falha; label `199` e aceito.
- One-hot e criado com shape `(n, num_classes)`.
- Nao ha chamada a `tensorflow.keras.utils.to_categorical`; a criacao e feita por `numpy.zeros((n, num_classes))` e atribuicao indexada.

Observacao:

- `labels_to_1d_integer()` nao contem conversao para boolean. Como o criterio e `astype(int64)` seguido de `allclose`, um vetor booleano poderia passar como labels `0/1`. Isso nao afeta o AppClassNet validado, mas deve ser coberto por teste se o contrato futuro quiser rejeitar `bool` explicitamente.

Pseudocodigo:

```python
integer_labels = labels.astype(int64)
if not allclose(labels, integer_labels):
    raise ValueError
if max_label >= num_classes:
    raise ValueError
encoded = zeros((n, num_classes), float32)
encoded[arange(n), integer_labels] = 1.0
```

### Treinamento dos modelos

- Arquivo: `Engine/Models/GenerativeModels.py`
- Funcoes: `_labels_to_one_hot()`, `_one_hot_batch_generator()`, `_batch_one_hot_dataset()`, `_fit_features_and_one_hot()`, `_fit_autoencode_with_one_hot()`
- Linhas aproximadas: ~97-190

Comportamento:

- Treino normal usa `_labels_to_one_hot(... number_samples_per_class["number_classes"])`.
- Treino batches usa `to_one_hot_batch(... num_classes)` por batch.
- `TensorSpec` do dataset batch tem segunda entrada `(None, num_classes)`.

Conclusao:

- One-hot de 200 posicoes funciona se `number_samples_per_class["number_classes"] == 200`.

### Embeddings de label em difusao

- Arquivo: `Engine/Models/LatentDiffusion/DiffusionModelUnet.py`
- Classe: `DiffusionModelUnet`
- Funcoes: `_label_embedding_MLP()`, `build_model()`
- Linhas aproximadas: ~287-304, ~403-423

Comportamento:

- `description_input` tem shape `(self._number_samples_per_class["number_classes"],)`, ou seja, a entrada de condicionamento por classe tem largura igual ao dominio configurado.
- `_label_embedding_MLP()` transforma esse vetor one-hot em embedding denso antes de blocos de atencao.
- Com `number_classes=200`, a entrada de descricao/label tem 200 posicoes.

Risco:

- Se o dominio cair para o default legado `2`, o embedding de label tambem passa a aceitar apenas 2 posicoes. O problema nasce na configuracao/inferencia de `number_classes`, nao no MLP em si.

## Criacao dos labels sinteticos

### Algoritmos condicionais

Os algoritmos principais iteram `number_samples_per_class["classes"].items()`, criam one-hot para `[label_class] * n`, geram features e salvam no dicionario sob a chave `label_class`.

Exemplos:

- `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py`, funcao `get_samples()`, linhas ~284-312.
- `Engine/Algorithms/VariationalAutoencoder/AlgorithmVariationalAutoencoder.py`, funcao `get_samples()`, linhas ~315-337.
- `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py`, funcao `get_samples()`, linhas ~335-361.
- `Engine/Algorithms/Wasserstein/AlgorithmWassersteinGAN.py`, funcao `get_samples()`, linhas ~205-225.
- `Engine/Algorithms/WassersteinGP/AlgorithmWassersteinGANGP.py`, funcao `get_samples()`, linhas ~340-360.
- `Engine/Algorithms/QuantizedVAE/AlgorithmQuantizedVAE.py`, funcao `get_samples()`, linhas ~262-283.
- `Engine/Algorithms/LatentDiffusion/AlgorithmLatentDiffusion.py`, funcao `get_samples()`, linhas ~427-442.
- `Engine/Algorithms/DenoisingDiffusion/AlgorithmDenoisingDiffusion.py`, funcao `get_samples()`, linhas ~435-448.

Conclusao:

- Labels sinteticos sao definidos pela classe solicitada no plano, nao por `argmax` de uma saida do gerador.
- `argmax` aparece em `RandomNoiseAlgorithm.fit()` para reconstruir classes a partir de one-hot de treino, mas nao e o mecanismo geral de salvamento de labels sinteticos.

### Batches e geracao particionada

- Arquivo: `main.py`
- Classe: `SynDataGen`
- Funcoes: `_synthesize_data_partitioned()`, `_synthesize_data_incremental()`
- Linhas aproximadas: ~1026-1156 e ~1216-1282

Comportamento:

- Para cada `label_class`, cria `batch_plan["classes"] = {label_class: batch_count}`.
- Chama `generator.get_samples(batch_plan)[label_class]`.
- Salva com `writer.write_batch(label_class, batch_index, generated_batch)`.
- Audita `requested_class=label_class` e `saved_label=label_class`.

Risco avaliado:

- Geração em batches nao mistura labels de batches anteriores no writer: cada chamada grava em `batches_by_class[str(class_label)]`.
- O gerador interno ainda pode ignorar condicionamento; nesse caso o label salvo continuara sendo o solicitado, e o problema aparece como baixa fidelidade/downstream, nao como troca de label.

## Salvamento e leitura posterior dos labels sinteticos

### CSV normal

- Arquivo: `Engine/DataIO/CSVLoader.py`
- Classe: `CSVDataProcessor`
- Funcao: `save_csv()`
- Linhas aproximadas: ~391-447

Comportamento:

- Para cada chave `label_class`, escreve uma coluna de label com `_decode_label(label_class)`.
- O arquivo `.space.json` registra `data_space`, `transform_id` e `transform_history`.

### Batches NPY/CSV

- Arquivo: `Engine/DataIO/SyntheticBatchIO.py`
- Classes: `SyntheticBatchWriter`, `SyntheticBatchReader`
- Funcoes: `write_batch()`, `close()`, `items()`, `iter_batches()`
- Linhas aproximadas: ~47-131 e ~134-171

Comportamento:

- Manifesto registra `num_classes`, `batches_by_class`, `data_space`, `transform_id`, `transform_history`.
- Cada batch fica sob `class_{label}` e `batches_by_class[str(label)]`.
- Na leitura, `SyntheticBatchReader.items()` retorna `(int(class_label), x_batch)`.
- `BatchClassifiers.iter_synthetic_labeled_batches()` cria `y_batch = full(n, int(class_label))`.

Conclusao:

- Labels de batches sao persistidos implicitamente no manifesto, nao como array separado.

## Respostas aos pontos verificados

1. `num_classes=200` e respeitado quando AppClassNet passa `--num_classes 200` e/ou campanhas setam `*_number_classes=200`. Evidencias: `NpyXYLoader._get_class_metadata()` linhas ~248-274; `run_appclassnet_top200.py` linhas ~151, ~171, ~213, ~234, ~260.
2. Sim, ainda existem defaults `number_classes=2` em argumentos legados de modelos. Evidencias: `Engine/Arguments/ArgumentsAutoencoder.py`, constante `DEFAULT_AUTOENCODER_NUMBER_CLASSES`, linha 49; `ArgumentsVariationalAutoencoder.py`, linha 41; `ArgumentsQuantizedVAE.py`, linha 42; `ArgumentsWassersteinGAN.py`, linha 40; `ArgumentsWassersteinGANGP.py`, linha 40.
3. Codigo binario ainda existe em caminhos legados e defaults, mas o one-hot central nao limita a `[0,1]` quando `number_classes=200`.
4. Nao foi encontrada conversao de labels para boolean no fluxo central de labels. `LabelUtils.labels_to_1d_integer()` converte para `int64` e valida integralidade, mas nao rejeita explicitamente dtype `bool`; evidencia em `Engine/DataIO/LabelUtils.py`, linhas ~14-34.
5. Labels nao sao arredondados nos helpers; `labels_to_1d_integer()` exige `allclose` contra inteiro. Arredondamento `rint` aparece em features geradas quando `data_type != continuous`; exemplos em `AutoencoderAlgorithm.py` linhas ~306-310.
6. Sim, CSV discreto armazena labels codificados como `float32`, e depois helpers convertem para `int64` se forem inteiros. Tambem ha `labels.astype(int)` em `main.py`, `SynDataGen._prepare_labels_for_conditional_generation()`, linhas ~524-527.
7. NPY cria mapping uma vez com labels coletados de todos os splits; CSV cria mapping uma vez sobre o CSV carregado.
8. A ordem interna de classes e zero-based/sorted no CSV e por dominio numerico no NPY. Batches leem labels pelo manifesto; nao ha ordenacao semantica externa garantida alem da chave numerica.
9. One-hot com 200 posicoes e criado quando `number_classes=200`; evidencia em `Engine/DataIO/LabelUtils.py`, `to_one_hot_batch()`, linhas ~106-128, e `Engine/Models/GenerativeModels.py`, `_labels_to_one_hot()`/datasets batch, linhas ~97-190. Nao ha `to_categorical` no fluxo auditado.
10. `argmax` nao e usado no fluxo principal para definir labels sinteticos; em `RandomNoiseAlgorithm.fit()` argmax converte one-hot de treino em classe interna, linhas ~123-136.
11. Labels sinteticos sao definidos pela classe solicitada/chave, nao pelo output do gerador.
12. Batches nao misturam labels no writer/reader quando cada batch e de uma unica classe; cada batch e registrado por `class_label`. Um batch fisico contendo multiplas classes seria rotulado inteiro por essa chave, portanto esse caso nao e suportado pelo contrato atual.
13. Classe 199 nao e perdida se `num_classes/number_classes=200`; label 199 indexa a ultima posicao do one-hot.
14. Classes com poucos exemplos nao sao ignoradas por `balanced_per_class`; podem receber zero em `match_train_distribution` se `total_rows` for pequeno; particionamento falha se faltarem classes reais no treino da unidade.
15. CSV registra labels na coluna final; batches registram labels no manifesto `batches_by_class` e reconstroem `y_batch` na leitura.

## Testes propostos

### `test_labels_zero_based_0_199_npy_loader`

Objetivo: `NpyXYLoader` com train/valid/test contendo labels `0..199` deve produzir `schema.num_classes == 200`, mapping identidade, `class_labels is None`, labels inteiros e sem remap.

Falha se classe 199 for perdida ou se `num_classes` virar 199/201.

### `test_labels_one_based_1_200_require_explicit_remap`

Objetivo: com labels `1..200`, sem remap, o loader deve sinalizar/warnar 1-based e preservar `class_labels=1..200`; com `remap_labels_to_zero_based=True`, deve converter para `0..199` usando mapping unico.

Falha se mapping for criado por split separado ou se label 200 passar para one-hot de 200 sem remap.

### `test_random_order_classes_preserve_numeric_domain`

Objetivo: labels em ordem aleatoria devem inferir o mesmo dominio `0..199`; `SamplePlan balanced_per_class` deve gerar classes ordenadas por `range(200)` e nao pela ordem de aparicao.

### `test_missing_class_detected`

Objetivo: remover uma classe do conjunto de labels e validar que auditoria AppClassNet ou plano/diagnostico detecta classe ausente.

Para sintéticos, `SyntheticLabelGenerationAudit(number_classes=200)` deve falhar se uma classe de `0..199` nao foi gerada.

### `test_duplicate_class_counts_do_not_duplicate_domain`

Objetivo: muitos exemplos duplicados de uma classe nao devem duplicar `class_labels` nem aumentar `number_classes`.

Validar `numpy.unique`/metadata e contagem por classe.

### `test_one_hot_200_positions_accepts_199_rejects_200`

Objetivo: `to_one_hot_batch([0, 199], 200)` retorna shape `(2, 200)` com posicoes corretas; `to_one_hot_batch([200], 200)` falha.

### `test_single_class_batch_labels`

Objetivo: `iter_synthetic_labeled_batches()` em um batch de classe `199` deve retornar `y_batch` todo `199`, shape alinhado a X e sem perder a classe.

### `test_multi_class_batches_labels`

Objetivo: manifesto com batches de classes `0`, `57` e `199` deve ser lido como tres pares `(class_label, x_batch)` e combinado sem trocar labels.

### `test_mixed_class_single_batch_is_not_supported_by_manifest_contract`

Objetivo: documentar o contrato atual: um unico batch fisico nao pode conter multiplas classes se a unica fonte de label e `batches_by_class[class_label]`.

Falha esperada/diagnostico: se um batch misto for salvo sob `class_57`, `iter_synthetic_labeled_batches()` reconstruira `y_batch` todo `57`. O teste deve impedir que esse formato seja usado silenciosamente para sintéticos multiclasse.

### `test_synthetic_save_read_roundtrip_labels`

Objetivo: gerar batches sintéticos com labels `0..199`, salvar via `SyntheticBatchWriter`, ler via `SyntheticBatchReader` e reconstruir labels via `iter_synthetic_labeled_batches()`; as contagens devem ser identicas ao plano original.

### `test_csv_synthetic_label_decode_roundtrip`

Objetivo: CSV com labels originais nao triviais deve codificar zero-based, gerar sintéticos por classes internas, salvar com `_decode_label()` e preservar labels originais esperados.

### `test_batch_no_label_leak_between_batches`

Objetivo: dois batches consecutivos de classes diferentes devem produzir y separado por batch; a classe do segundo batch nao pode herdar classe anterior.

### `test_boolean_labels_rejected_or_documented`

Objetivo: decidir e fixar o contrato para labels booleanos. O comportamento atual de `labels_to_1d_integer()` pode aceitar `False/True` como `0/1`; para AppClassNet top-200, o teste recomendado deve rejeitar dtype `bool` ou documentar explicitamente essa compatibilidade legada.

### `test_match_train_distribution_can_zero_small_classes`

Objetivo: documentar comportamento atual: com `total_rows < number_classes`, `match_train_distribution` produz zeros para algumas classes. O teste deve registrar esse comportamento e orientar uso de `balanced_per_class` para AppClassNet.

## Decisao recomendada

Para AppClassNet top-200, a regra operacional deve ser:

- sempre passar `--num_classes 200`;
- sempre garantir `*_number_classes=200` nos modelos condicionais quando executar fora do runner;
- preferir `--sample_plan balanced_per_class --samples_per_class N` para avaliacao 200 classes;
- validar `to_one_hot_batch(..., 200)` com classe 199 em teste unitario;
- tratar labels sinteticos como labels declarados pela classe solicitada e medir fidelidade por diagnosticos TR-TS/TS-TR, nao por argmax do gerador.
