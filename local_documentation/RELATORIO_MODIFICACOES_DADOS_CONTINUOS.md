# Relatorio de modificacoes para suporte a binary, multiclass e continuous

## Objetivo

A partir da analise em `ANALISE_DADOS_CONTINUOS.md`, o projeto foi ajustado para permitir que o usuario defina o modo de dados no inicio do treinamento e para reduzir o acoplamento original com dados binarios. O novo argumento operacional e:

```bash
--data_type {binary,multiclass,continuous}
```

O modo controla carregamento, codificacao de labels, metricas, metricas de distancia e pos-processamento das amostras geradas.

## Resumo das mudancas

| Area | Arquivos | Mudanca | Impacto |
| --- | --- | --- | --- |
| CLI/configuracao | `Engine/Arguments/ArgumentsDataLoader.py` | Adicionado `--data_type binary|multiclass|continuous` | O usuario escolhe o modo ao iniciar o treinamento. |
| Loader CSV | `Engine/DataIO/CSVLoader.py` | Removido `MinMaxScaler`, `fit_transform` e `inverse_transform`; features sao preservadas em escala original | Dados continuos deixam de ser comprimidos para `[0, 1]` pelo loader. |
| Labels discretos | `Engine/DataIO/CSVLoader.py` | Labels discretos sao codificados para classes zero-based e decodificados ao salvar | Evita falhas em `to_categorical` quando classes originais sao `1,2,...`; saida salva recupera os labels originais. |
| Cross-validation | `Engine/Evaluation/CrossValidation.py` | Metadados de classes agora sao inferidos do CSV; fallback para `KFold` quando labels nao sao discretos | Remove dependencia do default `number_samples_per_class` e evita estratificacao invalida em alvo continuo. |
| Metricas | `Engine/Metrics/Metrics.py` | `binary` usa metricas antigas; `multiclass` e `continuous` usam `Accuracy`, `BalancedAccuracy`, `PrecisionMacro`, `RecallMacro`, `F1Macro`, `F1Weighted` | Evita usar TP/TN/FP/FN binarios em cenarios multiclasse. |
| Distancias | `Engine/Metrics/Metrics.py` | `continuous` remove Hamming/Jaccard do conjunto de metricas de distancia | Evita metricas sem significado para features continuas. |
| Avaliacao | `Engine/Evaluation/TsTr.py`, `TrTs.py`, `TrTr.py` | Chamadas fixas de `get_binary_metrics()` substituidas por `get_task_metrics()` | A avaliacao passa a respeitar o modo escolhido. |
| Geracao | `main.py` | Preparacao centralizada de labels para geradores condicionais; `data_type` propagado para `get_samples()` | Remove casts espalhados e permite pos-processamento por modo. |
| Pos-processamento | `Engine/Algorithms/*/Algorithm*.py` | `numpy.rint()` agora so roda quando `data_type != continuous` | Amostras sinteticas continuas permanecem float. |

## Detalhamento por arquivo

### `Engine/Arguments/ArgumentsDataLoader.py`

Foi adicionado o argumento:

```bash
--data_type {binary,multiclass,continuous}
```

Por que:

- Antes o projeto assumia implicitamente dados binarios.
- Nao havia uma chave unica para metricas, split e pos-processamento saberem se deveriam operar como binario, multiclasse ou continuo.

Impacto:

- `binary` preserva o comportamento historico.
- `multiclass` ativa metricas macro/weighted e preserva arredondamento de amostras discretas.
- `continuous` preserva valores float gerados e remove metricas de distancia binarias.

### `Engine/DataIO/CSVLoader.py`

Mudancas:

- Removido o uso de `MinMaxScaler`.
- Removida a chamada automatica de normalizacao no fim de `load_csv()`.
- Removida a reversao de normalizacao no `save_csv()`.
- Adicionada codificacao de labels discretos para classes `0..n-1`.
- Adicionada decodificacao de labels ao salvar dados gerados.

Por que:

- O Markdown apontou que o loader aceitava `float32`, mas normalizava tudo para `[0, 1]`, o que escondia escala real das features.
- Os modelos condicionais usam `to_categorical`; labels como `1` e `2` com `number_classes=2` podem quebrar. A codificacao zero-based evita isso.

Impacto:

- Features continuas sao treinadas/salvas na escala original do CSV.
- O usuario passa a ser responsavel por normalizacao externa, se o modelo escolhido exigir.
- Labels salvos voltam ao valor original quando havia mapeamento discreto.

### `Engine/Evaluation/CrossValidation.py`

Mudancas:

- Adicionado `KFold` como fallback quando os labels nao sao discretos.
- `number_samples_per_class` passa a ser inferido do dataset carregado.
- O `data_type` e propagado no dicionario de metadados usado pelos geradores.

Por que:

- O default anterior `number_samples_per_class` podia nao representar o CSV real.
- `StratifiedKFold` exige classes discretas.

Impacto:

- Modelos condicionais passam a ser construidos com o numero real de classes do CSV.
- Datasets com labels nao discretos nao tentam usar estratificacao invalida.

### `Engine/Metrics/Metrics.py`

Mudancas:

- Adicionada selecao de metricas por `data_type`.
- `binary` mantem as metricas antigas: `Accuracy`, `Precision`, `Recall`, `F1Score`, `Specificity`, `TP`, `TN`, `FP`, `FN`, etc.
- `multiclass` e `continuous` usam metricas seguras para classificacao multiclasse: `Accuracy`, `BalancedAccuracy`, `PrecisionMacro`, `RecallMacro`, `F1Macro`, `F1Weighted`.
- `continuous` remove `HammingDistance` e `JaccardDistance` das distancias.
- Valores de metricas nao numericos sao convertidos para `0.0` para nao quebrar sumarizacao por media/desvio.

Por que:

- TP/TN/FP/FN e taxas derivadas sao binarios.
- Hamming/Jaccard nao sao adequados para features continuas.

Impacto:

- `Results.json` passa a refletir o modo de dados escolhido.
- Multiclasse deixa de depender de metricas binarias semanticamente incorretas.
- Continuo nao e penalizado por metricas de distancia binarias.

### `Engine/Evaluation/TsTr.py`, `TrTs.py`, `TrTr.py`

Mudancas:

- Substituida a chamada fixa de `get_binary_metrics()` por `get_task_metrics()`.
- Avaliacoes preditivas sao ignoradas quando os labels carregados nao sao discretos.

Por que:

- A avaliacao antiga era sempre binaria.
- Labels realmente continuos nao devem ser avaliados por classificadores sem um fluxo de regressao dedicado.

Impacto:

- `binary`, `multiclass` e `continuous` podem compartilhar a mesma interface de avaliacao.
- Quando o alvo e continuo, o projeto evita produzir metricas de classificacao invalidas.

### `main.py`

Mudancas:

- Criada `_prepare_labels_for_conditional_generation()`.
- Labels discretos sao convertidos para inteiros apenas em um ponto central.
- Labels nao discretos usam pseudo-classe unica para evitar falha estrutural em geradores condicionais.
- `data_type` e propagado para `number_samples_per_class` durante a geracao.

Por que:

- Antes havia `labels.astype(int)` diretamente em `synthesize_data()`, o que truncava labels continuos.
- Geradores condicionais ainda dependem de classes discretas.

Impacto:

- Reduz o risco de truncamento acidental.
- Mantem compatibilidade com os modelos condicionais existentes.
- Deixa claro o limite atual: alvo continuo completo ainda exige um fluxo regressivo dedicado.

### Geradores em `Engine/Algorithms/*`

Arquivos alterados:

- `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py`
- `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py`
- `Engine/Algorithms/VariationalAutoencoder/AlgorithmVariationalAutoencoder.py`
- `Engine/Algorithms/LatentDiffusion/AlgorithmLatentDiffusion.py`
- `Engine/Algorithms/LatentDiffusion/AlgorithmVAELatentDiffusion.py`
- `Engine/Algorithms/DenoisingDiffusion/AlgorithmDenoisingDiffusion.py`
- `Engine/Algorithms/QuantizedVAE/AlgorithmQuantizedVAE.py`
- `Engine/Algorithms/Wasserstein/AlgorithmWassersteinGAN.py`
- `Engine/Algorithms/WassersteinGP/AlgorithmWassersteinGANGP.py`

Mudanca:

```python
if number_samples_per_class.get("data_type") != "continuous":
    generated_samples = numpy.rint(generated_samples)
```

Por que:

- `numpy.rint()` era o principal ponto destrutivo para features continuas.

Impacto:

- `binary` e `multiclass` preservam o comportamento discreto anterior.
- `continuous` mantem valores sinteticos em ponto flutuante.

## Como usar

Modo binario, comportamento compativel com o historico:

```bash
python main.py --data_type binary
```

Modo multiclasse:

```bash
python main.py --data_type multiclass
```

Modo continuo para preservar features float e evitar Hamming/Jaccard:

```bash
python main.py --data_type continuous
```

No ambiente local deste projeto, o interpretador funcional foi:

```bash
./.venv/bin/python main.py --help
```

O Python global falhou por ausencia de `numpy`, entao validacoes runtime devem usar `.venv`.

## Validacoes executadas

Sintaxe dos arquivos alterados:

```bash
python3 -m py_compile Engine/Arguments/ArgumentsDataLoader.py Engine/DataIO/CSVLoader.py Engine/Evaluation/CrossValidation.py Engine/Evaluation/TsTr.py Engine/Evaluation/TrTs.py Engine/Evaluation/TrTr.py Engine/Metrics/Metrics.py main.py Engine/Algorithms/Adversarial/AdversarialAlgorithm.py Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py Engine/Algorithms/VariationalAutoencoder/AlgorithmVariationalAutoencoder.py Engine/Algorithms/Wasserstein/AlgorithmWassersteinGAN.py Engine/Algorithms/WassersteinGP/AlgorithmWassersteinGANGP.py Engine/Algorithms/LatentDiffusion/AlgorithmLatentDiffusion.py Engine/Algorithms/LatentDiffusion/AlgorithmVAELatentDiffusion.py Engine/Algorithms/DenoisingDiffusion/AlgorithmDenoisingDiffusion.py Engine/Algorithms/QuantizedVAE/AlgorithmQuantizedVAE.py
```

Parser do novo argumento:

```bash
./.venv/bin/python main.py --help | grep -A 3 -- '--data_type'
```

Teste isolado de `Metrics` para `binary`, `multiclass` e `continuous` usando `.venv`.

## Limites que permanecem

- O `XLSLoader` ainda normaliza com `MinMaxScaler`; a solicitacao foi restrita ao `CSVLoader`.
- Os modelos neurais ainda podem ter defaults como `sigmoid` ou `binary_crossentropy`; em dados continuos de escala ampla, o usuario deve ajustar hiperparametros ou criar defaults especificos por `data_type` em uma etapa futura.
- Alvo continuo completo ainda nao tem pipeline regressivo end-to-end. O ajuste atual evita truncamento e metricas invalidas, mas os geradores principais continuam arquiteturalmente condicionados por classes.
- `continuous` deve ser entendido principalmente como preservacao de features continuas no CSV e na geracao. Para regressao real, ainda e recomendada uma refatoracao dedicada de classificadores para regressores.
