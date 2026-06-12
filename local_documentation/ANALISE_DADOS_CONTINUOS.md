# Analise de suporte a dados continuos no MalDataGen

## Escopo

Esta analise parte de `Engine/Metrics/Metrics.py` e rastreia o fluxo geral do projeto para identificar pontos que impedem ou degradam o uso de dados continuos. O projeto foi estruturado originalmente para classificacao e geracao sintetica condicionada por classes, com forte vies para atributos binarios ou discretizados.

A conclusao depende de duas interpretacoes de "dados continuos":

- Features continuas com rotulo de classe discreto: o carregamento aceita esse caso, mas varios geradores destroem a continuidade no pos-processamento e as metricas de fidelidade misturam metricas inadequadas.
- Alvo/rotulo continuo: o pipeline atual nao suporta diretamente; ele usa `StratifiedKFold`, `to_categorical`, organizacao por classe e metricas de classificacao.

## Resumo executivo

O bloqueio mais critico para features continuas nao esta no loader. `CSVLoader` e `XLSLoader` convertem features para `float32`, normalizam com `MinMaxScaler` e aplicam `inverse_transform` ao salvar. O bloqueio principal esta no pos-processamento dos geradores: varios algoritmos aplicam `numpy.rint()` nos dados sinteticos, convertendo valores continuos em inteiros. Isso torna a saida efetivamente binaria/discreta quando combinada com ativacao `sigmoid`.

O segundo bloqueio critico esta em `Metrics.py` e nas avaliacoes `TS-TR`, `TR-TS` e `TR-TR`: a avaliacao de classificadores chama apenas `get_binary_metrics()`, e o dicionario de metricas inclui TP, TN, FP, FN, Recall, Specificity e FPR, que assumem rotulos binarios. Para rotulos multiclasse ou continuos, parte dessas metricas quebra, retorna resultado semanticamente errado ou deixa de representar o problema.

O terceiro bloqueio e arquitetural: a geracao e condicionada por classes discretas. `main.py` converte labels com `astype(int)`, calcula `numpy.unique` e cada gerador usa `to_categorical`. Isso impede alvo continuo e limita cenarios de regressao ou geracao condicional por variaveis continuas.

## Pontos criticos por prioridade

### P0 - Pos-processamento arredonda dados gerados

Arquivos afetados:

- `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py:358`
- `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py:308`
- `Engine/Algorithms/VariationalAutoencoder/AlgorithmVariationalAutoencoder.py:333`
- `Engine/Algorithms/LatentDiffusion/AlgorithmVAELatentDiffusion.py:302`
- `Engine/Algorithms/LatentDiffusion/AlgorithmLatentDiffusion.py:436`
- `Engine/Algorithms/DenoisingDiffusion/AlgorithmDenoisingDiffusion.py:444`
- `Engine/Algorithms/QuantizedVAE/AlgorithmQuantizedVAE.py:278`
- `Engine/Algorithms/Wasserstein/AlgorithmWassersteinGAN.py:220`
- `Engine/Algorithms/WassersteinGP/AlgorithmWassersteinGANGP.py:355`

Problema:

- `numpy.rint()` arredonda a saida gerada.
- Se a feature real era `0.37`, `12.8`, `0.004`, ou qualquer valor continuo normalizado, a amostra sintetica perde precisao.
- Com `sigmoid`, o modelo emite valores em `[0,1]`; apos `rint`, a saida vira majoritariamente `0` ou `1`.

Impacto no sistema:

- Inviabiliza geracao fiel de features continuas.
- Distorce distribuicoes, medias, variancias, correlacoes e outliers.
- Prejudica as metricas de distancia, pois elas passam a comparar real continuo contra sintetico discretizado.
- O arquivo salvo por `CSVLoader.save_csv()` pode aplicar `inverse_transform`, mas isso nao recupera a continuidade perdida pelo arredondamento.

Mudanca recomendada:

- Remover `numpy.rint()` quando o tipo de dado for continuo.
- Criar uma configuracao explicita de tipo de atributo, por exemplo `--data_feature_mode binary|continuous|mixed`.
- Para `mixed`, aplicar arredondamento apenas em colunas declaradas como binarias/categoricas, nunca globalmente.

### P0 - Metricas de classificacao sao binarizadas no centro do sistema

Arquivo principal:

- `Engine/Metrics/Metrics.py`

Pontos relevantes:

- `Engine/Metrics/Metrics.py:129` inicializa `_dictionary_binary_metrics`.
- `Engine/Metrics/Metrics.py:175` define `list_classifier_metrics` a partir de metricas binarias.
- `Engine/Metrics/Metrics.py:373` define `get_binary_metrics()`.
- `Engine/Metrics/Metrics.py:386` percorre apenas `_dictionary_binary_metrics`.

Arquivos que chamam essa avaliacao:

- `Engine/Evaluation/TsTr.py:95`
- `Engine/Evaluation/TrTs.py:95`
- `Engine/Evaluation/TrTr.py:93`

Problema:

- O sistema avalia classificadores apenas com metricas binarias.
- Varias classes em `Engine/Metrics/Binary` validam ou assumem valores `0` e `1`, especialmente TP/TN/FP/FN e taxas derivadas.
- Para multiclasse, algumas metricas do sklearn podem funcionar apenas se configuradas com `average`, mas o codigo nao define isso de forma consistente.
- Para alvo continuo, as metricas de classificacao nao se aplicam.

Impacto no sistema:

- Resultados podem quebrar em runtime ou serem matematicamente invalidos.
- `Results.json` fica preso a estrutura binaria, dificultando comparacao com regressao, multiclasse ou avaliacao de fidelidade continua.
- As estrategias `TS-TR`, `TR-TS` e `TR-TR` ficam semanticamente ligadas a classificacao.

Mudanca recomendada:

- Separar metricas em familias: `binary_classification`, `multiclass_classification`, `regression` e `synthetic_data_fidelity`.
- Renomear `get_binary_metrics()` para algo seletivo, como `get_classification_metrics()`, e selecionar metricas com base no tipo do problema.
- Para classificacao multiclasse, usar `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted` e matriz de confusao multiclasse.
- Para alvo continuo, usar `MAE`, `MSE`, `RMSE`, `R2`, erro relativo e validacao por regressao, nao TP/TN/FP/FN.

### P0 - Pipeline assume labels discretos/classes

Arquivos afetados:

- `main.py:534`
- `Engine/Evaluation/CrossValidation.py:103`
- Geradores com `to_categorical`, por exemplo `Engine/Algorithms/Adversarial/AdversarialAlgorithm.py:339`, `Engine/Algorithms/Autoencoder/AutoencoderAlgorithm.py:288`, `Engine/Algorithms/Wasserstein/AlgorithmWassersteinGAN.py:207`.

Problema:

- `main.py` converte labels com `labels.astype(int)` antes de gerar amostras por classe.
- `CrossValidation.py` usa `StratifiedKFold`, que exige labels discretos.
- Os geradores condicionais usam `to_categorical`, exigindo classes inteiras.

Impacto no sistema:

- O projeto nao suporta alvo continuo/regressao sem refatoracao.
- Labels numericos continuos seriam truncados para inteiros, criando classes artificiais.
- O numero de classes pode explodir se cada valor continuo for tratado como classe.

Mudanca recomendada:

- Adicionar conceito explicito de `target_type`: `classification` ou `regression`.
- Para regressao, trocar `StratifiedKFold` por `KFold` ou criar bins apenas para split estratificado, sem alterar o alvo real.
- Para geracao condicional por variavel continua, substituir `to_categorical` por entrada condicional numerica normalizada.

### P1 - Defaults de modelo favorecem dominio binario

Arquivos afetados:

- `Engine/Arguments/ArgumentsAdversarial.py:38`, `:48`, `:49`
- `Engine/Arguments/ArgumentsAutoencoder.py:49`, `:53`
- `Engine/Arguments/ArgumentsVariationalAutoencoder.py:41`, `:44`, `:46`, `:49`
- `Engine/Arguments/ArgumentsQuantizedVAE.py:42`, `:45`, `:47`, `:50`
- `Engine/Arguments/ArgumentsWassersteinGAN.py:40`, `:44`, `:46`
- `Engine/Arguments/ArgumentsWassersteinGANGP.py:40`, `:44`, `:46`
- `Engine/Arguments/ArgumentsLatentDiffusion.py:57`
- `Engine/Arguments/ArgumentsDenoisingDiffusion.py:57`

Problema:

- Muitos defaults usam `number_classes = 2`, `binary_crossentropy` e `sigmoid`.
- `sigmoid` limita a saida a `[0,1]`, adequado somente se todas as features forem normalizadas e permanecerem nesse dominio.
- `binary_crossentropy` e adequada para Bernoulli/binario; para features continuas ela so faz sentido em casos muito especificos.

Impacto no sistema:

- Induz configuracoes incorretas quando o usuario usa dados continuos sem alterar argumentos.
- Pode causar saturacao de saida, perda de variancia e reconstrucoes ruins.
- Para features continuas normalizadas, `sigmoid + MSE` pode ser aceitavel; `sigmoid + BCE + rint` nao.

Mudanca recomendada:

- Para features continuas normalizadas em `[0,1]`, usar `sigmoid` ou `linear` com `MSE`/`MAE`, sem arredondamento.
- Para features continuas padronizadas, usar `linear` na camada de saida e `MSE`, `MAE` ou perda robusta.
- Ajustar defaults conforme `data_feature_mode`.

### P1 - Metricas de distancia misturam continuas e binarias

Arquivo principal:

- `Engine/Metrics/Metrics.py:145-152`

Metricas problematicas para continuo:

- `Engine/Metrics/Distance/HammingDistance.py`
- `Engine/Metrics/Distance/JaccardDistance.py`

Metricas mais compativeis com continuo:

- `Engine/Metrics/Distance/EuclideanDistance.py`
- `Engine/Metrics/Distance/ManhattanDistance.py`
- `Engine/Metrics/Distance/HellingerDistance.py`, com cuidado porque Hellinger exige distribuicoes/probabilidades nao negativas e interpretaveis.

Problema:

- Hamming e Jaccard sao metricas para vetores binarios/conjuntos. Em dados continuos, medem desigualdade exata ou operacoes de conjunto sem significado estatistico adequado.
- O comentario de validacao binaria em Hamming/Jaccard aparece desativado, o que reduz erros explicitos, mas nao torna a metrica correta.

Impacto no sistema:

- `DistanceMetrics` pode produzir numeros com aparencia valida, mas interpretacao incorreta.
- A comparacao real-sintetico pode favorecer modelos discretizados em vez de modelos fieis a distribuicao continua.

Mudanca recomendada:

- Criar selecao de metricas de distancia por tipo de feature.
- Para continuo, considerar Wasserstein distance por coluna, KS statistic, MMD, energia, correlacao, diferenca de momentos, cobertura de faixa e metricas por coluna.
- Para misto, calcular metricas por subconjunto de colunas: binarias, categoricas e continuas.

### P1 - Loader aceita continuo, mas falta schema de dados

Arquivos afetados:

- `Engine/DataIO/CSVLoader.py:157`, `:356`, `:400`
- `Engine/DataIO/XLSLoader.py:125`, `:264`, `:248`
- `Engine/Arguments/ArgumentsDataLoader.py:38`

Situacao atual:

- O loader usa `MinMaxScaler(feature_range=(0, 1))` e preserva `float32`.
- O salvamento usa `inverse_transform`, o que e bom para features continuas.
- O dataset default aponta para `Datasets/binaries/kronodroid_emulador-balanced.csv`, reforcando o vies binario.

Problema:

- Nao existe schema de colunas para diferenciar binarias, categoricas, inteiras e continuas.
- O scaler e globalmente aplicado a todas as features, sem metadados suficientes para pos-processamento correto.

Impacto no sistema:

- O sistema nao sabe quais colunas podem ser arredondadas, clipadas, codificadas ou invertidas.
- Dados mistos tendem a ser tratados de forma uniforme, o que e inadequado para geracao tabular real.

Mudanca recomendada:

- Adicionar schema de dados via argumento ou arquivo, por exemplo JSON/YAML com listas de colunas `continuous`, `binary`, `categorical`, `integer`.
- Persistir transformadores por tipo de coluna.
- Aplicar pos-processamento seletivo por tipo.

## Arquivos que precisam ser modificados

| Prioridade | Arquivo | Mudanca necessaria | Impacto geral |
| --- | --- | --- | --- |
| P0 | `Engine/Algorithms/*/Algorithm*.py` | Remover ou condicionar `numpy.rint()` | Permite que os geradores emitam valores continuos reais. |
| P0 | `Engine/Metrics/Metrics.py` | Separar metricas binarias, multiclasse, regressao e fidelidade continua | Evita metricas invalidas e muda a estrutura de `Results.json`. |
| P0 | `Engine/Evaluation/TsTr.py`, `TrTs.py`, `TrTr.py` | Trocar chamada fixa de `get_binary_metrics()` por avaliacao baseada no tipo de problema | Torna as estrategias compativeis com multiclasse/regressao. |
| P0 | `main.py` | Evitar `labels.astype(int)` para alvo continuo e separar fluxo classification/regression | Impede truncamento de targets continuos. |
| P0 | `Engine/Evaluation/CrossValidation.py` | Usar `StratifiedKFold` apenas para classificacao; usar `KFold` para regressao | Permite validacao cruzada com alvo continuo. |
| P0 | Geradores que usam `to_categorical` | Permitir condicionamento numerico continuo ou desativar condicionamento por classe em regressao | Remove dependencia estrutural de classes inteiras. |
| P1 | `Engine/Arguments/Arguments*.py` | Ajustar defaults de `sigmoid`, `binary_crossentropy`, `number_classes=2` conforme tipo de dado | Reduz configuracao incorreta para dados continuos. |
| P1 | `Engine/Metrics/Distance/HammingDistance.py`, `JaccardDistance.py` | Excluir para continuo ou executar somente em colunas binarias | Evita metricas sem significado para features continuas. |
| P1 | `Engine/DataIO/CSVLoader.py`, `XLSLoader.py` | Adicionar schema de tipos de coluna e transformadores por tipo | Permite datasets tabulares mistos sem pos-processamento destrutivo. |
| P2 | `Tools/Plot*` | Adaptar graficos para metricas continuas/regressao | Mantem visualizacao coerente apos mudar metricas. |

## Recomendacao de arquitetura minima

Para suportar features continuas com rotulo discreto, o menor conjunto de mudancas e:

1. Adicionar argumento `--data_feature_mode binary|continuous|mixed`.
2. Remover `numpy.rint()` quando `data_feature_mode=continuous`.
3. Manter `MinMaxScaler` no loader, mas preservar valores float ate o `save_csv()`.
4. Em `Metrics.py`, desativar Hamming/Jaccard para continuo e adicionar metricas de distribuicao continua.
5. Manter `StratifiedKFold` e `to_categorical` apenas se o rotulo continuar sendo classe discreta.

Para suportar alvo continuo/regressao, a mudanca e maior:

1. Adicionar `--target_type classification|regression`.
2. Trocar `StratifiedKFold` por `KFold` no modo regressao.
3. Remover `labels.astype(int)` e `to_categorical` do fluxo de regressao.
4. Alterar geradores condicionais para aceitar condicao continua ou operar sem condicao de classe.
5. Substituir metricas binarias por metricas de regressao e fidelidade tabular.

## Observacoes sobre modelos especificos

- `copy`: preserva continuidade porque copia amostras reais; serve como baseline, mas nao gera novas distribuicoes.
- `smote`: naturalmente trabalha com interpolacao continua; e um dos mais compativeis com features continuas, desde que labels sejam classes discretas.
- `random`: aplica ruido salt-and-pepper com `1 - sample`, adequado para binario normalizado, mas inadequado para continuo geral.
- `autoencoder`: usa MSE no `train_step`, mas perde continuidade no `get_samples()` por causa de `numpy.rint()`.
- `variational` e `latent_diffusion` com VAE: usam `binary_crossentropy` na reconstrucao e arredondam a saida; precisam de perda continua e sem arredondamento.
- `adversarial`, `wasserstein`, `wasserstein_gp`: podem gerar continuo se a saida/pos-processamento forem ajustados, mas hoje usam defaults e pos-processamento voltados a binario.
- `copula`, `ctgan`, `tvae`: por serem SDV, tendem a ser mais adequados para tabular misto/continuo, mas o encapsulamento ainda devolve dados por classe e passa pela mesma avaliacao binaria do projeto.

## Conclusao

O projeto nao esta totalmente impedido de ler dados continuos; ele ja faz isso no carregamento. O impedimento real esta em tres camadas posteriores: pos-processamento dos geradores com `numpy.rint()`, avaliacao centralizada em metricas binarias e dependencia arquitetural de labels discretos/classes. Para features continuas com labels discretos, a correcao e moderada. Para alvo continuo/regressao, a refatoracao deve separar explicitamente os fluxos de classificacao e regressao.
