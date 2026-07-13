# Analise da Estrutura do Repositorio MalDataGen

Data da analise: 2026-07-03

## Visao Geral

Este repositorio contem o MalDataGen, um framework Python para geracao e avaliacao de dados sinteticos tabulares aplicados a deteccao de malware. O projeto combina modelos generativos, classificadores supervisionados, metricas, validacao cruzada, visualizacoes e scripts de campanha para reproducao experimental.

O fluxo principal e:

1. `run_campaign_sbseg.py` define campanhas, datasets, combinacoes de hiperparametros e chamadas de execucao.
2. `main.py` executa o pipeline de carregamento de dados, treino/generacao, classificacao e avaliacao.
3. `plots.py` consome resultados e monitores de treino para gerar graficos e relatorios visuais.
4. `Engine/` concentra a implementacao dos modelos, algoritmos, metricas, avaliacao e infraestrutura.
5. `outputs/` e `SBSEG25_Tests/` armazenam artefatos experimentais gerados.

## Raiz do Projeto

Arquivos e diretorios principais encontrados na raiz:

```text
.
|-- Datasets/
|-- Docs/
|-- Engine/
|-- Layout/
|-- SBSEG25_Tests/
|-- Scripts/
|-- Test/
|-- Tools/
|-- logs/
|-- outputs/
|-- Dockerfile
|-- FUNDING.yml
|-- LICENSE
|-- Pipfile
|-- README.md
|-- main.py
|-- pip_env_install.sh
|-- plots.py
|-- plots_svm.py
|-- pyproject.toml
|-- requirements.txt
|-- results.sh
|-- run_campaign_sbseg.py
|-- run_demo_docker.sh
|-- run_experiments_docker.sh
|-- run_experiments_docker_SDV.sh
|-- uv.lock
```

Tambem existem diretorios locais ou gerados que normalmente nao fazem parte da estrutura-fonte ideal:

```text
.git/
.idea/
.venv/
__pycache__/
Engine/**/__pycache__/
Tools/__pycache__/
```

## Arquivos de Entrada e Orquestracao

### `run_campaign_sbseg.py`

Script de campanha para execucao de experimentos SBSEG. Ele:

- define datasets usados nas campanhas, atualmente incluindo `Datasets/SBSeg_2025/reduced_balanced_androcrawl.csv`;
- define campanhas como `adversarial`, `autoencoder`, `variational`, `wasserstein`, `wasserstein_gp`, `latent_diffusion`, `denoising_diffusion`, `copula`, `tvae`, `ctgan` e variantes demo;
- monta comandos para `python3 main.py`;
- monta comandos para `python3 plots.py`;
- cria diretorios em `outputs/out_<timestamp>/`;
- salva logs em `outputs/<execucao>/evaluation_campaigns.log`;
- suporta `--campaign/-c`, `--dryrun/-d`, `--pipenv/-p`, `--verbosity/-v` e `--dataset/-a`.

Atalhos relevantes:

```bash
python3 run_campaign_sbseg.py -c sf
```

Executa as campanhas demo `variational_demo` e `adversarial_demo`.

```bash
python3 run_campaign_sbseg.py
```

Executa a campanha padrao completa definida em `DEFAULT_CAMPAIGN`.

### `main.py`

Ponto de entrada do framework. Define a classe `SynDataGen`, que combina:

- argumentos de execucao;
- processamento CSV;
- metricas;
- modelos generativos;
- classificadores;
- avaliacao.

O arquivo importa componentes centrais de `Engine/`, como `CSVDataProcessor`, `Metrics`, `Evaluation`, `Classifiers`, `GenerativeModels` e suporte de hardware.

### `plots.py`

Script de visualizacao e consolidacao de resultados. Ele:

- le arquivos `EvaluationResults/Results.json`;
- gera heatmaps;
- gera curvas de treinamento;
- gera matrizes de confusao;
- gera graficos de metricas de distancia e classificacao;
- pode consolidar resultados finais quando chamado com `--f_plot`.

### Outros scripts raiz

| Arquivo | Papel |
|---|---|
| `plots_svm.py` | Visualizacoes auxiliares para SVM. |
| `results.sh` | Script shell relacionado a resultados. |
| `run_demo_docker.sh` | Executa demo via Docker. |
| `run_experiments_docker.sh` | Executa experimentos completos via Docker. |
| `run_experiments_docker_SDV.sh` | Execucao Docker voltada a modelos/intefaces SDV. |
| `pip_env_install.sh` | Instalacao de ambiente/dependencias com pip. |

## Estrutura do `Engine/`

`Engine/` e o nucleo do framework. Foram encontrados 227 arquivos Python nesse diretorio.

```text
Engine/
|-- Activations/
|-- Algorithms/
|-- Arguments/
|-- Callbacks/
|-- Classifiers/
|-- DataIO/
|-- Evaluation/
|-- Exception/
|-- Layers/
|-- Loss/
|-- Metrics/
|-- Models/
|-- Optimizers/
|-- Support/
```

### `Engine/Activations/`

Implementa funcoes/camadas de ativacao usadas por modelos neurais:

```text
Activations.py
CeLU.py
ELU.py
Exponential.py
GELU.py
GLU.py
HardSigmoid.py
LeakyRelu.py
Linear.py
LogSigmoid.py
PReLU.py
ReLU.py
SELU.py
Sigmoid.py
SoftSign.py
Softmax.py
Softplus.py
Swish.py
Tanh.py
```

### `Engine/Algorithms/`

Contem os algoritmos de treinamento/geracao:

```text
Adversarial/AdversarialAlgorithm.py
Autoencoder/AutoencoderAlgorithm.py
Copy/CopyAlgorithm.py
DenoisingDiffusion/AlgorithmDenoisingDiffusion.py
DenoisingDiffusion/GaussianDenoisingDiffusion.py
LatentDiffusion/AlgorithmLatentDiffusion.py
LatentDiffusion/AlgorithmVAELatentDiffusion.py
LatentDiffusion/GaussianLatentDiffusion.py
QuantizedVAE/AlgorithmQuantizedVAE.py
RandomNoise/AlgorithmRandomNoise.py
SMOTE/AlgorithmSMOTE.py
ThirdParty/SDVInterfaceAlgorithm.py
VariationalAutoencoder/AlgorithmVariationalAutoencoder.py
Wasserstein/AlgorithmWassersteinGAN.py
WassersteinGP/AlgorithmWassersteinGANGP.py
```

### `Engine/Arguments/`

Define argumentos e configuracoes por familia de modelos e classificadores:

```text
Arguments.py
ArgumentsAdversarial.py
ArgumentsAutoencoder.py
ArgumentsDataLoader.py
ArgumentsDenoisingDiffusion.py
ArgumentsDiffusion.py
ArgumentsEarlyStop.py
ArgumentsFramework.py
ArgumentsLatentDiffusion.py
ArgumentsOptimizer.py
ArgumentsQuantizedVAE.py
ArgumentsRandomNoise.py
ArgumentsSMOTE.py
ArgumentsVariationalAutoencoder.py
ArgumentsWassersteinGAN.py
ArgumentsWassersteinGANGP.py
LoggerSetup.py
View.py
Classifiers/
```

`Engine/Arguments/Classifiers/` contem argumentos especificos para `RandomForest`, `SVM`, `KNN`, `DecisionTree`, `NaiveBayes`, `GradientBoosting`, `SGD`, entre outros.

### `Engine/Callbacks/`

Callbacks de treino e monitoramento:

```text
CallbackDiffusionModelSave.py
CallbackEarlyStop.py
CallbackModel.py
CallbackNan.py
CallbackResources.py
```

### `Engine/Classifiers/`

Camada de classificadores supervisionados usados na avaliacao:

```text
Classifiers.py
Algorithms/
```

Algoritmos presentes:

```text
AdaBoost.py
DecisionTree.py
GaussianProcess.py
GradientBoosting.py
KMeansClustering.py
KNearestNeighbors.py
LinearRegression.py
NaiveBayes.py
Perceptron.py
PerceptronModel.py
QuadraticDiscriminant.py
RandomForest.py
SpectralClustering.py
StochasticGradientDescent.py
SupportVectorMachine.py
```

### `Engine/DataIO/`

Entrada, saida e organizacao de dados:

```text
CSVLoader.py
DirectoryManager.py
XLSLoader.py
```

### `Engine/Evaluation/`

Estrategias de avaliacao e validacao:

```text
CrossValidation.py
Evaluation.py
TrTr.py
TrTs.py
TsTr.py
```

Os nomes indicam comparacoes como treino/teste em dados reais e sinteticos, incluindo estrategias TS-TR e TR-TS descritas no README.

### `Engine/Layers/`

Camadas customizadas para modelos neurais:

```text
AttentionBlock.py
AttentionBlockLayer.py
ClusteringLayer.py
ConvolutionalModuleLayer.py
CrossAttentionLayer.py
FeedforwardModuleLayer.py
GumbelVectorQuantizer.py
PositionalEncodingLayer.py
RelativePositionalEmbeddingLayer.py
RouterLayer.py
Sampling.py
SamplingLayer.py
SwitchLayer.py
TimeEmbedding.py
TimeEmbeddingLayer.py
TransformerDecoderLayer.py
TransformerEncoderLayer.py
TransformerLayer.py
TransposeLayer.py
VectorQuantizerLayer.py
```

### `Engine/Loss/`

Funcoes de perda:

```text
BinaryCrossEntropy.py
CategoricalCrossEntropy.py
ContrastiveLoss.py
CossineSimilarity.py
DiversityLoss.py
KLDivergenceLoss.py
MeanAbsoluteError.py
MeanSquaredError.py
```

### `Engine/Metrics/`

Metricas binarias e de distancia:

```text
Metrics.py
Binary/
Distance/
```

Metricas binarias incluem `Accuracy`, `AUC`, `F1`, `Precision`, `Recall`, `Specificity`, `FalsePositiveRate`, `TruePositive`, `TrueNegative`, `MCC`, `MAE` e `MSE`.

Metricas de distancia incluem `Euclidean`, `Hamming`, `Hellinger`, `Jaccard`, `Manhattan`, `LogLikelihood` e `PermutationTest`.

### `Engine/Models/`

Implementacoes dos modelos generativos:

```text
GenerativeModels.py
Adversarial/
Autoencoder/
DenoisingDiffusion/
DiffusionKernel/
LatentDiffusion/
QuantizedVAE/
VariationalAutoencoder/
Wasserstein/
WassersteinGP/
```

Familias principais:

- GAN adversarial comum;
- Autoencoder;
- Variational Autoencoder;
- Quantized/VQ-VAE;
- Wasserstein GAN;
- Wasserstein GAN com gradient penalty;
- Denoising Diffusion;
- Latent Diffusion.

### `Engine/Optimizers/`

Wrappers/configuracoes de otimizadores:

```text
AdaDelta.py
Adam.py
FTRL.py
Nadam.py
Optimizer_Test.py
Optimizers.py
RSMProp.py
SGD.py
```

### `Engine/Support/`

Infraestrutura auxiliar:

```text
HardwareManager.py
```

## `Tools/`

Diretorio de visualizacao e utilitarios graficos.

```text
Tools/
|-- ClusteringVisualizer.py
|-- EvaluationMetrics.py
|-- Plot.py
|-- PlotClassificationMetrics.py
|-- PlotClasssificationMetrics.py
|-- PlotConfusionMatrix.py
|-- PlotDistanceMetrics.py
|-- PlotFidelityMetrics.py
|-- PlotHeatMap.py
|-- PlotLossCurve.py
|-- PlotRocCurve.py
|-- PlotTrainingCurve.py
|-- config.py
|-- utils.py
|-- Plot/
```

`Tools/Plot/` contem versoes/organizacao adicional para componentes de plotagem, configuracoes e exemplos de resultados.

## `Datasets/`

Foram encontrados 20 arquivos em `Datasets/`.

```text
Datasets/
|-- SBSeg_2025/
|   |-- reduced_balanced_androcrawl.csv
|   |-- reduced_balanced_drebin215.csv
|-- AppClassNet/
|   |-- processed/
|       |-- without_f14/
|       |-- without_f17/
|       |-- without_f17_f14/
```

Os subdiretorios de `AppClassNet/processed/` contem splits `train`, `valid` e `test`, separados em arquivos `_x.csv` e `_y.csv`.

## `Docs/`

Documentacao do projeto:

```text
Docs/
|-- README.md
|-- Overview.md
|-- index.html
|-- Diagrams/
|-- Documentation/
```

`Docs/Diagrams/` possui diagramas Mermaid para:

```text
01_system_architecture.md
02_core_class_hierarchy.md
03_evaluation_strategy.md
04_model_training_pipeline.md
05_metrics_framework.md
06_data_flow_architecture.md
07_generative_models_comparison.md
08_deployment_architecture.md
README.md
```

`Docs/Documentation/` contem documentacao HTML/API gerada para classes e componentes especificos.

## `Test/` e `Scripts/`

Testes shell encontrados:

```text
Test/test_adversarial.sh
Test/test_autoencoder.sh
Test/test_diffusion.sh
Test/test_variational.sh
Test/test_wasserstein.sh
```

Scripts auxiliares:

```text
Scripts/app_run.sh
Scripts/commit.sh
Scripts/docker.sh
```

## Saidas e Artefatos Experimentais

### `outputs/`

Foram encontrados 114 arquivos em `outputs/`. A estrutura segue o padrao:

```text
outputs/
|-- out_YYYY-MM-DD_HH-MM-SS/
|   |-- evaluation_campaigns.log
|   |-- <dataset>/
|       |-- <campaign>/
|           |-- combination_1/
|               |-- DataGenerated/
|               |-- EvaluationResults/
|               |-- Logs/
|               |-- Monitor/
```

Exemplos de artefatos:

- `DataGenerated/data_training_fold_<k>.csv`;
- `DataGenerated/data_evaluation_fold_<k>.csv`;
- `DataGenerated/DataOutput_K_fold_<k>_<model>.txt`;
- `EvaluationResults/Results.json`;
- `EvaluationResults/*.pdf`;
- `Monitor/monitor_model_<k>_fold.json`;
- `Monitor/training_curves - <model> - <dataset>.pdf`;
- `Logs/logging.log`.

### `SBSEG25_Tests/`

Foram encontrados 784 arquivos em `SBSEG25_Tests/`. O diretorio parece armazenar resultados de campanhas/reproducoes SBSEG, incluindo subpastas por modelo e dataset:

```text
SBSEG25_Tests/
|-- Results/
|-- adversarial/
|-- autoencoder/
|-- latent_diffusion/
|-- reduced_balanced_kronodroid_emulator/
|-- variational/
|-- wasserstein/
|-- wasserstein_gp/
```

Assim como `outputs/`, contem CSVs gerados, logs, monitores, JSONs de resultado e PDFs de avaliacao.

## Configuracao e Dependencias

Arquivos principais:

```text
pyproject.toml
requirements.txt
Pipfile
uv.lock
Dockerfile
```

`pyproject.toml` define o pacote `maldatagen`, Python `>=3.10` e dependencias relevantes como:

- `tensorflow`;
- `keras`;
- `numpy`;
- `pandas`;
- `scikit-learn`;
- `scipy`;
- `sdv`;
- `matplotlib`;
- `seaborn`;
- `plotly`;
- `xgboost`;
- `mlflow`;
- `neptune`.

## Fluxo de Execucao Resumido

```text
Dataset CSV
   |
   v
run_campaign_sbseg.py
   |
   |-- chama main.py com parametros de campanha
   v
main.py / SynDataGen
   |
   |-- DataIO: carrega dados
   |-- Algorithms + Models: treina e gera dados sinteticos
   |-- Classifiers: avalia classificadores
   |-- Metrics + Evaluation: calcula metricas
   v
outputs/<execucao>/<dataset>/<campanha>/combination_<n>/
   |
   |-- DataGenerated/
   |-- EvaluationResults/Results.json
   |-- Monitor/
   |-- Logs/
   v
plots.py
   |
   v
PDFs, heatmaps, curvas, matrizes e graficos consolidados
```

## Observacoes do Workspace

- O `git status` indicou muitos arquivos `.pyc` modificados em `__pycache__`, provavelmente gerados por execucoes locais.
- Ha arquivos/diretorios nao rastreados como `.idea/`, `outputs/`, `pyproject.toml`, `uv.lock`, `Datasets/AppClassNet/`, `Tools/__pycache__/` e `__pycache__/`.
- O status tambem indica remocao local de templates em `.github/ISSUE_TEMPLATE/`.
- Para documentacao de estrutura, os diretorios `outputs/`, `SBSEG25_Tests/`, `.venv/`, `.idea/` e `__pycache__/` devem ser tratados como artefatos locais ou gerados, nao como nucleo de implementacao.

## Arquivos Mais Relevantes Para Entender o Projeto

```text
README.md
run_campaign_sbseg.py
main.py
plots.py
Engine/Arguments/Arguments.py
Engine/Models/GenerativeModels.py
Engine/Algorithms/*/
Engine/Evaluation/Evaluation.py
Engine/Evaluation/CrossValidation.py
Engine/Classifiers/Classifiers.py
Engine/DataIO/CSVLoader.py
Engine/Metrics/Metrics.py
Docs/Overview.md
Docs/Diagrams/README.md
```

