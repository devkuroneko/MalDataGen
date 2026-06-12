# Relatorio - Runner AppClassNet top200 com dados continuos

## Objetivo

Centralizar a execucao dos experimentos AppClassNet top200 no arquivo `run_appclassnet_top200.py` e garantir que as campanhas enviem `--data_type continuous` para o `main.py`.

Apos a renomeacao feita no workspace, `run_appclassnet_top200.py` passou a ser o runner principal. O arquivo antigo `run_campaign_sbseg.py` nao e mais necessario para executar AppClassNet.

## Arquivos modificados

### `run_appclassnet_top200.py`

O arquivo foi recriado como runner especifico para AppClassNet top200.

Principais mudancas:

- Centralizacao dos caminhos AppClassNet em constantes: `Datasets/raw/AppClassNet/top200` e `Datasets/converted/AppClassNet/top200`.
- Suporte aos splits `train`, `valid`, `test` e `all` por meio de `--dataset_split`.
- Preparacao automatica de CSV unico a partir dos pares `.npy` `*_x.npy` e `*_y.npy`.
- CSV preparado com colunas `f0` ate `f19` e coluna final `label`, formato esperado pelo `CSVDataProcessor`.
- `--data_type` padrao definido como `continuous` e repassado ao `main.py`.
- `--prepare_max_samples` para materializar subconjuntos sem converter o dataset completo.
- `--prepare_sampling balanced` como padrao quando `--prepare_max_samples` e usado.
- `--prepare_sampling head` disponivel para reproducao sequencial simples.
- Escrita de CSV em chunks com `numpy.load(..., mmap_mode="r")` para reduzir pico de memoria.
- Reuso de CSV preparado quando o arquivo ja existe.
- `--force_prepare` para reconstruir o CSV.
- `--prepare_only` para converter/preparar sem executar treinamento.
- `--skip_plots` para evitar a etapa `plots.py`.
- `--list_campaigns` para listar campanhas disponiveis.
- Preferencia por `.venv/bin/python` quando existir.
- Reexecucao automatica com o Python configurado quando o runner for chamado com `python3` global sem `numpy/pandas`.
- Montagem de comandos com lista de argumentos, nao concatenacao de string, para evitar erros em parametros multi-valor.
- Campanhas AppClassNet ajustadas para 200 classes.
- Perdas e ativacoes ajustadas para dados continuos onde o parser/modelo aceita: `mse`, `mean_squared_error` e `linear`.
- Aliases preservados: `sf` executa demos (`variational_demo`, `adversarial_demo`) e `sf2` executa SDV (`copula`, `tvae`, `ctgan`).

Impacto no sistema:

- O `main.py` passa a receber um CSV valido para AppClassNet sem depender de conversao manual externa.
- Os atributos continuos nao sao arredondados, porque o runner injeta `--data_type continuous` por padrao.
- Os modelos condicionais recebem configuracoes compativeis com 200 classes.
- Campanhas pequenas ficam mais viaveis com `--prepare_max_samples`, sem ler/escrever todo o AppClassNet.
- A execucao antiga SBSeg foi substituida por um fluxo AppClassNet top200.

### `main.py`

O erro abaixo ocorria durante a sintese com VAE em folds que nao continham todas as 200 classes:

```text
Input 1 of layer "Decoder" is incompatible with the layer: expected shape=(None, 200), found shape=(2, 193)
```

Causa:

- `synthesize_data()` recalculava `number_classes` com base apenas nas classes presentes no fold atual.
- Quando o fold continha 193 classes, o one-hot de labels sinteticos tinha largura 193.
- O decoder havia sido configurado com `variational_autoencoder_number_classes=200`, portanto esperava largura 200.

Correcoes aplicadas:

- Adicionado `_prepare_labels_for_conditional_generation()` para padronizar labels condicionais antes de treino/sintese.
- Adicionado `_get_configured_number_classes()` para preservar o dominio total configurado por argumentos e metadados do experimento.
- Adicionado `_build_generation_metadata()` para usar as classes presentes no fold apenas para contagem, sem reduzir a largura total do dominio de classes.
- `train_model()` agora envia labels condicionais preparadas para os modelos generativos.
- `synthesize_data()` agora monta `number_samples_per_class` com o total configurado de classes, nao com `len(unique_classes)` do fold.

Impacto no sistema:

- Evita incompatibilidade de shape entre labels sinteticos e entrada condicional do decoder/generator.
- Permite que folds incompletos continuem usando o dominio AppClassNet completo de 200 classes.
- Mantem a contagem por fold para definir quantas amostras sintetizar por classe presente.
- Nao altera o CSV nem o runner; a correcao fica no nucleo de treino/sintese.

## Como executar

### Demo sem plots, com 1000 amostras balanceadas

```bash
python3 run_appclassnet_top200.py --campaign sf --dataset_split train --prepare_max_samples 1000 --skip_plots
```

Comportamento esperado:

- Prepara `Datasets/converted/AppClassNet/top200/train_balanced_max1000.csv`.
- Usa 1000 linhas balanceadas, 5 amostras por classe para 200 classes.
- Executa `variational_demo` e `adversarial_demo`.
- Repassa `--data_type continuous` para o `main.py`.

### Apenas preparar CSV

```bash
python3 run_appclassnet_top200.py --prepare_only --dataset_split train --prepare_max_samples 1000
```

### Campanha especifica

```bash
python3 run_appclassnet_top200.py --campaign wasserstein_gp --dataset_split train --prepare_max_samples 5000 --skip_plots
```

### Usando `run_experiments.py`

```bash
python3 run_experiments.py --campaign sf --dataset_split train --prepare_max_samples 1000 --skip_plots
```

## Observacoes tecnicas

- `--prepare_max_samples` com amostragem balanceada deve ser pelo menos `200 * number_k_folds` para evitar classes com menos amostras que o numero de folds.
- Para as campanhas completas com `number_k_folds=5`, o minimo recomendado e `1000` amostras.
- Para os demos `sf`, que usam `number_k_folds=2`, o minimo recomendado e `400` amostras.
- O split `train` completo tem 4.347.270 linhas; `test` tem 4.830.012 linhas. Converter splits completos para CSV pode gerar arquivos grandes.
- `plots.py` ainda possui visualizacoes orientadas a labels 0/1 em algumas funcoes de heatmap. Por isso, `--skip_plots` e recomendado para validacoes iniciais no AppClassNet top200 multiclasses.

## Validacoes executadas

Comandos executados:

```bash
python3 -m py_compile run_appclassnet_top200.py run_experiments.py
python3 run_appclassnet_top200.py --help
python3 run_appclassnet_top200.py --list_campaigns
python3 run_appclassnet_top200.py --dryrun --campaign sf --dataset_split train --prepare_max_samples 1000 --skip_plots --verbosity 20
python3 run_appclassnet_top200.py --dryrun --pipenv --campaign adversarial_demo --dataset_split train --prepare_max_samples 1000 --skip_plots --verbosity 20
python3 run_appclassnet_top200.py --prepare_only --dataset_split train --prepare_max_samples 1000 --converted_root /tmp/maldatagen_appclassnet_test --force_prepare --verbosity 20
python3 run_experiments.py --dryrun --campaign adversarial_demo --dataset_split train --prepare_max_samples 10 --skip_plots --verbosity 20
python3 -m py_compile main.py run_appclassnet_top200.py
python3 run_appclassnet_top200.py --campaign sf --dataset_split train --prepare_max_samples 1000 --skip_plots --verbosity 20
```

Resultado das validacoes:

- Compilacao Python passou.
- `--help` mostra os novos argumentos AppClassNet.
- `--list_campaigns` listou 13 campanhas.
- `--dryrun` gerou comandos apontando para `.venv/bin/python`, `main.py`, CSV AppClassNet preparado e `--data_type continuous`.
- `--dryrun --pipenv` gerou comando no formato `pipenv run python3 ...`.
- Preparacao em `/tmp` gerou CSV com 1000 linhas, 21 colunas, 200 classes e 5 amostras por classe.
- `run_appclassnet_top200.py` e o ponto de entrada principal do fluxo AppClassNet.
- Os dry-runs geraram logs de campanha em `outputs/` sem iniciar treinamento.
- O treinamento real `sf` com 1000 amostras foi concluido com codigo `0`.
- Durante a validacao real, um fold continha menos classes que o dominio total e o log confirmou a preservacao do dominio configurado: `Generation fold contains 191/200 configured classes; preserving total class domain.`
- O erro `expected shape=(None, 200), found shape=(2, 193)` nao reapareceu.

## Limitacoes nao resolvidas nesta tarefa

- `plots.py` ainda tem pressupostos binarios em algumas visualizacoes, principalmente heatmaps por labels 0 e 1.
- O projeto ainda usa validacao cruzada interna do `main.py`; os splits originais `train/valid/test` do AppClassNet sao centralizados para selecao/conversao, mas nao substituem a logica de folds do `main.py`.
- O teste real executado foi a campanha demo `sf`; campanhas completas com mais epocas ainda podem exigir validacao separada por custo computacional.
