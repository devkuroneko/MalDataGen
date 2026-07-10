# Backward Compatibility Report

Data da revisao: 2026-07-10

## Objetivo

Validar que o suporte a `npy_xy`/AppClassNet nao quebrou o fluxo original do MalDataGen baseado em CSV unico.

## Resultado geral

Status: aprovado com uma correcao aplicada durante a revisao.

O fluxo CSV legado continua sendo o padrao:

- `data_format='csv'`
- `split_mode='cross_validation'`
- `target_type='auto'`
- `feature_type='auto'`
- `sample_plan='legacy'`
- `number_samples_per_class='1:256,2:256'`

O `CSVLoader` nao foi alterado nesta revisao; `git diff -- Engine/DataIO/CSVLoader.py` nao retornou diferencas.

## Correcao aplicada

Foi encontrada uma regressao potencial no parser: o default de `--number_samples_per_class` havia passado a ser interpretado como estrutura parseada em vez de permanecer exatamente como o valor legado `"1:256,2:256"`.

Correcao:

- o default voltou a ser a string legada `"1:256,2:256"`;
- quando o usuario passa `--number_samples_per_class` explicitamente, o valor continua sendo parseado para o formato historicamente usado internamente;
- foi adicionada uma assercao de regressao em `tests/test_data_loader_arguments.py` para garantir que o comando antigo sem argumentos novos preserve esse default.

## Testes executados

### Suite completa

Comando:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests
```

Resultado:

```text
Ran 48 tests in 0.059s
OK
```

Observacoes:

- TensorFlow emitiu avisos de registro de plugins CUDA/cuDNN/cuBLAS no stderr.
- Matplotlib criou cache temporario porque `/home/matheus/.config/matplotlib` nao estava gravavel no sandbox.
- Esses avisos nao impediram os testes.

### Subconjunto de testes novos e de regressao

Comando:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_data_loader_arguments \
  tests.test_sample_planner \
  tests.test_npy_xy_loader \
  tests.test_dataset_contracts \
  tests.test_multiclass_pipeline \
  tests.test_metrics_target_types \
  tests.test_cross_validation_data_loading
```

Resultado:

```text
Ran 48 tests in 0.054s
OK
```

Cobertura relevante:

- defaults de CLI legados;
- validacao de argumentos `npy_xy`;
- contratos `DatasetSchema`, `SplitData`, `DatasetBundle`;
- `NpyXYLoader`;
- `SamplePlan`;
- multiclass com muitas classes;
- metricas binarias, multiclass e casos `not_applicable`;
- integracao inicial com `CrossValidation`.

### Smoke test CSV antigo

Comando:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python main.py \
  -i Datasets/raw/maldatagen_sbseg/reduced_balanced_androcrawl.csv \
  --data_load_label_column -1 \
  --data_load_max_samples 40 \
  --number_k_folds 2 \
  --model_type random \
  --classifier KNN \
  --output_dir outputs/maldatagen_csv_smoke
```

Resultado:

- execucao concluiu com `All experiments completed`;
- gerou folds em `outputs/maldatagen_csv_smoke/DataGenerated`;
- gerou `outputs/maldatagen_csv_smoke/EvaluationResults/Results.json`;
- argumentos salvos confirmam `data_format="csv"`, `split_mode="cross_validation"`, `data_type="binary"` e `number_samples_per_class="1:256,2:256"`;
- metricas binarias antigas continuaram com nomes como `Accuracy`, `Precision`, `Recall`, `F1Score`, `Specificity`, `FalsePositiveRate`, `TrueNegativeRate`, `MeanSquareError` e `MeanAbsoluteError`.

### Smoke test npy_xy

Comando:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/smoke_test_npy_xy.py \
  --dry-run \
  --num_classes 10 \
  --samples_per_class 1
```

Resultado:

- criou `train_x.npy`, `train_y.npy`, `valid_x.npy`, `valid_y.npy`, `test_x.npy`, `test_y.npy` em diretorio temporario;
- carregou `DatasetBundle` com `source_format=npy_xy`;
- validou `feature_type=continuous`, `target_type=multiclass`, `num_classes=10`;
- validou splits com `X=(10, 20)` e `y=(10,)`;
- construiu `SamplePlan` `balanced_per_class`.

Comando adicional para simular AppClassNet top-200:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/smoke_test_npy_xy.py \
  --dry-run \
  --num_classes 200 \
  --samples_per_class 1
```

Resultado:

- carregou `DatasetBundle` com `num_classes=200`;
- validou splits com `X=(200, 20)` e `y=(200,)`;
- construiu plano balanceado com 1 amostra por classe.

Tambem foi executado o smoke sem `--dry-run`:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/smoke_test_npy_xy.py \
  --num_classes 5 \
  --samples_per_class 1
```

Resultado:

- criou arquivos `.npy` pequenos;
- carregou e validou schema/splits;
- imprimiu comando recomendado para executar o `main.py`;
- nao alterou o pipeline CSV.

### Verificacao de defaults do CLI

Comando:

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from Engine.Arguments.ArgumentsFramework import add_argument_framework
from Engine.Arguments.ArgumentsDataLoader import add_argument_data_load, validate_data_load_arguments

p = add_argument_data_load(add_argument_framework())
old = validate_data_load_arguments(p.parse_args([]))
for key in [
    'data_format',
    'split_mode',
    'target_type',
    'feature_type',
    'number_samples_per_class',
    'sample_plan',
    'samples_per_class',
    'total_synthetic_rows',
    'mmap_npy',
    '_legacy_number_samples_per_class_explicit',
]:
    print(f'{key}={getattr(old, key)!r}')
PY
```

Resultado:

```text
data_format='csv'
split_mode='cross_validation'
target_type='auto'
feature_type='auto'
number_samples_per_class='1:256,2:256'
sample_plan='legacy'
samples_per_class=None
total_synthetic_rows=None
mmap_npy=False
_legacy_number_samples_per_class_explicit=False
```

Verificacao de uso explicito:

```text
--number_samples_per_class 0:2,1:3
```

Resultado interno:

```text
{'classes': {0: 2, 1: 3}, 'number_classes': 2}
_legacy_number_samples_per_class_explicit=True
```

## Analise de pontos perigosos

### Argumentos antigos removidos

Nao foi identificado argumento antigo removido. Os argumentos novos sao opcionais e `data_format` permanece `csv` por default.

### Defaults modificados

Foi identificada e corrigida a regressao no default de `--number_samples_per_class`. A suite agora cobre esse comportamento.

### CSVLoader alterado sem necessidade

Nao ha diferenca em `Engine/DataIO/CSVLoader.py`.

### main.py redirecionando CSV para fluxo novo

O fluxo CSV segue por `CSVDataProcessor`/`load_csv()` via adaptacao localizada em `load_dataset_from_args`. O `DatasetBundle` so e retornado quando `data_format='npy_xy'`.

### Metricas antigas mudando nome/formato

As metricas binarias executadas continuam sendo gravadas com os nomes historicos. A mudanca intencional foi reservar `not_applicable` para avaliacoes que nao podem ser executadas, em vez de preencher zero falso.

### number_samples_per_class quebrado

O modo antigo sem argumentos novos preserva a string default. O uso explicito ainda gera o dicionario de classes usado pelo pipeline. O plano `legacy` continua sendo o padrao.

### Labels binarios tratados como multiclass indevidamente

No modo CSV padrao, `target_type='auto'` e `data_type='binary'` mantem metricas binarias. O modo multiclass so e ativado explicitamente ou por metadados do loader `npy_xy`.

## Como rodar o CSV antigo

Exemplo minimo preservando o fluxo legado:

```bash
python main.py \
  -i Datasets/raw/maldatagen_sbseg/reduced_balanced_androcrawl.csv \
  --data_load_label_column -1 \
  --number_k_folds 2 \
  --model_type random \
  --classifier KNN
```

Sem `--data_format`, o valor usado e `csv`.

## Como rodar npy_xy AppClassNet

Exemplo AppClassNet top-200:

```bash
python main.py \
  --data_format npy_xy \
  --split_mode provided \
  --train_x_path /path/train_x.npy \
  --train_y_path /path/train_y.npy \
  --valid_x_path /path/valid_x.npy \
  --valid_y_path /path/valid_y.npy \
  --test_x_path /path/test_x.npy \
  --test_y_path /path/test_y.npy \
  --target_type multiclass \
  --feature_type continuous \
  --num_classes 200 \
  --sample_plan balanced_per_class \
  --samples_per_class 1000 \
  --mmap_npy
```

Smoke test sem AppClassNet real:

```bash
python scripts/smoke_test_npy_xy.py --dry-run --num_classes 200 --samples_per_class 1
```

## Riscos remanescentes

- O smoke CSV usou `model_type=random` para reduzir custo; modelos neurais pesados devem ser validados separadamente antes de campanhas longas.
- AppClassNet completo pode exigir RAM/tempo elevados; usar `--mmap_npy` e comecar por subconjuntos.
- Avaliacao final usando split `test` separado ainda e marcada como `not_applicable` quando o pipeline TR-TS/TS-TR nao suporta a etapa diretamente; isso evita metricas falsas.
- Existem arquivos modificados fora do escopo desta revisao no worktree, como `Tools/`, `plots.py`, `outputs/` e documentacao local. Eles nao foram revertidos.
