# Final Evaluation Validation Report

Data: 2026-07-13

Escopo: validacao final dos modulos TR-TR, TR-TS e TS-TR para AppClassNet top-200, mantendo compatibilidade com o fluxo CSV original do MalDataGen.

## Auditoria pre-correcao

Nenhuma correcao foi aplicada antes desta auditoria. O repositório ja continha alteracoes nao commitadas em avaliadores, argumentos, metricas e testes; elas foram tratadas como estado existente e nao foram revertidas.

### Evidencias do pipeline

- TR-TR usa `EvaluationRunner` para montar treino real e teste real: arquivo `Engine/Evaluation/TrTr.py`, classe `TrTr`, funcao `evaluation_TR_TR`, linhas aproximadas 85-112.
- TR-TS em modo normal usa `EvaluationRunner`; em protocolo AppClassNet estrito treina em `x_training_real/y_training_real` e testa em sintetico materializado: arquivo `Engine/Evaluation/EvaluationRunner.py`, classe `EvaluationRunner`, funcao `build_evaluation_dataset`, linhas aproximadas 103-122.
- TS-TR em modo normal usa sintetico como treino e `x_evaluation_real/y_evaluation_real` como teste: arquivo `Engine/Evaluation/EvaluationRunner.py`, classe `EvaluationRunner`, funcao `build_evaluation_dataset`, linhas aproximadas 124-135.
- TR-TS em batches valida metadados de escala contra o manifesto sintetico antes da avaliacao: arquivo `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS`, linhas aproximadas 75-88.
- TS-TR em batches valida metadados de escala contra o manifesto sintetico antes da avaliacao: arquivo `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR`, linhas aproximadas 98-109.
- O runner AppClassNet usa `source_profile=appclassnet_top200`, `feature_transform=preserve`, `classifier_transform=preserve` e `evaluation_space=source` por padrao: arquivo `run_appclassnet_top200.py`, funcao `build_parser`, linhas aproximadas 2076-2103.
- O baseline real-real usa `DecisionTreeClassifier(random_state=0)` e selecao estratificada deterministica com seeds fixas: arquivo `run_appclassnet_top200.py`, funcoes `build_baseline_classifier` e `run_baseline_real_only`, linhas aproximadas 785-799 e 1031-1075.
- Metricas preditivas multiclasses incluem Accuracy, MacroF1, WeightedF1 e BalancedAccuracy: arquivo `Engine/Metrics/Metrics.py`, classe `Metrics`, funcoes `_get_classifier_metric_names` e `_get_multiclass_metric_values`, linhas aproximadas 310-384.
- Metricas nao executadas sao inicializadas como `not_applicable`, nao como zero, para avaliacao preditiva/distancia/SDV: arquivo `Engine/Metrics/Metrics.py`, classe `Metrics`, funcao `__initialize_dictionary`, linhas aproximadas 202-290.

### Lacunas encontradas

- Diagnostico incompleto: `Metrics._record_predictive_diagnostics` registrava contagem de classes verdadeiras/preditas e suspeita de nivel aleatorio, mas nao registrava "classe mais prevista" nem "taxa da classe mais prevista", exigidas nesta validacao. Evidencia: arquivo `Engine/Metrics/Metrics.py`, classe `Metrics`, funcao `_record_predictive_diagnostics`, linhas aproximadas 398-432.
- Determinismo incompleto em batches: `BatchClassifiers.make_batch_classifier` e `_collect_stratified_subset` fixavam `random_state=42`/`default_rng(42)`, enquanto a validacao exige `random_state=0`. Evidencia: arquivo `Engine/Classifiers/BatchClassifiers.py`, funcoes `make_batch_classifier` e `_collect_stratified_subset`, linhas aproximadas 42-74 e 114-116.
- Determinismo incompleto no wrapper legado de DecisionTree: `DecisionTree.get_model` criava `DecisionTreeClassifier` sem `random_state`. Evidencia: arquivo `Engine/Classifiers/Algorithms/DecisionTree.py`, classe `DecisionTree`, funcao `get_model`, linhas aproximadas 109-115.

### Correcoes planejadas

- Registrar em diagnosticos: numero de classes previstas, classe mais prevista, contagem e taxa da classe mais prevista.
- Fazer o batch classifier respeitar `arguments.random_state` quando existir, com default `0`, sem remover argumentos existentes.
- Adicionar `--random_state` aos argumentos de framework e ao runner AppClassNet, preservando compatibilidade por default.
- Fazer o wrapper legado `DecisionTree` encaminhar `random_state`, com default `0`.

## Correcoes aplicadas

- `Engine/Arguments/ArgumentsFramework.py`, funcao `add_argument_framework`, linha aproximada 229: adicionado `--random_state` com default `0`.
- `run_appclassnet_top200.py`, funcoes `build_main_command`, `build_batch_main_command` e `build_parser`, linhas aproximadas 1588-1602, 1683-1698 e 2193-2205: `--random_state` e encaminhado para `main.py`, com fallback compativel via `getattr(..., 0)`.
- `Engine/Classifiers/BatchClassifiers.py`, funcoes `make_batch_classifier` e `_collect_stratified_subset`, linhas aproximadas 35-76 e 114-116: classificadores batch/subset e reservoir sampling usam `arguments.random_state` ou `0`.
- `Engine/Classifiers/Algorithms/DecisionTree.py`, classe `DecisionTree`, funcoes `__init__` e `get_model`, linhas aproximadas 67-71 e 109-115: `DecisionTreeClassifier` recebe `random_state`.
- `Engine/Metrics/Metrics.py`, classe `Metrics`, funcao `_record_predictive_diagnostics`, linhas aproximadas 398-432: diagnosticos agora incluem `predicted_class_count`, `most_predicted_class`, `most_predicted_count` e `most_predicted_rate`.

## Verificacao automatizada

Comando:

```bash
PYTHONPATH=. pytest -q tests/test_evaluation_runner.py tests/test_evaluation_integration_spec.py tests/test_metrics_target_types.py tests/test_appclassnet_evaluation_mode.py tests/test_data_loader_arguments.py tests/test_batch_npy_dataset.py tests/test_synthetic_batch_io.py
```

Resultado: `35 passed, 18 skipped in 4.17s`.

## Dados reais AppClassNet

Comando:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
root=Path('Datasets/raw/AppClassNet/top200')
for split in ['train','valid','test']:
    x=np.load(root/f'{split}_x.npy', mmap_mode='r', allow_pickle=False)
    y=np.load(root/f'{split}_y.npy', mmap_mode='r', allow_pickle=False)
    u,c=np.unique(y, return_counts=True)
    print(split, x.shape, y.shape, len(u), int(u.min()), int(u.max()), int(c.min()), int(c.max()), float(np.min(x)), float(np.max(x)))
PY
```

Resultado:

| Split | X shape | y shape | Classes | Labels | Min count | Max count | Feature min | Feature max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | `(4347270, 20)` | `(4347270,)` | 200 | 0..199 | 1388 | 450392 | -0.4999829788 | 0.4999150418 |
| valid | `(482489, 20)` | `(482489,)` | 200 | 0..199 | 151 | 50041 | -0.4999829788 | 0.4999150418 |
| test | `(4830012, 20)` | `(4830012,)` | 200 | 0..199 | 1539 | 500431 | -0.4999994815 | 0.4999150418 |

Conclusao: os arquivos NPY reais atendem ao contrato de 20 features, 200 classes, labels `0..199` e escala `source` aproximadamente `[-0.5, 0.5]`.

## Execucoes finais

Observacao: TR-TS e TS-TR foram validados com sintetico deterministico tipo copy em `source`. Isso valida os contratos de avaliacao, espaco numerico, labels, shapes e metricas; nao e uma avaliacao de qualidade de geradores neurais.

### 1. TR-TR baseline real-real

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_trtr.time python run_appclassnet_top200.py --baseline_real_only --use_mmap --train_samples_per_class 1000 --test_samples_per_class 500 --baseline_classifier decision_tree --classifier_transform preserve --feature_transform preserve --generator_transform preserve --evaluation_space source --random_state 0 --skip_plots
```

Resultado:

| Campo | Valor |
|---|---:|
| Duracao wall | 6.10 s |
| Pico memoria RSS | 2040412 kB |
| Train shape | `(200000, 20)` |
| Test shape | `(100000, 20)` |
| Classes treino/teste | 200 / 200 |
| data_space | source |
| transform_id | null |
| Classificador | DecisionTreeClassifier |
| Parametros principais | `criterion=gini`, `random_state=0`, `max_depth=None` |
| Accuracy | 0.68673 |
| Macro-F1 | 0.6862981265 |
| Weighted-F1 | 0.6862981265 |
| BalancedAccuracy | 0.68673 |

Conclusao: passou. O resultado permanece no baseline validado.

### 2. TR-TS

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_modules.time env PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-final-validation python /tmp/maldatagen_final_validation_runner.py
```

Artefato: `results/appclassnet_top200/final_validation/module_validation_metrics.json`.

Resultado TR-TS:

| Campo | Valor |
|---|---:|
| Duracao interna | 4.483630 s |
| Pico memoria RSS do processo | 2078892 kB |
| Train shape | `(200000, 20)` real |
| Test shape | `(100000, 20)` sintetico copy |
| Classes treino/teste | 200 / 200 |
| data_space | source |
| transform_id | null |
| Classificador | DecisionTreeClassifier |
| Parametros principais | `criterion=gini`, `random_state=0`, `max_depth=None` |
| Accuracy | 0.68673 |
| Macro-F1 | 0.6862981265 |
| Weighted-F1 | 0.6862981265 |
| BalancedAccuracy | 0.68673 |
| Classes previstas | 200 |
| Classe mais prevista | 153 |
| Taxa da classe mais prevista | 0.00565 |

Conclusao: passou. Real e sintetico foram comparados em `source`, sem transformacao automatica.

### 3. TS-TR

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_modules.time env PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-final-validation python /tmp/maldatagen_final_validation_runner.py
```

Resultado TS-TR:

| Campo | Valor |
|---|---:|
| Duracao interna | 4.432678 s |
| Pico memoria RSS do processo | 2078892 kB |
| Train shape | `(200000, 20)` sintetico copy |
| Test shape | `(100000, 20)` real |
| Classes treino/teste | 200 / 200 |
| data_space | source |
| transform_id | null |
| Classificador | DecisionTreeClassifier |
| Parametros principais | `criterion=gini`, `random_state=0`, `max_depth=None` |
| Accuracy | 0.68673 |
| Macro-F1 | 0.6862981265 |
| Weighted-F1 | 0.6862981265 |
| BalancedAccuracy | 0.68673 |
| Classes previstas | 200 |
| Classe mais prevista | 153 |
| Taxa da classe mais prevista | 0.00565 |

Conclusao: passou. O treino sintetico e o teste real estavam no mesmo espaco `source`.

### 4. Modo normal em subconjunto menor

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_modules.time env PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-final-validation python /tmp/maldatagen_final_validation_runner.py
```

Resultado normal:

| Campo | Valor |
|---|---:|
| Duracao interna | 0.116633 s |
| Pico memoria RSS do processo | 2078892 kB |
| Train shape | `(4000, 20)` |
| Test shape | `(2000, 20)` |
| Classes treino/teste | 200 / 200 |
| data_space | source |
| transform_id | null |
| Classificador | DecisionTreeClassifier |
| Accuracy | 0.349 |
| Macro-F1 | 0.3429347545 |
| Weighted-F1 | 0.3429347545 |
| BalancedAccuracy | 0.349 |
| Classes previstas | 200 |
| Classe mais prevista | 79 |
| Taxa da classe mais prevista | 0.009 |

### 5. Modo batches no mesmo subconjunto

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_modules.time env PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-final-validation python /tmp/maldatagen_final_validation_runner.py
```

Resultado batches:

| Campo | Valor |
|---|---:|
| Duracao interna | 0.120718 s |
| Pico memoria RSS do processo | 2078892 kB |
| Train shape | `(4000, 20)` |
| Test shape | `(2000, 20)` |
| Classes treino/teste | 200 / 200 |
| data_space | source |
| transform_id | null |
| Classificador | DecisionTreeSubset |
| Parametros principais | `criterion=gini`, `random_state=0`, `max_depth=None` |
| Accuracy | 0.349 |
| Macro-F1 | 0.3429347545 |
| Weighted-F1 | 0.3429347545 |
| BalancedAccuracy | 0.349 |
| Classes previstas | 200 |
| Classe mais prevista | 79 |
| Taxa da classe mais prevista | 0.009 |
| Predicoes iguais ao modo normal | true |
| Deltas de metricas | 0.0 para Accuracy, Macro-F1, Weighted-F1 e BalancedAccuracy |

Conclusao: passou. No mesmo subconjunto, normal e batches foram semanticamente equivalentes.

### 6. CSV original MalDataGen

O repositório nao contem o CSV unico padrao `Datasets/converted/train_x.csv`; para validar o fluxo legado `data_format=csv`, foi criado um CSV temporario unico em `/tmp/maldatagen_legacy_validation.csv` com 200 linhas, 6 features e coluna `class` binaria.

Comando:

```bash
/usr/bin/time -v -o /tmp/maldatagen_final_csv.time env PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-final-validation python main.py --data_load_path_file_input /tmp/maldatagen_legacy_validation.csv --data_load_label_column -1 --data_type binary --target_type binary --model_type copy --number_samples_per_class 0:20,1:20 --sample_plan class_counts --number_k_folds 2 -c DecisionTree --random_state 0 --output_dir outputs/final_validation_csv
```

Resultado:

| Campo | Valor |
|---|---:|
| Duracao wall | 4.79 s |
| Pico memoria RSS | 655772 kB |
| Output | `outputs/final_validation_csv_2026-07-13_22-21-18/EvaluationResults/Results.json` |
| Train/test por fold | 100 / 100 |
| Features | 6 |
| Classes | 2 |
| data_format | csv |
| Classificador | DecisionTree |
| TR-TS Accuracy/F1 por fold | 1.0 / 1.0 |
| TS-TR Accuracy/F1 por fold | 1.0 / 1.0 |
| TR-TR | `not_applicable`, pois `--run_tr_tr` nao foi solicitado no fluxo legado |

Conclusao: passou. O fluxo CSV legado carrega CSV unico, executa K-fold, gera sintetico copy e avalia TR-TS/TS-TR. A avaliacao nao executada aparece como `not_applicable`, nao como zero.

## Criterios

- TR-TR permaneceu no baseline validado: Accuracy `0.68673`, Macro-F1 `0.6862981265`.
- AppClassNet preservou `source`; `transform_id=null` em todas as avaliacoes finais.
- Nenhuma normalizacao automatica foi aplicada aos dados AppClassNet.
- TR-TS e TS-TR usaram sintetico copy correto no mesmo espaco numerico dos reais.
- Todas as execucoes AppClassNet observaram 200 classes em treino/teste e 200 classes previstas.
- Metricas proximas de 0.005 nao ocorreram; o diagnostico `chance_level_suspected` ficou `false`.
- Normal e batches foram equivalentes no mesmo subconjunto: predicoes iguais e deltas de metricas `0.0`.
- O fluxo CSV legado continuou funcionando.
- Avaliacoes nao executadas foram registradas como `not_applicable`, nao como zeros.

## Conclusao final

Validacao final aprovada para os contratos de TR-TR, TR-TS, TS-TR, modo normal, modo batches com mmap e CSV legado. As correcoes foram pequenas, testadas e compativeis com o comportamento original.
