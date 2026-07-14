# Pipeline Refactor Implementation Report

Data: 2026-07-14.

Escopo: primeira fatia de correcoes aprovadas nos relatorios de auditoria para alinhar TR-TR, TR-TS e TS-TR ao AppClassNet top-200 sem quebrar o caminho legado CSV.

## Problemas corrigidos nesta etapa

### P1. TR-TS treinava no conjunto real de avaliacao

- **Evidencia:** `TR_TS_AUDIT.md` e `EVALUATION_FLOW_AUDIT.md`; `Engine/Evaluation/TrTs.py`, classe `TrTs`, funcao `evaluation_TR_TS()`, linhas aproximadas antigas ~96-111 em batches e ~188-193 em normal.
- **Correcao:** adicionado `evaluation_protocol`. O perfil legado permanece `legacy`; o perfil AppClassNet resolve para `appclassnet_strict`. Nesse protocolo, TR-TS treina em `x_training_real/y_training_real` e testa no sintetico.
- **Arquivos:** `Engine/Evaluation/TrTs.py`, `Engine/Evaluation/EvaluationRunner.py`, `Engine/Arguments/Arguments.py`, `Engine/Arguments/ArgumentsFramework.py`.
- **Risco mitigado:** mudanca nao altera CSV legado por default.

### P2. TS-TR normal truncava metricas pelo total sintetico

- **Evidencia:** `TS_TR_AUDIT.md` e `METRICS_AUDIT.md`; `Engine/Evaluation/TsTr.py`, classe `TsTr`, funcao `evaluation_TS_TR()`, linhas aproximadas antigas ~207-216.
- **Correcao:** TS-TR normal agora treina no sintetico e calcula metricas contra todo `x_evaluation_real/y_evaluation_real`, sem `[:total_samples]`.
- **Arquivos:** `Engine/Evaluation/TsTr.py`, `Engine/Evaluation/EvaluationRunner.py`.

### P0/P11/P13. Falta de interface comum, metadados e diagnosticos

- **Evidencia:** `PIPELINE_LIMITATIONS_AND_IMPROVEMENTS.md`, problemas P0, P11 e P13.
- **Correcao:** criada interface incremental `EvaluationDataset`, `EvaluationRunner` e `EvaluationMode` com separacao de dataset, validacao, fit, predicao, metricas e persistencia de metadados.
- **Metadados registrados:** origem de treino/teste, shapes, hashes SHA-256, contagens por classe, labels observadas, `schema_version=evaluation_dataset/v1`.
- **Arquivos:** `Engine/Evaluation/EvaluationRunner.py`, `Engine/Evaluation/TrTr.py`, `Engine/Evaluation/TrTs.py`, `Engine/Evaluation/TsTr.py`, `Engine/Metrics/Metrics.py`.

### P12. Falta de diagnostico de chance level

- **Evidencia:** `METRICS_AUDIT.md`; recomendacao para marcar `chance_level_suspected` quando `num_classes=200` e Accuracy fica entre `0.004` e `0.006`.
- **Correcao:** `Metrics.get_task_metrics()` registra bloco `Diagnostics` por fold/avaliacao/classificador com `chance_level_suspected`, classes observadas e classes ausentes.
- **Arquivos:** `Engine/Metrics/Metrics.py`.

### TR-TR principal

- **Evidencia:** `TR_TR_AUDIT.md`; `main.py`, classe `SynDataGen`, funcao `run_experiments()`, chamada TR-TR comentada.
- **Correcao:** `TrTr.evaluation_TR_TR()` passou a usar o runner comum e registrar metadados. O pipeline principal ganhou `--run_tr_tr` opt-in. O default continua falso para preservar o output legado; o golden AppClassNet continua sendo `--baseline_real_only`.
- **Arquivos:** `Engine/Evaluation/TrTr.py`, `Engine/Arguments/ArgumentsFramework.py`, `main.py`.

## Arquivos modificados

- `Engine/Arguments/Arguments.py`
- `Engine/Arguments/ArgumentsFramework.py`
- `Engine/Evaluation/EvaluationRunner.py`
- `Engine/Evaluation/TrTr.py`
- `Engine/Evaluation/TrTs.py`
- `Engine/Evaluation/TsTr.py`
- `Engine/Metrics/Metrics.py`
- `main.py`
- `tests/test_data_loader_arguments.py`
- `tests/test_evaluation_label_shapes.py`
- `tests/test_evaluation_runner.py`
- `tests/test_metrics_target_types.py`

## Testes criados ou atualizados

- `tests/test_evaluation_runner.py`
  - valida metadados, hashes, contagens por classe e rejeicao de X/y desalinhado.
- `tests/test_evaluation_label_shapes.py`
  - adiciona TR-TS estrito treinando no split real de treino.
  - adiciona TS-TR normal usando todo o real de avaliacao.
- `tests/test_metrics_target_types.py`
  - adiciona diagnostico `chance_level_suspected` para 200 classes.
- `tests/test_data_loader_arguments.py`
  - valida que AppClassNet resolve `evaluation_protocol=appclassnet_strict`.
  - valida que CSV legado permanece `evaluation_protocol=legacy`.

## Resultados dos testes

### Suíte completa

Comando:

```bash
python -m unittest discover -s tests
```

Resultado:

```text
Ran 125 tests in 0.212s
OK (skipped=18)
```

### Baseline AppClassNet real-real

Comando:

```bash
python run_appclassnet_top200.py \
  --baseline_real_only \
  --baseline_classifier decision_tree \
  --train_samples_per_class 1000 \
  --test_samples_per_class 500 \
  --skip_plots
```

Resultado observado:

- Accuracy: `0.68673`
- Macro-F1: `0.6862981264884368`
- Weighted-F1: `0.6862981264884368`
- BalancedAccuracy: `0.6867300000000001`
- Train: `200000 x 20`
- Test: `100000 x 20`
- Classes: 200, com 1000 treino/classe e 500 teste/classe

Comparacao: igual ao baseline validado informado no contexto do projeto.

## Compatibilidade

- O comportamento CSV legado permanece `evaluation_protocol=legacy`.
- O AppClassNet `source_profile=appclassnet_top200` passa a usar `appclassnet_strict` quando o usuario nao fornece `--evaluation_protocol`.
- `--run_tr_tr` e opt-in; portanto TR-TR do pipeline principal nao altera resultados antigos por padrao.
- O schema legado de metricas preditivas foi preservado. Novos blocos foram adicionados:
  - `Schema`
  - `EvaluationMetadata`
  - `Diagnostics`

## Mudancas pequenas aplicadas

1. Criacao do helper comum de avaliacao.
2. Refatoracao local de TR-TR para usar helper comum.
3. Correcao local de TR-TS com protocolo estrito AppClassNet.
4. Correcao local de TS-TR sem truncamento.
5. Adicao de metadados e diagnosticos de metricas.
6. Testes unitarios focados.
7. Execucao da suíte completa e baseline real-real.

## Itens nao implementados nesta etapa

- Nao foi reescrito o modo batches para streaming end-to-end.
- Nao foram alterados geradores.
- Nao foram alterados defaults globais legados `number_classes=2`.
- Nao foram adicionadas metricas multiclass de dominio fixo como novas chaves numericas; nesta etapa foi adicionado diagnostico de classes ausentes e chance-level.
- Nao foi alterado o contrato `valid` versus `test` de `split_mode=provided`.

## Proximas etapas recomendadas

1. Adicionar `evaluation_split=valid|test` para `split_mode=provided`.
2. Uniformizar quotas `synthetic_train_samples_per_class` e `synthetic_test_samples_per_class` entre normal e batches.
3. Adicionar metricas com dominio fixo `0..199` como chaves novas, sem substituir as metricas legadas.
4. Tornar sanity checks de sinteticos gates configuraveis para AppClassNet.
5. Evoluir batches para streaming real de X e predicoes por matriz de confusao acumulada.
