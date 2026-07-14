# Evaluation Mode Dispatch Fix

## Escopo

Auditoria e correcao do despacho de `evaluation_mode` para o runner AppClassNet top-200 em batches, com foco nos modos TR-TS, TS-TR, `both`, baseline real-only e serializacao final.

## Ocorrencias Localizadas

| Termo | Arquivo | Funcao | Condicao atual auditada |
| --- | --- | --- | --- |
| `evaluation_mode` | `main.py:110-128` | `run_synthetic_evaluation_modes()` | Usa dois `if` independentes: TR-TS roda para `tr_ts` ou `both`; TS-TR roda para `ts_tr` ou `both`. |
| `evaluation_mode` | `run_appclassnet_top200.py:579-582` | `effective_evaluation_mode()` | Forca `none` quando `baseline_real_only=True`. |
| `evaluation_mode` | `run_appclassnet_top200.py:1390-1416` | `write_batches_metrics()` | O agregador final decide quais blocos devem estar completos usando `run_tr_ts = mode in {"tr_ts", "both"}` e `run_ts_tr = mode in {"ts_tr", "both"}`. |
| `status: not_run` | `run_appclassnet_top200.py:1255-1271` | `_empty_evaluation_summary()` | Placeholder padrao para avaliacoes nao solicitadas ou ainda nao agregadas. |
| `reason evaluation_mode=both` | `run_appclassnet_top200.py` | `write_batches_metrics()` | Antes da correcao, TR-TS e TS-TR eram inicializados como `not_run` com `reason=evaluation_mode=both`; se o JSON filho nao fosse extraido, o placeholder sobrevivia silenciosamente. |
| `Synthetic run does not execute TR-TR` | `run_appclassnet_top200.py:1390` | `write_batches_metrics()` | Fluxo sintetico continua marcando TR-TR como `not_run`; TR-TR so roda no caminho separado `--baseline_real_only` ou se solicitado no `main.py` via `run_tr_tr`. |
| Impressao/serializacao | `run_appclassnet_top200.py:1343-1344` | `print_results_payload()` | Responsavel unico por imprimir o JSON consolidado final. |
| Impressao/serializacao | `run_appclassnet_top200.py:2343-2348` e `2510-2519` | `main()` | Baseline e batches chamam `print_results_payload()` uma vez depois de escrever o payload. |
| Impressao/serializacao | `main.py:510-511` | `SynDataGen.run_experiments()` | Salvamento intermediario do `Results.json` so ocorre quando ainda ha folds pendentes; o salvamento final permanece apos `update_mean_std_fold()`. |

## Por Que `both` Nao Executava no Resultado Final

O despacho real dentro do subprocesso `main.py` ja era correto: `run_synthetic_evaluation_modes()` chamava TR-TS e TS-TR com dois blocos `if` independentes. O problema observado vinha do agregador AppClassNet em batches:

1. `write_batches_metrics()` inicializava TR-TS e TS-TR como `not_run` usando `reason=evaluation_mode=both`.
2. O payload so era substituido se `_extract_evaluation_metrics()` encontrasse metricas no `EvaluationResults/Results.json` do subprocesso.
3. Se o resultado filho estivesse ausente, incompleto, com metricas nao extraiveis ou sem metadata batch, o placeholder continuava no JSON final.
4. Nao havia fail-fast para avaliacoes solicitadas; por isso o comando podia terminar com `status: not_run` e metricas `null`.

## Por Que Havia Impressao Duplicada

Havia dois problemas de responsabilidade:

1. `run_real_real_baseline()` imprimia suas metricas diretamente enquanto o runner tambem precisava consolidar/escrever metricas.
2. `main.py::SynDataGen.run_experiments()` salvava `Results.json` dentro do loop de folds e novamente depois de `update_mean_std_fold()`. Em execucoes de um fold isso repetia a emissao de salvamento do mesmo resultado.

A correcao removeu a impressao direta do baseline e definiu `run_appclassnet_top200.py::print_results_payload()` como unico ponto de impressao do JSON final consolidado. O salvamento intermediario no `main.py` agora e pulado quando nao ha folds restantes.

## Confirmacao do Fluxo Atual

No fluxo sintetico de batches:

1. O subprocesso `main.py` carrega dados reais e prepara o fold.
2. O gerador global e treinado, salvo quando `generation_strategy` usa geradores particionados; nesse caso os subgeradores sao treinados durante a geracao.
3. `synthesize_data()` gera um artefato sintetico por fold. Em batches sem materializacao, ele retorna um `SyntheticBatchReader` baseado em `synthetic_batches/manifest.json`.
4. O mesmo objeto sintetico e passado para `run_synthetic_evaluation_modes()`.
5. TR-TS treina classificador em dados reais e prediz nos sinteticos.
6. TS-TR treina classificador nos sinteticos e prediz nos dados reais.
7. TR-TR nao roda no fluxo sintetico padrao.

Antes da correcao, quando o agregador nao encontrava metricas completas, ele apenas deixava placeholders. Agora avaliacoes solicitadas precisam terminar com `status="completed"` ou o runner levanta `RuntimeError`.

## Pre-condicoes Adicionadas

As validacoes batch foram centralizadas em `Engine/Classifiers/BatchClassifiers.py`:

- X/y reais presentes e alinhados.
- Dados sinteticos presentes.
- Numero de classes esperado.
- Quantidade minima solicitada por classe.
- Mesmo numero de features.
- `data_space` sintetico compativel com `source`.
- Ausencia de NaN e inf.

TR-TS chama essas validacoes antes do `fit` real e antes do `predict` sintetico. TS-TR chama essas validacoes antes do `fit` sintetico e antes do `predict` real.

## Comportamento Corrigido

| Modo | TR-TR | TR-TS | TS-TR |
| --- | --- | --- | --- |
| `evaluation_mode=none` | `not_run` | `not_run` | `not_run` |
| `evaluation_mode=tr_ts` | `not_run` | `completed` ou erro | `not_run` |
| `evaluation_mode=ts_tr` | `not_run` | `not_run` | `completed` ou erro |
| `evaluation_mode=both` | `not_run` | `completed` ou erro | `completed` ou erro |
| `baseline_real_only` | `completed` | `not_run` | `not_run` |

Metricas `null` permanecem apenas em avaliacoes realmente nao solicitadas.

## Testes

Foram adicionados testes em `tests/test_appclassnet_evaluation_mode.py` cobrindo:

- `test_evaluation_mode_none`
- `test_evaluation_mode_tr_ts`
- `test_evaluation_mode_ts_tr`
- `test_evaluation_mode_both`
- `test_baseline_real_only`
- `test_requested_evaluation_cannot_finish_not_run`
- `test_result_is_serialized_once`
- `test_both_reuses_synthetic_artifacts`
- `test_missing_synthetic_train_fails_ts_tr`
- `test_missing_synthetic_test_fails_tr_ts`

Dataset/fixture: 3 classes, features separaveis e artefatos sinteticos pequenos em `.npy`.
