# Final Release Gate - MalDataGen

Data: 2026-07-15

Classificacao: **aprovado com ressalvas**

## Resultado

O gate final passou nos itens criticos: instalacao limpa apos correcao minima de empacotamento, imports pela raiz, coleta pytest, suite de testes, dryrun, AppClassNet TR-TR, controles `real_resample` e `label_permutation`, TS-TR com gerador, TR+TS-TR identificado como misto, 10 classes, 200 classes, batches, mmap, modo normal, CSV legado, manifests e reprodutibilidade por seed.

A aprovacao fica com ressalvas porque o gate encontrou e documentou limitacoes nao bloqueadoras: `copy` nao suporta geracao incremental em batches, CSV legado aceita indice inteiro de label e nao nome de coluna, e `main.py` direto com paths crus top200 exige `--num_classes 200` antes de qualquer subset.

## Checks De Base

- Instalacao limpa: passou apos adicionar descoberta explicita de pacotes em `pyproject.toml`.
- Imports a partir da raiz: passou.
- Pytest collect: 269 testes coletados.
- Testes: 251 passed, 18 skipped, 9 warnings, 5 subtests passed.
- Dryrun e dry_run_memory: passaram sem iniciar treinamento.

## Controles

- `real_resample` 10 classes: TR-TS accuracy 0.6333, TS-TR accuracy 0.6550, chance 0.1000.
- `label_permutation` 10 classes: TR-TS accuracy 0.0933, TS-TR accuracy 0.1450, chance 0.1000.
- `real_resample` 200 classes: TR-TS accuracy 0.2213, TS-TR accuracy 0.1892, chance 0.0050.
- TR-TR seed 0 reproduzivel: metricas identicas entre `real_resample_10` e `real_resample_10_repeat`.
- TR+TS-TR aparece com `train_sources=[real_train, synthetic_train]`, portanto fica explicitamente misto.

## Integridade

- Nenhum manifest avaliado aponta para arquivo inexistente.
- Nenhum resultado `completed` sem metricas.
- Nenhum uso de VALID como TEST encontrado nos logs do gate.
- `DecisionTreeSubset` registra `discarded_rows`, `effective_fit_rows` e parametros.
- `preserve/source` foi usado com `allow_double_transform=false`; nenhuma dupla transformacao foi observada.
- Erros controlados retornaram codigos nao-zero e mensagens explicitas.

## Arquivos De Apoio

- Matriz: `docs/audits/final_test_matrix.json`
- Ressalvas: `docs/audits/final_known_issues.json`

## Comandos

Os comandos exatos dos experimentos principais estao em `docs/audits/final_test_matrix.json`, campo `baseline_checks.*.command` e `cases.*.command`.
