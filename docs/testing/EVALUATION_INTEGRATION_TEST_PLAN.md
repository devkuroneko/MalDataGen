# Evaluation Integration Test Plan

Data: 2026-07-14.

Escopo: plano de testes de integracao para TR-TR, TR-TS e TS-TR no AppClassNet top-200, baseado nos relatorios de auditoria em `docs/audits/`. Esta etapa cria apenas o plano e esqueletos de testes. Nenhuma correcao funcional do pipeline deve ser feita junto com estes testes.

## Objetivos

1. Travar a semantica esperada das tres avaliacoes:
   - TR-TR: treinar em real independente e testar em real independente.
   - TR-TS: treinar em real e testar em sintetico no mesmo espaco numerico.
   - TS-TR: treinar em sintetico e testar em real independente.
2. Detectar regressao para nivel aleatorio quando as classes sao separaveis.
3. Detectar vazamento, reuso indevido, desalinhamento de labels, classes ausentes e divergencia de espaco numerico.
4. Comparar modo normal e modo batches sem exigir que batches alterem a semantica do experimento.
5. Preservar compatibilidade com o caminho CSV legado.

## Dataset controlado

Todos os testes devem usar datasets artificiais pequenos, gerados em memoria e salvos em NPY/CSV temporario quando o pipeline exigir arquivos.

Formato comum:

- `num_features=20`;
- labels inteiras zero-based;
- escala final em `[-0.5, 0.5]`;
- splits `train`, `valid`, `test` independentes;
- classes com centroides bem separados;
- ruído pequeno por feature;
- opcao de seed fixa para reproducibilidade.

Tamanhos propostos:

| Perfil | Classes | Uso |
|---|---:|---|
| `quick_3` | 3 | testes rapidos de semantica, labels, metricas e bloqueios |
| `medium_10` | 10 | comparacao normal vs batches e deteccao de primeiro bloco |
| `structural_200` | 200 | validacao estrutural de dominio `0..199`, one-hot e contagem de classes |

Contrato de geracao:

```text
for class_id in range(num_classes):
    centroid = deterministic_vector(class_id, num_features, range=[-0.4, 0.4])
    X_split[class_id] = centroid + noise(seed, split, class_id)
    clip X_split to [-0.5, 0.5]
    y_split[class_id] = class_id
```

O split `test` nunca deve compartilhar linhas com `train`. O split `valid` deve ser independente e usado apenas quando o contrato da avaliacao pedir validacao.

## Sinteticos controlados

As avaliacoes sinteticas devem usar tres familias de sintetico controlado:

1. `compatible_copy`: sintetico por classe amostrado em torno do mesmo centroide real, com seed diferente e sem reutilizar linhas exatas do real.
2. `preserves_classes`: sintetico com labels corretas e distribuicao separavel por classe, adequado para TS-TR.
3. `label_broken`: sintetico com X independente dos labels ou labels permutadas, esperado para cair ao acaso.

Todos devem carregar metadata minima:

- `data_space`;
- `transform_id`;
- `num_classes`;
- `num_features`;
- contagem por classe;
- seed;
- origem do split usado para gerar sintetico.

## Classificador de referencia

Usar `DecisionTreeClassifier` como classificador primario de integracao, alinhado ao baseline AppClassNet validado.

Regras:

- o classificador deve ser recriado a cada avaliacao;
- metricas calculadas sobre todas as predicoes;
- para testes de 3 e 10 classes, a acuracia esperada em dados separaveis deve ficar alta;
- para labels quebradas, a acuracia esperada deve ficar proxima de `1/num_classes`.

## Casos de teste

| ID | Nome | Perfil | Avaliacao | Resultado esperado |
|---|---|---|---|---|
| T01 | TR-TR aprende classes separaveis | `quick_3` | TR-TR | accuracy e macro-F1 altos; train/test independentes |
| T02 | TR-TR falha em labels embaralhados | `quick_3` | TR-TR | accuracy proxima de `1/3`; diagnostico de label shuffle |
| T03 | TR-TS funciona com sintetico compatível | `quick_3` | TR-TS | accuracy alta em sintetico compatível |
| T04 | TS-TR funciona quando sintetico preserva classes | `quick_3` | TS-TR | accuracy alta no real test |
| T05 | TR-TS e TS-TR caem ao acaso com labels quebradas | `quick_3` | TR-TS/TS-TR | accuracy proxima de `1/3` |
| T06 | Espacos diferentes bloqueiam avaliacao | `quick_3` | TR-TS/TS-TR | erro `PreprocessingSpaceMismatchError` ou equivalente documentado |
| T07 | Classes ausentes geram erro ou warning | `quick_3` | TR-TS/TS-TR | falha documentada ou warning com classes ausentes |
| T08 | Labels desalinhadas sao detectadas | `quick_3` | TR-TS/TS-TR | falha antes da metrica ou diagnostico explicito |
| T09 | Normal e batches produzem resultados proximos | `medium_10` | TR-TS/TS-TR | diferenca dentro de tolerancia |
| T10 | Batches nao descartam classes | `medium_10` | batches | todas as classes presentes em batches |
| T11 | Batches nao usam apenas o primeiro bloco | `medium_10` | batches | amostras de blocos posteriores afetam contagem/metricas |
| T12 | Scaler ajustado apenas no treino | `quick_3` | TR-TR/TR-TS/TS-TR | nenhum `fit` em valid/test/sintetico |
| T13 | Nenhuma transformacao ocorre duas vezes | `quick_3` | pipeline | `transform_history` nao duplica o mesmo transform |
| T14 | Metricas nao executadas ficam `not_applicable` | `quick_3` | evaluation_mode parcial | sem zeros falsos |
| T15 | 200 classes usam labels `0..199` corretamente | `structural_200` | estrutural | dominio completo preservado |
| T16 | Sintetico de treino nao e reutilizado como teste sintetico | `quick_3` | TR-TS/TS-TR | IDs/origem diferentes para synthetic train/test |
| T17 | Seeds produzem reproducibilidade | `quick_3` | pipeline | mesmas metricas e hashes com mesma seed |
| T18 | Comando antigo CSV continua funcionando | `quick_3` | legado CSV | fluxo CSV nao quebra |

## Esqueletos criados

Arquivo: `tests/test_evaluation_integration_spec.py`.

Estado inicial:

- todos os testes ficam marcados com `@unittest.skip`;
- cada teste contem o contrato, fixture alvo e assercoes esperadas;
- a classe inclui uma fabrica de dataset controlado para ser usada quando os testes forem habilitados.

## Criterios para habilitar

1. Habilitar primeiro T01, T02, T15 e T18, pois validam contrato estrutural e regressao basica.
2. Habilitar T03, T04 e T05 depois de formalizar a API de avaliacao TR-TS/TS-TR para usar splits independentes.
3. Habilitar T06, T12 e T13 junto com os guards de espaco e transformacao.
4. Habilitar T09, T10 e T11 somente apos definir equivalencia entre normal e batches.
5. Habilitar T16 quando synthetic train/test tiverem origem identificavel em metadata.
6. Habilitar T17 apos padronizar seed para amostragem, geracao e classificadores.

## Observacoes das auditorias

- `EVALUATION_FLOW_AUDIT.md`: TR-TS atual treina em `x_evaluation_real`; TS-TR normal pode truncar metricas pelo total sintetico; TR-TR principal nao roda no fluxo sintetico.
- `PREPROCESSING_AND_DATA_SPACE_AUDIT.md`: AppClassNet deve preservar `source` por default e nao aplicar scaler automatico.
- `BATCH_AND_MEMORY_AUDIT.md`: batches devem reduzir memoria sem descartar classes, usar apenas primeiro bloco ou alterar distribuicao.
- `METRICS_AUDIT.md`: avaliacoes nao executadas devem ser `not_applicable`, nao zero.
- `LABEL_AND_CONDITIONING_AUDIT.md`: dominio `0..199` precisa ser preservado em labels reais, one-hot e labels sinteticos.
- `SYNTHETIC_GENERATION_AUDIT.md`: sinteticos devem registrar `data_space`, classes presentes e sanidade de escala antes de TR-TS/TS-TR.

