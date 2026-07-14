# CLI Sample Parameter Precedence

## Escopo

Este ajuste refatora o runner `run_appclassnet_top200.py` para resolver parametros de amostragem com precedencia explicita:

1. argumento informado na CLI;
2. configuracao da campanha;
3. padrao global do runner.

O comportamento original do `main.py` foi preservado: ele continua recebendo o argumento legado `--number_samples_per_class`.

## Parametros Cobertos

- `train_samples_per_class`
- `test_samples_per_class`
- `synthetic_train_samples_per_class`
- `synthetic_test_samples_per_class`
- `generated_samples_per_class`

No `argparse` do runner, esses parametros usam `default=None`. Isso permite distinguir valor omitido de valor informado pelo usuario.

## Resolucao Central

Foi criada a funcao:

```python
resolve_argument(cli_value, campaign_value, default_value)
```

Ela retorna o valor efetivo e a origem (`cli`, `campaign` ou `default`) usando a ordem:

```text
CLI > campanha > padrao
```

O runner constroi um plano central em `resolve_effective_sample_arguments()`, registra a origem de cada parametro e usa esse plano para serializar o subprocesso.

## Plano De Geracao

O total necessario por classe e calculado como:

```text
required_generated_per_class =
    synthetic_train_samples_per_class
    + synthetic_test_samples_per_class
```

Quando `generated_samples_per_class` nao e informado pela CLI nem pela campanha, o runner usa automaticamente `required_generated_per_class`.

Quando `generated_samples_per_class` e menor que o necessario, o runner interrompe antes de chamar o subprocesso com:

```text
InsufficientGeneratedSamplesPerClass
```

## Argumento Legado

O runner constroi automaticamente:

```text
--number_samples_per_class 0:N,1:N,...,199:N
```

onde `N = generated_samples_per_class`.

Assim, o caminho AppClassNet nao usa mais o limite fixo `256` para o plano legado.

## Validacoes

- Todos os valores efetivos precisam ser inteiros maiores ou iguais a zero.
- `evaluation_mode` com `TS-TR` exige `synthetic_train_samples_per_class > 0`.
- `evaluation_mode` com `TR-TS` exige `synthetic_test_samples_per_class > 0`.
- O plano de geracao precisa cobrir o total necessario por classe.
- A montagem dos comandos passa por deteccao de argumentos duplicados e conflitantes antes de `subprocess.run`.

## Compatibilidade

- `--number_samples_per_class` continua sendo enviado ao `main.py`.
- O fluxo CSV original continua coberto pelos testes de `ArgumentsDataLoader`.
- As configuracoes de campanha nao foram removidas; parametros de quota sao resolvidos pela nova precedencia.
- O runner continua ignorando `number_samples_per_class` da campanha AppClassNet ao montar o plano sintetico, pois o valor legado agora e derivado de `generated_samples_per_class`.

## Testes

Comando executado:

```bash
pytest
```

Resultado:

```text
142 passed, 18 skipped, 2 warnings
```

Coberturas adicionadas ou reforcadas:

- nenhum argumento usa defaults efetivos;
- campanha sobrescreve defaults;
- CLI sobrescreve campanha;
- `50/50`, `100/100`, `200/200` e `500/500` geram `100`, `200`, `400` e `1000` por classe;
- `generated_samples_per_class` suficiente e respeitado;
- `generated_samples_per_class` insuficiente falha antes do subprocesso;
- `train_samples_per_class=1000` chega como `1000`;
- comandos batch nao contem argumentos duplicados;
- plano legado `0:N,...,199:N` e construido corretamente;
- parser do runner preserva `default=None` para parametros configuraveis;
- CSV legado continua passando nos testes existentes.
