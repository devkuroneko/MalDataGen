# Real Class Count CLI Implementation

## Escopo

Implementacao integral dos argumentos:

- `--real_class_count_policy {strict,uniform_min,available_cap}`
- `--samples_per_class_scope {split,fold}`

Os argumentos foram adicionados ao wrapper `run_appclassnet_top200.py` e ao parser interno usado por `main.py`.

## Defaults e Precedencia

No wrapper AppClassNet:

- `real_class_count_policy`: `default=None`
- `samples_per_class_scope`: `default=None`

Isso preserva a precedencia:

```text
CLI explicito > campanha > padrao AppClassNet
```

Defaults efetivos AppClassNet:

```text
real_class_count_policy=strict
samples_per_class_scope=split
```

O wrapper resolve os valores efetivos por combinacao de campanha e repassa ao `main.py` exatamente uma vez. Os parametros tambem foram adicionados a `GENERATION_QUOTA_PARAMETERS`, impedindo que valores vindos de campanha sejam anexados novamente no fim do comando.

## Politicas

Implementado em `Engine/DataIO/RealClassCountPolicy.py`.

### strict

Exige a quantidade solicitada por classe. Se uma classe tiver menos amostras do que o pedido, gera erro claro:

```text
<split> split has fewer than requested <N> samples per class with real_class_count_policy=strict: {...}
```

### uniform_min

Usa a mesma quantidade para todas as classes:

```text
min(solicitado, menor quantidade disponivel entre classes)
```

### available_cap

Usa cota individual por classe:

```text
min(solicitado, disponivel_na_classe)
```

Nenhuma politica usa duplicacao ou `replace=True`.

## Roteamento dos Splits

Mantido o contrato corrigido para `split_mode=provided`:

- `train`: treino real;
- `valid`: validacao interna;
- `test`: avaliacao final;
- TS-TR usa `test`, nao `valid`;
- `test` externo nao e dividido novamente por padrao.

A politica e aplicada somente depois da escolha do split correto:

- `CrossValidation._apply_stratified_split_selection()` recebe `split_name` ja resolvido;
- TS-TR registra e aplica a politica sobre `real_test_data`;
- baseline real-only seleciona diretamente `train` e `test`.

## Logs

Foram adicionados logs com:

- `real_test_split`
- `requested_samples_per_class`
- `minimum_available_per_class`
- `effective_samples_per_class`
- `real_class_count_policy`
- `samples_per_class_scope`

Exemplo esperado no TS-TR:

```text
real_test_split=test
requested_samples_per_class=500
minimum_available_per_class=1539
effective_samples_per_class=500
real_class_count_policy=strict
samples_per_class_scope=split
```

## Verificacoes

Comandos executados:

```bash
python3 run_appclassnet_top200.py -h
python3 main.py -h
python3 run_appclassnet_top200.py --dryrun --execution_mode batches --evaluation_mode none --real_class_count_policy strict --samples_per_class_scope split --campaign adversarial_demo --skip_plots
pytest -q
```

Resultados:

```text
run_appclassnet_top200.py -h lista --real_class_count_policy e --samples_per_class_scope
main.py -h lista --real_class_count_policy e --samples_per_class_scope
dry-run aceita os argumentos e propaga ambos uma unica vez ao main.py
165 passed, 18 skipped, 2 warnings, 5 subtests passed
```

Observacao: `main.py -h` emite logs de importacao do TensorFlow/Matplotlib no ambiente atual, mas retorna codigo 0 e lista os argumentos.
