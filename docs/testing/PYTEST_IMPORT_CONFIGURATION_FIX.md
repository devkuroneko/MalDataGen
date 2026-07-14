# Pytest Import Configuration Fix

Data: 2026-07-13

## Problema

`pytest -q` interrompia a coleta com erros de importacao:

- `ModuleNotFoundError: No module named 'Engine'`
- `ModuleNotFoundError: No module named 'main'`

Nenhum teste era executado quando a raiz do repositorio nao estava no caminho de imports do Python.

## Inspecao

- `Engine/` existe na raiz do repositorio.
- `main.py` existe na raiz do repositorio.
- `Engine/Classifiers/__init__.py` e `Engine/Preprocessing/__init__.py` existem; varios outros diretorios sob `Engine/` funcionam como namespace packages quando a raiz esta no `sys.path`.
- Nao havia `tests/conftest.py`.
- `pyproject.toml` nao tinha `[tool.pytest.ini_options]`.
- Nenhum `--import-mode` customizado estava configurado.

## Correcao

Configuracao centralizada adicionada em `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Nao foi necessario criar `tests/conftest.py`.

## Como executar

Execute os testes a partir da raiz do repositorio com:

```bash
python -m pytest -q
```

Isso garante que o pytest use o interpretador Python selecionado para o projeto e leia a configuracao do `pyproject.toml`.

## Verificacao

Comando:

```bash
python -c "import Engine"
```

Resultado: passou.

Comando:

```bash
python -c "import main"
```

Resultado: passou. O ambiente emitiu avisos de TensorFlow/Matplotlib, mas sem erro de importacao.

Comando:

```bash
python -m pytest --collect-only -q
```

Resultado: `125 tests collected in 5.51s`.

Comando:

```bash
python -m pytest -q
```

Resultado: `107 passed, 18 skipped in 6.13s`.

## Conclusao

Os erros `ModuleNotFoundError` de configuracao desapareceram. A coleta nao retorna zero testes e a suite completa passa no ambiente atual.
