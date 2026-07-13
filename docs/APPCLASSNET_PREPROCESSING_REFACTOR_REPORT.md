# AppClassNet Preprocessing Refactor Report

## Arquivos modificados

- `Engine/Preprocessing/FeatureTransformManager.py`: novo módulo central com `FeatureTransformPolicy`, `FeatureTransformManager`, `TransformManifest`, `ScaleGuard` e `ModelInputAdapter`.
- `Engine/Preprocessing/__init__.py`: pacote de preprocessing.
- `Engine/DataIO/DatasetContracts.py`: `DatasetSchema` passou a registrar perfil, ranges, histórico, `transform_id`, dtype e `data_space`.
- `Engine/DataIO/NpyXYLoader.py`: adiciona metadados de perfil AppClassNet sem aplicar scaler.
- `Engine/DataIO/CSVLoader.py`: mantém CSV sem normalização e grava sidecar `.space.json` para sintéticos salvos no modo normal.
- `Engine/DataIO/XLSLoader.py`: remove `MinMaxScaler` interno; o loader preserva valores e deixa transformações para a camada central.
- `Engine/DataIO/SyntheticBatchIO.py`: manifesto de batches sintéticos registra `data_space`, `transform_id` e `transform_history`.
- `Engine/Arguments/ArgumentsDataLoader.py`: adiciona CLI central de política de transformação, preservando `legacy_csv` como default.
- `Engine/Arguments/Arguments.py`: traduz `--scaler` legado para política AppClassNet quando `main.py` é chamado diretamente.
- `Engine/Evaluation/CrossValidation.py`: encaminha `source_profile` para o loader `npy_xy`.
- `Engine/Evaluation/TrTs.py` e `Engine/Evaluation/TsTr.py`: bloqueiam avaliação normal e batch quando real/sintético estão em espaços incompatíveis.
- `main.py`: usa `ModelInputAdapter` para separar espaço do gerador e espaço de avaliação.
- `run_appclassnet_top200.py`: substitui scaler manual por `FeatureTransformManager`, muda o default AppClassNet para preservar escala e grava `preprocessing_manifest.json`.
- `tests/test_feature_transform_manager.py`: testes novos para contrato de escala, fit somente no treino, dupla transformação, manifesto e compatibilidade.

## Política anterior

- O runner AppClassNet usava `DEFAULT_SCALER = "minmax"`.
- O pré-processamento do AppClassNet materializava splits transformados por `--scaler`, com risco de tratar `[0,1]` como novo espaço de origem.
- O caminho baseline usava `scaler.fit_transform(train_x)` e `scaler.transform(test_x)`.
- Os sintéticos não tinham metadado explícito informando o espaço numérico salvo.
- Avaliações TS-TR/TR-TS não verificavam contrato de espaço real versus sintético.

## Política nova

- Para `source_profile=appclassnet_top200`, o default é preservar a escala pública aproximada `[-0.5, 0.5]`.
- `--scaler` continua existindo como argumento legado; `--scaler minmax` é tratado como solicitação explícita e emite aviso de depreciação.
- `FeatureTransformManager` ajusta transformadores somente no treino e reutiliza o mesmo objeto para train/valid/test.
- Transformações do gerador são separadas por `ModelInputAdapter`; sintéticos voltam ao espaço `source` quando `inverse_transform_synthetic` está habilitado.
- Manifests registram política, ranges, histórico, `transform_id`, fit split, espaços de entrada/saída e validações.
- Sintéticos em CSV e batches agora declaram explicitamente `data_space`.

## Bugs corrigidos

- AppClassNet era renormalizado por padrão para `[0,1]`, apesar dos arquivos públicos já estarem normalizados em torno de `[-0.5, 0.5]`.
- Não havia proteção formal contra aplicação repetida de um MinMax equivalente.
- O espaço numérico dos sintéticos não era persistido.
- Avaliações batch podiam consumir sintéticos em espaço diferente sem falhar cedo.
- Avaliações normais também dependiam do chamador externo para checar o espaço dos dados.
- O baseline real-real tinha lógica própria de scaler fora da camada central.
- `XLSLoader` ainda continha `MinMaxScaler.fit_transform` interno, violando o novo contrato de loaders puros.

## Compatibilidade

- `CSVLoader` não foi removido e continua sem aplicar scaler.
- Defaults do MalDataGen original permanecem `source_profile=legacy_csv`, `--scaler none`, `feature_transform preserve`.
- Comandos antigos com `--scaler` continuam aceitos.
- Loaders carregam dados e preservam valores; transformação ocorre no módulo central ou no adaptador do pipeline.
- O modo normal e o modo batches usam a mesma política e o mesmo contrato de manifesto.

## Comandos atualizados

AppClassNet preservando escala por padrão:

```bash
python run_appclassnet_top200.py --campaign adversarial_demo --skip_plots
```

Gerador com escala interna `[0,1]` e sintético salvo em source space:

```bash
python run_appclassnet_top200.py --campaign adversarial_demo --generator_transform minmax --inverse_transform_synthetic
```

Uso legado explícito:

```bash
python run_appclassnet_top200.py --campaign adversarial_demo --scaler minmax
```

## Testes executados

```bash
PYTHONPATH=. pytest -q
```

Resultado final: `93 passed`.

Validação focada executada durante a refatoração:

```bash
PYTHONPATH=. pytest -q tests/test_feature_transform_manager.py tests/test_dataset_contracts.py tests/test_data_loader_arguments.py tests/test_evaluation_label_shapes.py tests/test_batch_npy_dataset.py tests/test_synthetic_batch_io.py tests/test_npy_xy_loader.py
```

Resultado: `51 passed`.

Também foram executados:

```bash
python -m compileall Engine/DataIO Engine/Preprocessing Engine/Arguments Engine/Evaluation main.py run_appclassnet_top200.py
```

```bash
rg -n "MinMaxScaler|StandardScaler|fit_transform\(|scaler\.fit|_data_scaler = MinMaxScaler|_data_scaler = StandardScaler" Engine/DataIO Engine/Preprocessing main.py run_appclassnet_top200.py tests
```

A varredura confirmou scalers apenas em `Engine/Preprocessing/FeatureTransformManager.py` e nos testes que verificam o contrato de fit somente no treino.
