# Synthetic Generation Audit

Data da auditoria: 2026-07-14.

Escopo: auditoria da qualidade e da implementacao da geracao sintetica usada pela campanha `sf` do AppClassNet top-200. Nenhum arquivo de codigo foi modificado nesta etapa.

Contexto operacional assumido para AppClassNet:

- 20 features numericas.
- 200 classes, labels esperadas `0..199`.
- dados reais em escala original aproximada `[-0.5, 0.5]`.
- espaco padrao de avaliacao: `source`.
- loaders nao devem normalizar automaticamente.
- dados reais e sinteticos so podem ser comparados no mesmo espaco numerico.

## Resumo executivo

Achado critico: a campanha `sf` tem dois significados no runner. Em `run_appclassnet_top200.py`, `choose_campaigns()` retorna apenas `["variational_demo", "adversarial_demo"]` quando o token e `sf` sem `--full`; com `--full`, retorna `DEFAULT_CAMPAIGN` com `wasserstein`, `wasserstein_gp`, `variational`, `autoencoder`, `adversarial`, `latent_diffusion`, `denoising_diffusion`, `ctgan`, `tvae`, `copula` (linhas 58-70 e 360-368). Qualquer relatorio de campanha `sf` precisa registrar se foi demo ou full. A campanha `quantized` existe em `campaigns_available`, mas nao entra em `sf --full`.

Achado alto: os defaults legados de varios geradores ainda sao binarios (`number_classes=2`, `sigmoid`, perdas binarios) em arquivos de argumentos. A campanha AppClassNet sobrescreve a maior parte desses parametros para 200 classes e saida `linear`, mas execucoes diretas de `main.py` ou campanhas incompletas podem voltar ao comportamento binario. Evidencias: `ArgumentsAutoencoder.py::add_argument_autoencoder` linhas 49-53 e 99-101; `ArgumentsVariationalAutoencoder.py::add_argument_variation_autoencoder` linhas 41-49 e 95-100; `ArgumentsWassersteinGAN.py::add_argument_wasserstein_gan` linhas 40-46 e 110-112; `ArgumentsWassersteinGANGP.py::add_argument_wasserstein_gan_gp` linhas 40-46 e 111-113; `ArgumentsQuantizedVAE.py::add_argument_quantized_vae` linhas 42-47 e 104-106.

Achado alto: `--data_type continuous` e essencial. Todos os geradores neurais auditados arredondam com `numpy.rint` quando `number_samples_per_class["data_type"] != "continuous"`. Para AppClassNet isso destruiria features continuas em `[-0.5, 0.5]`. O runner declara `DEFAULT_DATA_TYPE="continuous"` e encaminha `--data_type` em modo normal e batches (`run_appclassnet_top200.py` linhas 9, 1556-1574, 1628-1658, 2194-2198). Exemplos de arredondamento: `AdversarialAlgorithm.get_samples()` linhas 356-358; `AutoencoderAlgorithm.get_samples()` linhas 306-309; `AlgorithmVariationalAutoencoder.get_samples()` linhas 331-334; `AlgorithmWassersteinGAN.get_samples()` linhas 220-223; `AlgorithmWassersteinGANGP.get_samples()` linhas 355-358; `AlgorithmLatentDiffusion.get_samples()` linhas 437-439; `AlgorithmDenoisingDiffusion.get_samples()` linhas 445-448.

Achado alto: parametros de learning rate configurados para Wasserstein no runner nao sao os usados no treino. `run_appclassnet_top200.py` configura `wasserstein_optimizer_*_learning_rate=0.001` e `wasserstein_gp_optimizer_*_learning_rate=0.001` (linhas 240-243 e 266-269), mas `GenerativeModels._training_wasserstein_model()` e `_training_wasserstein_gp_model()` instanciam Adam com `learning_rate=0.0002` fixo (linhas 2116-2117 e 2554-2555). Isso afeta reprodutibilidade e interpretacao dos resultados da campanha.

Achado alto: SDV (`copula`, `ctgan`, `tvae`) materializa um `DataFrame` real completo com features e coluna `class`, usa metadata detectada automaticamente e gera amostras condicionadas por `Condition`. A implementacao ainda chama `sample(num_rows=number_instances*5)` dentro do loop e descarta o resultado, depois usa `sample_from_conditions()` (linhas 168-178). Se a condicao nao produzir linhas para uma classe, a classe e omitida do `generated_data` (linhas 200-226). A auditoria de labels deve detectar ausencia de classes em `number_classes=200`, mas a qualidade do synthetic set pode ficar incompleta.

Achado medio: as difusoes usam clipping na amostragem reversa (`clip_denoised=True`) e a campanha configura limites `[-0.5, 0.5]` (`run_appclassnet_top200.py` linhas 290-294 e 322-326; `AlgorithmLatentDiffusion.generate_data()` linhas 304-310; `AlgorithmDenoisingDiffusion.generate_data()` linhas 313-319). Isso e coerente com AppClassNet, mas pode mascarar erro de escala e reduzir variancia.

Achado medio: nao ha seed global evidente para treino e amostragem dos geradores neurais. O codigo usa `tensorflow.random.normal` no treino e `numpy.random.normal`/`numpy.random.choice` na geracao sem `np.random.seed`, `tensorflow.random.set_seed` ou `default_rng` local nos algoritmos auditados. Isso nao indica repeticao de seeds; indica nao determinismo entre execucoes. Evidencia negativa por busca em `Engine/Algorithms`, `Engine/Models`, `main.py` e `run_appclassnet_top200.py`: seeds aparecem em selecao/avaliacao e manifestos, nao em geracao neural.

Achado medio: os sanity checks existentes cobrem contagem de classes, range real/sintetico, distancia de medias por classe e um DecisionTree real-para-sintetico, mas ainda nao cobrem KS, Wasserstein, correlacao, nearest-neighbor, duplicacao, entropia de predicoes ou diversidade intra-classe. Evidencias: `SyntheticSanityChecks.py::SyntheticSanityChecker.run()` linhas 80-106, `_scale_report()` linhas 224-239, `_class_mean_distances()` linhas 241-275, `_classifier_report()` linhas 277-303.

## Campanha sf

Arquivo: `run_appclassnet_top200.py`.

Funcao: `choose_campaigns()`.

Entradas:

- `--campaign sf`
- `--full` opcional

Saidas:

- `sf` sem `--full`: `variational_demo`, `adversarial_demo`.
- `sf --full`: `wasserstein`, `wasserstein_gp`, `variational`, `autoencoder`, `adversarial`, `latent_diffusion`, `denoising_diffusion`, `ctgan`, `tvae`, `copula`.

Risco alto: resultados chamados de campanha `sf` sao ambiguos se o log nao registrar `--full`.

## Tabela operacional por algoritmo

| Algoritmo em `sf` | Arquitetura | Loss principal | Ativacao de saida em AppClassNet | Classes/condicao | Epocas e batch | Quantidade real usada | Quantidade sintetica | Estrategia de geracao | Escala e salvamento |
|---|---|---|---|---|---|---|---|---|---|
| `adversarial_demo` | cGAN denso condicional | `BinaryCrossentropy` no treino adversarial | `linear` | one-hot `200`; label concatena em generator e discriminator | 1 epoca, batch 128 | `len(x_real_samples)` recebido em `SynDataGen.train_model()` | 256 por classe por default | loop por classe, ruido normal, chunks por `generation_batch_size` | source por default; salvo via `CSVDataProcessor.save_csv()` ou `SyntheticBatchWriter` |
| `variational_demo` | VAE condicional denso | `mse` configuravel no ponto de entrada, com objetivo VAE interno | `linear` | one-hot `200`; decoder usa embedding de label de 8 unidades | 1 epoca, batch 128 | `len(x_real_samples)` recebido em `SynDataGen.train_model()` | 256 por classe por default | loop por classe, ruido normal, decoder condicional | source por default; inverse para source se adapter existir antes de salvar |
| `adversarial` | cGAN denso condicional | `BinaryCrossentropy` | `linear` | one-hot `200`; `number_classes` vem do plano | 300 epocas, batch 128 | dados reais do fold/split de treino | 256 por classe por default | igual ao demo, com maior capacidade/epocas | source por default; metadata de espaco gravada quando aplicavel |
| `variational` | VAE condicional denso | `mse` configuravel no compile | `linear` | one-hot `200`; decoder condicionado por label | 300 epocas, batch 128 | dados reais do fold/split de treino | 256 por classe por default | decoder condicionado por classe solicitada | source por default; arredonda somente se `data_type != continuous` |
| `autoencoder` | autoencoder condicional denso | `mse` | `linear` | one-hot `200`; decoder concatena label | 300 epocas, batch 256 | dados reais do fold/split de treino | 256 por classe por default | amostra latente normal e decodifica por classe | source por default; risco de latente normal fora da manifold |
| `wasserstein` | WGAN condicional denso | Wasserstein critic/generator loss | `linear` | one-hot `200` | 300 epocas, batch 128 | dados reais do fold/split de treino | 256 por classe por default | loop por classe, ruido normal, generator condicional | source por default; learning rate da campanha nao e usado no treino |
| `wasserstein_gp` | WGAN-GP condicional denso | Wasserstein + gradient penalty | `linear` | one-hot `200` | 300 epocas, batch 128 | dados reais do fold/split de treino | 256 por classe por default | loop por classe, ruido normal, generator condicional | source por default; learning rate da campanha nao e usado no treino |
| `latent_diffusion` | autoencoder condicional + U-Net de difusao latente | VAE `mse` + MSE do ruido predito | U-Net `linear`, decoder `linear` | one-hot `200`; U-Net usa embedding/attention | 300 epocas AE + 300 epocas U-Net, batch 128 | dados reais do fold/split de treino | 256 por classe por default | ruido latente, reverse diffusion, decoder condicional | clipping `[-0.5, 0.5]`; source por default |
| `denoising_diffusion` | U-Net de difusao no espaco de features | MSE do ruido predito | U-Net `linear` | one-hot `200` | 300 epocas, batch 128 | dados reais do fold/split de treino | 256 por classe por default | ruido em features, reverse diffusion por classe | clipping `[-0.5, 0.5]`; source por default |
| `copula`/`ctgan`/`tvae` | SDV `GaussianCopulaSynthesizer`, `CTGANSynthesizer`, `TVAESynthesizer` | defaults da biblioteca SDV | nao aplicavel no codigo do projeto | coluna `class`, nao one-hot | nao configurado pela campanha | DataFrame real completo recebido no wrapper SDV | 256 por classe solicitado via `Condition` | `sample_from_conditions()` por classe e filtro posterior | aprende escala do DataFrame; salva dict por classe se classe retornou amostras |

## Contrato comum de geracao

### Planejamento de amostras

Arquivo: `run_appclassnet_top200.py`.

Funcao: `_base_campaign()`.

Evidencia: `SAMPLES_TOP200` usa `class_id:DEFAULT_SAMPLES_PER_CLASS` para `class_id in range(200)`, e `_base_campaign()` passa esse plano como `number_samples_per_class` (linhas 77, 86-92). O default e `256` sinteticos por classe, total `51.200` linhas, salvo override.

Arquivo: `Engine/DataIO/SamplePlanner.py`.

Funcoes: `build_sample_plan_from_args()`, `sample_plan_to_legacy_metadata()`.

Evidencias:

- modo `legacy` usa contagens explicitas se `number_samples_per_class` foi passado (linhas 36-40);
- para `data_format=npy_xy`, o modo legacy sem plano explicito falha e pede plano explicito (linhas 42-47);
- `sample_plan_to_legacy_metadata()` gera dict com `classes`, `number_classes`, `data_type`, `sample_plan` (linhas 92-103);
- `_validate_counts()` e `_class_domain()` preservam dominio `0..number_classes-1` quando `number_classes` e conhecido (linhas 146-149).

Shapes esperados:

- `X_real`: `(n, 20)`, `float32`/float numerico.
- `y_real`: `(n,)`, inteiros `0..199`.
- `number_samples_per_class["classes"]`: `{0: 256, ..., 199: 256}` por default da campanha.
- `number_samples_per_class["number_classes"]`: `200`.

Risco medio: em modo `legacy` CSV, se nao houver plano explicito, a distribuicao de treino pode ser usada como plano; para AppClassNet NPY, o codigo exige plano explicito.

### Labels e one-hot

Arquivo: `Engine/DataIO/LabelUtils.py`.

Funcoes: `one_hot_encode_labels()`, `to_one_hot_batch()`.

Evidencias:

- labels sao convertidas para inteiro e validadas como zero-based (linhas 14-34 e 74-95);
- one-hot tem shape `(n, num_classes)` e dtype `float32` (linhas 98-103);
- batch one-hot valida labels em `[0, num_classes-1]` (linhas 106-128).

Arquivo: `Engine/Models/GenerativeModels.py`.

Funcoes: `_labels_to_one_hot()`, `_fit_features_and_one_hot()`, `_fit_autoencode_with_one_hot()`.

Evidencias:

- treino usa `number_samples_per_class["number_classes"]` para one-hot (linhas 97-102);
- modo batches cria `TensorSpec(shape=(None, num_classes), dtype=float32)` (linhas 123-153);
- modelos GAN/difusao treinam com `(x_real, y_one_hot)` ou `x_real, y_one_hot` (linhas 156-174);
- autoencoders condicionais treinam com entrada `(x_real, y_one_hot)` e target `x_real` (linhas 177-195).

Risco baixo no contrato: `num_classes=200` e respeitado quando o plano contem `number_classes=200`.

Risco medio: `main._get_configured_number_classes()` tambem consulta argumentos especificos de modelo e `max(label)+1`/`unique(labels)` (linhas 532-558). Para classes ausentes em um fold, o maximo dos candidatos preserva dominio se algum argumento/plano tiver 200, mas execucoes parciais sem esse metadado podem reduzir o dominio.

### Treino generativo

Arquivo: `main.py`.

Classe: `SynDataGen`.

Funcao: `train_model()`.

Entradas:

- `x_real_samples`: dados reais do fold/split de treino recebido pelo pipeline.
- `y_real_samples`: labels reais correspondentes.
- `monitor_path`, `k_fold`.

Saidas:

- modelo treinado armazenado no atributo do algoritmo selecionado.

Evidencias:

- loga numero de amostras reais e `model_type` (linhas 732-738);
- SDV recebe `x_real_samples`, `y_real_samples`, headers e algoritmo (linhas 743-753);
- demais modelos convertem labels por `_prepare_labels_for_conditional_generation()` e chamam `training_model()` (linhas 755-761).

Quantidade real usada no treino:

- modo normal: exatamente `len(x_real_samples)` recebido pelo fold/split do pipeline.
- campanha full normal: `_base_campaign()` define `number_k_folds=5`; demos definem `number_k_folds=2` (linhas 86-92, 120-123, 180-183).
- modo batches com `split_mode=provided`: runner passa `train_x/train_y`, `valid_x/valid_y`, `test_x/test_y`, `num_classes=200` e `execution_mode=batches` (linhas 1628-1664).
- estrategias `per_class`/`grouped_classes`: cada gerador e treinado com subset real das classes da unidade (`_subset_by_classes`) e registra `training_rows` (main.py linhas 1067-1106 e 1135-1147).

Risco medio: `per_class`/`grouped_classes` reduzem memoria, mas alteram a semantica estatistica do treino porque cada modelo ve apenas uma classe ou grupo.

### Geracao e salvamento

Arquivo: `main.py`.

Classe: `SynDataGen`.

Funcao: `synthesize_data()`.

Entradas:

- `x_real_samples`, `y_real_samples` usados para construir o plano e sanity checks.
- modelo treinado previamente.

Saidas:

- modo normal materializado: `dict[int, ndarray]` com uma matriz por classe.
- modo incremental: `SyntheticBatchReader` apontando para manifestos/batches em disco.

Evidencias:

- cria `number_samples_per_class = self._build_generation_metadata(y_real_samples)` (linha 821);
- escolhe algoritmo por `model_type` e chama `get_samples(number_samples_per_class)` (linhas 855-941);
- se houver `ModelInputAdapter`, transforma sinteticos para source e registra metadata de `data_space`, `transform_id`, `transform_history` (linhas 951-963);
- executa `audit_synthetic_label_generation()` (linhas 965-974);
- executa `run_synthetic_sanity_checks()` (linhas 976-987);
- salva quando `save_data=True` (linhas 998-1000).

Arquivo: `Engine/DataIO/CSVLoader.py`.

Classe: `CSVDataProcessor`.

Funcao: `save_csv()`.

Evidencias:

- materializa `labels` e `data` em listas, cria `DataFrame` e salva `DataOutput_K_fold_{fold}_{generator}.txt` (linhas 391-424);
- salva metadata `.space.json` com `data_space`, `transform_id`, `transform_history` (linhas 426-447).

Arquivo: `Engine/DataIO/SyntheticBatchIO.py`.

Classe: `SyntheticBatchWriter`.

Funcoes: `write_batch()`, `close()`.

Evidencias:

- valida shape `(n, num_features)` (linhas 71-76);
- salva `npy_batches`, `csv_batches` ou `single_npy` (linhas 82-113);
- registra `batches_by_class`, `total_rows`, `data_space`, `transform_id`, `transform_history` no manifest (linhas 47-59, 115-130).

Risco medio: modo normal materializa todos os sinteticos em memoria antes do CSV; para 51.200 x 20 isso e pequeno, mas para campanhas maiores cresce rapidamente.

## Algoritmos usados em sf

### `adversarial` e `adversarial_demo`

Severidade geral: medio.

Arquitetura:

- cGAN denso condicional.
- `VanillaGenerator.get_generator()` cria input latente e `label_input` com shape `(number_classes,)`, concatena latente+label, aplica Dense para latent dimension e passa pelo dense generator (linhas 187-213).
- `VanillaDiscriminator.get_discriminator()` tambem recebe `label_input` com shape `(number_classes,)` (linhas 222-225).

Loss:

- treino compila `BinaryCrossentropy()` para gerador e discriminador em `GenerativeModels._training_adversarial_modelo()` (linhas 347-354).
- campanha configura `adversarial_loss_generator="binary_crossentropy"` e `adversarial_loss_discriminator="binary_crossentropy"`, mas o codigo de treino usa `BinaryCrossentropy()` diretamente.

Ativacao de saida:

- default legado: `Sigmoid` em `ArgumentsAdversarial.py` linha 38.
- campanha `sf`: `linear` em `run_appclassnet_top200.py` linhas 118 e 139.

Numero de classes:

- nao ha argumento `adversarial_number_classes`.
- dimensao de classes vem de `number_samples_per_class["number_classes"]`, passado para `AdversarialModel` em `GenerativeModels._get_adversarial_model()` linhas 295-306.

Entrada condicional e labels:

- treino usa one-hot via `_fit_features_and_one_hot()` (linhas 363-372).
- geracao usa `to_one_hot_batch([label_class] * batch_instances, num_classes=number_samples_per_class["number_classes"])` em `AdversarialAlgorithm.get_samples()` linhas 341-344.

Epocas e batch size:

- full: 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 102-104).
- demo: 1 epoca, batch 128 (`run_appclassnet_top200.py` linhas 122-125).

Stopping:

- `use_early_stop=False` por default (`ArgumentsEarlyStop.py` linhas 33-45).
- callback de early stop so entra se `arguments.use_early_stop` for verdadeiro (`GenerativeModels.py` linhas 356-360).

Quantidade real usada:

- `len(x_real_samples)` do fold/split recebido por `main.train_model()` (main.py linhas 732-761).

Quantidade sintetica:

- default `256` por classe, 200 classes, total `51.200`, salvo override (`run_appclassnet_top200.py` linhas 77, 86-92).

Estrategia de geracao:

- loop por classe; chunks por `generation_batch_size`; ruído normal `numpy.random.normal(mean, std, (batch_instances, latent_dimension))`; `generator.predict([latent_noise, label_one_hot])`; concatena chunks; arredonda se nao-continuo (`AdversarialAlgorithm.get_samples()` linhas 335-358).

Escala de entrada/saida:

- entrada: source por default do runner (`feature_transform preserve`, `generator_transform` conforme politica; `run_appclassnet_top200.py` linhas 1588-1601 e 1683-1693).
- saida: linear no espaco do gerador; se houver adapter, `main.synthesize_data()` aplica transformacao para source (linhas 951-963).

Riscos:

- medio: demo com 1 epoca quase certamente nao mede qualidade sintetica final.
- medio: sem seed global, resultados nao sao deterministicos.
- medio: condicionamento existe, mas o modelo pode ignorar o one-hot; precisa de diagnostico por predicoes/classes.
- baixo: labels sinteticos sao definidos pela classe solicitada no loop, nao por uma saida do gerador.

### `variational` e `variational_demo`

Severidade geral: medio.

Arquitetura:

- VAE condicional denso.
- decoder recebe latente e `label_input` `(number_classes,)`; aplica embedding denso de 8 unidades no label e concatena com latente (`VanillaDecoder.get_decoder_model()` linhas 228-249).

Loss:

- campanha configura `mse` (`run_appclassnet_top200.py` linhas 171-174 e 192-195).
- treino compila `self._variational_algorithm.compile(loss=self._variational_autoencoder_loss_function, optimizer=Adam())` (`GenerativeModels._training_variational_autoencoder_model()` linhas 2966-2969). A classe do algoritmo implementa o objetivo VAE sobre reconstrucao/regularizacao, mas o ponto de entrada de loss configuravel e `mse`.

Ativacao de saida:

- default legado: `sigmoid` (`ArgumentsVariationalAutoencoder.py` linhas 44-49).
- campanha: `linear` (`run_appclassnet_top200.py` linhas 174 e 195).

Numero de classes:

- campanha: `variational_autoencoder_number_classes=200` (`run_appclassnet_top200.py` linhas 171 e 192).
- one-hot de treino e geracao usa `number_samples_per_class["number_classes"]`.

Entrada condicional e labels:

- treino autoencoding com entrada `(x_real, y_one_hot)` e target `x_real` via `_fit_autoencode_with_one_hot()` (linhas 177-195 e 2977-2987).
- geracao cria one-hot por classe e chama decoder (`AlgorithmVariationalAutoencoder.get_samples()` linhas 315-327).

Epocas e batch size:

- full: 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 162-170).
- demo: 1 epoca, batch 128 (`run_appclassnet_top200.py` linhas 182-191).

Stopping:

- early stop off por default; condicional no treino (`GenerativeModels.py` linhas 2971-2975).

Quantidade real usada:

- `len(x_real_samples)` do fold/split.

Quantidade sintetica:

- default 256 por classe.

Estrategia de geracao:

- loop por classe; chunks por `generation_batch_size`; ruído `numpy.random.normal(size=(batch_instances, latent_dim))`; decoder; concatena chunks; arredonda se nao-continuo (`AlgorithmVariationalAutoencoder.get_samples()` linhas 315-334).

Escala:

- source por default; saida linear; inverse para source se adapter estiver ativo.

Riscos:

- alto: embedding de label com 8 unidades para 200 classes pode limitar separacao condicional e facilitar colapso/ignorancia da classe.
- medio: demo com 1 epoca e inadequado para qualidade final.
- medio: defaults legados do VAE sao binarios e sigmoid se campanha nao sobrescrever.

### `autoencoder`

Severidade geral: medio.

Arquitetura:

- autoencoder condicional denso.
- decoder concatena latente e `label_input` `(number_classes,)` antes das camadas densas (`Autoencoder/VanillaDecoder.py` linhas 196-220).

Loss:

- campanha: `mse` (`run_appclassnet_top200.py` linha 152).
- treino: `self._autoencoder_algorithm.compile(loss=arguments.autoencoder_loss_function)` (`GenerativeModels._training_autoencoder_model()` linhas 678-680).

Ativacao de saida:

- default legado: `sigmoid` (`ArgumentsAutoencoder.py` linhas 49-53).
- campanha: `linear` (`run_appclassnet_top200.py` linha 154).

Numero de classes:

- campanha: `autoencoder_number_classes=200` (`run_appclassnet_top200.py` linha 151).
- arquitetura usa `number_samples_per_class["number_classes"]`.

Entrada condicional e labels:

- treino `(x_real, y_one_hot) -> x_real` (`GenerativeModels.py` linhas 687-697).
- geracao `to_one_hot_batch([label_class] * number_instances, num_classes=...)` e decoder (`AutoencoderAlgorithm.get_samples()` linhas 284-304).

Epocas e batch size:

- 300 epocas, batch 256 (`run_appclassnet_top200.py` linhas 143-150).

Stopping:

- early stop off por default; condicional no treino (`GenerativeModels.py` linhas 681-685).

Quantidade sintetica:

- default 256 por classe.

Riscos:

- medio: autoencoder amostra latente de uma normal parametrica, nao necessariamente da distribuicao posterior por classe; pode gerar pontos fora da manifold real.
- medio: se `data_type` nao for continuous, arredonda features continuas.

### `wasserstein`

Severidade geral: alto.

Arquitetura:

- WGAN condicional denso.
- generator recebe latente e `label_input` `(number_classes,)`, concatena e projeta para latent dimension antes do dense generator (`Wasserstein/VanillaGenerator.py` linhas 213-243).
- critic/discriminator tambem e condicional no label.

Loss:

- treino define `discriminator_loss = mean(fake) - mean(real)` e `generator_loss = -mean(fake)` em `GenerativeModels._training_wasserstein_model()` linhas 2109-2114.
- treino aplica clipping de pesos no critic em `AlgorithmWassersteinGAN.train_step()` linhas 164-166.

Ativacao de saida:

- default legado: `sigmoid` (`ArgumentsWassersteinGAN.py` linhas 40-46).
- campanha: `linear` (`run_appclassnet_top200.py` linha 237).

Numero de classes:

- campanha: `wasserstein_number_classes=200` (`run_appclassnet_top200.py` linha 234).
- arquitetura usa `number_samples_per_class["number_classes"]`.

Entrada condicional:

- treino via one-hot (`GenerativeModels.py` linhas 2130-2140).
- geracao por classe com one-hot e ruído normal (`AlgorithmWassersteinGAN.get_samples()` linhas 205-218).

Epocas e batch size:

- 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 225-233).

Stopping:

- early stop off por default; condicional no treino (`GenerativeModels.py` linhas 2125-2128).

Quantidade sintetica:

- default 256 por classe.

Riscos:

- alto: learning rate da campanha (`0.001`) nao e usado pelo treino, que fixa `0.0002`.
- medio: weight clipping pode limitar critic e favorecer baixa diversidade.
- medio: condicionamento pode ser ignorado; precisa diagnostico por classe.

### `wasserstein_gp`

Severidade geral: alto.

Arquitetura:

- WGAN-GP condicional denso.
- generator com `label_input` `(number_classes,)`, concatena latente e label (`WassersteinGP/VanillaGenerator.py` linhas 213-243).

Loss:

- treino define loss Wasserstein e adiciona gradient penalty em `AlgorithmWassersteinGANGP.train_step()` linhas 278-289.
- treino compila com losses custom em `GenerativeModels._training_wasserstein_gp_model()` linhas 2547-2561.

Ativacao de saida:

- default legado: `sigmoid` (`ArgumentsWassersteinGANGP.py` linhas 40-46).
- campanha: `linear` (`run_appclassnet_top200.py` linha 263).

Numero de classes:

- campanha: `wasserstein_gp_number_classes=200` (`run_appclassnet_top200.py` linha 260).

Entrada condicional:

- treino via one-hot (`GenerativeModels.py` linhas 2569-2579).
- geracao por classe com one-hot e ruído normal (`AlgorithmWassersteinGANGP.get_samples()` linhas 340-353).

Epocas e batch size:

- 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 251-259).

Stopping:

- early stop off por default.

Riscos:

- alto: learning rate da campanha (`0.001`) nao e usado; treino fixa `0.0002`.
- medio: custo alto para 200 classes; possibilidade de colapso sem metricas por classe.

### `latent_diffusion`

Severidade geral: alto.

Arquitetura:

- VAE condicional para criar embedding latente, seguido de U-Net de difusao no espaco latente.
- `DiffusionModelUnet.get_unet_model()` recebe `description_input` com shape `(number_classes,)`, cria embedding de label e usa cross-attention em blocos configurados (`DiffusionModelUnet.py` linhas 402-423 e 432-468).
- decoder VAE reconstrói features a partir do embedding denoised e labels.

Loss:

- VAE: `mse` pela campanha (`run_appclassnet_top200.py` linha 295) e compile em `GenerativeModels._training_latent_diffusion_model()` linhas 1484-1499.
- difusao: `MeanSquaredError()` sobre ruido predito (`GenerativeModels.py` linhas 1522-1543; `AlgorithmLatentDiffusion.train_step()` linhas 253-257).

Ativacao de saida:

- campanha: U-Net `linear`, autoencoder `linear`, encoder output `linear` (`run_appclassnet_top200.py` linhas 280, 298, 304).

Numero de classes:

- usa `number_samples_per_class["number_classes"]` no one-hot e no U-Net.

Entrada condicional:

- VAE recebe one-hot em treino e embedding (`GenerativeModels.py` linhas 1493-1499 e 1525-1528).
- U-Net recebe labels one-hot como `description_input`.

Epocas e batch size:

- VAE 300 epocas, U-Net 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 278-301).

Stopping:

- early stop off por default; condicional nos dois treinos (`GenerativeModels.py` linhas 1490-1491 e 1536-1537).

Quantidade sintetica:

- default 256 por classe.

Estrategia de geracao:

- por classe, cria one-hot, inicia ruido normal no embedding, itera `time_steps` reversos, chama U-Net e `p_sample(..., clip_denoised=True)`, decodifica para dados (`AlgorithmLatentDiffusion.generate_data()` linhas 288-316; `get_samples()` linhas 427-439).

Escala:

- clipping configurado em `[-0.5, 0.5]` pela campanha.

Riscos:

- alto: `execution_mode=batches` nao implementado para latent diffusion sem one-hot global; o metodo levanta `NotImplementedError` (`GenerativeModels.py` linhas 1465-1469).
- alto: cria embeddings globais (`data_embedding = numpy.array(data_embedding)` e `expand_dims`) em memoria (`GenerativeModels.py` linhas 1525-1531).
- medio: clipping pode reduzir variancia e esconder saida fora de escala.

### `denoising_diffusion`

Severidade geral: medio-alto.

Arquitetura:

- U-Net de difusao diretamente no espaco de features, condicionado por one-hot.
- mesmo `DiffusionModelUnet` com `description_input` `(number_classes,)`.

Loss:

- `MeanSquaredError()` sobre ruido predito (`GenerativeModels._training_denoising_diffusion_model()` linhas 3364-3386; `AlgorithmDenoisingDiffusion.train_step()` linhas 263-266).

Ativacao de saida:

- campanha: U-Net `linear` (`run_appclassnet_top200.py` linha 312).

Numero de classes:

- one-hot de 200 via `number_samples_per_class`.

Entrada condicional:

- treino via `_fit_features_and_one_hot()` depois de expandir `X` para `(n, 20, 1)` (`GenerativeModels.py` linhas 3374-3386).
- geracao cria one-hot por classe (`AlgorithmDenoisingDiffusion.get_samples()` linhas 435-443).

Epocas e batch size:

- 300 epocas, batch 128 (`run_appclassnet_top200.py` linhas 311-316).

Stopping:

- early stop off por default.

Estrategia de geracao:

- inicia ruido normal em shape `(n, output_shape, 1)`, itera time steps reversos, aplica `p_sample(..., clip_denoised=True)`, recorta para shape original e remove ultimo eixo (`AlgorithmDenoisingDiffusion.generate_data()` linhas 297-325; `get_samples()` linhas 435-448).

Riscos:

- medio: clipping ativo em `[-0.5, 0.5]`.
- medio: custo alto para 200 classes; qualidade depende de diagnosticos por classe.

### `copula`, `ctgan`, `tvae`

Severidade geral: alto.

Arquitetura:

- wrappers SDV `GaussianCopulaSynthesizer`, `CTGANSynthesizer`, `TVAESynthesizer` (`SDVInterfaceAlgorithm.training_model()` linhas 130-139).
- metadata inferida automaticamente de um `DataFrame` com features e coluna `class` (linhas 122-128).

Loss, ativacao de saida, epocas, batch size:

- nao sao configurados pela campanha `sf`; `_base_campaign("copula")`, `_base_campaign("tvae")`, `_base_campaign("ctgan")` nao passa parametros especificos (run_appclassnet_top200.py linhas 331-333).
- o comportamento fica nos defaults da versao instalada do SDV, portanto nao e explicitamente reprodutivel pelo arquivo de campanha.

Numero de classes e labels:

- nao usa one-hot no codigo do projeto; usa coluna `class` no DataFrame.
- geracao usa `Condition(num_rows=number_instances, column_values={"class": label_class})` para cada classe (linhas 168-178).
- depois filtra `sdv_y_samples == label_class` e salva somente features no dict da classe solicitada (linhas 189-221).

Quantidade sintetica:

- default 256 por classe solicitado via `number_samples_per_class`.

Estrategia de geracao:

- cria lista de condicoes por classe, chama `sample_from_conditions()`, separa X/y, substitui strings por `1`, e seleciona indices por classe com reposicao (`SDVInterfaceAlgorithm.get_samples()` linhas 178-221).

Escala:

- SDV aprende a escala do DataFrame real recebido; nao ha output activation nem inverse transform especifico dentro do wrapper.

Riscos:

- alto: chamada `sample(num_rows=number_instances*5)` dentro do loop e descartada, desperdicando tempo/memoria (linha 170).
- alto: se `sample_from_conditions()` nao devolver classe solicitada, a classe e pulada; o audit de labels deve falhar para 200 classes, mas a geracao fica incompleta (linhas 200-226).
- alto: `numpy.where(... isinstance(str), 1, sdv_x_samples)` pode corromper features se SDV inferir/gerar strings (linhas 182-187).
- medio: parametros de SDV nao sao fixados no runner; qualidade depende de defaults externos.

## Investigacao dos problemas solicitados

| Problema investigado | Evidencia atual | Classificacao |
|---|---|---|
| Default binario em problema de 200 classes | Defaults `number_classes=2` e `sigmoid` existem em AE/VAE/WGAN/WGAN-GP/VQ-VAE; campanha sobrescreve para principais modelos de `sf --full`. | Alto |
| `number_classes=2` persistente | Persistente nos defaults dos arquivos de argumentos; nao persistente no plano AppClassNet quando runner passa `SAMPLES_TOP200` e args de 200 classes. | Alto |
| Condicao one-hot incorreta | Helpers validam zero-based e shape `(n, 200)` quando `number_classes=200`; nao ha evidencia de one-hot binario na campanha. | Baixo |
| Gerador ignorando a condicao | Arquiteturas concatenam ou usam attention com label; ignorancia semantica so pode ser detectada por metricas/predicoes. | Medio |
| Colapso para uma unica distribuicao | Nao provavel por codigo estatico; possivel em treino, precisa diagnostico. | Medio |
| Mesmos sinteticos para varias classes | Geracao usa ruído novo e one-hot por classe; nao ha seed fixa por classe. Colapso ainda pode produzir amostras parecidas. | Medio |
| Seeds repetidas gerando amostras identicas | Nao encontrei seed fixa nos algoritmos neurais; risco principal e nao determinismo, nao repeticao. | Medio |
| Saida sigmoid incompatível com source space | Campanha usa `linear`; defaults legados usam `sigmoid`. Runner detecta sigmoid e resolve `generator_transform auto` para `minmax` em AppClassNet (`run_appclassnet_top200.py` linhas 544-571 e 585-604). | Alto |
| Falta de inverse_transform | `main.synthesize_data()` aplica adapter para source quando existe; batches aplicam `inverse_synthetic_batch()` antes de salvar. Risco se `inverse_transform_synthetic` for desligado com generator transform nao-preserve. | Medio |
| Arredondamento de features continuas | Todos os geradores neurais arredondam se `data_type` nao for `continuous`; runner encaminha `continuous`. | Alto |
| Clipping excessivo | Difusoes usam `clip_denoised=True`; limites configurados `[-0.5, 0.5]`. | Medio |
| Dados sinteticos constantes | Nao detectavel estaticamente; sanity checks atuais nao medem variancia por classe de forma suficiente. | Medio |
| Variancia muito baixa | Mesmo risco; difusoes e autoencoders podem reduzir variancia. | Medio |
| Features fora da distribuicao real | `SyntheticSanityChecks._scale_report()` mede fora do range real e alerta se >20%. | Medio |
| Classes sem diversidade | Nao medido diretamente; label audit mede presenca/contagem, nao diversidade intra-classe. | Medio |
| Geracao por classe com labels incorretos | Labels salvos sao a classe solicitada no dict; `SyntheticLabelAudit` registra `requested_class` e `saved_label` iguais. | Baixo |
| Treinamento com poucas amostras para 200 condicoes | Depende de quotas/folds; demos usam 1 epoca. `per_class` treina modelos com subset por classe/grupo. | Alto |
| Modelo originalmente desenhado para binario | Defaults e perdas historicas indicam origem binaria; campanha adapta para 200 classes. | Alto |
| Geracao em batches alterando qualidade | `single_conditional` mantem um gerador; `per_class`/`grouped_classes` mudam o treino e podem alterar qualidade. | Medio |
| Reutilizacao incorreta do estado do modelo | Normal e incremental reutilizam o modelo treinado para todas as classes, correto. Particionado chama `_release_current_generator()` apos unidade. | Baixo |

## Sanity checks propostos

Antes de TR-TS e TS-TR, executar e salvar por fold/modelo:

- `real_feature_mean_by_class`: media por feature e classe.
- `synthetic_feature_mean_by_class`: media por feature e classe.
- `real_feature_std_by_class` e `synthetic_feature_std_by_class`.
- quantis por feature: `0.01`, `0.05`, `0.25`, `0.50`, `0.75`, `0.95`, `0.99`.
- KS por feature, global e por classe quando houver amostras suficientes.
- Wasserstein por feature, global e por classe.
- matriz de correlacao real vs sintetica e distancia Frobenius entre correlacoes.
- nearest-neighbor distance real->synthetic e synthetic->real, por classe.
- taxa de duplicacao: linhas sinteticas identicas, linhas sinteticas que coincidem com reais, e percentual de linhas repetidas por classe.
- C2ST com classificador simples e reinicializado, reportando accuracy e `chance_level_suspected` quando aplicavel.
- cobertura de classes: contagens, classes ausentes, classes extras.
- diversidade intra-classe: variancia media, distancia media ao centroide sintetico, distancia media ao centroide real da classe.
- distancia entre centroides reais e sinteticos por classe.
- distribuicao das predicoes de um DecisionTree treinado no real: numero de labels previstas, classe majoritaria prevista, entropia das predicoes, classes nunca previstas, recall por classe.
- taxa de NaN/inf e taxa fora do range real por feature.
- comparacao `data_space`, `transform_id`, `transform_history` entre real e sintetico.

## Regras de regressao recomendadas

1. Para AppClassNet, falhar a geracao se `number_samples_per_class["number_classes"] != 200`.
2. Falhar se qualquer classe `0..199` solicitada nao aparecer no sintetico salvo.
3. Falhar se `data_type != "continuous"` em `source_profile=appclassnet_top200`.
4. Falhar se `data_space` real e sintetico divergirem na avaliacao.
5. Marcar diagnostico se mais de 20% dos valores sinteticos ficarem fora do range real por feature.
6. Marcar diagnostico se a classe majoritaria prevista por classificador real->sintetico passar de 90%.
7. Marcar diagnostico se variancia sintetica media por classe for proxima de zero.
8. Marcar diagnostico se distancia entre centroides reais e sinteticos por classe for extrema em relacao a distribuicao real.

## Conclusao

A implementacao atual tem um contrato condicional razoavelmente claro para os geradores neurais da campanha `sf`: labels zero-based sao convertidas para one-hot com `number_classes`, a geracao itera por classe solicitada e os labels salvos sao derivados da chave de classe, nao de um output ambíguo do gerador. Para AppClassNet, os parametros do runner corrigem os maiores defaults legados: `number_classes=200`, `last_activation=linear`, `data_type=continuous` e `feature_transform=preserve`.

Os principais riscos nao estao no ato de criar labels sinteticos, mas na qualidade efetiva e na reprodutibilidade: defaults binarios fora do runner, parametros de Wasserstein ignorados pelo treino, SDV sem parametros fixos, difusoes com clipping, demos com 1 epoca e falta de metricas fortes de diversidade/distribuicao. Antes de interpretar TR-TS ou TS-TR, o pipeline deve registrar sanity checks por classe e por feature para provar que os sinteticos cobrem as 200 classes, estao no espaco `source`, preservam escala e nao colapsaram para uma distribuicao unica.
