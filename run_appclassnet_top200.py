#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AppClassNet top200 campaign runner.

This script centralizes execution of MalDataGen campaigns for the AppClassNet
`top200` split. It materializes the original `.npy` feature/label files into a
single CSV accepted by `main.py`, then launches the selected campaigns with
`--data_type continuous` by default so generated features are not rounded.
"""

import argparse
import datetime
import itertools
import logging
import shlex
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_VERBOSITY_LEVEL = logging.INFO
DEFAULT_NUM_EPOCHS = 300
DEFAULT_DEMO_EPOCHS = 1
DEFAULT_K_FOLDS = 5
DEFAULT_DATA_TYPE = "continuous"
DEFAULT_SAVE_DATA = "True"
DEFAULT_SAMPLES_PER_CLASS = 256
DEFAULT_CHUNK_SIZE = 100_000
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

APPCLASSNET_NUM_CLASSES = 200
APPCLASSNET_NUM_FEATURES = 20
APPCLASSNET_SPLITS = ("train", "valid", "test")
APPCLASSNET_LABEL_COLUMN = "label"

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = Path("Datasets/raw/AppClassNet/top200")
DEFAULT_CONVERTED_ROOT = Path("Datasets/converted/AppClassNet/top200")
PATH_LOG = "logs"
PATHS = [PATH_LOG]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

DEFAULT_CAMPAIGN = [
    "wasserstein",
    "wasserstein_gp",
    "variational",
    "autoencoder",
    "adversarial",
    "latent_diffusion",
    "denoising_diffusion",
    "ctgan",
    "tvae",
    "copula",
]
DEMO_CAMPAIGNS = ["variational_demo", "adversarial_demo"]
SDV_CAMPAIGNS = ["copula", "tvae", "ctgan"]
NO_TRAINING_PLOT_MODELS = {"copy", "copula", "ctgan", "tvae"}

DEFAULT_CLASSIFIER = [
    "RandomForest SupportVectorMachine KNN DecisionTree NaiveBayes GradientBoosting StochasticGradientDescent"
]
SAMPLES_TOP200 = ",".join(f"{class_id}:{DEFAULT_SAMPLES_PER_CLASS}" for class_id in range(APPCLASSNET_NUM_CLASSES))

arguments = None


def _list(value):
    return value if isinstance(value, list) else [value]


def _base_campaign(model_type, **params):
    campaign = {
        "classifier": DEFAULT_CLASSIFIER,
        "model_type": [model_type],
        "number_samples_per_class": [SAMPLES_TOP200],
        "number_k_folds": [DEFAULT_K_FOLDS],
        "save_data": [DEFAULT_SAVE_DATA],
    }
    for key, value in params.items():
        campaign[key] = _list(value)
    return campaign


campaigns_available = {
    "adversarial": _base_campaign(
        "adversarial",
        adversarial_number_epochs=DEFAULT_NUM_EPOCHS,
        adversarial_batch_size=128,
        adversarial_dense_layer_sizes_g="256 128",
        adversarial_dense_layer_sizes_d="128 64",
        adversarial_dropout_decay_rate_g=0.2,
        adversarial_dropout_decay_rate_d=0.4,
        adversarial_initializer_deviation=0.125,
        adversarial_initializer_mean=0.0,
        adversarial_latent_dimension=64,
        adversarial_latent_mean_distribution=0.0,
        adversarial_latent_stander_deviation=1.0,
        adversarial_training_algorithm="Adam",
        adversarial_activation_function="leakyrelu",
        adversarial_loss_generator="binary_crossentropy",
        adversarial_loss_discriminator="binary_crossentropy",
        adversarial_smoothing_rate=0.15,
        adversarial_last_layer_activation="linear",
    ),
    "adversarial_demo": _base_campaign(
        "adversarial",
        number_k_folds=2,
        adversarial_number_epochs=DEFAULT_DEMO_EPOCHS,
        adversarial_batch_size=128,
        adversarial_dense_layer_sizes_g="128",
        adversarial_dense_layer_sizes_d="64",
        adversarial_dropout_decay_rate_g=0.2,
        adversarial_dropout_decay_rate_d=0.4,
        adversarial_initializer_deviation=0.125,
        adversarial_initializer_mean=0.0,
        adversarial_latent_dimension=32,
        adversarial_latent_mean_distribution=0.0,
        adversarial_latent_stander_deviation=1.0,
        adversarial_training_algorithm="Adam",
        adversarial_activation_function="leakyrelu",
        adversarial_loss_generator="binary_crossentropy",
        adversarial_loss_discriminator="binary_crossentropy",
        adversarial_smoothing_rate=0.15,
        adversarial_last_layer_activation="linear",
    ),
    "autoencoder": _base_campaign(
        "autoencoder",
        autoencoder_number_epochs=DEFAULT_NUM_EPOCHS,
        autoencoder_latent_dimension=64,
        autoencoder_activation_function="elu",
        autoencoder_dropout_decay_rate_encoder=0.10,
        autoencoder_dropout_decay_rate_decoder=0.10,
        autoencoder_dense_layer_sizes_encoder="128 64",
        autoencoder_dense_layer_sizes_decoder="64 128",
        autoencoder_batch_size=256,
        autoencoder_number_classes=APPCLASSNET_NUM_CLASSES,
        autoencoder_loss_function="mse",
        autoencoder_momentum=0.8,
        autoencoder_last_activation_layer="linear",
        autoencoder_initializer_mean=0.0,
        autoencoder_initializer_deviation=0.125,
        autoencoder_latent_mean_distribution=0.0,
        autoencoder_latent_stander_deviation=1.0,
    ),
    "variational": _base_campaign(
        "variational",
        variational_autoencoder_number_epochs=DEFAULT_NUM_EPOCHS,
        variational_autoencoder_latent_dimension=64,
        variational_autoencoder_training_algorithm="Adam",
        variational_autoencoder_activation_function="elu",
        variational_autoencoder_dropout_decay_rate_encoder=0.2,
        variational_autoencoder_dropout_decay_rate_decoder=0.2,
        variational_autoencoder_dense_layer_sizes_encoder="128 64",
        variational_autoencoder_dense_layer_sizes_decoder="64 128",
        variational_autoencoder_batch_size=128,
        variational_autoencoder_number_classes=APPCLASSNET_NUM_CLASSES,
        variational_autoencoder_loss_function="mse",
        variational_autoencoder_momentum=0.8,
        variational_autoencoder_last_activation_layer="linear",
        variational_autoencoder_initializer_mean=0.0,
        variational_autoencoder_initializer_deviation=0.125,
        variational_autoencoder_mean_distribution=0.0,
        variational_autoencoder_stander_deviation=1.0,
    ),
    "variational_demo": _base_campaign(
        "variational",
        number_k_folds=2,
        variational_autoencoder_number_epochs=DEFAULT_DEMO_EPOCHS,
        variational_autoencoder_latent_dimension=32,
        variational_autoencoder_training_algorithm="Adam",
        variational_autoencoder_activation_function="elu",
        variational_autoencoder_dropout_decay_rate_encoder=0.2,
        variational_autoencoder_dropout_decay_rate_decoder=0.2,
        variational_autoencoder_dense_layer_sizes_encoder="64 32",
        variational_autoencoder_dense_layer_sizes_decoder="32 64",
        variational_autoencoder_batch_size=128,
        variational_autoencoder_number_classes=APPCLASSNET_NUM_CLASSES,
        variational_autoencoder_loss_function="mse",
        variational_autoencoder_momentum=0.8,
        variational_autoencoder_last_activation_layer="linear",
        variational_autoencoder_initializer_mean=0.0,
        variational_autoencoder_initializer_deviation=0.125,
        variational_autoencoder_mean_distribution=0.0,
        variational_autoencoder_stander_deviation=1.0,
    ),
    "quantized": _base_campaign(
        "quantized",
        quantized_vae_number_epochs=DEFAULT_NUM_EPOCHS,
        quantized_vae_latent_dimension=32,
        quantized_vae_number_embedding=64,
        quantized_vae_training_algorithm="Adam",
        quantized_vae_activation_function="elu",
        quantized_vae_dropout_decay_rate_encoder=0.2,
        quantized_vae_dropout_decay_rate_decoder=0.2,
        quantized_vae_dense_layer_sizes_encoder="128 64",
        quantized_vae_dense_layer_sizes_decoder="64 128",
        quantized_vae_batch_size=128,
        quantized_vae_number_classes=APPCLASSNET_NUM_CLASSES,
        quantized_vae_loss_function="mse",
        quantized_vae_momentum=0.8,
        quantized_vae_last_activation_layer="linear",
        quantized_vae_initializer_mean=0.0,
        quantized_vae_initializer_deviation=0.125,
        quantized_vae_mean_distribution=0.0,
        quantized_vae_stander_deviation=1.0,
        quantized_vae_train_variance=0.5,
    ),
    "wasserstein": _base_campaign(
        "wasserstein",
        wasserstein_number_epochs=DEFAULT_NUM_EPOCHS,
        wasserstein_latent_dimension=64,
        wasserstein_training_algorithm="Adam",
        wasserstein_activation_function="elu",
        wasserstein_dropout_decay_rate_g=0.0,
        wasserstein_dropout_decay_rate_d=0.0,
        wasserstein_dense_layer_sizes_generator="256 128",
        wasserstein_dense_layer_sizes_discriminator="128 64",
        wasserstein_batch_size=128,
        wasserstein_number_classes=APPCLASSNET_NUM_CLASSES,
        wasserstein_loss_function="mean_squared_error",
        wasserstein_momentum=0.8,
        wasserstein_last_activation_layer="linear",
        wasserstein_initializer_mean=0.0,
        wasserstein_initializer_deviation=0.125,
        wasserstein_optimizer_generator_learning_rate=0.001,
        wasserstein_optimizer_discriminator_learning_rate=0.001,
        wasserstein_optimizer_generator_beta=0.5,
        wasserstein_optimizer_discriminator_beta=0.5,
        wasserstein_discriminator_steps=3,
        wasserstein_smoothing_rate=0.10,
        wasserstein_latent_mean_distribution=0.0,
        wasserstein_latent_stander_deviation=1.0,
    ),
    "wasserstein_gp": _base_campaign(
        "wasserstein_gp",
        wasserstein_gp_number_epochs=DEFAULT_NUM_EPOCHS,
        wasserstein_gp_latent_dimension=64,
        wasserstein_gp_training_algorithm="Adam",
        wasserstein_gp_activation_function="leakyrelu",
        wasserstein_gp_dropout_decay_rate_g=0.0,
        wasserstein_gp_dropout_decay_rate_d=0.0,
        wasserstein_gp_dense_layer_sizes_generator="256 128",
        wasserstein_gp_dense_layer_sizes_discriminator="128 64",
        wasserstein_gp_batch_size=128,
        wasserstein_gp_number_classes=APPCLASSNET_NUM_CLASSES,
        wasserstein_gp_loss_function="mean_squared_error",
        wasserstein_gp_momentum=0.8,
        wasserstein_gp_last_activation_layer="linear",
        wasserstein_gp_initializer_mean=0.0,
        wasserstein_gp_initializer_deviation=0.125,
        wasserstein_gp_optimizer_generator_learning_rate=0.001,
        wasserstein_gp_optimizer_discriminator_learning_rate=0.001,
        wasserstein_gp_optimizer_generator_beta=0.5,
        wasserstein_gp_optimizer_discriminator_beta=0.5,
        wasserstein_gp_discriminator_steps=3,
        wasserstein_gp_smoothing_rate=0.10,
        wasserstein_gp_latent_mean_distribution=0.0,
        wasserstein_gp_latent_stander_deviation=1.0,
        wasserstein_gp_gradient_penalty=10.0,
    ),
    "latent_diffusion": _base_campaign(
        "latent_diffusion",
        latent_diffusion_unet_epochs=DEFAULT_NUM_EPOCHS,
        latent_diffusion_autoencoder_epochs=DEFAULT_NUM_EPOCHS,
        latent_diffusion_unet_last_layer_activation="linear",
        latent_diffusion_latent_dimension=64,
        latent_diffusion_unet_num_embedding_channels=1,
        latent_diffusion_unet_channels_per_level="1 2 4",
        latent_diffusion_unet_batch_size=128,
        latent_diffusion_unet_attention_mode="False True True",
        latent_diffusion_unet_num_residual_blocks=2,
        latent_diffusion_unet_group_normalization=1,
        latent_diffusion_unet_intermediary_activation="elu",
        latent_diffusion_unet_intermediary_activation_alpha=0.05,
        latent_diffusion_gaussian_beta_start=1e-4,
        latent_diffusion_gaussian_beta_end=0.02,
        latent_diffusion_gaussian_time_steps=300,
        latent_diffusion_gaussian_clip_min=-0.5,
        latent_diffusion_gaussian_clip_max=0.5,
        latent_diffusion_autoencoder_loss="mse",
        latent_diffusion_autoencoder_encoder_filters="128 64",
        latent_diffusion_autoencoder_decoder_filters="64 128",
        latent_diffusion_autoencoder_last_layer_activation="linear",
        latent_diffusion_autoencoder_latent_dimension=64,
        latent_diffusion_autoencoder_batch_size_create_embedding=128,
        latent_diffusion_autoencoder_batch_size_training=128,
        latent_diffusion_autoencoder_intermediary_activation_function="elu",
        latent_diffusion_autoencoder_intermediary_activation_alpha=0.05,
        latent_diffusion_autoencoder_activation_output_encoder="linear",
        latent_diffusion_margin=0.5,
        latent_diffusion_ema=0.999,
        latent_diffusion_time_steps=300,
    ),
    "denoising_diffusion": _base_campaign(
        "denoising_diffusion",
        denoising_diffusion_unet_epochs=DEFAULT_NUM_EPOCHS,
        denoising_diffusion_unet_last_layer_activation="linear",
        denoising_diffusion_latent_dimension=64,
        denoising_diffusion_unet_num_embedding_channels=1,
        denoising_diffusion_unet_channels_per_level="1 2 4",
        denoising_diffusion_unet_batch_size=128,
        denoising_diffusion_unet_attention_mode="False True True",
        denoising_diffusion_unet_num_residual_blocks=2,
        denoising_diffusion_unet_group_normalization=1,
        denoising_diffusion_unet_intermediary_activation="elu",
        denoising_diffusion_unet_intermediary_activation_alpha=0.05,
        denoising_diffusion_gaussian_beta_start=1e-4,
        denoising_diffusion_gaussian_beta_end=0.02,
        denoising_diffusion_gaussian_time_steps=300,
        denoising_diffusion_gaussian_clip_min=-0.5,
        denoising_diffusion_gaussian_clip_max=0.5,
        denoising_diffusion_margin=0.5,
        denoising_diffusion_ema=0.999,
        denoising_diffusion_time_steps=300,
    ),
    "copula": _base_campaign("copula"),
    "tvae": _base_campaign("tvae"),
    "ctgan": _base_campaign("ctgan"),
}


def print_all_settings(parsed_arguments):
    logging.info("Campaign Command:\n\t%s\n", " ".join(sys.argv))
    logging.info("Campaign Settings:")
    max_length = max(len(key) for key in vars(parsed_arguments).keys())
    for key_item, values in sorted(vars(parsed_arguments).items()):
        logging.info("\t%s : %s", key_item.ljust(max_length, " "), values)
    logging.info("")


def resolve_project_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def split_campaign_tokens(campaigns):
    tokens = []
    for campaign in campaigns:
        tokens.extend(item for item in campaign.split(",") if item)
    return tokens


def choose_campaigns(campaigns):
    if not campaigns:
        return list(campaigns_available.keys())

    tokens = split_campaign_tokens(campaigns)
    if tokens == ["sf"]:
        return DEMO_CAMPAIGNS
    if tokens == ["sf2"]:
        return SDV_CAMPAIGNS

    unknown = [name for name in tokens if name not in campaigns_available]
    if unknown:
        logging.error("ERROR: Campaign(s) not found: %s", ", ".join(unknown))
        logging.error("Available campaigns: %s", ", ".join(sorted(campaigns_available)))
        sys.exit(1)
    return tokens


def warn_prepare_limit_if_needed(parsed_arguments, campaigns_chosen):
    if not parsed_arguments.prepare_max_samples or parsed_arguments.prepare_max_samples <= 0:
        return

    max_k_folds = 1
    for campaign_name in campaigns_chosen:
        campaign_folds = campaigns_available[campaign_name].get("number_k_folds", [DEFAULT_K_FOLDS])
        max_k_folds = max(max_k_folds, max(int(value) for value in campaign_folds))

    minimum_recommended = APPCLASSNET_NUM_CLASSES * max_k_folds
    if parsed_arguments.prepare_max_samples < minimum_recommended:
        logging.warning(
            "prepare_max_samples=%d is below the recommended minimum %d for %d classes and %d folds. "
            "StratifiedKFold may fail if any class has fewer samples than folds.",
            parsed_arguments.prepare_max_samples,
            minimum_recommended,
            APPCLASSNET_NUM_CLASSES,
            max_k_folds,
        )


def expected_csv_path(split, output_root, max_samples, sampling="balanced"):
    suffix = split
    if max_samples and max_samples > 0:
        suffix = f"{suffix}_{sampling}_max{max_samples}"
    return output_root / f"{suffix}.csv"


def validate_raw_split(raw_root, split):
    x_path = raw_root / f"{split}_x.npy"
    y_path = raw_root / f"{split}_y.npy"
    missing = [str(path) for path in (x_path, y_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AppClassNet file(s): " + ", ".join(missing))
    return x_path, y_path


def iter_splits(split):
    if split == "all":
        return APPCLASSNET_SPLITS
    return (split,)


def select_balanced_indices(numpy, labels, max_samples, seed=42):
    labels_array = numpy.asarray(labels)
    unique_labels = numpy.unique(labels_array)
    if max_samples >= labels_array.shape[0]:
        return numpy.arange(labels_array.shape[0])

    samples_per_class = max_samples // len(unique_labels)
    remainder = max_samples % len(unique_labels)
    random_generator = numpy.random.default_rng(seed)
    selected_indices = []

    for position, label in enumerate(unique_labels):
        label_indices = numpy.flatnonzero(labels_array == label)
        sample_count = samples_per_class + (1 if position < remainder else 0)
        sample_count = min(sample_count, label_indices.shape[0])
        if sample_count > 0:
            selected_indices.append(random_generator.choice(label_indices, size=sample_count, replace=False))

    if not selected_indices:
        return numpy.array([], dtype=int)

    selected = numpy.concatenate(selected_indices)
    random_generator.shuffle(selected)
    return selected


def write_rows_to_csv(pandas, output_path, data, labels, feature_columns, header_written):
    data_frame = pandas.DataFrame(data, columns=feature_columns)
    data_frame[APPCLASSNET_LABEL_COLUMN] = labels
    data_frame.to_csv(output_path, mode="a", header=not header_written, index=False)
    return True


def materialize_appclassnet_csv(
    split,
    raw_root,
    output_root,
    max_samples=-1,
    chunk_size=DEFAULT_CHUNK_SIZE,
    force=False,
    sampling="balanced",
):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = expected_csv_path(split, output_root, max_samples, sampling)

    if csv_path.is_file() and not force:
        logging.info("Using prepared AppClassNet CSV: %s", csv_path)
        return csv_path

    try:
        import numpy
        import pandas
    except ImportError as error:
        raise RuntimeError(
            "Preparing AppClassNet CSV requires numpy and pandas in the runner environment."
        ) from error

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    feature_columns = [f"f{index}" for index in range(APPCLASSNET_NUM_FEATURES)]
    remaining = max_samples if max_samples and max_samples > 0 else None
    header_written = False
    total_rows = 0

    for split_name in iter_splits(split):
        x_path, y_path = validate_raw_split(raw_root, split_name)
        logging.info("Preparing split %s from %s and %s", split_name, x_path, y_path)

        x_values = numpy.load(x_path, mmap_mode="r")
        y_values = numpy.load(y_path, mmap_mode="r")

        if x_values.ndim != 2 or x_values.shape[1] != APPCLASSNET_NUM_FEATURES:
            raise ValueError(
                f"Expected {APPCLASSNET_NUM_FEATURES} features in {x_path}, got shape {x_values.shape}."
            )
        if y_values.ndim != 1 or y_values.shape[0] != x_values.shape[0]:
            raise ValueError(
                f"Feature/label shape mismatch for split {split_name}: {x_values.shape} vs {y_values.shape}."
            )

        rows_to_write = x_values.shape[0]
        selected_indices = None
        if remaining is not None:
            rows_to_write = min(rows_to_write, remaining)
            if sampling == "balanced":
                selected_indices = select_balanced_indices(numpy, y_values, rows_to_write)

        if selected_indices is None:
            for start in range(0, rows_to_write, chunk_size):
                end = min(start + chunk_size, rows_to_write)
                header_written = write_rows_to_csv(
                    pandas,
                    tmp_path,
                    x_values[start:end],
                    y_values[start:end],
                    feature_columns,
                    header_written,
                )
                total_rows += end - start
        else:
            for start in range(0, selected_indices.shape[0], chunk_size):
                end = min(start + chunk_size, selected_indices.shape[0])
                chunk_indices = selected_indices[start:end]
                header_written = write_rows_to_csv(
                    pandas,
                    tmp_path,
                    x_values[chunk_indices],
                    y_values[chunk_indices],
                    feature_columns,
                    header_written,
                )
                total_rows += chunk_indices.shape[0]

        if remaining is not None:
            remaining -= rows_to_write
            if remaining <= 0:
                break

    if not header_written:
        raise ValueError("No rows were written while preparing AppClassNet CSV.")

    tmp_path.replace(csv_path)
    logging.info("Prepared AppClassNet CSV: %s (%d rows)", csv_path, total_rows)
    return csv_path


def append_cli_value(command, parameter, value):
    command.append(f"--{parameter}")
    if isinstance(value, (list, tuple)):
        command.extend(str(item) for item in value)
    elif isinstance(value, str) and " " in value:
        command.extend(value.split())
    else:
        command.append(str(value))


def build_main_command(python_executable, dataset_path, output_dir_run, combination, verbosity, data_type, data_load_max_samples):
    command = [
        python_executable,
        str(REPO_ROOT / "main.py"),
        "--data_load_path_file_input",
        str(dataset_path),
        "--data_type",
        data_type,
        "--verbosity",
        str(verbosity),
        "--output_dir",
        str(output_dir_run),
    ]

    if data_load_max_samples and data_load_max_samples > 0:
        command.extend(["--data_load_max_samples", str(data_load_max_samples)])

    for parameter, value in combination.items():
        append_cli_value(command, parameter, value)

        if (
            parameter == "variational_autoencoder_dense_layer_sizes_encoder"
            and "variational_autoencoder_dense_layer_sizes_decoder" not in combination
        ):
            layers = str(value).split()[::-1]
            command.extend(["--variational_autoencoder_dense_layer_sizes_decoder", *layers])

    return command


def build_plot_command(python_executable, dataset_path, output_dir_run, combination, plot_title, results_paths, final_plot):
    command = [
        python_executable,
        str(REPO_ROOT / "plots.py"),
        "--results",
        ",".join(str(path) for path in results_paths) if final_plot else str(results_paths[-1]),
        "--title",
        *plot_title.split(),
        "--folds",
        str(combination["number_k_folds"]),
        "--dataset",
        str(dataset_path),
        "--output_dir",
        str(output_dir_run),
        "--model",
        str(combination["model_type"]),
    ]

    if final_plot:
        command.append("--f_plot")

    if combination["model_type"] not in NO_TRAINING_PLOT_MODELS:
        training_files = [
            output_dir_run / "Monitor" / f"monitor_model_{fold}_fold.json"
            for fold in range(combination["number_k_folds"])
        ]
        command.extend(["--training", *map(str, training_files)])

    return command


def run_cmd(command):
    logging.info("Command line: %s", shlex.join(map(str, command)))
    if not arguments.dryrun:
        subprocess.run(command, check=True)


def prepare_dependencies_available():
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


def reexec_with_configured_python_if_needed(parsed_arguments):
    if parsed_arguments.dryrun or prepare_dependencies_available():
        return

    if parsed_arguments.python == sys.executable:
        return

    command = [parsed_arguments.python, str(Path(__file__).resolve()), *sys.argv[1:]]
    completed_process = subprocess.run(command, check=False)
    sys.exit(completed_process.returncode)


def configure_logging(output_dir, verbosity):
    logging_filename = output_dir / "evaluation_campaigns.log"
    logging_format = "%(asctime)s\t---\t%(message)s"
    if verbosity == logging.DEBUG:
        logging_format = "%(asctime)s\t---\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s"

    logging.basicConfig(format=logging_format, level=verbosity)
    rotating_file_handler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotating_file_handler.setLevel(verbosity)
    rotating_file_handler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotating_file_handler)


def build_parser():
    parser = argparse.ArgumentParser(description="MalDataGen AppClassNet top200 campaign runner")

    parser.add_argument(
        "--campaign",
        "-c",
        help=f"Campaign list, comma separated list, sf demo alias, or sf2 SDV alias. Default: {DEFAULT_CAMPAIGN}",
        default=DEFAULT_CAMPAIGN,
        type=str,
        nargs="+",
    )
    parser.add_argument("--dryrun", "-d", help="show commands without running them", action="store_true")
    parser.add_argument("--pipenv", "-p", help="prefix subprocesses with pipenv run", action="store_true")
    parser.add_argument("--verbosity", "-v", default=DEFAULT_VERBOSITY_LEVEL, type=int)
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable),
        help="Python executable used to run main.py and plots.py",
    )
    parser.add_argument("--skip_plots", action="store_true", help="skip plots.py after each campaign run")
    parser.add_argument("--prepare_only", action="store_true", help="only materialize the selected AppClassNet CSV")
    parser.add_argument("--force_prepare", action="store_true", help="rebuild the prepared AppClassNet CSV even if it exists")
    parser.add_argument("--list_campaigns", action="store_true", help="list campaign names and exit")

    parser.add_argument(
        "--dataset_split",
        choices=[*APPCLASSNET_SPLITS, "all"],
        default="train",
        help="AppClassNet split materialized and passed to main.py",
    )
    parser.add_argument(
        "--dataset",
        "-a",
        dest="dataset_index",
        default=-1,
        type=int,
        help="compatibility option; ignored because AppClassNet is selected by --dataset_split",
    )
    parser.add_argument("--raw_root", default=str(DEFAULT_RAW_ROOT), help="path to AppClassNet top200 npy files")
    parser.add_argument("--converted_root", default=str(DEFAULT_CONVERTED_ROOT), help="path for prepared AppClassNet CSV files")
    parser.add_argument("--prepare_max_samples", default=-1, type=int, help="max rows written to the prepared CSV")
    parser.add_argument("--data_load_max_samples", default=-1, type=int, help="max rows passed to main.py after CSV loading")
    parser.add_argument("--prepare_chunk_size", default=DEFAULT_CHUNK_SIZE, type=int, help="rows per CSV write chunk")
    parser.add_argument(
        "--prepare_sampling",
        choices=["balanced", "head"],
        default="balanced",
        help="sampling strategy used when --prepare_max_samples is positive",
    )
    parser.add_argument(
        "--data_type",
        default=DEFAULT_DATA_TYPE,
        choices=["binary", "multiclass", "continuous"],
        help="forwarded to main.py; continuous is required to preserve AppClassNet feature values",
    )

    return parser


def main():
    global arguments
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.list_campaigns:
        print("\n".join(sorted(campaigns_available)))
        return 0

    reexec_with_configured_python_if_needed(arguments)

    for path in PATHS:
        (REPO_ROOT / path).mkdir(parents=True, exist_ok=True)

    output_dir = REPO_ROOT / "outputs" / f"appclassnet_top200_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    campaigns_chosen = choose_campaigns(arguments.campaign)
    if set(campaigns_chosen) == set(DEMO_CAMPAIGNS):
        output_dir = REPO_ROOT / "outputs" / "appclassnet_top200_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, arguments.verbosity)
    print_all_settings(arguments)
    warn_prepare_limit_if_needed(arguments, campaigns_chosen)

    raw_root = resolve_project_path(arguments.raw_root)
    converted_root = resolve_project_path(arguments.converted_root)

    if arguments.dryrun:
        dataset_path = expected_csv_path(
            arguments.dataset_split,
            converted_root,
            arguments.prepare_max_samples,
            arguments.prepare_sampling,
        )
        logging.info("Dry run: AppClassNet CSV preparation skipped. Expected dataset: %s", dataset_path)
    else:
        dataset_path = materialize_appclassnet_csv(
            split=arguments.dataset_split,
            raw_root=raw_root,
            output_root=converted_root,
            max_samples=arguments.prepare_max_samples,
            chunk_size=arguments.prepare_chunk_size,
            force=arguments.force_prepare,
            sampling=arguments.prepare_sampling,
        )

    if arguments.prepare_only:
        logging.info("Preparation-only mode completed. Dataset: %s", dataset_path)
        print(dataset_path)
        return 0

    default_python = str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable)
    child_python = arguments.python
    if arguments.pipenv and child_python == default_python:
        child_python = "python3"

    logging.info("datasets: [%s]", dataset_path)
    logging.info("campaigns: %s", campaigns_chosen)
    logging.info("data_type: %s", arguments.data_type)

    time_start_campaign = datetime.datetime.now()
    logging.info("\n\n\n")
    logging.info("##########################################")
    logging.info(" APPCLASSNET TOP200 EVALUATION ")
    logging.info("##########################################")
    time_start_evaluation = datetime.datetime.now()

    dataset_name = dataset_path.stem
    results_grouping = []

    for count_campaign, campaign_name in enumerate(campaigns_chosen, start=1):
        logging.info("\tCampaign %s %d/%d", campaign_name, count_campaign, len(campaigns_chosen))
        campaign = campaigns_available[campaign_name]
        params, values = zip(*campaign.items())
        combinations = [dict(zip(params, values_set)) for values_set in itertools.product(*values)]
        campaign_dir = output_dir / dataset_name / campaign_name

        for count_combination, combination in enumerate(combinations, start=1):
            output_dir_run = campaign_dir / f"combination_{count_combination}"
            logging.info("\t\tcombination %d/%d", count_combination, len(combinations))
            logging.info("\t\t%s", combination)

            command = build_main_command(
                child_python,
                dataset_path,
                output_dir_run,
                combination,
                arguments.verbosity,
                arguments.data_type,
                arguments.data_load_max_samples,
            )
            if arguments.pipenv:
                command = ["pipenv", "run", *command]

            time_start_experiment = datetime.datetime.now()
            logging.info("\t\t\tBegin Experiment: %s", time_start_experiment.strftime(TIME_FORMAT))
            run_cmd(command)
            time_end_experiment = datetime.datetime.now()
            logging.info("\t\t\tEnd                : %s", time_end_experiment.strftime(TIME_FORMAT))
            logging.info("\t\t\tExperiment duration: %s", time_end_experiment - time_start_experiment)

            results_path = output_dir_run / "EvaluationResults" / "Results.json"
            results_grouping.append(results_path)

            if arguments.skip_plots:
                continue

            final_plot = count_campaign == len(campaigns_chosen)
            plot_title = f"{combination['model_type']} {dataset_name}"
            plot_command = build_plot_command(
                child_python,
                dataset_path,
                output_dir_run,
                combination,
                plot_title,
                results_grouping,
                final_plot,
            )
            if arguments.pipenv:
                plot_command = ["pipenv", "run", *plot_command]

            time_start_plot = datetime.datetime.now()
            logging.info("\t\t\tBegin Plot: %s", time_start_plot.strftime(TIME_FORMAT))
            run_cmd(plot_command)
            time_end_plot = datetime.datetime.now()
            logging.info("\t\t\tEnd Plot     : %s", time_end_plot.strftime(TIME_FORMAT))
            logging.info("\t\t\tPlot duration: %s", time_end_plot - time_start_plot)

        time_end_campaign = datetime.datetime.now()
        logging.info("\t Campaign duration: %s", time_end_campaign - time_start_campaign)

    time_end_evaluation = datetime.datetime.now()
    logging.info("Evaluation duration: %s", time_end_evaluation - time_start_evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
