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
import json
import logging
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy
from Engine.Preprocessing.FeatureTransformManager import TransformManifest


DEFAULT_VERBOSITY_LEVEL = logging.INFO
DEFAULT_NUM_EPOCHS = 300
DEFAULT_DEMO_EPOCHS = 1
DEFAULT_K_FOLDS = 5
DEFAULT_DATA_TYPE = "continuous"
DEFAULT_SAVE_DATA = "True"
DEFAULT_SAMPLES_PER_CLASS = 256
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_BATCH_SIZE = 8192
DEFAULT_EVAL_BATCH_SIZE = 16384
DEFAULT_GENERATION_BATCH_SIZE = 8192
DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS = 1000
DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS = 500
DEFAULT_SCALER = "none"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

APPCLASSNET_NUM_CLASSES = 200
APPCLASSNET_NUM_FEATURES = 20
APPCLASSNET_SPLITS = ("train", "valid", "test")
APPCLASSNET_LABEL_COLUMN = "label"

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_ROOT = Path("Datasets/raw/AppClassNet/top200")
DEFAULT_CONVERTED_ROOT = Path("Datasets/converted/AppClassNet/top200")
RESULTS_ROOT = Path("results/appclassnet_top200")
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


@dataclass(frozen=True)
class RunArtifacts:
    model_name: str
    fold: int | None
    combination_dir: Path
    results_json_path: Path
    synthetic_train_manifest_path: Path | None
    synthetic_test_manifest_path: Path | None
    generated_data_dir: Path
    monitor_dir: Path
    data_space: str | None = None
    transform_id: str | None = None


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


def choose_campaigns(campaigns, full=False):
    if not campaigns:
        return list(campaigns_available.keys())

    tokens = split_campaign_tokens(campaigns)
    if tokens == ["sf"]:
        if full:
            return list(DEFAULT_CAMPAIGN)
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


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def estimate_array_bytes(shape, dtype):
    try:
        import numpy
    except ImportError:
        return None
    return int(numpy.prod(shape) * numpy.dtype(dtype).itemsize)


def log_array_summary(name, array):
    array_bytes = estimate_array_bytes(array.shape, array.dtype)
    logging.info(
        "%s shape=%s dtype=%s approx_ram=%s",
        name,
        array.shape,
        array.dtype,
        format_bytes(array_bytes) if array_bytes is not None else "unknown",
    )


def _json_scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_list(values):
    return [_json_scalar(value) for value in values]


def _array_has_nan_or_inf(numpy, values):
    if not numpy.issubdtype(values.dtype, numpy.number):
        return {"nan": False, "inf": False, "non_finite": False}
    finite_mask = numpy.isfinite(values)
    non_finite = bool((~finite_mask).any())
    return {
        "nan": bool(numpy.isnan(values).any()) if numpy.issubdtype(values.dtype, numpy.floating) else False,
        "inf": bool(numpy.isinf(values).any()) if numpy.issubdtype(values.dtype, numpy.number) else False,
        "non_finite": non_finite,
    }


def infer_feature_scale(min_value, max_value):
    if min_value is None or max_value is None or not math.isfinite(min_value) or not math.isfinite(max_value):
        return "unknown"
    if min_value >= 0.0 and max_value <= 1.0:
        return "[0,1]"
    if min_value >= -1.0 and max_value <= 1.0:
        return "[-1,1]"
    return "other"


def summarize_feature_matrix(numpy, x_values):
    if x_values.ndim != 2:
        return {
            "feature_count": None,
            "per_feature_min": [],
            "per_feature_max": [],
            "per_feature_mean": [],
            "per_feature_std": [],
            "global_min": None,
            "global_max": None,
            "scale_guess": "unknown",
        }

    feature_min = numpy.nanmin(x_values, axis=0)
    feature_max = numpy.nanmax(x_values, axis=0)
    feature_mean = numpy.nanmean(x_values, axis=0, dtype=numpy.float64)
    feature_std = numpy.nanstd(x_values, axis=0, dtype=numpy.float64)
    global_min = _json_scalar(float(numpy.nanmin(feature_min)))
    global_max = _json_scalar(float(numpy.nanmax(feature_max)))
    return {
        "feature_count": int(x_values.shape[1]),
        "per_feature_min": _json_list(feature_min),
        "per_feature_max": _json_list(feature_max),
        "per_feature_mean": _json_list(feature_mean),
        "per_feature_std": _json_list(feature_std),
        "global_min": global_min,
        "global_max": global_max,
        "scale_guess": infer_feature_scale(global_min, global_max),
    }


def summarize_labels(numpy, y_values, expected_classes):
    labels = numpy.asarray(y_values).reshape(-1)
    if not numpy.issubdtype(labels.dtype, numpy.number):
        return {
            "min": None,
            "max": None,
            "unique_class_count": 0,
            "unique_classes": [],
            "missing_classes": list(range(expected_classes)),
            "class_counts": {},
            "zero_based": False,
            "within_expected_interval": False,
            "one_based_1_to_num_classes": False,
        }

    finite_labels = labels[numpy.isfinite(labels)]
    if finite_labels.size == 0:
        unique_labels = numpy.array([], dtype=numpy.int64)
        counts = numpy.array([], dtype=numpy.int64)
        min_label = None
        max_label = None
    else:
        unique_labels, counts = numpy.unique(finite_labels, return_counts=True)
        min_label = _json_scalar(numpy.min(finite_labels))
        max_label = _json_scalar(numpy.max(finite_labels))

    expected_label_set = set(range(expected_classes))
    observed_label_set = {
        int(label)
        for label in unique_labels.tolist()
        if float(label).is_integer() and 0 <= int(label) < expected_classes
    }
    missing_classes = sorted(expected_label_set - observed_label_set)
    return {
        "min": min_label,
        "max": max_label,
        "unique_class_count": int(unique_labels.shape[0]),
        "unique_classes": _json_list(unique_labels),
        "missing_classes": missing_classes,
        "class_counts": {str(_json_scalar(label)): int(count) for label, count in zip(unique_labels, counts)},
        "zero_based": bool(min_label == 0) if min_label is not None else False,
        "within_expected_interval": (
            bool(min_label >= 0 and max_label < expected_classes)
            if min_label is not None and max_label is not None
            else False
        ),
        "one_based_1_to_num_classes": bool(min_label == 1 and max_label == expected_classes),
    }


def selected_campaigns_use_sigmoid(campaigns_chosen):
    output_activation_markers = ("last_activation", "last_layer_activation", "activation_output")
    for campaign_name in campaigns_chosen:
        campaign = campaigns_available[campaign_name]
        params, values = zip(*campaign.items())
        for values_set in itertools.product(*values):
            combination = dict(zip(params, values_set))
            for parameter, value in combination.items():
                if (
                    any(marker in parameter for marker in output_activation_markers)
                    and str(value).lower() == "sigmoid"
                ):
                    return True
    return False


def validate_scaler_for_campaigns(parsed_arguments, campaigns_chosen):
    if parsed_arguments.scaler not in {"none", "minmax", "standard"}:
        raise ValueError(f"Unsupported scaler: {parsed_arguments.scaler}")
    if (
        selected_campaigns_use_sigmoid(campaigns_chosen)
        and parsed_arguments.generator_transform == "preserve"
        and _option_was_provided("--generator_transform")
    ):
        raise ValueError(
            "Selected generator uses sigmoid output. Use --generator_transform minmax so only the generator "
            "input/output path uses [0,1], then inverse-transform synthetic data before evaluation."
        )


def _option_was_provided(option_name):
    prefix = f"{option_name}="
    return any(argument == option_name or argument.startswith(prefix) for argument in sys.argv[1:])


def effective_evaluation_mode(parsed_arguments) -> str:
    if getattr(parsed_arguments, "baseline_real_only", False):
        return "none"
    return getattr(parsed_arguments, "evaluation_mode", "both")


def normalize_preprocessing_arguments(parsed_arguments, campaigns_chosen):
    if _option_was_provided("--scaler"):
        logging.warning(
            "--scaler is a legacy argument. Prefer --classifier_transform or --generator_transform to avoid "
            "applying the same scaler to unrelated pipeline stages."
        )

    if parsed_arguments.source_profile == "appclassnet_top200":
        if parsed_arguments.scaler == "none" and not _option_was_provided("--feature_transform"):
            parsed_arguments.feature_transform = "preserve"
        elif parsed_arguments.scaler in {"minmax", "standard"} and not _option_was_provided("--feature_transform"):
            parsed_arguments.feature_transform = parsed_arguments.scaler

        if selected_campaigns_use_sigmoid(campaigns_chosen) and parsed_arguments.generator_transform == "auto":
            logging.warning(
                "Selected generator uses sigmoid output; resolving --generator_transform auto to minmax. "
                "Synthetic output will be inverse-transformed before evaluation when enabled."
            )
            parsed_arguments.generator_transform = "minmax"

    parsed_arguments._preprocessing_policy = FeatureTransformPolicy.for_profile(
        parsed_arguments.source_profile,
        feature_transform=parsed_arguments.feature_transform,
        generator_transform=parsed_arguments.generator_transform,
        classifier_transform=parsed_arguments.classifier_transform,
        evaluation_space=parsed_arguments.evaluation_space,
        allow_refit=parsed_arguments.allow_scaler_refit or parsed_arguments.source_profile == "legacy_csv",
        allow_double_transform=(
            parsed_arguments.allow_double_transform or parsed_arguments.source_profile == "legacy_csv"
        ),
        inverse_transform_synthetic=parsed_arguments.inverse_transform_synthetic,
    )
    return parsed_arguments


def run_input_diagnostics(parsed_arguments, raw_root, output_dir, campaigns_chosen):
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError("NumPy is required for AppClassNet input diagnostics.") from error

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diagnostics_dir / "input_diagnostics.json"

    mmap_mode = "r" if parsed_arguments.use_mmap or parsed_arguments.execution_mode == "batches" else None
    expected_classes = APPCLASSNET_NUM_CLASSES
    diagnostics = {
        "dataset": "AppClassNet top200",
        "raw_root": str(raw_root),
        "num_classes": expected_classes,
        "expected_label_interval": [0, expected_classes - 1],
        "expected_num_features": APPCLASSNET_NUM_FEATURES,
        "execution_mode": parsed_arguments.execution_mode,
        "use_mmap": parsed_arguments.use_mmap,
        "splits": {},
        "errors": [],
        "warnings": [],
    }

    any_non_normalized = False
    for split_name in APPCLASSNET_SPLITS:
        x_path, y_path = validate_raw_split(raw_root, split_name)
        x_values = numpy.load(x_path, mmap_mode=mmap_mode, allow_pickle=False)
        y_values = numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)

        x_finiteness = _array_has_nan_or_inf(numpy, x_values)
        y_finiteness = _array_has_nan_or_inf(numpy, y_values)
        feature_summary = summarize_feature_matrix(numpy, x_values)
        label_summary = summarize_labels(numpy, y_values, expected_classes)
        any_non_normalized = any_non_normalized or feature_summary["scale_guess"] == "other"

        split_diagnostics = {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "x_shape": list(x_values.shape),
            "y_shape": list(y_values.shape),
            "x_dtype": str(x_values.dtype),
            "y_dtype": str(y_values.dtype),
            "x_nan_or_inf": x_finiteness,
            "y_nan_or_inf": y_finiteness,
            "features": feature_summary,
            "labels": label_summary,
        }
        diagnostics["splits"][split_name] = split_diagnostics

        if x_values.ndim != 2 or x_values.shape[1] != APPCLASSNET_NUM_FEATURES:
            diagnostics["errors"].append(
                f"{split_name}_x must have shape (n, {APPCLASSNET_NUM_FEATURES}); got {tuple(x_values.shape)}."
            )
        if y_values.ndim != 1 or y_values.shape[0] != x_values.shape[0]:
            diagnostics["errors"].append(
                f"{split_name}_y must be one-dimensional with the same row count as {split_name}_x; "
                f"got X={tuple(x_values.shape)} y={tuple(y_values.shape)}."
            )
        if x_finiteness["non_finite"]:
            diagnostics["errors"].append(f"{split_name}_x contains NaN or inf values.")
        if y_finiteness["non_finite"]:
            diagnostics["errors"].append(f"{split_name}_y contains NaN or inf values.")
        if not label_summary["within_expected_interval"]:
            diagnostics["errors"].append(
                f"{split_name}_y labels must be in [0, {expected_classes - 1}] for num_classes={expected_classes}; "
                f"got min={label_summary['min']} max={label_summary['max']}."
            )
        if label_summary["one_based_1_to_num_classes"]:
            diagnostics["errors"].append(
                f"{split_name}_y appears to be 1-based (1..{expected_classes}); use --remap_labels_to_zero_based."
            )
        if split_name == "train" and label_summary["missing_classes"]:
            diagnostics["errors"].append(
                f"train split is missing {len(label_summary['missing_classes'])} class(es): "
                f"{label_summary['missing_classes']}."
            )
        if split_name == "test" and label_summary["missing_classes"]:
            diagnostics["warnings"].append(
                f"test split is missing {len(label_summary['missing_classes'])} class(es): "
                f"{label_summary['missing_classes']}."
            )

    if any_non_normalized and selected_campaigns_use_sigmoid(campaigns_chosen):
        diagnostics["warnings"].append(
            "X does not appear normalized to [0,1] or [-1,1], but a selected model uses sigmoid output."
        )

    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    with diagnostics_path.open("w", encoding="utf-8") as diagnostics_file:
        json.dump(diagnostics, diagnostics_file, indent=2, sort_keys=True)
        diagnostics_file.write("\n")

    logging.info("Input diagnostics saved to %s", diagnostics_path)
    for warning in diagnostics["warnings"]:
        logging.warning("DIAGNOSTIC WARNING: %s", warning)
    for error in diagnostics["errors"]:
        logging.error("DIAGNOSTIC ERROR: %s", error)

    return diagnostics_path, diagnostics


def _validate_samples_per_class(value, field_name):
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def select_stratified_indices(numpy, labels, samples_per_class, num_classes, seed, split_name):
    samples_per_class = _validate_samples_per_class(samples_per_class, f"{split_name}_samples_per_class")
    labels_array = numpy.asarray(labels).reshape(-1)
    random_generator = numpy.random.default_rng(seed)
    selected_by_class = {}
    counts_by_class = {}
    missing_classes = []
    short_classes = {}

    for class_id in range(num_classes):
        class_indices = numpy.flatnonzero(labels_array == class_id)
        if class_indices.shape[0] == 0:
            missing_classes.append(class_id)
            selected = numpy.array([], dtype=numpy.int64)
        else:
            sample_count = min(samples_per_class, int(class_indices.shape[0]))
            if sample_count < samples_per_class:
                short_classes[class_id] = int(class_indices.shape[0])
            selected = random_generator.choice(class_indices, size=sample_count, replace=False)
        selected_by_class[class_id] = selected
        counts_by_class[str(class_id)] = int(selected.shape[0])

    if missing_classes:
        raise ValueError(f"{split_name} split is missing class(es): {missing_classes}")

    selected_indices = numpy.concatenate([selected_by_class[class_id] for class_id in range(num_classes)])
    random_generator.shuffle(selected_indices)
    return selected_indices.astype(numpy.int64, copy=False), counts_by_class, short_classes


def load_selected_rows(numpy, x_values, y_values, indices, seed):
    order = numpy.argsort(indices)
    sorted_indices = indices[order]
    x_selected = numpy.asarray(x_values[sorted_indices], dtype=numpy.float32)
    y_selected = numpy.asarray(y_values[sorted_indices], dtype=numpy.int64)
    permutation = numpy.random.default_rng(seed).permutation(indices.shape[0])
    return x_selected[permutation], y_selected[permutation]


def build_baseline_classifier(parsed_arguments):
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for --baseline_real_only.") from error

    classifier_name = parsed_arguments.baseline_classifier
    if parsed_arguments.n_estimators is not None and parsed_arguments.n_estimators <= 0:
        raise ValueError("--n_estimators must be a positive integer when provided.")
    if parsed_arguments.max_depth is not None and parsed_arguments.max_depth <= 0:
        raise ValueError("--max_depth must be a positive integer when provided.")

    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(random_state=0)
    if classifier_name == "extra_trees":
        n_estimators = parsed_arguments.n_estimators or 50
        return ExtraTreesClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    if classifier_name == "random_forest_light":
        n_estimators = parsed_arguments.n_estimators or 30
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=parsed_arguments.max_depth,
            random_state=0,
            n_jobs=-1,
        )
    if classifier_name == "sgd":
        return SGDClassifier(random_state=0)
    raise ValueError(f"Unsupported baseline classifier: {classifier_name}")


def _feature_min_max(numpy, values, chunk_size):
    feature_min = None
    feature_max = None
    for start in range(0, values.shape[0], chunk_size):
        end = min(start + chunk_size, values.shape[0])
        chunk = numpy.asarray(values[start:end], dtype=numpy.float32)
        chunk_min = numpy.nanmin(chunk, axis=0)
        chunk_max = numpy.nanmax(chunk, axis=0)
        feature_min = chunk_min if feature_min is None else numpy.minimum(feature_min, chunk_min)
        feature_max = chunk_max if feature_max is None else numpy.maximum(feature_max, chunk_max)
    return feature_min, feature_max


def _iter_x_batches(numpy, x_values, chunk_size):
    for start in range(0, x_values.shape[0], chunk_size):
        end = min(start + chunk_size, x_values.shape[0])
        yield numpy.asarray(x_values[start:end], dtype=numpy.float32)


def _write_transformed_split(numpy, manager, x_values, y_values, output_x_path, output_y_path, chunk_size, split_name):
    output_x_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_x = numpy.lib.format.open_memmap(
        output_x_path,
        mode="w+",
        dtype=numpy.float32,
        shape=x_values.shape,
    )
    after_min = None
    after_max = None
    for start in range(0, x_values.shape[0], chunk_size):
        end = min(start + chunk_size, x_values.shape[0])
        transformed = manager.transform(
            x_values[start:end],
            split_name=split_name,
            input_space="source",
            output_space="transformed",
        )
        transformed_x[start:end] = transformed
        chunk_min = numpy.nanmin(transformed, axis=0)
        chunk_max = numpy.nanmax(transformed, axis=0)
        after_min = chunk_min if after_min is None else numpy.minimum(after_min, chunk_min)
        after_max = chunk_max if after_max is None else numpy.maximum(after_max, chunk_max)
    transformed_x.flush()
    numpy.save(output_y_path, numpy.asarray(y_values))
    return after_min, after_max


def preprocess_appclassnet_splits(parsed_arguments, raw_root, mode_name):
    try:
        import joblib
        import numpy
    except ImportError as error:
        raise RuntimeError("NumPy and joblib are required for AppClassNet preprocessing.") from error

    policy = getattr(parsed_arguments, "_preprocessing_policy", None)
    if policy is None:
        policy = FeatureTransformPolicy.for_profile(
            parsed_arguments.source_profile,
            feature_transform=parsed_arguments.feature_transform,
            generator_transform=parsed_arguments.generator_transform,
            classifier_transform=parsed_arguments.classifier_transform,
            evaluation_space=parsed_arguments.evaluation_space,
            allow_refit=parsed_arguments.allow_scaler_refit,
            allow_double_transform=parsed_arguments.allow_double_transform,
            inverse_transform_synthetic=parsed_arguments.inverse_transform_synthetic,
        )
    preprocessing_dir = REPO_ROOT / RESULTS_ROOT / mode_name / "preprocessing"
    scaled_root = preprocessing_dir / "scaled_npy"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    scaled_root.mkdir(parents=True, exist_ok=True)

    mmap_mode = "r" if parsed_arguments.use_mmap or parsed_arguments.execution_mode == "batches" else None
    train_x_path, _ = validate_raw_split(raw_root, "train")
    train_x_values = numpy.load(train_x_path, mmap_mode=mmap_mode, allow_pickle=False)
    chunk_size = max(1, int(parsed_arguments.prepare_chunk_size))

    logging.info(
        "AppClassNet preprocessing: feature_transform=%s mode=%s",
        policy.feature_transform,
        mode_name,
    )
    manager = FeatureTransformManager(policy, stage="feature", output_space="transformed")
    manager.partial_fit_batches(_iter_x_batches(numpy, train_x_values, chunk_size), split_name="train")
    train_feature_min, train_feature_max = _feature_min_max(numpy, train_x_values, chunk_size)

    scaler_path = preprocessing_dir / "scaler.joblib"
    manager.save(scaler_path)

    preprocessing_report = {
        "scaler": parsed_arguments.scaler,
        "source_profile": policy.source_profile,
        "feature_transform": policy.feature_transform,
        "generator_transform": policy.generator_transform,
        "classifier_transform": policy.classifier_transform,
        "evaluation_space": policy.evaluation_space,
        "transform_id": manager.transform_id,
        "mode": mode_name,
        "fit_split": "train",
        "scaler_path": str(scaler_path),
        "scaled_root": str(scaled_root),
        "splits": {},
    }

    for split_name in APPCLASSNET_SPLITS:
        x_path, y_path = validate_raw_split(raw_root, split_name)
        x_values = numpy.load(x_path, mmap_mode=mmap_mode, allow_pickle=False)
        y_values = numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)
        before_min, before_max = _feature_min_max(numpy, x_values, chunk_size)
        scaled_x_path = scaled_root / f"{split_name}_x.npy"
        scaled_y_path = scaled_root / f"{split_name}_y.npy"
        after_min, after_max = _write_transformed_split(
            numpy,
            manager,
            x_values,
            y_values,
            scaled_x_path,
            scaled_y_path,
            chunk_size,
            split_name,
        )
        preprocessing_report["splits"][split_name] = {
            "x_path": str(scaled_x_path),
            "y_path": str(scaled_y_path),
            "shape": list(x_values.shape),
            "feature_min_before_scaling": _json_list(before_min),
            "feature_max_before_scaling": _json_list(before_max),
            "feature_min_after_scaling": _json_list(after_min),
            "feature_max_after_scaling": _json_list(after_max),
        }
        logging.info("AppClassNet preprocessing: wrote %s split to %s", split_name, scaled_x_path)

    train_stats = preprocessing_report["splits"]["train"]
    preprocessing_report["feature_min_before_scaling"] = train_stats["feature_min_before_scaling"]
    preprocessing_report["feature_max_before_scaling"] = train_stats["feature_max_before_scaling"]
    preprocessing_report["feature_min_after_scaling"] = train_stats["feature_min_after_scaling"]
    preprocessing_report["feature_max_after_scaling"] = train_stats["feature_max_after_scaling"]

    report_path = preprocessing_dir / "preprocessing_stats.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(preprocessing_report, report_file, indent=2, sort_keys=True)
        report_file.write("\n")

    logging.info("AppClassNet preprocessing scaler saved to %s", scaler_path)
    logging.info("AppClassNet preprocessing stats saved to %s", report_path)

    manifest = TransformManifest(
        source_profile=policy.source_profile,
        paths={
            "raw_root": str(raw_root),
            "preprocessed_root": str(scaled_root),
            "scaler_path": str(scaler_path),
            "stats_path": str(report_path),
        },
        original_ranges={
            "train": [_json_list(train_feature_min), _json_list(train_feature_max)],
        },
        current_ranges={
            split_name: [
                preprocessing_report["splits"][split_name]["feature_min_after_scaling"],
                preprocessing_report["splits"][split_name]["feature_max_after_scaling"],
            ]
            for split_name in APPCLASSNET_SPLITS
        },
        transformations=manager.transform_history,
        transform_id=manager.transform_id,
        train_fit={
            "fit_split": "train",
            "operation": manager.operation,
            "input_range": manager.input_range,
            "output_range": manager.output_range,
        },
        split_usage={
            split_name: {
                "x_path": preprocessing_report["splits"][split_name]["x_path"],
                "y_path": preprocessing_report["splits"][split_name]["y_path"],
                "used_scaler_fit_split": "train",
                "transform_id": manager.transform_id,
            }
            for split_name in APPCLASSNET_SPLITS
        },
        generator_input_space="generator" if policy.generator_transform != "preserve" else "source",
        synthetic_output_space=(
            "source"
            if policy.generator_transform == "preserve" or policy.inverse_transform_synthetic
            else "generator"
        ),
        evaluation_space=policy.evaluation_space,
        classifier_input_space=(
            "classifier" if policy.classifier_transform not in {"preserve", "auto"} else "source"
        ),
        inverse_transform_synthetic=policy.inverse_transform_synthetic,
        warnings=[],
        validations=[
            {
                "name": "fit_only_train",
                "status": "passed",
                "detail": "FeatureTransformManager fitted on train and reused for valid/test.",
            }
        ],
    )
    manifest_path = preprocessing_dir / "preprocessing_manifest.json"
    manifest.save(manifest_path)
    logging.info("AppClassNet preprocessing manifest saved to %s", manifest_path)
    return scaled_root, scaler_path, manifest_path


def run_real_real_baseline(parsed_arguments, raw_root, output_dir):
    try:
        import numpy
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.metrics import f1_score
    except ImportError as error:
        raise RuntimeError("NumPy and scikit-learn are required for --baseline_real_only.") from error

    time_start = datetime.datetime.now()
    baseline_dir = output_dir / "baseline_real_only"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = baseline_dir / "metrics.json"

    mmap_mode = "r" if parsed_arguments.use_mmap or parsed_arguments.execution_mode == "batches" else None
    train_x_path, train_y_path = validate_raw_split(raw_root, "train")
    test_x_path, test_y_path = validate_raw_split(raw_root, "test")
    train_x_values = numpy.load(train_x_path, mmap_mode=mmap_mode, allow_pickle=False)
    train_y_values = numpy.load(train_y_path, mmap_mode=mmap_mode, allow_pickle=False)
    test_x_values = numpy.load(test_x_path, mmap_mode=mmap_mode, allow_pickle=False)
    test_y_values = numpy.load(test_y_path, mmap_mode=mmap_mode, allow_pickle=False)

    logging.info("Baseline real-real: selecting train samples per class=%s", parsed_arguments.train_samples_per_class)
    train_indices, train_counts, train_short_classes = select_stratified_indices(
        numpy,
        train_y_values,
        parsed_arguments.train_samples_per_class,
        APPCLASSNET_NUM_CLASSES,
        seed=0,
        split_name="train",
    )
    logging.info("Baseline real-real: selecting test samples per class=%s", parsed_arguments.test_samples_per_class)
    test_indices, test_counts, test_short_classes = select_stratified_indices(
        numpy,
        test_y_values,
        parsed_arguments.test_samples_per_class,
        APPCLASSNET_NUM_CLASSES,
        seed=1,
        split_name="test",
    )

    train_x, train_y = load_selected_rows(numpy, train_x_values, train_y_values, train_indices, seed=2)
    test_x, test_y = load_selected_rows(numpy, test_x_values, test_y_values, test_indices, seed=3)

    policy = getattr(parsed_arguments, "_preprocessing_policy", None)
    if policy is None:
        policy = FeatureTransformPolicy.for_profile(parsed_arguments.source_profile)
    classifier_policy = FeatureTransformPolicy.for_profile(
        policy.source_profile,
        feature_transform="preserve",
        generator_transform="preserve",
        classifier_transform=parsed_arguments.classifier_transform,
        evaluation_space=policy.evaluation_space,
        allow_refit=policy.allow_refit,
        allow_double_transform=policy.allow_double_transform,
        inverse_transform_synthetic=policy.inverse_transform_synthetic,
    )
    classifier_manager = FeatureTransformManager(classifier_policy, stage="classifier", output_space="classifier")
    classifier_manager.fit(train_x, split_name="train")
    train_x = classifier_manager.transform(train_x, split_name="train", input_space="source", output_space="classifier")
    test_x = classifier_manager.transform(test_x, split_name="test", input_space="source", output_space="classifier")

    classifier = build_baseline_classifier(parsed_arguments)
    logging.info("Baseline real-real: training %s on %s", classifier.__class__.__name__, train_x.shape)
    classifier.fit(train_x, train_y)
    logging.info("Baseline real-real: predicting %s", test_x.shape)
    predictions = classifier.predict(test_x)

    metrics = {
        "mode": "baseline_real_only",
        "classifier": parsed_arguments.baseline_classifier,
        "classifier_params": classifier.get_params(),
        "num_classes": APPCLASSNET_NUM_CLASSES,
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "train_samples_per_class_requested": parsed_arguments.train_samples_per_class,
        "test_samples_per_class_requested": parsed_arguments.test_samples_per_class,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "data_space": "source",
        "feature_range": {
            "train": summarize_feature_matrix(numpy, train_x),
            "test": summarize_feature_matrix(numpy, test_x),
        },
        "train_short_classes": {str(key): value for key, value in train_short_classes.items()},
        "test_short_classes": {str(key): value for key, value in test_short_classes.items()},
        "scaler": {
            "legacy_name": parsed_arguments.scaler,
            "classifier_transform": parsed_arguments.classifier_transform,
            "transform_id": classifier_manager.transform_id,
            "feature_range": [0, 1] if parsed_arguments.classifier_transform == "minmax" else None,
            "data_min": (
                _json_list(classifier_manager.scaler.data_min_)
                if hasattr(classifier_manager.scaler, "data_min_") else None
            ),
            "data_max": (
                _json_list(classifier_manager.scaler.data_max_)
                if hasattr(classifier_manager.scaler, "data_max_") else None
            ),
            "mean": (
                _json_list(classifier_manager.scaler.mean_)
                if hasattr(classifier_manager.scaler, "mean_") else None
            ),
            "scale": (
                _json_list(classifier_manager.scaler.scale_)
                if hasattr(classifier_manager.scaler, "scale_") else None
            ),
        },
        "metrics": {
            "Accuracy": float(accuracy_score(test_y, predictions)),
            "MacroF1": float(f1_score(test_y, predictions, average="macro", zero_division=0)),
            "WeightedF1": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
            "BalancedAccuracy": float(balanced_accuracy_score(test_y, predictions)),
        },
        "paths": {
            "train_x": str(train_x_path),
            "train_y": str(train_y_path),
            "test_x": str(test_x_path),
            "test_y": str(test_y_path),
            "metrics": str(metrics_path),
        },
        "duration_seconds": (datetime.datetime.now() - time_start).total_seconds(),
    }

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")

    logging.info("Baseline real-real metrics saved to %s", metrics_path)
    return metrics_path, metrics


def _load_json_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as error:
        logging.warning("Could not read JSON %s: %s", path, error)
        return None


def _metric_value(metrics_block, metric_name):
    value = metrics_block.get(metric_name)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return None


def _extract_evaluation_metrics(results_data, evaluation_name):
    evaluation_block = results_data.get(evaluation_name)
    if not isinstance(evaluation_block, dict):
        return None, {}
    for classifier_name, classifier_block in evaluation_block.items():
        if not isinstance(classifier_block, dict):
            continue
        fold_metrics = None
        for key, value in classifier_block.items():
            if key.endswith("-Fold") and isinstance(value, dict):
                fold_metrics = value
                break
        if fold_metrics is None:
            summary = classifier_block.get("Summary")
            if isinstance(summary, dict):
                fold_metrics = summary
        if fold_metrics is None:
            continue
        return classifier_name, {
            "Accuracy": _metric_value(fold_metrics, "Accuracy"),
            "MacroF1": _metric_value(fold_metrics, "MacroF1"),
            "WeightedF1": _metric_value(fold_metrics, "WeightedF1"),
            "BalancedAccuracy": _metric_value(fold_metrics, "BalancedAccuracy"),
        }
    return None, {}


def _extract_batch_metadata(results_data, evaluation_name):
    batch_block = results_data.get("BatchClassifier")
    if not isinstance(batch_block, dict):
        return {}
    for fold_block in batch_block.values():
        if isinstance(fold_block, dict) and isinstance(fold_block.get(evaluation_name), dict):
            return fold_block[evaluation_name]
    return {}


def _synthetic_manifest_summary(output_dir_run=None, manifest_path=None, max_batches=200):
    try:
        import numpy
    except ImportError:
        return {"data_space": "unknown", "feature_range": None, "manifest_path": None}

    if manifest_path is not None:
        manifests = [Path(manifest_path)]
    else:
        manifests = sorted(Path(output_dir_run).rglob("synthetic_batches/manifest.json"))
        if not manifests:
            manifests = sorted(Path(output_dir_run).rglob("synthetic_batches/*/manifest.json"))
    manifests = [path for path in manifests if path and Path(path).is_file()]
    if not manifests:
        return {"data_space": "unknown", "feature_range": None, "manifest_path": None}
    manifest_path = Path(manifests[0])
    manifest = _load_json_file(manifest_path) or {}
    feature_min = None
    feature_max = None
    batches_seen = 0
    for batches in manifest.get("batches_by_class", {}).values():
        for batch in batches:
            if batches_seen >= max_batches:
                break
            batch_path = Path(batch.get("path", ""))
            if not batch_path.is_absolute():
                batch_path = (manifest_path.parent / batch_path).resolve()
            if not batch_path.is_file():
                batch_path = Path(batch.get("path", ""))
            if not batch_path.is_file():
                continue
            try:
                if batch_path.suffix == ".npy":
                    values = numpy.load(batch_path, mmap_mode="r", allow_pickle=False)
                    if "offset_start" in batch:
                        values = values[int(batch["offset_start"]):int(batch["offset_end"])]
                else:
                    values = numpy.loadtxt(batch_path, delimiter=",", dtype=numpy.float32)
                values = numpy.asarray(values, dtype=numpy.float32)
                if values.ndim == 1:
                    values = values.reshape(1, -1)
                current_min = float(numpy.nanmin(values))
                current_max = float(numpy.nanmax(values))
                feature_min = current_min if feature_min is None else min(feature_min, current_min)
                feature_max = current_max if feature_max is None else max(feature_max, current_max)
                batches_seen += 1
            except Exception as error:
                logging.warning("Could not summarize synthetic batch %s: %s", batch_path, error)
    return {
        "data_space": manifest.get("data_space", "unknown"),
        "transform_id": manifest.get("transform_id"),
        "transform_history": manifest.get("transform_history", []),
        "feature_range": [feature_min, feature_max] if feature_min is not None else None,
        "manifest_path": str(manifest_path),
        "sampled_batches": int(batches_seen),
    }


def _empty_evaluation_summary(status, reason):
    return {
        "status": status,
        "reason": reason,
        "classifier": None,
        "classifier_params": None,
        "real_samples_by_class": {},
        "synthetic_samples_by_class": {},
        "data_space": None,
        "feature_range": None,
        "synthetic_manifest": None,
        "Accuracy": None,
        "MacroF1": None,
        "WeightedF1": None,
        "BalancedAccuracy": None,
        "duration_seconds": None,
    }


def _build_method_summary(results_data, evaluation_name, output_dir_run, synthetic_summary, active):
    if not active:
        return _empty_evaluation_summary("not_run", f"Skipped by evaluation_mode.")
    classifier_name, metric_values = _extract_evaluation_metrics(results_data, evaluation_name)
    metadata = _extract_batch_metadata(results_data, evaluation_name)
    required_metrics = ("Accuracy", "BalancedAccuracy", "MacroF1", "WeightedF1")
    missing_metrics = [metric for metric in required_metrics if metric_values.get(metric) is None]
    status = "completed" if classifier_name and not missing_metrics else "not_run"
    reason = None
    if status != "completed":
        if not classifier_name:
            reason = f"{evaluation_name} requested but no classifier metrics were found in Results.json."
        else:
            reason = f"{evaluation_name} requested but missing metric(s): {', '.join(missing_metrics)}."
    duration_seconds = None
    if metadata:
        duration_seconds = float(metadata.get("training_time_seconds", 0.0) or 0.0) + float(
            metadata.get("evaluation_time_seconds", 0.0) or 0.0
        )
    return {
        "status": status,
        "reason": reason,
        "classifier": classifier_name or metadata.get("classifier"),
        "classifier_params": {
            key: value
            for key, value in metadata.items()
            if key in {"n_estimators", "max_depth", "max_samples", "class_weight", "subset_quota_per_class"}
        } if metadata else None,
        "real_samples_by_class": metadata.get("real_samples_used_by_class", {}),
        "synthetic_samples_by_class": metadata.get("synthetic_samples_used_by_class", {}),
        "data_space": synthetic_summary.get("data_space", "source"),
        "feature_range": synthetic_summary.get("feature_range"),
        "synthetic_manifest": synthetic_summary.get("manifest_path"),
        "Accuracy": metric_values.get("Accuracy"),
        "MacroF1": metric_values.get("MacroF1"),
        "WeightedF1": metric_values.get("WeightedF1"),
        "BalancedAccuracy": metric_values.get("BalancedAccuracy"),
        "duration_seconds": duration_seconds,
    }


def _artifact_results_path(artifact_or_path):
    return Path(getattr(artifact_or_path, "results_json_path", artifact_or_path))


def _artifact_combination_dir(artifact_or_path, results_path):
    return Path(getattr(artifact_or_path, "combination_dir", Path(results_path).parents[1]))


def _artifact_manifest_path(artifact_or_path, evaluation_name):
    if evaluation_name == "TR-TS":
        return getattr(artifact_or_path, "synthetic_test_manifest_path", None)
    if evaluation_name == "TS-TR":
        return getattr(artifact_or_path, "synthetic_train_manifest_path", None)
    return None


def _requested_evaluations(evaluation_mode):
    return {
        "TR-TS": evaluation_mode in {"tr_ts", "both"},
        "TS-TR": evaluation_mode in {"ts_tr", "both"},
    }


def validate_requested_evaluations_completed(payload, parsed_arguments, results_paths, stopped_function):
    for evaluation_name, requested in _requested_evaluations(parsed_arguments.evaluation_mode).items():
        if not requested:
            continue
        summary = payload.get(evaluation_name, {})
        if summary.get("status") == "completed":
            continue
        missing_artifacts = []
        normalized_results_paths = [_artifact_results_path(item) for item in results_paths]
        if not normalized_results_paths:
            missing_artifacts.append("EvaluationResults/Results.json")
        else:
            missing_artifacts.extend(str(path) for path in normalized_results_paths if not Path(path).is_file())
        if not summary.get("synthetic_manifest"):
            expected_split = "test" if evaluation_name == "TR-TS" else "train"
            missing_artifacts.append(f"synthetic_batches/{expected_split}/manifest.json")
        raise RuntimeError(
            f"Requested evaluation {evaluation_name} was not completed; "
            f"status={summary.get('status')}; reason={summary.get('reason')}; "
            f"missing_artifact={missing_artifacts or None}; stopped_function={stopped_function}."
        )


def print_results_payload(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_baseline_batches_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "TR-TR": {
            "status": "completed",
            "reason": None,
            "classifier": metrics.get("classifier"),
            "classifier_params": metrics.get("classifier_params"),
            "real_samples_by_class": {
                "train": metrics.get("train_class_counts", {}),
                "test": metrics.get("test_class_counts", {}),
            },
            "synthetic_samples_by_class": {},
            "data_space": metrics.get("data_space", "source"),
            "feature_range": metrics.get("feature_range", {}),
            "synthetic_manifest": None,
            "duration_seconds": metrics.get("duration_seconds"),
            **metrics.get("metrics", {}),
        },
        "TR-TS": _empty_evaluation_summary("not_run", "baseline_real_only executes only TR-TR."),
        "TS-TR": _empty_evaluation_summary("not_run", "baseline_real_only executes only TR-TR."),
    }
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    logging.info("AppClassNet batches metrics saved to %s", output_path)
    return output_path, payload


def write_batches_metrics(results_paths, output_path, parsed_arguments):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "TR-TR": _empty_evaluation_summary("not_run", "Synthetic run does not execute TR-TR."),
        "TR-TS": _empty_evaluation_summary("not_run", "TR-TS was not requested."),
        "TS-TR": _empty_evaluation_summary("not_run", "TS-TR was not requested."),
    }
    run_tr_ts = parsed_arguments.evaluation_mode in {"tr_ts", "both"}
    run_ts_tr = parsed_arguments.evaluation_mode in {"ts_tr", "both"}
    if run_tr_ts:
        payload["TR-TS"] = _empty_evaluation_summary("not_run", "TR-TS requested but not yet executed.")
    if run_ts_tr:
        payload["TS-TR"] = _empty_evaluation_summary("not_run", "TS-TR requested but not yet executed.")
    for artifact_or_path in results_paths:
        results_path = _artifact_results_path(artifact_or_path)
        results_data = _load_json_file(results_path)
        if not isinstance(results_data, dict):
            continue
        output_dir_run = _artifact_combination_dir(artifact_or_path, results_path)
        tr_ts_manifest = _artifact_manifest_path(artifact_or_path, "TR-TS")
        ts_tr_manifest = _artifact_manifest_path(artifact_or_path, "TS-TR")
        if tr_ts_manifest is None:
            candidate = output_dir_run / "DataGenerated" / "synthetic_batches" / "test" / "manifest.json"
            tr_ts_manifest = candidate if candidate.is_file() else None
        if ts_tr_manifest is None:
            candidate = output_dir_run / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json"
            ts_tr_manifest = candidate if candidate.is_file() else None
        tr_ts_synthetic_summary = _synthetic_manifest_summary(
            output_dir_run,
            tr_ts_manifest,
        )
        ts_tr_synthetic_summary = _synthetic_manifest_summary(
            output_dir_run,
            ts_tr_manifest,
        )
        payload["TR-TS"] = _build_method_summary(
            results_data, "TR-TS", output_dir_run, tr_ts_synthetic_summary, run_tr_ts
        )
        payload["TS-TR"] = _build_method_summary(
            results_data, "TS-TR", output_dir_run, ts_tr_synthetic_summary, run_ts_tr
        )
    validate_requested_evaluations_completed(payload, parsed_arguments, results_paths, "write_batches_metrics")
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    logging.info("AppClassNet batches metrics saved to %s", output_path)
    return output_path, payload


def log_appclassnet_memory_plan(parsed_arguments, raw_root):
    try:
        import numpy
    except ImportError:
        logging.warning("NumPy is unavailable; memory dry-run cannot inspect .npy shapes.")
        return

    logging.info("Memory plan: execution_mode=%s", parsed_arguments.execution_mode)
    logging.info("Memory plan: batch_size=%s", parsed_arguments.batch_size)
    logging.info("Memory plan: eval_batch_size=%s", parsed_arguments.eval_batch_size)
    logging.info("Memory plan: generation_batch_size=%s", parsed_arguments.generation_batch_size)
    logging.info("Memory plan: use_mmap=%s", parsed_arguments.use_mmap)
    logging.info("Memory plan: max_train_samples=%s", parsed_arguments.max_train_samples)
    logging.info("Memory plan: max_samples_per_class=%s", parsed_arguments.max_samples_per_class)

    total_feature_bytes = 0
    total_label_bytes = 0
    for split_name in iter_splits(parsed_arguments.dataset_split):
        x_path, y_path = validate_raw_split(raw_root, split_name)
        mmap_mode = "r" if parsed_arguments.use_mmap or parsed_arguments.execution_mode == "batches" else None
        x_values = numpy.load(x_path, mmap_mode=mmap_mode, allow_pickle=False)
        y_values = numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)
        log_array_summary(f"AppClassNet {split_name} X", x_values)
        log_array_summary(f"AppClassNet {split_name} y", y_values)
        total_feature_bytes += estimate_array_bytes(x_values.shape, x_values.dtype) or 0
        total_label_bytes += estimate_array_bytes(y_values.shape, y_values.dtype) or 0

    rows_for_one_hot = parsed_arguments.max_train_samples
    if rows_for_one_hot is None:
        rows_for_one_hot = max(
            numpy.load(validate_raw_split(raw_root, split_name)[1], mmap_mode="r", allow_pickle=False).shape[0]
            for split_name in iter_splits(parsed_arguments.dataset_split)
        )
    one_hot_bytes = int(rows_for_one_hot * APPCLASSNET_NUM_CLASSES * numpy.dtype(numpy.float32).itemsize)
    batch_feature_bytes = int(parsed_arguments.batch_size * APPCLASSNET_NUM_FEATURES * numpy.dtype(numpy.float32).itemsize)
    batch_one_hot_bytes = int(parsed_arguments.batch_size * APPCLASSNET_NUM_CLASSES * numpy.dtype(numpy.float32).itemsize)

    logging.info("Memory plan: total raw feature bytes=%s", format_bytes(total_feature_bytes))
    logging.info("Memory plan: total raw label bytes=%s", format_bytes(total_label_bytes))
    logging.info(
        "Memory plan: global one-hot upper bound for %s rows would be %s; batch one-hot is %s",
        rows_for_one_hot,
        format_bytes(one_hot_bytes),
        format_bytes(batch_one_hot_bytes),
    )
    logging.info("Memory plan: feature batch approx=%s", format_bytes(batch_feature_bytes))


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


def annotate_explicit_cli_arguments(parsed_arguments, raw_args):
    explicit = set()
    for index, argument in enumerate(raw_args):
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        explicit.add(option[2:].replace("-", "_"))
    parsed_arguments._explicit_cli_options = explicit
    return parsed_arguments


def resolve_effective_argument(parsed_arguments, combination, parameter):
    explicit = set(getattr(parsed_arguments, "_explicit_cli_options", set()))
    requested_value = getattr(parsed_arguments, parameter, None)
    campaign_has_value = parameter in combination
    campaign_value = combination.get(parameter)

    if parameter in explicit:
        effective_value = requested_value
        source = "cli"
    elif campaign_has_value:
        effective_value = campaign_value
        source = "campaign"
    else:
        effective_value = requested_value
        source = "default"

    logging.info(
        "Argument resolution: %s requested=%s campaign=%s effective=%s source=%s",
        parameter,
        requested_value,
        campaign_value,
        effective_value,
        source,
    )
    return effective_value, source


def synthetic_required_samples_per_class(parsed_arguments, combination):
    train_value, _ = resolve_effective_argument(parsed_arguments, combination, "synthetic_train_samples_per_class")
    test_value, _ = resolve_effective_argument(parsed_arguments, combination, "synthetic_test_samples_per_class")
    if train_value is None and test_value is None:
        return None
    return int(train_value or 0) + int(test_value or 0)


def build_number_samples_per_class_plan(samples_per_class):
    return ",".join(f"{class_id}:{int(samples_per_class)}" for class_id in range(APPCLASSNET_NUM_CLASSES))


def _remove_command_option(command, parameter):
    option = f"--{parameter}"
    cleaned = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == option:
            index += 1
            while index < len(command) and not str(command[index]).startswith("--"):
                index += 1
            continue
        cleaned.append(token)
        index += 1
    command[:] = cleaned


def deduplicate_command_options(command):
    seen = {}
    prefix = []
    option_parts = []
    index = 0
    while index < len(command):
        token = str(command[index])
        if not token.startswith("--"):
            prefix.append(command[index])
            index += 1
            continue
        option = token
        values = []
        index += 1
        while index < len(command) and not str(command[index]).startswith("--"):
            values.append(str(command[index]))
            index += 1
        values_tuple = tuple(values)
        if option in seen:
            if seen[option] != values_tuple:
                raise ValueError(
                    f"DuplicateCommandArgumentConflict: {option} values={seen[option]} and {values_tuple}"
                )
            continue
        seen[option] = values_tuple
        option_parts.extend([option, *values])
    return [*prefix, *option_parts]


GENERATION_QUOTA_PARAMETERS = {
    "number_samples_per_class",
    "sample_plan",
    "train_samples_per_class",
    "test_samples_per_class",
    "synthetic_train_samples_per_class",
    "synthetic_test_samples_per_class",
}


def build_main_command(
    python_executable,
    dataset_path,
    output_dir_run,
    combination,
    verbosity,
    data_type,
    data_load_max_samples,
    normal_classifier,
    batch_classifier_subset_size,
    parsed_arguments,
):
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

    if normal_classifier:
        command.extend(["--normal_classifier", normal_classifier])
        command.extend(["--batch_classifier_subset_size", str(batch_classifier_subset_size)])

    command.extend([
        "--source_profile",
        parsed_arguments.source_profile,
        "--feature_transform",
        "preserve",
        "--generator_transform",
        parsed_arguments.generator_transform,
        "--classifier_transform",
        parsed_arguments.classifier_transform,
        "--evaluation_space",
        parsed_arguments.evaluation_space,
        "--evaluation_mode",
        parsed_arguments.evaluation_mode,
        "--random_state",
        str(getattr(parsed_arguments, "random_state", 0)),
    ])
    if parsed_arguments.allow_double_transform:
        command.append("--allow_double_transform")
    if parsed_arguments.allow_scaler_refit:
        command.append("--allow_scaler_refit")
    if parsed_arguments.inverse_transform_synthetic:
        command.append("--inverse_transform_synthetic")
    for parameter in (
            "train_samples_per_class",
            "test_samples_per_class",
            "synthetic_train_samples_per_class",
            "synthetic_test_samples_per_class"):
        effective_value, _ = resolve_effective_argument(parsed_arguments, combination, parameter)
        if effective_value is not None:
            command.extend([f"--{parameter}", str(effective_value)])

    required_synthetic = synthetic_required_samples_per_class(parsed_arguments, combination)
    if required_synthetic is not None:
        command.extend(["--number_samples_per_class", build_number_samples_per_class_plan(required_synthetic)])

    for parameter, value in combination.items():
        if parameter in GENERATION_QUOTA_PARAMETERS:
            continue
        append_cli_value(command, parameter, value)

        if (
            parameter == "variational_autoencoder_dense_layer_sizes_encoder"
            and "variational_autoencoder_dense_layer_sizes_decoder" not in combination
        ):
            layers = str(value).split()[::-1]
            command.extend(["--variational_autoencoder_dense_layer_sizes_decoder", *layers])

    return deduplicate_command_options(command)


def build_batch_main_command(python_executable, raw_root, output_dir_run, combination, verbosity, data_type, parsed_arguments):
    train_x_path, train_y_path = validate_raw_split(raw_root, parsed_arguments.dataset_split if parsed_arguments.dataset_split != "all" else "train")
    valid_x_path, valid_y_path = validate_raw_split(raw_root, "valid")
    test_x_path, test_y_path = validate_raw_split(raw_root, "test")
    command = [
        python_executable,
        str(REPO_ROOT / "main.py"),
        "--data_format",
        "npy_xy",
        "--train_x_path",
        str(train_x_path),
        "--train_y_path",
        str(train_y_path),
        "--valid_x_path",
        str(valid_x_path),
        "--valid_y_path",
        str(valid_y_path),
        "--test_x_path",
        str(test_x_path),
        "--test_y_path",
        str(test_y_path),
        "--target_type",
        "multiclass",
        "--feature_type",
        "continuous",
        "--num_classes",
        str(APPCLASSNET_NUM_CLASSES),
        "--split_mode",
        "provided",
        "--data_type",
        data_type,
        "--verbosity",
        str(verbosity),
        "--output_dir",
        str(output_dir_run),
        "--execution_mode",
        "batches",
        "--batch_size",
        str(parsed_arguments.batch_size),
        "--eval_batch_size",
        str(parsed_arguments.eval_batch_size),
        "--generation_batch_size",
        str(parsed_arguments.generation_batch_size),
        "--eval_classifier",
        parsed_arguments.eval_classifier,
        "--batch_classifier",
        parsed_arguments.eval_classifier,
        "--batch_classifier_subset_size",
        str(parsed_arguments.batch_classifier_subset_size),
        "--min_samples_per_class_required",
        str(parsed_arguments.min_samples_per_class_required),
        "--generation_strategy",
        parsed_arguments.generation_strategy,
        "--classes_per_group",
        str(parsed_arguments.classes_per_group),
        "--source_profile",
        parsed_arguments.source_profile,
        "--feature_transform",
        "preserve",
        "--generator_transform",
        parsed_arguments.generator_transform,
        "--classifier_transform",
        parsed_arguments.classifier_transform,
        "--evaluation_space",
        parsed_arguments.evaluation_space,
        "--evaluation_mode",
        parsed_arguments.evaluation_mode,
        "--random_state",
        str(getattr(parsed_arguments, "random_state", 0)),
    ]

    if parsed_arguments.use_mmap:
        command.append("--mmap_npy")

    if parsed_arguments.max_train_samples is not None:
        command.extend(["--max_train_samples", str(parsed_arguments.max_train_samples)])

    if parsed_arguments.max_samples_per_class is not None:
        command.extend(["--max_samples_per_class", str(parsed_arguments.max_samples_per_class)])

    for parameter in (
            "train_samples_per_class",
            "test_samples_per_class",
            "synthetic_train_samples_per_class",
            "synthetic_test_samples_per_class"):
        effective_value, _ = resolve_effective_argument(parsed_arguments, combination, parameter)
        if effective_value is not None:
            command.extend([f"--{parameter}", str(effective_value)])

    required_synthetic = synthetic_required_samples_per_class(parsed_arguments, combination)
    if required_synthetic is not None:
        command.extend([
            "--sample_plan",
            "class_counts",
            "--number_samples_per_class",
            build_number_samples_per_class_plan(required_synthetic),
        ])

    if parsed_arguments.n_estimators is not None:
        command.extend(["--n_estimators", str(parsed_arguments.n_estimators)])

    if parsed_arguments.max_depth is not None:
        command.extend(["--max_depth", str(parsed_arguments.max_depth)])

    if parsed_arguments.max_samples is not None:
        command.extend(["--max_samples", str(parsed_arguments.max_samples)])

    if parsed_arguments.class_weight is not None:
        command.extend(["--class_weight", str(parsed_arguments.class_weight)])

    if parsed_arguments.full:
        command.append("--strict_min_samples_per_class")

    if parsed_arguments.dry_run_memory:
        command.append("--dry_run_memory")

    synthetic_format = parsed_arguments.save_synthetic_format or "npy_batches"
    command.extend(["--save_synthetic_format", synthetic_format])
    if parsed_arguments.materialize_synthetic:
        command.append("--materialize_synthetic")
    if parsed_arguments.allow_double_transform:
        command.append("--allow_double_transform")
    if parsed_arguments.allow_scaler_refit:
        command.append("--allow_scaler_refit")
    if parsed_arguments.inverse_transform_synthetic:
        command.append("--inverse_transform_synthetic")

    for parameter, value in combination.items():
        if parameter in GENERATION_QUOTA_PARAMETERS:
            continue
        append_cli_value(command, parameter, value)

        if (
            parameter == "variational_autoencoder_dense_layer_sizes_encoder"
            and "variational_autoencoder_dense_layer_sizes_decoder" not in combination
        ):
            layers = str(value).split()[::-1]
            command.extend(["--variational_autoencoder_dense_layer_sizes_decoder", *layers])

    model_type = combination.get("model_type")
    batch_overrides = {
        "adversarial": ["adversarial_batch_size"],
        "autoencoder": ["autoencoder_batch_size"],
        "variational": ["variational_autoencoder_batch_size"],
        "quantized": ["quantized_vae_batch_size"],
        "wasserstein": ["wasserstein_batch_size"],
        "wasserstein_gp": ["wasserstein_gp_batch_size"],
        "latent_diffusion": [
            "latent_diffusion_unet_batch_size",
            "latent_diffusion_autoencoder_batch_size_create_embedding",
            "latent_diffusion_autoencoder_batch_size_training",
        ],
        "denoising_diffusion": ["denoising_diffusion_unet_batch_size"],
    }
    for parameter in batch_overrides.get(model_type, []):
        _remove_command_option(command, parameter)
        command.extend([f"--{parameter}", str(parsed_arguments.batch_size)])

    return deduplicate_command_options(command)


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
    command = deduplicate_command_options(command)
    logging.info("Command line: %s", shlex.join(map(str, command)))
    if not arguments.dryrun:
        subprocess.run(command, check=True)


def resolve_actual_run_dir(requested_output_dir):
    requested_output_dir = Path(requested_output_dir)
    if (requested_output_dir / "EvaluationResults" / "Results.json").is_file():
        return requested_output_dir

    candidates = [
        path
        for path in requested_output_dir.parent.glob(f"{requested_output_dir.name}*")
        if path.is_dir() and (path / "EvaluationResults" / "Results.json").is_file()
    ]
    if not candidates:
        return requested_output_dir
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_run_artifacts(requested_output_dir, combination):
    combination_dir = resolve_actual_run_dir(requested_output_dir)
    generated_data_dir = combination_dir / "DataGenerated"
    train_manifest = generated_data_dir / "synthetic_batches" / "train" / "manifest.json"
    test_manifest = generated_data_dir / "synthetic_batches" / "test" / "manifest.json"
    legacy_manifest = generated_data_dir / "synthetic_batches" / "manifest.json"
    if not train_manifest.is_file() and legacy_manifest.is_file():
        train_manifest = legacy_manifest
    if not test_manifest.is_file() and legacy_manifest.is_file():
        test_manifest = legacy_manifest

    manifest_data = _load_json_file(train_manifest) if train_manifest.is_file() else {}
    return RunArtifacts(
        model_name=str(combination.get("model_type")),
        fold=None,
        combination_dir=combination_dir,
        results_json_path=combination_dir / "EvaluationResults" / "Results.json",
        synthetic_train_manifest_path=train_manifest if train_manifest.is_file() else None,
        synthetic_test_manifest_path=test_manifest if test_manifest.is_file() else None,
        generated_data_dir=generated_data_dir,
        monitor_dir=combination_dir / "Monitor",
        data_space=manifest_data.get("data_space") if isinstance(manifest_data, dict) else None,
        transform_id=manifest_data.get("transform_id") if isinstance(manifest_data, dict) else None,
    )


def _args_without_option(raw_args, option_names, option_takes_value):
    filtered = []
    skip_next = False
    for argument in raw_args:
        if skip_next:
            skip_next = False
            continue
        if argument in option_names:
            skip_next = argument in option_takes_value
            continue
        if any(argument.startswith(f"{option}=") for option in option_names):
            continue
        filtered.append(argument)
    return filtered


def run_compare_modes(parsed_arguments):
    base_args = _args_without_option(
        sys.argv[1:],
        option_names={"--compare_modes", "--execution_mode"},
        option_takes_value={"--execution_mode"},
    )
    script_path = str(Path(__file__).resolve())
    normal_command = [
        sys.executable,
        script_path,
        *base_args,
        "--execution_mode",
        "normal",
        "--output_suffix",
        "normal",
    ]
    batches_command = [
        sys.executable,
        script_path,
        *base_args,
        "--execution_mode",
        "batches",
        "--use_mmap",
        "--output_suffix",
        "batches",
    ]

    logging.info("Compare modes: running normal mode.")
    if parsed_arguments.dryrun:
        logging.info("Command line: %s", shlex.join(map(str, normal_command)))
    else:
        subprocess.run(normal_command, check=True)

    logging.info("Compare modes: running batches mode.")
    if parsed_arguments.dryrun:
        logging.info("Command line: %s", shlex.join(map(str, batches_command)))
    else:
        subprocess.run(batches_command, check=True)


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
    parser.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="inspect AppClassNet input arrays, save diagnostics JSON, and exit before training or synthetic generation",
    )
    parser.add_argument(
        "--baseline_real_only",
        action="store_true",
        help="train and evaluate a real-data classifier baseline without training generators or synthetic data",
    )
    parser.add_argument("--force_prepare", action="store_true", help="rebuild the prepared AppClassNet CSV even if it exists")
    parser.add_argument("--list_campaigns", action="store_true", help="list campaign names and exit")
    parser.add_argument(
        "--full",
        action="store_true",
        help="when used with -c sf, run the complete AppClassNet campaign set instead of the demo campaigns",
    )
    parser.add_argument(
        "--compare_modes",
        action="store_true",
        help="run the selected campaign twice: once in normal mode and once in batches mode",
    )
    parser.add_argument("--output_suffix", default=None, help=argparse.SUPPRESS)

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
        "--execution_mode",
        choices=["normal", "batches"],
        default="normal",
        help="normal preserves the current CSV flow; batches uses AppClassNet .npy files directly",
    )
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int, help="training batch size for batches mode")
    parser.add_argument(
        "--eval_batch_size",
        default=DEFAULT_EVAL_BATCH_SIZE,
        type=int,
        help="evaluation batch size for batches mode",
    )
    parser.add_argument(
        "--generation_batch_size",
        default=DEFAULT_GENERATION_BATCH_SIZE,
        type=int,
        help="generation batch size for batches mode",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=False,
        help="use numpy mmap for .npy files; recommended with --execution_mode batches",
    )
    parser.add_argument("--max_train_samples", default=None, type=int, help="optional cap for training rows in batches mode")
    parser.add_argument(
        "--max_samples_per_class",
        default=None,
        type=int,
        help="optional per-class cap for training rows in batches mode",
    )
    parser.add_argument(
        "--min_samples_per_class_required",
        default=1,
        type=int,
        help="minimum rows required per class after batches stratified selection; use 500 or 1000 for experiments",
    )
    parser.add_argument(
        "--dry_run_memory",
        action="store_true",
        help="log AppClassNet shapes and memory estimates without running campaigns",
    )
    parser.add_argument(
        "--save_synthetic_format",
        choices=["npy_batches", "csv_batches", "single_npy"],
        default=None,
        help="synthetic output format for batches mode; default is npy_batches in batches mode",
    )
    parser.add_argument(
        "--materialize_synthetic",
        action="store_true",
        help="materialize synthetic data in memory in batches mode; high RAM usage",
    )
    parser.add_argument(
        "--batch_classifier",
        choices=[
            "sgd",
            "passive_aggressive",
            "naive_bayes",
            "mlp_small",
            "decision_tree_subset",
            "extra_trees_subset",
            "random_forest_light",
            "random_forest_subset",
        ],
        default=None,
        help="legacy alias for --eval_classifier in batches mode",
    )
    parser.add_argument(
        "--eval_classifier",
        choices=["decision_tree_subset", "extra_trees_subset", "random_forest_light", "sgd"],
        default="decision_tree_subset",
        help="batch-mode evaluation classifier; tree subsets are recommended first for AppClassNet",
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=["none", "tr_ts", "ts_tr", "both"],
        default="both",
        help="synthetic evaluation mode; baseline_real_only forces effective none",
    )
    parser.add_argument(
        "--batch_classifier_subset_size",
        default=100000,
        type=int,
        help="maximum rows loaded by *_subset classifiers",
    )
    parser.add_argument(
        "--normal_classifier",
        choices=["decision_tree", "random_forest", "decision_tree_subset", "random_forest_subset"],
        default=None,
        help="optional normal-mode classifier override; omitted preserves the legacy classifier list",
    )
    parser.add_argument(
        "--scaler",
        choices=["none", "minmax", "standard"],
        default=DEFAULT_SCALER,
        help="Legacy AppClassNet feature scaler alias; default none preserves the public top-200 feature scale",
    )
    parser.add_argument(
        "--source_profile",
        choices=["legacy_csv", "appclassnet_top200", "custom"],
        default="appclassnet_top200",
        help="preprocessing source profile; AppClassNet runner defaults to appclassnet_top200",
    )
    parser.add_argument(
        "--feature_transform",
        choices=["preserve", "auto", "minmax", "standard"],
        default="preserve",
        help="source feature transform before materialization; AppClassNet default preserve",
    )
    parser.add_argument(
        "--generator_transform",
        choices=["preserve", "auto", "minmax", "standard"],
        default="auto",
        help="internal generator transform; auto resolves only when a model requires a documented internal scale",
    )
    parser.add_argument(
        "--classifier_transform",
        choices=["preserve", "auto", "minmax", "standard"],
        default="preserve",
        help="classifier input transform; tree classifiers preserve AppClassNet scale by default",
    )
    parser.add_argument(
        "--evaluation_space",
        choices=["source", "transformed"],
        default="source",
        help="space used by evaluation; AppClassNet default source",
    )
    parser.add_argument(
        "--allow_double_transform",
        action="store_true",
        help="allow applying an equivalent transform more than once",
    )
    parser.add_argument(
        "--allow_scaler_refit",
        action="store_true",
        help="allow refitting a previously fitted preprocessing scaler",
    )
    parser.add_argument(
        "--inverse_transform_synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="inverse-transform generator-space synthetic data back to source space before saving/evaluation",
    )
    parser.add_argument(
        "--baseline_classifier",
        choices=["decision_tree", "extra_trees", "random_forest_light", "sgd"],
        default="decision_tree",
        help="real-real baseline classifier used by --baseline_real_only",
    )
    parser.add_argument(
        "--train_samples_per_class",
        default=DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS,
        type=int,
        help="stratified real train samples per class used by --baseline_real_only",
    )
    parser.add_argument(
        "--test_samples_per_class",
        default=DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS,
        type=int,
        help="stratified real test samples per class used by --baseline_real_only",
    )
    parser.add_argument(
        "--synthetic_train_samples_per_class",
        default=None,
        type=int,
        help="per-class synthetic quota used to train TS-TR classifiers",
    )
    parser.add_argument(
        "--synthetic_test_samples_per_class",
        default=None,
        type=int,
        help="per-class synthetic cap used to evaluate TR-TS classifiers",
    )
    parser.add_argument(
        "--max_depth",
        default=None,
        type=int,
        help="optional max_depth for tree baseline/evaluation classifiers",
    )
    parser.add_argument(
        "--n_estimators",
        default=None,
        type=int,
        help="optional estimator count for extra_trees or random_forest_light classifiers",
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=float,
        help="optional RandomForest max_samples for random_forest_light",
    )
    parser.add_argument(
        "--class_weight",
        choices=["balanced", "balanced_subsample"],
        default=None,
        help="optional class_weight for tree ensemble eval classifiers",
    )
    parser.add_argument(
        "--generation_strategy",
        choices=["single_conditional", "per_class", "grouped_classes"],
        default="single_conditional",
        help="batch generation strategy: one conditional generator, one generator per class, or one per class group",
    )
    parser.add_argument(
        "--classes_per_group",
        default=10,
        type=int,
        help="number of classes per generator when --generation_strategy grouped_classes",
    )
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
    parser.add_argument(
        "--random_state",
        default=0,
        type=int,
        help="random seed forwarded to deterministic evaluators and subset classifiers",
    )

    return parser


def main():
    global arguments
    parser = build_parser()
    arguments = parser.parse_args()
    annotate_explicit_cli_arguments(arguments, sys.argv[1:])
    if arguments.batch_classifier:
        arguments.eval_classifier = (
            "random_forest_light"
            if arguments.batch_classifier == "random_forest_subset"
            else arguments.batch_classifier
        )
    arguments.batch_classifier = arguments.eval_classifier

    if arguments.list_campaigns:
        print("\n".join(sorted(campaigns_available)))
        return 0

    if arguments.classes_per_group <= 0:
        raise ValueError("--classes_per_group must be a positive integer.")

    reexec_with_configured_python_if_needed(arguments)

    for path in PATHS:
        (REPO_ROOT / path).mkdir(parents=True, exist_ok=True)

    if arguments.compare_modes:
        output_dir = REPO_ROOT / "outputs" / f"appclassnet_top200_compare_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(output_dir, arguments.verbosity)
        print_all_settings(arguments)
        run_compare_modes(arguments)
        return 0

    campaigns_chosen = choose_campaigns(arguments.campaign, full=arguments.full)
    normalize_preprocessing_arguments(arguments, campaigns_chosen)
    if arguments.diagnostic_only or arguments.baseline_real_only:
        output_dir = REPO_ROOT / RESULTS_ROOT
    else:
        output_dir = REPO_ROOT / "outputs" / f"appclassnet_top200_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not arguments.diagnostic_only and not arguments.baseline_real_only and set(campaigns_chosen) == set(DEMO_CAMPAIGNS):
        output_dir = REPO_ROOT / "outputs" / "appclassnet_top200_demo"
    if arguments.output_suffix and not arguments.diagnostic_only and not arguments.baseline_real_only:
        output_dir = output_dir.parent / f"{output_dir.name}_{arguments.output_suffix}"
    elif arguments.execution_mode == "batches" and not arguments.diagnostic_only and not arguments.baseline_real_only:
        output_dir = output_dir.parent / f"{output_dir.name}_batches"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, arguments.verbosity)
    if arguments.baseline_real_only:
        if arguments.evaluation_mode != "none":
            logging.warning(
                "--baseline_real_only executes only TR-TR; ignoring --evaluation_mode=%s and using none.",
                arguments.evaluation_mode,
            )
        arguments.evaluation_mode = effective_evaluation_mode(arguments)
    print_all_settings(arguments)
    warn_prepare_limit_if_needed(arguments, campaigns_chosen)

    raw_root = resolve_project_path(arguments.raw_root)
    converted_root = resolve_project_path(arguments.converted_root)
    effective_raw_root = raw_root
    effective_converted_root = converted_root

    if arguments.diagnostic_only:
        diagnostics_path, diagnostics = run_input_diagnostics(arguments, raw_root, output_dir, campaigns_chosen)
        logging.info("Diagnostic-only mode completed. Diagnostics: %s", diagnostics_path)
        return 1 if diagnostics["errors"] else 0

    if arguments.baseline_real_only:
        metrics_path, metrics = run_real_real_baseline(arguments, raw_root, output_dir)
        write_result = write_baseline_batches_metrics(metrics, REPO_ROOT / RESULTS_ROOT / "batches" / "metrics.json")
        payload = write_result[1] if isinstance(write_result, tuple) else metrics
        print_results_payload(payload)
        logging.info("Baseline real-only mode completed. Metrics: %s", metrics_path)
        return 0

    validate_scaler_for_campaigns(arguments, campaigns_chosen)

    if arguments.execution_mode == "batches" and not arguments.use_mmap:
        logging.warning("execution_mode=batches was selected without --use_mmap; .npy arrays may be loaded into RAM.")
    if arguments.execution_mode == "batches":
        logging.info("Batches mode synthetic format: %s", arguments.save_synthetic_format or "npy_batches")
        logging.info("Batches mode eval classifier: %s", arguments.eval_classifier)
        logging.info("Batches mode generation strategy: %s", arguments.generation_strategy)
        if arguments.generation_strategy == "grouped_classes":
            logging.info("Batches mode classes per generator group: %d", arguments.classes_per_group)
    if arguments.execution_mode == "batches" and arguments.materialize_synthetic:
        logging.warning("--materialize_synthetic in batches mode can use high memory.")

    if arguments.dry_run_memory:
        log_appclassnet_memory_plan(arguments, raw_root)
        logging.info("dry_run_memory enabled; stopping after memory estimate.")
        return 0

    if not arguments.dryrun:
        effective_raw_root, scaler_path, preprocessing_report_path = preprocess_appclassnet_splits(
            arguments,
            raw_root,
            arguments.execution_mode,
        )
        if arguments.execution_mode == "normal":
            effective_converted_root = REPO_ROOT / RESULTS_ROOT / "normal" / "preprocessing" / "converted_csv"
        logging.info("Using preprocessed AppClassNet root: %s", effective_raw_root)
        logging.info("Using preprocessing scaler: %s", scaler_path)
        logging.info("Using preprocessing stats: %s", preprocessing_report_path)

    log_appclassnet_memory_plan(arguments, effective_raw_root)

    if arguments.execution_mode == "normal" and arguments.dryrun:
        dataset_path = expected_csv_path(
            arguments.dataset_split,
            effective_converted_root,
            arguments.prepare_max_samples,
            arguments.prepare_sampling,
        )
        logging.info("Dry run: AppClassNet CSV preparation skipped. Expected dataset: %s", dataset_path)
    elif arguments.execution_mode == "normal":
        dataset_path = materialize_appclassnet_csv(
            split=arguments.dataset_split,
            raw_root=effective_raw_root,
            output_root=effective_converted_root,
            max_samples=arguments.prepare_max_samples,
            chunk_size=arguments.prepare_chunk_size,
            force=arguments.force_prepare,
            sampling=arguments.prepare_sampling,
        )
    else:
        selected_split = arguments.dataset_split if arguments.dataset_split != "all" else "train"
        train_x_path, train_y_path = validate_raw_split(effective_raw_root, selected_split)
        dataset_path = train_x_path
        logging.info("Batches mode: CSV preparation skipped.")
        logging.info("Batches mode: train_x_path=%s", train_x_path)
        logging.info("Batches mode: train_y_path=%s", train_y_path)
        if arguments.dataset_split == "all":
            logging.warning("Batches mode does not concatenate train/valid/test; using train split for training.")

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
    pending_plot_commands = []

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

            if arguments.execution_mode == "normal":
                command = build_main_command(
                    child_python,
                    dataset_path,
                    output_dir_run,
                    combination,
                    arguments.verbosity,
                    arguments.data_type,
                    arguments.data_load_max_samples,
                    arguments.normal_classifier,
                    arguments.batch_classifier_subset_size,
                    arguments,
                )
            else:
                command = build_batch_main_command(
                    child_python,
                    effective_raw_root,
                    output_dir_run,
                    combination,
                    arguments.verbosity,
                    arguments.data_type,
                    arguments,
                )
            if arguments.pipenv:
                command = ["pipenv", "run", *command]

            time_start_experiment = datetime.datetime.now()
            logging.info("\t\t\tBegin Experiment: %s", time_start_experiment.strftime(TIME_FORMAT))
            run_cmd(command)
            time_end_experiment = datetime.datetime.now()
            logging.info("\t\t\tEnd                : %s", time_end_experiment.strftime(TIME_FORMAT))
            logging.info("\t\t\tExperiment duration: %s", time_end_experiment - time_start_experiment)

            if arguments.execution_mode == "batches":
                run_artifacts = build_run_artifacts(output_dir_run, combination)
                results_grouping.append(run_artifacts)
                actual_output_dir_run = run_artifacts.combination_dir
            else:
                actual_output_dir_run = output_dir_run
                results_path = output_dir_run / "EvaluationResults" / "Results.json"
                results_grouping.append(results_path)

            if arguments.skip_plots:
                continue

            final_plot = count_campaign == len(campaigns_chosen)
            plot_title = f"{combination['model_type']} {dataset_name}"
            plot_results_paths = [_artifact_results_path(item) for item in results_grouping]
            plot_command = build_plot_command(
                child_python,
                dataset_path,
                actual_output_dir_run,
                combination,
                plot_title,
                plot_results_paths,
                final_plot,
            )
            if arguments.pipenv:
                plot_command = ["pipenv", "run", *plot_command]
            if arguments.execution_mode == "batches":
                pending_plot_commands.append(plot_command)
                continue

            time_start_plot = datetime.datetime.now()
            logging.info("\t\t\tBegin Plot: %s", time_start_plot.strftime(TIME_FORMAT))
            try:
                run_cmd(plot_command)
            except subprocess.CalledProcessError as error:
                logging.warning("Optional plot command failed and results remain valid: %s", error)
            time_end_plot = datetime.datetime.now()
            logging.info("\t\t\tEnd Plot     : %s", time_end_plot.strftime(TIME_FORMAT))
            logging.info("\t\t\tPlot duration: %s", time_end_plot - time_start_plot)

        time_end_campaign = datetime.datetime.now()
        logging.info("\t Campaign duration: %s", time_end_campaign - time_start_campaign)

    time_end_evaluation = datetime.datetime.now()
    if arguments.execution_mode == "batches":
        write_result = write_batches_metrics(
            results_grouping,
            REPO_ROOT / RESULTS_ROOT / "batches" / "metrics.json",
            arguments,
        )
        payload = write_result[1] if isinstance(write_result, tuple) else write_result
        print_results_payload(payload)
        for plot_command in pending_plot_commands:
            time_start_plot = datetime.datetime.now()
            logging.info("\t\t\tBegin Plot: %s", time_start_plot.strftime(TIME_FORMAT))
            try:
                run_cmd(plot_command)
            except subprocess.CalledProcessError as error:
                logging.warning("Optional plot command failed and metrics remain valid: %s", error)
            time_end_plot = datetime.datetime.now()
            logging.info("\t\t\tEnd Plot     : %s", time_end_plot.strftime(TIME_FORMAT))
            logging.info("\t\t\tPlot duration: %s", time_end_plot - time_start_plot)
    logging.info("Evaluation duration: %s", time_end_evaluation - time_start_evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
