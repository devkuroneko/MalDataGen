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
import numbers
import shlex
import subprocess
import sys
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy
from Engine.Preprocessing.FeatureTransformManager import TransformManifest
from Engine.DataIO.RealClassCountPolicy import select_stratified_indices_from_labels
from Engine.DataIO.RealClassCountPolicy import validate_real_class_count_policy
from Engine.DataIO.RealClassCountPolicy import validate_samples_per_class_scope
from Engine.DataIO.DatasetContracts import materialize_npy_class_subset
from Engine.DataIO.DatasetContracts import resolve_class_mapping
from Engine.Evaluation.ExperimentProtocol import EVALUATION_PROTOCOL_CHOICES
from Engine.Evaluation.ExperimentProtocol import is_canonical_protocol_selector
from Engine.Evaluation.ExperimentProtocol import normalize_protocol_selector


DEFAULT_VERBOSITY_LEVEL = logging.INFO
DEFAULT_NUM_EPOCHS = 300
DEFAULT_DEMO_EPOCHS = 10
DEFAULT_K_FOLDS = 5
DEFAULT_DATA_TYPE = "continuous"
DEFAULT_SAVE_DATA = "True"
DEFAULT_SAMPLES_PER_CLASS = 2500
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_BATCH_SIZE = 8192
DEFAULT_EVAL_BATCH_SIZE = 16384
DEFAULT_GENERATION_BATCH_SIZE = 8192
DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS = 100_000
DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS = 5_000
DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS = 50
DEFAULT_SCALER = "none"
DEFAULT_REAL_CLASS_COUNT_POLICY = "strict"
DEFAULT_SAMPLES_PER_CLASS_SCOPE = "split"
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


@dataclass(frozen=True)
class RunConfig:
    run_mode: str
    pipeline: str
    execution_mode: str
    data_type: str
    verbosity: int
    random_state: int


@dataclass(frozen=True)
class DatasetConfig:
    source_profile: str
    split_mode: str
    dataset_split: str
    raw_root: str | None
    dataset_path: str | None
    effective_number_k_folds: int
    requested_number_k_folds: int
    effective_num_classes: int


@dataclass(frozen=True)
class GeneratorConfig:
    model_type: str
    generation_strategy: str
    classes_per_group: int
    generated_samples_per_class: int


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_protocol: str
    evaluation_mode: str
    run_tr_tr: bool
    synthetic_control: str


@dataclass(frozen=True)
class ClassifierConfig:
    requested_classifier: str | None
    effective_classifier: str | None
    normal_classifier: str | None
    batch_classifier_subset_size: int


@dataclass(frozen=True)
class TransformConfig:
    feature_transform: str
    generator_transform: str
    classifier_transform: str
    evaluation_space: str
    inverse_transform_synthetic: bool
    allow_double_transform: bool
    allow_scaler_refit: bool


@dataclass(frozen=True)
class SamplePlan:
    train_samples_per_class: int
    test_samples_per_class: int
    synthetic_train_samples_per_class: int
    synthetic_test_samples_per_class: int
    generated_samples_per_class: int
    required_generated_per_class: int
    number_samples_per_class: str


@dataclass(frozen=True)
class ResolvedConfig:
    run: RunConfig
    dataset: DatasetConfig
    generator: GeneratorConfig
    evaluation: EvaluationConfig
    classifier: ClassifierConfig
    transform: TransformConfig
    sample_plan: SamplePlan
    effective_parameters: dict


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    pipeline: str
    execution_mode: str
    use_mmap: bool
    skip_plots: bool
    save_synthetic_format: str | None
    train_samples_per_class: int | None
    test_samples_per_class: int | None
    synthetic_train_samples_per_class: int | None
    synthetic_test_samples_per_class: int | None
    generated_samples_per_class: int | None
    prepare_max_samples: int
    data_load_max_samples: int
    batch_size: int
    eval_batch_size: int
    generation_batch_size: int
    vae_epochs: int | None
    gan_epochs: int | None
    model_list: tuple[str, ...]
    campaign_list: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedValue:
    value: object
    origin: str


COMMAND_MANIFESTS = {}


DEMO_PROFILE = ExecutionProfile(
    name="demo",
    pipeline="all",
    execution_mode="batches",
    use_mmap=True,
    skip_plots=True,
    save_synthetic_format="npy_batches",
    train_samples_per_class=1000,
    test_samples_per_class=500,
    synthetic_train_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    synthetic_test_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    generated_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS * 2,
    prepare_max_samples=-1,
    data_load_max_samples=-1,
    batch_size=DEFAULT_BATCH_SIZE,
    eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    vae_epochs=DEFAULT_DEMO_EPOCHS,
    gan_epochs=DEFAULT_DEMO_EPOCHS,
    model_list=("variational", "adversarial"),
    campaign_list=tuple(DEMO_CAMPAIGNS),
)

FULL_PROFILE = ExecutionProfile(
    name="full",
    pipeline="all",
    execution_mode="normal",
    use_mmap=False,
    skip_plots=False,
    save_synthetic_format=None,
    train_samples_per_class=DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS,
    test_samples_per_class=DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS,
    synthetic_train_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    synthetic_test_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    generated_samples_per_class=DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS * 2,
    prepare_max_samples=-1,
    data_load_max_samples=-1,
    batch_size=DEFAULT_BATCH_SIZE,
    eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    vae_epochs=DEFAULT_NUM_EPOCHS,
    gan_epochs=DEFAULT_NUM_EPOCHS,
    model_list=tuple(DEFAULT_CAMPAIGN),
    campaign_list=tuple(DEFAULT_CAMPAIGN),
)

PROFILES = {
    "demo": DEMO_PROFILE,
    "full": FULL_PROFILE,
}

GLOBAL_PARAMETER_DEFAULTS = {
    "pipeline": "all",
    "execution_mode": "normal",
    "use_mmap": False,
    "skip_plots": False,
    "save_synthetic_format": None,
    "prepare_max_samples": -1,
    "data_load_max_samples": -1,
    "batch_size": DEFAULT_BATCH_SIZE,
    "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
    "generation_batch_size": DEFAULT_GENERATION_BATCH_SIZE,
    "vae_epochs": None,
    "gan_epochs": None,
    "train_samples_per_class": DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS,
    "test_samples_per_class": DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS,
    "synthetic_train_samples_per_class": DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    "synthetic_test_samples_per_class": DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS,
    "generated_samples_per_class": None,
    "real_class_count_policy": DEFAULT_REAL_CLASS_COUNT_POLICY,
    "samples_per_class_scope": DEFAULT_SAMPLES_PER_CLASS_SCOPE,
}

EXPERIMENT_BUDGET_SCENARIOS = {
    "r200_r500": {
        "label": "A. R200 -> R500",
        "evaluation_protocol": "tr_tr",
        "train_samples_per_class": 200,
        "test_samples_per_class": 500,
    },
    "s200_r500": {
        "label": "B. S200 -> R500",
        "evaluation_protocol": "ts_tr",
        "train_samples_per_class": 200,
        "synthetic_train_samples_per_class": 200,
        "generated_samples_per_class": 200,
        "test_samples_per_class": 500,
    },
    "r50_r500": {
        "label": "C. R50 -> R500",
        "evaluation_protocol": "tr_tr",
        "train_samples_per_class": 50,
        "test_samples_per_class": 500,
    },
    "r50_s150_r500": {
        "label": "D. R50 + S150 -> R500",
        "evaluation_protocol": "tr_plus_ts_tr",
        "train_samples_per_class": 50,
        "synthetic_train_samples_per_class": 150,
        "generated_samples_per_class": 150,
        "test_samples_per_class": 500,
    },
    "r200_s200_r500": {
        "label": "E. R200 + S200 -> R500",
        "evaluation_protocol": "tr_plus_ts_tr",
        "train_samples_per_class": 200,
        "synthetic_train_samples_per_class": 200,
        "generated_samples_per_class": 200,
        "test_samples_per_class": 500,
    },
    "real_resample_r500": {
        "label": "F. real_resample -> R500",
        "evaluation_protocol": "all",
        "synthetic_control": "real_resample",
        "train_samples_per_class": 200,
        "synthetic_train_samples_per_class": 200,
        "synthetic_test_samples_per_class": 200,
        "test_samples_per_class": 500,
        "generated_samples_per_class": 200,
    },
    "label_permutation_r500": {
        "label": "G. label_permutation -> R500",
        "evaluation_protocol": "all",
        "synthetic_control": "label_permutation",
        "train_samples_per_class": 200,
        "synthetic_train_samples_per_class": 200,
        "synthetic_test_samples_per_class": 200,
        "test_samples_per_class": 500,
        "generated_samples_per_class": 200,
    },
}


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
    if not campaigns:
        return tokens
    for campaign in campaigns:
        tokens.extend(item for item in campaign.split(",") if item)
    return tokens


def choose_campaigns(campaigns, full=False):
    if not campaigns:
        return list(DEFAULT_CAMPAIGN)

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


def _explicit_options(parsed_arguments):
    return set(getattr(parsed_arguments, "_explicit_cli_options", set()) or set())


def _argument_was_explicit(parsed_arguments, parameter):
    if not hasattr(parsed_arguments, "_explicit_cli_options"):
        return True
    return parameter in _explicit_options(parsed_arguments)


def resolve_run_mode(parsed_arguments):
    if _argument_was_explicit(parsed_arguments, "run_mode"):
        return ResolvedValue(parsed_arguments.run_mode, "cli")
    if getattr(parsed_arguments, "full", False):
        warnings.warn(
            "--full is deprecated; use --run_mode full instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ResolvedValue("full", "legacy_full")
    if _argument_was_explicit(parsed_arguments, "campaign"):
        return ResolvedValue("demo", "legacy_campaign")
    return ResolvedValue("full", "default")


def resolve_pipeline(parsed_arguments, profile):
    if _argument_was_explicit(parsed_arguments, "pipeline"):
        return ResolvedValue(parsed_arguments.pipeline, "cli")
    if _argument_was_explicit(parsed_arguments, "evaluation_mode"):
        evaluation_mode = getattr(parsed_arguments, "evaluation_mode", "both")
        if evaluation_mode == "none":
            return ResolvedValue("tr_tr", "legacy_evaluation_mode")
        if evaluation_mode == "tr_ts_tr":
            return ResolvedValue("augmentation", "legacy_evaluation_mode")
        if evaluation_mode == "all":
            return ResolvedValue("all", "legacy_evaluation_mode")
        return ResolvedValue("synthetic", "legacy_evaluation_mode")
    return ResolvedValue(profile.pipeline, "profile")


def _record_effective_parameter(
        parsed_arguments,
        parameter,
        *,
        cli_value,
        campaign_value,
        profile_value,
        global_default,
        effective_value,
        origin):
    if not hasattr(parsed_arguments, "_effective_parameters"):
        parsed_arguments._effective_parameters = {}
    parsed_arguments._effective_parameters[parameter] = {
        "parameter": parameter,
        "cli_value": cli_value,
        "campaign_value": campaign_value,
        "profile_value": profile_value,
        "default_value": global_default,
        "global_default": global_default,
        "effective_value": effective_value,
        "origin": origin,
    }


def resolve_configurable_argument(parsed_arguments, parameter, *, campaign_value=None, profile_value=None, global_default=None):
    cli_value = getattr(parsed_arguments, parameter, None) if _argument_was_explicit(parsed_arguments, parameter) else None
    if cli_value is not None:
        effective_value, origin = cli_value, "cli"
    elif campaign_value is not None:
        effective_value, origin = campaign_value, "campaign"
    elif profile_value is not None:
        effective_value, origin = profile_value, "profile"
    else:
        effective_value, origin = global_default, "default"
    _record_effective_parameter(
        parsed_arguments,
        parameter,
        cli_value=cli_value,
        campaign_value=campaign_value,
        profile_value=profile_value,
        global_default=global_default,
        effective_value=effective_value,
        origin=origin,
    )
    logging.info(
        "Parameter resolution: %s cli=%s campaign=%s profile=%s default=%s effective=%s origin=%s",
        parameter,
        cli_value,
        campaign_value,
        profile_value,
        global_default,
        effective_value,
        origin,
    )
    return effective_value, origin


def _profile_parameter_value(parsed_arguments, parameter):
    profile_values = getattr(parsed_arguments, "_execution_profile_parameters", {}) or {}
    return profile_values.get(parameter)


def apply_experiment_budget_scenario(parsed_arguments):
    scenario_key = getattr(parsed_arguments, "experiment_budget_scenario", None)
    if not scenario_key:
        return
    scenario = EXPERIMENT_BUDGET_SCENARIOS[scenario_key]
    parsed_arguments.experiment_budget_scenario_label = scenario["label"]
    for parameter, value in scenario.items():
        if parameter == "label":
            continue
        if not _argument_was_explicit(parsed_arguments, parameter):
            setattr(parsed_arguments, parameter, value)
            if parameter in getattr(parsed_arguments, "_execution_profile_parameters", {}):
                parsed_arguments._execution_profile_parameters[parameter] = value
    if not _argument_was_explicit(parsed_arguments, "evaluation_protocol"):
        protocol = normalize_protocol_selector(scenario["evaluation_protocol"])
        parsed_arguments.evaluation_protocol = protocol
        if protocol == "tr_tr":
            parsed_arguments.pipeline_effective = "tr_tr"
            parsed_arguments.evaluation_mode = "none"
            parsed_arguments.run_tr_tr_effective = True
            parsed_arguments.baseline_real_only = True
        elif protocol == "ts_tr":
            parsed_arguments.pipeline_effective = "synthetic"
            parsed_arguments.evaluation_mode = "ts_tr"
            parsed_arguments.run_tr_tr_effective = False
        elif protocol == "tr_plus_ts_tr":
            parsed_arguments.pipeline_effective = "augmentation"
            parsed_arguments.evaluation_mode = "tr_ts_tr"
            parsed_arguments.run_tr_tr_effective = False
        elif protocol == "all":
            parsed_arguments.pipeline_effective = "all"
            parsed_arguments.evaluation_mode = "all"
            parsed_arguments.run_tr_tr_effective = True
    parsed_arguments._effective_parameters["experiment_budget_scenario"] = {
        "effective": scenario_key,
        "label": scenario["label"],
    }


def apply_execution_profile(parsed_arguments):
    run_mode = resolve_run_mode(parsed_arguments)
    if run_mode.value not in PROFILES:
        raise ValueError(f"Unsupported run_mode: {run_mode.value}")
    profile = PROFILES[run_mode.value]
    pipeline = resolve_pipeline(parsed_arguments, profile)
    if getattr(parsed_arguments, "baseline_real_only", False) and not _argument_was_explicit(parsed_arguments, "pipeline"):
        pipeline = ResolvedValue("tr_tr", "legacy_baseline_real_only")

    parsed_arguments.run_mode_effective = run_mode.value
    parsed_arguments.run_mode_origin = run_mode.origin
    parsed_arguments.execution_profile = profile
    parsed_arguments.pipeline_effective = pipeline.value
    parsed_arguments.pipeline_origin = pipeline.origin
    parsed_arguments._execution_profile_parameters = {
        field: getattr(profile, field)
        for field in GLOBAL_PARAMETER_DEFAULTS
        if hasattr(profile, field)
    }
    if pipeline.value == "augmentation":
        parsed_arguments._execution_profile_parameters["synthetic_test_samples_per_class"] = 0
        parsed_arguments._execution_profile_parameters["generated_samples_per_class"] = None
    parsed_arguments._effective_parameters = {}
    _record_effective_parameter(
        parsed_arguments,
        "run_mode",
        cli_value=parsed_arguments.run_mode if _argument_was_explicit(parsed_arguments, "run_mode") else None,
        campaign_value=None,
        profile_value=None,
        global_default="full",
        effective_value=run_mode.value,
        origin=run_mode.origin,
    )
    _record_effective_parameter(
        parsed_arguments,
        "pipeline",
        cli_value=parsed_arguments.pipeline if _argument_was_explicit(parsed_arguments, "pipeline") else None,
        campaign_value=None,
        profile_value=profile.pipeline,
        global_default=GLOBAL_PARAMETER_DEFAULTS["pipeline"],
        effective_value=pipeline.value,
        origin=pipeline.origin,
    )

    if not _argument_was_explicit(parsed_arguments, "campaign"):
        parsed_arguments.campaign = list(profile.campaign_list)

    for parameter in (
            "execution_mode",
            "use_mmap",
            "skip_plots",
            "save_synthetic_format",
            "prepare_max_samples",
            "data_load_max_samples",
            "batch_size",
            "eval_batch_size",
            "generation_batch_size"):
        profile_value = getattr(profile, parameter)
        global_default = GLOBAL_PARAMETER_DEFAULTS[parameter]
        effective_value, _ = resolve_configurable_argument(
            parsed_arguments,
            parameter,
            profile_value=profile_value,
            global_default=global_default,
        )
        setattr(parsed_arguments, parameter, effective_value)

    for parameter in ("vae_epochs", "gan_epochs"):
        profile_value = getattr(profile, parameter)
        global_default = GLOBAL_PARAMETER_DEFAULTS[parameter]
        cli_value = getattr(parsed_arguments, parameter, None) if _argument_was_explicit(parsed_arguments, parameter) else None
        effective_value = cli_value if cli_value is not None else profile_value
        origin = "cli" if cli_value is not None else "profile"
        _record_effective_parameter(
            parsed_arguments,
            parameter,
            cli_value=cli_value,
            campaign_value=None,
            profile_value=profile_value,
            global_default=global_default,
            effective_value=effective_value,
            origin=origin,
        )
        if cli_value is None:
            setattr(parsed_arguments, parameter, None)

    if pipeline.value == "tr_tr":
        parsed_arguments.evaluation_mode = "none"
        parsed_arguments.run_tr_tr_effective = True
        parsed_arguments.baseline_real_only = True
    elif pipeline.value == "synthetic":
        if pipeline.origin == "cli":
            parsed_arguments.evaluation_mode = "both"
        parsed_arguments.run_tr_tr_effective = False
    elif pipeline.value == "augmentation":
        if pipeline.origin == "cli":
            parsed_arguments.evaluation_mode = "tr_ts_tr"
        parsed_arguments.run_tr_tr_effective = False
    elif pipeline.value == "all":
        parsed_arguments.evaluation_mode = "all"
        parsed_arguments.run_tr_tr_effective = True
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline.value}")

    if (
            _argument_was_explicit(parsed_arguments, "evaluation_protocol")
            and is_canonical_protocol_selector(getattr(parsed_arguments, "evaluation_protocol", None))
    ):
        protocol = normalize_protocol_selector(parsed_arguments.evaluation_protocol)
        parsed_arguments.evaluation_protocol = protocol
        if protocol == "tr_tr":
            parsed_arguments.pipeline_effective = "tr_tr"
            parsed_arguments.evaluation_mode = "none"
            parsed_arguments.run_tr_tr_effective = True
            parsed_arguments.baseline_real_only = True
        elif protocol == "tr_ts":
            parsed_arguments.pipeline_effective = "synthetic"
            parsed_arguments.evaluation_mode = "tr_ts"
            parsed_arguments.run_tr_tr_effective = False
        elif protocol == "ts_tr":
            parsed_arguments.pipeline_effective = "synthetic"
            parsed_arguments.evaluation_mode = "ts_tr"
            parsed_arguments.run_tr_tr_effective = False
        elif protocol == "tr_plus_ts_tr":
            parsed_arguments.pipeline_effective = "augmentation"
            parsed_arguments.evaluation_mode = "tr_ts_tr"
            parsed_arguments.run_tr_tr_effective = False
            parsed_arguments._execution_profile_parameters["synthetic_test_samples_per_class"] = 0
            parsed_arguments._execution_profile_parameters["generated_samples_per_class"] = None
        elif protocol == "all":
            parsed_arguments.pipeline_effective = "all"
            parsed_arguments.evaluation_mode = "all"
            parsed_arguments.run_tr_tr_effective = True

    return profile


def build_pipeline_plan(pipeline):
    if pipeline not in {"tr_tr", "synthetic", "augmentation", "all"}:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    return {
        "prepare_data": "completed",
        "validate_dataset": "completed",
        "run_tr_tr": "completed" if pipeline in {"tr_tr", "all"} else "not_run",
        "train_generator": "completed" if pipeline in {"synthetic", "augmentation", "all"} else "not_run",
        "generate_synthetic_train": "completed" if pipeline in {"synthetic", "augmentation", "all"} else "not_run",
        "generate_synthetic_test": "completed" if pipeline in {"synthetic", "all"} else "not_run",
        "run_tr_ts": "completed" if pipeline in {"synthetic", "all"} else "not_run",
        "run_ts_tr": "completed" if pipeline in {"synthetic", "all"} else "not_run",
        "run_tr_ts_tr": "completed" if pipeline in {"augmentation", "all"} else "not_run",
        "consolidate_results": "completed",
        "run_optional_plots": "optional",
        "preprocessing_runs": 1,
        "synthetic_generation_runs": 1 if pipeline in {"synthetic", "augmentation", "all"} else 0,
    }


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


def normalize_classifier_arguments(parsed_arguments):
    batch_classifier = getattr(parsed_arguments, "batch_classifier", None)
    eval_classifier = getattr(parsed_arguments, "eval_classifier", None)
    if batch_classifier:
        warnings.warn(
            "--batch_classifier is deprecated; use --eval_classifier.",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized_batch = "random_forest_light" if batch_classifier == "random_forest_subset" else batch_classifier
        if (
            _argument_was_explicit(parsed_arguments, "batch_classifier")
            and _argument_was_explicit(parsed_arguments, "eval_classifier")
            and eval_classifier is not None
            and normalized_batch != eval_classifier
        ):
            raise ValueError(
                "ConflictingClassifierArguments: --batch_classifier and --eval_classifier resolve to different values "
                f"({normalized_batch!r} != {eval_classifier!r})."
            )
        parsed_arguments.eval_classifier = normalized_batch
    parsed_arguments.batch_classifier = parsed_arguments.eval_classifier
    parsed_arguments.requested_classifier = batch_classifier or parsed_arguments.eval_classifier
    parsed_arguments.effective_classifier = parsed_arguments.eval_classifier
    return parsed_arguments


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


def select_stratified_indices(
        numpy,
        labels,
        samples_per_class,
        num_classes,
        seed,
        split_name,
        real_class_count_policy=DEFAULT_REAL_CLASS_COUNT_POLICY,
        samples_per_class_scope=DEFAULT_SAMPLES_PER_CLASS_SCOPE):
    samples_per_class = _validate_samples_per_class(samples_per_class, f"{split_name}_samples_per_class")
    policy = validate_real_class_count_policy(real_class_count_policy)
    scope = validate_samples_per_class_scope(samples_per_class_scope)
    selected_indices, report, short_classes = select_stratified_indices_from_labels(
        labels,
        samples_per_class,
        num_classes,
        seed,
        split_name,
        policy,
        require_all_classes=True,
    )
    logging.info(
        "Real class count policy: real_%s_split=%s requested_samples_per_class=%s "
        "minimum_available_per_class=%s effective_samples_per_class=%s "
        "real_class_count_policy=%s samples_per_class_scope=%s",
        "test" if split_name == "test" else split_name,
        split_name,
        report.get("requested_samples_per_class"),
        report.get("minimum_available_per_class"),
        report.get("effective_samples_per_class"),
        policy,
        scope,
    )
    return selected_indices, report["selected_counts_by_class"], short_classes


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
        real_class_count_policy=getattr(parsed_arguments, "real_class_count_policy", DEFAULT_REAL_CLASS_COUNT_POLICY),
        samples_per_class_scope=getattr(parsed_arguments, "samples_per_class_scope", DEFAULT_SAMPLES_PER_CLASS_SCOPE),
    )
    logging.info("Baseline real-real: selecting test samples per class=%s", parsed_arguments.test_samples_per_class)
    test_indices, test_counts, test_short_classes = select_stratified_indices(
        numpy,
        test_y_values,
        parsed_arguments.test_samples_per_class,
        APPCLASSNET_NUM_CLASSES,
        seed=1,
        split_name="test",
        real_class_count_policy=getattr(parsed_arguments, "real_class_count_policy", DEFAULT_REAL_CLASS_COUNT_POLICY),
        samples_per_class_scope=getattr(parsed_arguments, "samples_per_class_scope", DEFAULT_SAMPLES_PER_CLASS_SCOPE),
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
        "real_class_count_policy": getattr(parsed_arguments, "real_class_count_policy", DEFAULT_REAL_CLASS_COUNT_POLICY),
        "samples_per_class_scope": getattr(parsed_arguments, "samples_per_class_scope", DEFAULT_SAMPLES_PER_CLASS_SCOPE),
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


def _build_real_method_summary(results_data, evaluation_name, active):
    if not active:
        return _empty_evaluation_summary("not_run", f"{evaluation_name} was not requested.")
    classifier_name, metric_values = _extract_evaluation_metrics(results_data, evaluation_name)
    required_metrics = ("Accuracy", "BalancedAccuracy", "MacroF1", "WeightedF1")
    missing_metrics = [metric for metric in required_metrics if metric_values.get(metric) is None]
    status = "completed" if classifier_name and not missing_metrics else "not_run"
    reason = None
    if status != "completed":
        if not classifier_name:
            reason = f"{evaluation_name} requested but no classifier metrics were found in Results.json."
        else:
            reason = f"{evaluation_name} requested but missing metric(s): {', '.join(missing_metrics)}."
    summary = _empty_evaluation_summary(status, reason)
    summary.update({
        "classifier": classifier_name,
        "Accuracy": metric_values.get("Accuracy"),
        "MacroF1": metric_values.get("MacroF1"),
        "WeightedF1": metric_values.get("WeightedF1"),
        "BalancedAccuracy": metric_values.get("BalancedAccuracy"),
    })
    return summary


def _artifact_results_path(artifact_or_path):
    return Path(getattr(artifact_or_path, "results_json_path", artifact_or_path))


def _artifact_combination_dir(artifact_or_path, results_path):
    return Path(getattr(artifact_or_path, "combination_dir", Path(results_path).parents[1]))


def _artifact_manifest_path(artifact_or_path, evaluation_name):
    if evaluation_name == "TR-TS":
        return getattr(artifact_or_path, "synthetic_test_manifest_path", None)
    if evaluation_name in {"TS-TR", "TR+TS-TR"}:
        return getattr(artifact_or_path, "synthetic_train_manifest_path", None)
    return None


def _requested_evaluations(evaluation_mode):
    return {
        "TR-TS": evaluation_mode in {"tr_ts", "both", "all"},
        "TS-TR": evaluation_mode in {"ts_tr", "both", "all"},
        "TR+TS-TR": evaluation_mode in {"tr_ts_tr", "all"},
    }


def validate_requested_evaluations_completed(payload, parsed_arguments, results_paths, stopped_function):
    if (
            getattr(parsed_arguments, "run_tr_tr_effective", False)
            or getattr(parsed_arguments, "pipeline_effective", None) in {"tr_tr", "all"}
    ):
        summary = payload.get("TR-TR", {})
        if summary.get("status") != "completed":
            raise RuntimeError(
                f"Requested evaluation TR-TR was not completed; "
                f"status={summary.get('status')}; reason={summary.get('reason')}; "
                f"stopped_function={stopped_function}."
            )
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
        "TR+TS-TR": _empty_evaluation_summary("not_run", "baseline_real_only executes only TR-TR."),
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
        "TR+TS-TR": _empty_evaluation_summary("not_run", "TR+TS-TR was not requested."),
    }
    run_tr_tr = (
        getattr(parsed_arguments, "run_tr_tr_effective", False)
        or getattr(parsed_arguments, "pipeline_effective", None) in {"tr_tr", "all"}
    )
    run_tr_ts = parsed_arguments.evaluation_mode in {"tr_ts", "both", "all"}
    run_ts_tr = parsed_arguments.evaluation_mode in {"ts_tr", "both", "all"}
    run_tr_ts_tr = parsed_arguments.evaluation_mode in {"tr_ts_tr", "all"}
    if run_tr_tr:
        payload["TR-TR"] = _empty_evaluation_summary("not_run", "TR-TR requested but not yet executed.")
    if run_tr_ts:
        payload["TR-TS"] = _empty_evaluation_summary("not_run", "TR-TS requested but not yet executed.")
    if run_ts_tr:
        payload["TS-TR"] = _empty_evaluation_summary("not_run", "TS-TR requested but not yet executed.")
    if run_tr_ts_tr:
        payload["TR+TS-TR"] = _empty_evaluation_summary("not_run", "TR+TS-TR requested but not yet executed.")
    for artifact_or_path in results_paths:
        results_path = _artifact_results_path(artifact_or_path)
        results_data = _load_json_file(results_path)
        if not isinstance(results_data, dict):
            continue
        output_dir_run = _artifact_combination_dir(artifact_or_path, results_path)
        tr_ts_manifest = _artifact_manifest_path(artifact_or_path, "TR-TS")
        ts_tr_manifest = _artifact_manifest_path(artifact_or_path, "TS-TR")
        tr_ts_tr_manifest = _artifact_manifest_path(artifact_or_path, "TR+TS-TR")
        if tr_ts_manifest is None:
            candidate = output_dir_run / "DataGenerated" / "synthetic_batches" / "test" / "manifest.json"
            tr_ts_manifest = candidate if candidate.is_file() else None
        if ts_tr_manifest is None:
            candidate = output_dir_run / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json"
            ts_tr_manifest = candidate if candidate.is_file() else None
        if tr_ts_tr_manifest is None:
            candidate = output_dir_run / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json"
            tr_ts_tr_manifest = candidate if candidate.is_file() else None
        tr_ts_synthetic_summary = _synthetic_manifest_summary(
            output_dir_run,
            tr_ts_manifest,
        )
        ts_tr_synthetic_summary = _synthetic_manifest_summary(
            output_dir_run,
            ts_tr_manifest,
        )
        tr_ts_tr_synthetic_summary = _synthetic_manifest_summary(
            output_dir_run,
            tr_ts_tr_manifest,
        )
        payload["TR-TR"] = _build_real_method_summary(results_data, "TR-TR", run_tr_tr)
        payload["TR-TS"] = _build_method_summary(
            results_data, "TR-TS", output_dir_run, tr_ts_synthetic_summary, run_tr_ts
        )
        payload["TS-TR"] = _build_method_summary(
            results_data, "TS-TR", output_dir_run, ts_tr_synthetic_summary, run_ts_tr
        )
        payload["TR+TS-TR"] = _build_method_summary(
            results_data, "TR+TS-TR", output_dir_run, tr_ts_tr_synthetic_summary, run_tr_ts_tr
        )
    validate_requested_evaluations_completed(payload, parsed_arguments, results_paths, "write_batches_metrics")
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    logging.info("AppClassNet batches metrics saved to %s", output_path)
    return output_path, payload


def collect_normal_metrics(results_paths, parsed_arguments):
    payload = {
        "TR-TR": _empty_evaluation_summary("not_run", "TR-TR was not requested."),
        "TR-TS": _empty_evaluation_summary("not_run", "TR-TS was not requested."),
        "TS-TR": _empty_evaluation_summary("not_run", "TS-TR was not requested."),
        "TR+TS-TR": _empty_evaluation_summary("not_run", "TR+TS-TR was not requested."),
    }
    run_tr_tr = (
        getattr(parsed_arguments, "run_tr_tr_effective", False)
        or getattr(parsed_arguments, "pipeline_effective", None) in {"tr_tr", "all"}
    )
    run_tr_ts = parsed_arguments.evaluation_mode in {"tr_ts", "both", "all"}
    run_ts_tr = parsed_arguments.evaluation_mode in {"ts_tr", "both", "all"}
    run_tr_ts_tr = parsed_arguments.evaluation_mode in {"tr_ts_tr", "all"}
    for results_path in results_paths:
        results_data = _load_json_file(results_path)
        if not isinstance(results_data, dict):
            continue
        payload["TR-TR"] = _build_real_method_summary(results_data, "TR-TR", run_tr_tr)
        payload["TR-TS"] = _build_method_summary(results_data, "TR-TS", Path(results_path).parents[1], {}, run_tr_ts)
        payload["TS-TR"] = _build_method_summary(results_data, "TS-TR", Path(results_path).parents[1], {}, run_ts_tr)
        payload["TR+TS-TR"] = _build_method_summary(
            results_data,
            "TR+TS-TR",
            Path(results_path).parents[1],
            {},
            run_tr_ts_tr,
        )
    validate_requested_evaluations_completed(payload, parsed_arguments, results_paths, "collect_normal_metrics")
    return payload


def _artifact_path_strings(results_grouping, attribute):
    paths = []
    for artifact in results_grouping:
        value = getattr(artifact, attribute, None)
        if value:
            paths.append(str(value))
    return paths


def write_run_results(output_dir, parsed_arguments, campaigns_chosen, metrics_payload, results_grouping):
    output_dir = Path(output_dir)
    canonical_summary = {
        "TR_TR": metrics_payload.get("TR-TR", _empty_evaluation_summary("not_run", "not collected")),
        "TR_TS": metrics_payload.get("TR-TS", _empty_evaluation_summary("not_run", "not collected")),
        "TS_TR": metrics_payload.get("TS-TR", _empty_evaluation_summary("not_run", "not collected")),
        "TR_PLUS_TS_TR": metrics_payload.get("TR+TS-TR", _empty_evaluation_summary("not_run", "not collected")),
        "controls": {
            "synthetic_control": getattr(parsed_arguments, "synthetic_control", "none"),
            "real_resample": getattr(parsed_arguments, "synthetic_control", "none") == "real_resample",
            "label_permutation": getattr(parsed_arguments, "synthetic_control", "none") == "label_permutation",
        },
        "budgets": {
            "generator_training_real_samples_per_class": getattr(parsed_arguments, "train_samples_per_class", None),
            "classifier_training_samples_per_class": {
                "real": getattr(parsed_arguments, "train_samples_per_class", None),
                "synthetic_train": getattr(parsed_arguments, "synthetic_train_samples_per_class", None),
                "synthetic_test": getattr(parsed_arguments, "synthetic_test_samples_per_class", None),
            },
            "test_samples_per_class": getattr(parsed_arguments, "test_samples_per_class", None),
            "generated_samples_per_class": getattr(parsed_arguments, "generated_samples_per_class", None),
            "experiment_budget_scenario": getattr(parsed_arguments, "experiment_budget_scenario", None),
            "experiment_budget_scenario_label": getattr(parsed_arguments, "experiment_budget_scenario_label", None),
        },
        "artifacts": {
            "synthetic_train_manifest": _artifact_path_strings(results_grouping, "synthetic_train_manifest_path"),
            "synthetic_test_manifest": _artifact_path_strings(results_grouping, "synthetic_test_manifest_path"),
            "results_json": [str(_artifact_results_path(item)) for item in results_grouping],
        },
    }
    protocol_payload = {
        "protocol_id": normalize_protocol_selector(getattr(parsed_arguments, "evaluation_protocol", "appclassnet_strict")),
        "train_sources": {
            "TR_TR": ["real_train"],
            "TR_TS": ["real_train"],
            "TS_TR": ["synthetic_train"],
            "TR_PLUS_TS_TR": ["real_train", "synthetic_train"],
        },
        "test_source": {
            "TR_TR": "real_test",
            "TR_TS": "synthetic_test",
            "TS_TR": "real_test",
            "TR_PLUS_TS_TR": "real_test",
        },
        "real_rows_per_class": {},
        "synthetic_rows_per_class": {},
        "generator_training_budget": {
            "real_samples_per_class": getattr(parsed_arguments, "train_samples_per_class", None),
        },
        "classifier_training_budget": canonical_summary["budgets"]["classifier_training_samples_per_class"],
        "test_budget": {"real_samples_per_class": getattr(parsed_arguments, "test_samples_per_class", None)},
        "class_count": getattr(parsed_arguments, "num_classes_subset", None) or APPCLASSNET_NUM_CLASSES,
        "feature_count": APPCLASSNET_NUM_FEATURES,
        "classifier": {
            "requested_classifier": getattr(parsed_arguments, "requested_classifier", None) or getattr(parsed_arguments, "eval_classifier", None),
            "effective_classifier": getattr(parsed_arguments, "eval_classifier", None),
            "random_state": getattr(parsed_arguments, "random_state", None),
        },
        "seeds": {"random_state": getattr(parsed_arguments, "random_state", None)},
        "transforms": {
            "feature_transform": getattr(parsed_arguments, "feature_transform", None),
            "generator_transform": getattr(parsed_arguments, "generator_transform", None),
            "classifier_transform": getattr(parsed_arguments, "classifier_transform", None),
            "evaluation_space": getattr(parsed_arguments, "evaluation_space", None),
        },
        "dataset_hashes": {},
        "leakage_checks": {
            "TS_TR_test_is_real": True,
            "TR_PLUS_TS_TR_test_is_real": True,
            "synthetic_train_derives_from_test": False,
        },
        "run_id": getattr(parsed_arguments, "run_id", output_dir.name),
    }
    payload = {
        "run_id": getattr(parsed_arguments, "run_id", output_dir.name),
        "run_mode": parsed_arguments.run_mode_effective,
        "pipeline": parsed_arguments.pipeline_effective,
        "pipeline_plan": build_pipeline_plan(parsed_arguments.pipeline_effective),
        "campaigns": list(campaigns_chosen),
        "effective_parameters": getattr(parsed_arguments, "_effective_parameters", {}),
        "TR-TR": metrics_payload.get("TR-TR", _empty_evaluation_summary("not_run", "not collected")),
        "TR-TS": metrics_payload.get("TR-TS", _empty_evaluation_summary("not_run", "not collected")),
        "TS-TR": metrics_payload.get("TS-TR", _empty_evaluation_summary("not_run", "not collected")),
        "TR+TS-TR": metrics_payload.get("TR+TS-TR", _empty_evaluation_summary("not_run", "not collected")),
        "artifacts": {
            "synthetic_train_manifest": _artifact_path_strings(results_grouping, "synthetic_train_manifest_path"),
            "synthetic_test_manifest": _artifact_path_strings(results_grouping, "synthetic_test_manifest_path"),
            "model_checkpoints": [],
            "results_json": [str(_artifact_results_path(item)) for item in results_grouping],
        },
    }
    if parsed_arguments.pipeline_effective == "all":
        for evaluation_name in ("TR-TR", "TR-TS", "TS-TR", "TR+TS-TR"):
            if payload[evaluation_name].get("status") != "completed":
                raise RuntimeError(
                    f"pipeline=all requires {evaluation_name}.status == completed; "
                    f"got {payload[evaluation_name].get('status')}."
                )
    output_path = output_dir / "RunResults.json"
    with output_path.open("w", encoding="utf-8") as results_file:
        json.dump(payload, results_file, indent=2, sort_keys=True)
        results_file.write("\n")
    (output_dir / "results_summary.json").write_text(
        json.dumps(canonical_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment_protocol.json").write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logging.info("Consolidated run results saved to %s", output_path)
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
    aliases = {
        "mode": "run_mode",
        "c": "campaign",
    }
    for index, argument in enumerate(raw_args):
        if argument in {"-c"}:
            explicit.add("campaign")
            continue
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        normalized = option[2:].replace("-", "_")
        explicit.add(aliases.get(normalized, normalized))
    parsed_arguments._explicit_cli_options = explicit
    return parsed_arguments


def _evaluation_mode_includes(evaluation_mode, requested_mode):
    if evaluation_mode == "all":
        return True
    if requested_mode == "tr_ts_tr":
        return evaluation_mode == "tr_ts_tr"
    return evaluation_mode in {requested_mode, "both"}


def resolve_argument(cli_value, campaign_value, default_value):
    if cli_value is not None:
        return cli_value, "cli"
    if campaign_value is not None:
        return campaign_value, "campaign"
    return default_value, "default"


def _non_negative_int(parameter, value):
    if isinstance(value, bool):
        raise ValueError(f"--{parameter} must be an integer greater than or equal to zero.")
    if isinstance(value, numbers.Integral):
        integer_value = int(value)
    elif isinstance(value, str):
        value_text = value.strip()
        digits_text = value_text[1:] if value_text.startswith(("+", "-")) else value_text
        if not digits_text.isdigit():
            raise ValueError(f"--{parameter} must be an integer greater than or equal to zero.")
        try:
            integer_value = int(value_text)
        except ValueError as error:
            raise ValueError(f"--{parameter} must be an integer greater than or equal to zero.") from error
    else:
        raise ValueError(f"--{parameter} must be an integer greater than or equal to zero.")
    if integer_value < 0:
        raise ValueError(f"--{parameter} must be an integer greater than or equal to zero.")
    return integer_value


def _default_sample_value(parameter, evaluation_mode):
    defaults = {
        "train_samples_per_class": DEFAULT_BASELINE_TRAIN_SAMPLES_PER_CLASS,
        "test_samples_per_class": DEFAULT_BASELINE_TEST_SAMPLES_PER_CLASS,
        "synthetic_train_samples_per_class": (
            DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS
            if (
                _evaluation_mode_includes(evaluation_mode, "ts_tr")
                or _evaluation_mode_includes(evaluation_mode, "tr_ts_tr")
            )
            else 0
        ),
        "synthetic_test_samples_per_class": (
            DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS
            if _evaluation_mode_includes(evaluation_mode, "tr_ts")
            else 0
        ),
    }
    return defaults[parameter]


def resolve_effective_sample_arguments(parsed_arguments, combination):
    evaluation_mode = getattr(parsed_arguments, "evaluation_mode", "both")
    parameters = (
        "train_samples_per_class",
        "test_samples_per_class",
        "synthetic_train_samples_per_class",
        "synthetic_test_samples_per_class",
    )
    values = {}
    origins = {}
    details = {}

    for parameter in parameters:
        campaign_value = combination.get(parameter)
        profile_value = _profile_parameter_value(parsed_arguments, parameter)
        default_value = _default_sample_value(parameter, evaluation_mode)
        cli_value = getattr(parsed_arguments, parameter, None) if _argument_was_explicit(parsed_arguments, parameter) else None
        effective_value, origin = resolve_configurable_argument(
            parsed_arguments,
            parameter,
            campaign_value=campaign_value,
            profile_value=profile_value,
            global_default=default_value,
        )
        effective_value = _non_negative_int(parameter, effective_value)
        values[parameter] = effective_value
        origins[parameter] = origin
        details[parameter] = {
            "cli_value": cli_value,
            "campaign_value": campaign_value,
            "profile_value": profile_value,
            "global_default": default_value,
            "effective": effective_value,
            "origin": origin,
        }
        logging.info(
            "Argument resolution: %s cli=%s campaign=%s default=%s effective=%s origin=%s",
            parameter,
            cli_value,
            campaign_value,
            default_value,
            effective_value,
            origin,
        )

    if (
        (
            _evaluation_mode_includes(evaluation_mode, "ts_tr")
            or _evaluation_mode_includes(evaluation_mode, "tr_ts_tr")
        )
        and values["synthetic_train_samples_per_class"] <= 0
    ):
        raise ValueError(
            "--synthetic_train_samples_per_class must be greater than zero when evaluation_mode includes TS-TR or TR+TS-TR."
        )
    if (
        _evaluation_mode_includes(evaluation_mode, "tr_ts")
        and values["synthetic_test_samples_per_class"] <= 0
    ):
        raise ValueError(
            "--synthetic_test_samples_per_class must be greater than zero when evaluation_mode includes TR-TS."
        )

    required_generated_per_class = (
        values["synthetic_train_samples_per_class"]
        + values["synthetic_test_samples_per_class"]
    )
    generated_cli_value = (
        getattr(parsed_arguments, "generated_samples_per_class", None)
        if _argument_was_explicit(parsed_arguments, "generated_samples_per_class")
        else None
    )
    generated_campaign_value = combination.get("generated_samples_per_class")
    generated_profile_value = _profile_parameter_value(parsed_arguments, "generated_samples_per_class")
    generated_default_value = required_generated_per_class
    generated_value, generated_origin = resolve_configurable_argument(
        parsed_arguments,
        "generated_samples_per_class",
        campaign_value=generated_campaign_value,
        profile_value=generated_profile_value,
        global_default=generated_default_value,
    )
    generated_value = _non_negative_int("generated_samples_per_class", generated_value)
    values["generated_samples_per_class"] = generated_value
    origins["generated_samples_per_class"] = generated_origin
    details["generated_samples_per_class"] = {
        "cli_value": generated_cli_value,
        "campaign_value": generated_campaign_value,
        "profile_value": generated_profile_value,
        "global_default": generated_default_value,
        "effective": generated_value,
        "origin": generated_origin,
    }
    logging.info(
        "Argument resolution: generated_samples_per_class cli=%s campaign=%s default=%s effective=%s origin=%s",
        generated_cli_value,
        generated_campaign_value,
        generated_default_value,
        generated_value,
        generated_origin,
    )
    logging.info(
        "Synthetic generation plan: synthetic_train_per_class=%s synthetic_test_per_class=%s "
        "required_generated_per_class=%s planned_generated_per_class=%s",
        values["synthetic_train_samples_per_class"],
        values["synthetic_test_samples_per_class"],
        required_generated_per_class,
        generated_value,
    )
    if generated_value < required_generated_per_class:
        raise ValueError(
            "InsufficientGeneratedSamplesPerClass: "
            f"generated_samples_per_class={generated_value} is smaller than required_generated_per_class="
            f"{required_generated_per_class} "
            f"(synthetic_train_samples_per_class={values['synthetic_train_samples_per_class']} + "
            f"synthetic_test_samples_per_class={values['synthetic_test_samples_per_class']})."
        )

    return {
        "values": values,
        "origins": origins,
        "details": details,
        "required_generated_per_class": required_generated_per_class,
        "planned_generated_per_class": generated_value,
    }


def resolve_effective_real_class_count_arguments(parsed_arguments, combination):
    values = {}
    origins = {}
    details = {}
    defaults = {
        "real_class_count_policy": DEFAULT_REAL_CLASS_COUNT_POLICY,
        "samples_per_class_scope": DEFAULT_SAMPLES_PER_CLASS_SCOPE,
    }
    validators = {
        "real_class_count_policy": validate_real_class_count_policy,
        "samples_per_class_scope": validate_samples_per_class_scope,
    }
    for parameter, default_value in defaults.items():
        campaign_value = combination.get(parameter)
        profile_value = _profile_parameter_value(parsed_arguments, parameter)
        cli_value = getattr(parsed_arguments, parameter, None) if _argument_was_explicit(parsed_arguments, parameter) else None
        effective_value, origin = resolve_configurable_argument(
            parsed_arguments,
            parameter,
            campaign_value=campaign_value,
            profile_value=profile_value,
            global_default=default_value,
        )
        effective_value = validators[parameter](effective_value)
        values[parameter] = effective_value
        origins[parameter] = origin
        details[parameter] = {
            "cli_value": cli_value,
            "campaign_value": campaign_value,
            "profile_value": profile_value,
            "global_default": default_value,
            "effective": effective_value,
            "origin": origin,
        }
        logging.info(
            "Argument resolution: %s cli=%s campaign=%s default=%s effective=%s origin=%s",
            parameter,
            cli_value,
            campaign_value,
            default_value,
            effective_value,
            origin,
        )
    return {"values": values, "origins": origins, "details": details}


def resolve_effective_argument(parsed_arguments, combination, parameter):
    plan = resolve_effective_sample_arguments(parsed_arguments, combination)
    return plan["values"][parameter], plan["origins"][parameter]


def synthetic_required_samples_per_class(parsed_arguments, combination):
    plan = resolve_effective_sample_arguments(parsed_arguments, combination)
    return plan["required_generated_per_class"]


def build_number_samples_per_class_plan(samples_per_class, num_classes=APPCLASSNET_NUM_CLASSES):
    return ",".join(f"{class_id}:{int(samples_per_class)}" for class_id in range(int(num_classes)))


def _parameter_record(parsed_arguments, parameter, cli_value, campaign_value, profile_value, default_value,
                      effective_value, origin):
    _record_effective_parameter(
        parsed_arguments,
        parameter,
        cli_value=cli_value,
        campaign_value=campaign_value,
        profile_value=profile_value,
        global_default=default_value,
        effective_value=effective_value,
        origin=origin,
    )
    return getattr(parsed_arguments, "_effective_parameters", {})[parameter]


def _class_domain(parsed_arguments):
    mapping = resolve_class_mapping(
        getattr(parsed_arguments, "class_subset", None),
        getattr(parsed_arguments, "num_classes_subset", None),
        total_num_classes=APPCLASSNET_NUM_CLASSES,
    )
    if mapping is None:
        return APPCLASSNET_NUM_CLASSES
    return mapping.effective_num_classes


def _canonicalize_epoch_aliases(parsed_arguments, combination):
    canonical = dict(combination)
    if getattr(parsed_arguments, "vae_epochs", None) is not None:
        canonical["variational_autoencoder_number_epochs"] = int(parsed_arguments.vae_epochs)
        _parameter_record(
            parsed_arguments,
            "variational_autoencoder_number_epochs",
            int(parsed_arguments.vae_epochs),
            combination.get("variational_autoencoder_number_epochs"),
            _profile_parameter_value(parsed_arguments, "vae_epochs"),
            GLOBAL_PARAMETER_DEFAULTS["vae_epochs"],
            int(parsed_arguments.vae_epochs),
            "cli",
        )
    if getattr(parsed_arguments, "gan_epochs", None) is not None:
        gan_epochs = int(parsed_arguments.gan_epochs)
        for parameter in (
                "adversarial_number_epochs",
                "wasserstein_number_epochs",
                "wasserstein_gp_number_epochs"):
            if parameter in canonical or canonical.get("model_type") in {"adversarial", "wasserstein", "wasserstein_gp"}:
                canonical[parameter] = gan_epochs
                _parameter_record(
                    parsed_arguments,
                    parameter,
                    gan_epochs,
                    combination.get(parameter),
                    _profile_parameter_value(parsed_arguments, "gan_epochs"),
                    GLOBAL_PARAMETER_DEFAULTS["gan_epochs"],
                    gan_epochs,
                    "cli",
                )
    return canonical


def resolve_config_for_command(parsed_arguments, combination, *, split_mode, raw_root=None, dataset_path=None):
    combination = with_k_fold_metadata(_canonicalize_epoch_aliases(parsed_arguments, combination), split_mode)
    sample_arguments = resolve_effective_sample_arguments(parsed_arguments, combination)
    real_count_arguments = resolve_effective_real_class_count_arguments(parsed_arguments, combination)
    sample_values = sample_arguments["values"]
    effective_classifier = getattr(parsed_arguments, "eval_classifier", None)
    requested_classifier = getattr(parsed_arguments, "requested_classifier", None) or effective_classifier
    effective_num_classes = _class_domain(parsed_arguments)
    number_samples_per_class = build_number_samples_per_class_plan(
        sample_values["generated_samples_per_class"],
        num_classes=effective_num_classes,
    )
    _parameter_record(
        parsed_arguments,
        "effective_num_classes",
        getattr(parsed_arguments, "num_classes_subset", None),
        combination.get("num_classes"),
        APPCLASSNET_NUM_CLASSES,
        APPCLASSNET_NUM_CLASSES,
        effective_num_classes,
        "cli" if getattr(parsed_arguments, "num_classes_subset", None) is not None or getattr(parsed_arguments, "class_subset", None) else "profile",
    )
    resolved = ResolvedConfig(
        run=RunConfig(
            run_mode=getattr(parsed_arguments, "run_mode_effective", "full"),
            pipeline=getattr(parsed_arguments, "pipeline_effective", "all"),
            execution_mode=getattr(parsed_arguments, "execution_mode", "normal"),
            data_type=getattr(parsed_arguments, "data_type", DEFAULT_DATA_TYPE),
            verbosity=int(getattr(parsed_arguments, "verbosity", DEFAULT_VERBOSITY_LEVEL)),
            random_state=int(getattr(parsed_arguments, "random_state", 0)),
        ),
        dataset=DatasetConfig(
            source_profile=getattr(parsed_arguments, "source_profile", "appclassnet_top200"),
            split_mode=split_mode,
            dataset_split=getattr(parsed_arguments, "dataset_split", "train"),
            raw_root=str(raw_root) if raw_root is not None else None,
            dataset_path=str(dataset_path) if dataset_path is not None else None,
            effective_number_k_folds=int(combination["effective_number_k_folds"]),
            requested_number_k_folds=int(combination["requested_number_k_folds"]),
            effective_num_classes=effective_num_classes,
        ),
        generator=GeneratorConfig(
            model_type=str(combination.get("model_type")),
            generation_strategy=getattr(parsed_arguments, "generation_strategy", "single_conditional"),
            classes_per_group=int(getattr(parsed_arguments, "classes_per_group", 10)),
            generated_samples_per_class=int(sample_values["generated_samples_per_class"]),
        ),
        evaluation=EvaluationConfig(
            evaluation_protocol=getattr(parsed_arguments, "evaluation_protocol", "appclassnet_strict"),
            evaluation_mode=getattr(parsed_arguments, "evaluation_mode", "both"),
            run_tr_tr=bool(getattr(parsed_arguments, "run_tr_tr_effective", False)),
            synthetic_control=getattr(parsed_arguments, "synthetic_control", "none"),
        ),
        classifier=ClassifierConfig(
            requested_classifier=requested_classifier,
            effective_classifier=effective_classifier,
            normal_classifier=getattr(parsed_arguments, "normal_classifier", None),
            batch_classifier_subset_size=int(getattr(parsed_arguments, "batch_classifier_subset_size", 100000)),
        ),
        transform=TransformConfig(
            feature_transform=getattr(parsed_arguments, "feature_transform", "preserve"),
            generator_transform=getattr(parsed_arguments, "generator_transform", "preserve"),
            classifier_transform=getattr(parsed_arguments, "classifier_transform", "preserve"),
            evaluation_space=getattr(parsed_arguments, "evaluation_space", "source"),
            inverse_transform_synthetic=bool(getattr(parsed_arguments, "inverse_transform_synthetic", True)),
            allow_double_transform=bool(getattr(parsed_arguments, "allow_double_transform", False)),
            allow_scaler_refit=bool(getattr(parsed_arguments, "allow_scaler_refit", False)),
        ),
        sample_plan=SamplePlan(
            train_samples_per_class=int(sample_values["train_samples_per_class"]),
            test_samples_per_class=int(sample_values["test_samples_per_class"]),
            synthetic_train_samples_per_class=int(sample_values["synthetic_train_samples_per_class"]),
            synthetic_test_samples_per_class=int(sample_values["synthetic_test_samples_per_class"]),
            generated_samples_per_class=int(sample_values["generated_samples_per_class"]),
            required_generated_per_class=int(sample_arguments["required_generated_per_class"]),
            number_samples_per_class=number_samples_per_class,
        ),
        effective_parameters=dict(getattr(parsed_arguments, "_effective_parameters", {})),
    )
    return resolved, combination, sample_arguments, real_count_arguments


def _resolved_config_payload(resolved_config):
    return asdict(resolved_config)


K_FOLD_METADATA_PARAMETERS = {
    "requested_number_k_folds",
    "effective_number_k_folds",
    "number_k_folds_origin",
    "split_mode",
    "origin",
}


def resolve_k_fold_metadata(combination, split_mode):
    requested_folds = int(combination.get("requested_number_k_folds", combination.get("number_k_folds", DEFAULT_K_FOLDS)))
    origin = combination.get("number_k_folds_origin", "campaign" if "number_k_folds" in combination else "default")
    effective_folds = 1 if split_mode == "provided" else requested_folds
    metadata = {
        "requested_number_k_folds": requested_folds,
        "effective_number_k_folds": effective_folds,
        "split_mode": split_mode,
        "number_k_folds_origin": origin,
        "origin": origin,
    }
    logging.info(
        "K-fold resolution: requested_number_k_folds=%s effective_number_k_folds=%s split_mode=%s origin=%s",
        requested_folds,
        effective_folds,
        split_mode,
        origin,
    )
    return metadata


def with_k_fold_metadata(combination, split_mode):
    metadata = resolve_k_fold_metadata(combination, split_mode)
    return {**combination, **metadata}


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


def find_duplicate_command_options(command):
    seen = set()
    duplicates = []
    for token in map(str, command):
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in seen and option not in duplicates:
            duplicates.append(option)
        seen.add(option)
    return duplicates


def _is_main_command(command):
    return any(Path(str(token)).name == "main.py" for token in command)


def _main_command_argv(command):
    tokens = [str(token) for token in command]
    for index, token in enumerate(tokens):
        if Path(token).name == "main.py":
            return tokens[index + 1:]
    return []


@lru_cache(maxsize=1)
def _build_main_parser_for_validation():
    from Engine.Arguments import Arguments as arguments_module

    parser = arguments_module.add_argument_framework()
    parser = arguments_module.add_argument_adversarial(parser)
    parser = arguments_module.add_argument_smote(parser)
    parser = arguments_module.add_argument_optimizers(parser)
    parser = arguments_module.add_argument_early_stop(parser)
    parser = arguments_module.add_argument_data_load(parser)
    parser = arguments_module.add_argument_random_noise(parser)
    parser = arguments_module.add_argument_autoencoder(parser)
    parser = arguments_module.add_argument_latent_diffusion(parser)
    parser = arguments_module.add_argument_denoising_diffusion(parser)
    parser = arguments_module.add_argument_quantized_vae(parser)
    parser = arguments_module.add_argument_variation_autoencoder(parser)
    parser = arguments_module.add_argument_wasserstein_gan_gp(parser)
    parser = arguments_module.add_argument_wasserstein_gan(parser)
    parser = arguments_module.add_argument_decision_tree(parser)
    parser = arguments_module.add_argument_gaussian_process(parser)
    parser = arguments_module.add_argument_gradient_boosting(parser)
    parser = arguments_module.add_argument_k_means(parser)
    parser = arguments_module.add_argument_knn(parser)
    parser = arguments_module.add_argument_naive_bayes(parser)
    parser = arguments_module.add_argument_linear_regression(parser)
    parser = arguments_module.add_argument_spectral_clustering(parser)
    parser = arguments_module.add_argument_perceptron(parser)
    parser = arguments_module.add_argument_quadratic_discriminant_analysis(parser)
    parser = arguments_module.add_argument_random_forest(parser)
    parser = arguments_module.add_argument_stochastic_gradient_descent(parser)
    parser = arguments_module.add_argument_support_vector_machine(parser)
    return parser


def _main_parser_option_strings():
    parser = _build_main_parser_for_validation()
    return {
        option
        for action in parser._actions
        for option in action.option_strings
    }


def validate_command_before_subprocess(command):
    duplicates = find_duplicate_command_options(command)
    if duplicates:
        raise ValueError(f"DuplicateCommandArgument: {', '.join(sorted(duplicates))}")
    if _is_main_command(command):
        number_k_folds_count = sum(1 for token in command if str(token).split("=", 1)[0] == "--number_k_folds")
        if number_k_folds_count != 1:
            raise ValueError(
                f"InvalidMainCommand: --number_k_folds must appear exactly once; found {number_k_folds_count}."
            )
        if any(str(token).split("=", 1)[0] == "--effective_number_k_folds" for token in command):
            raise ValueError("InvalidMainCommand: --effective_number_k_folds must not be passed to main.py.")
        allowed_options = _main_parser_option_strings()
        unknown_options = sorted({
            str(token).split("=", 1)[0]
            for token in _main_command_argv(command)
            if str(token).startswith("--") and str(token).split("=", 1)[0] not in allowed_options
        })
        if unknown_options:
            raise ValueError(f"UnknownMainCommandArgument: {', '.join(unknown_options)}")
        parser = _build_main_parser_for_validation()
        try:
            parser.parse_args(_main_command_argv(command))
        except SystemExit as error:
            raise ValueError(
                f"InvalidMainCommand: parser rejected command with exit_code={error.code}: "
                f"{shlex.join(map(str, command))}"
            ) from None


GENERATION_QUOTA_PARAMETERS = {
    "number_samples_per_class",
    "sample_plan",
    "train_samples_per_class",
    "test_samples_per_class",
    "synthetic_train_samples_per_class",
    "synthetic_test_samples_per_class",
    "generated_samples_per_class",
    "real_class_count_policy",
    "samples_per_class_scope",
}

INTERNAL_COMBINATION_PARAMETERS = GENERATION_QUOTA_PARAMETERS | K_FOLD_METADATA_PARAMETERS


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
    resolved_config, combination, sample_arguments, real_count_arguments = resolve_config_for_command(
        parsed_arguments,
        combination,
        split_mode="cross_validation",
        dataset_path=dataset_path,
    )
    sample_values = sample_arguments["values"]
    real_count_values = real_count_arguments["values"]
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
        "--number_k_folds",
        str(combination["effective_number_k_folds"]),
    ]

    if data_load_max_samples and data_load_max_samples > 0:
        command.extend(["--data_load_max_samples", str(data_load_max_samples)])

    if normal_classifier:
        command.extend(["--normal_classifier", normal_classifier])
        command.extend(["--batch_classifier_subset_size", str(batch_classifier_subset_size)])

    command.extend([
        "--source_profile",
        resolved_config.dataset.source_profile,
        "--feature_transform",
        resolved_config.transform.feature_transform,
        "--generator_transform",
        resolved_config.transform.generator_transform,
        "--classifier_transform",
        resolved_config.transform.classifier_transform,
        "--evaluation_space",
        resolved_config.transform.evaluation_space,
        "--evaluation_protocol",
        resolved_config.evaluation.evaluation_protocol,
        "--evaluation_mode",
        resolved_config.evaluation.evaluation_mode,
        "--random_state",
        str(resolved_config.run.random_state),
    ])
    if parsed_arguments.allow_double_transform:
        command.append("--allow_double_transform")
    if parsed_arguments.allow_scaler_refit:
        command.append("--allow_scaler_refit")
    if parsed_arguments.inverse_transform_synthetic:
        command.append("--inverse_transform_synthetic")
    if getattr(parsed_arguments, "run_tr_tr_effective", False):
        command.append("--run_tr_tr")
    for parameter in (
            "train_samples_per_class",
            "test_samples_per_class",
            "synthetic_train_samples_per_class",
            "synthetic_test_samples_per_class"):
        command.extend([f"--{parameter}", str(sample_values[parameter])])
    for parameter in ("real_class_count_policy", "samples_per_class_scope"):
        command.extend([f"--{parameter}", str(real_count_values[parameter])])

    command.extend([
        "--number_samples_per_class",
        resolved_config.sample_plan.number_samples_per_class,
    ])

    for parameter, value in combination.items():
        if parameter in INTERNAL_COMBINATION_PARAMETERS:
            continue
        if parameter == "number_k_folds":
            continue
        append_cli_value(command, parameter, value)

        if (
            parameter == "variational_autoencoder_dense_layer_sizes_encoder"
            and "variational_autoencoder_dense_layer_sizes_decoder" not in combination
        ):
            layers = str(value).split()[::-1]
            command.extend(["--variational_autoencoder_dense_layer_sizes_decoder", *layers])

    validate_command_before_subprocess(command)
    register_command_manifest(command, resolved_config, combination)
    return command


def build_batch_main_command(python_executable, raw_root, output_dir_run, combination, verbosity, data_type, parsed_arguments):
    resolved_config, combination, sample_arguments, real_count_arguments = resolve_config_for_command(
        parsed_arguments,
        combination,
        split_mode="provided",
        raw_root=raw_root,
    )
    sample_values = sample_arguments["values"]
    real_count_values = real_count_arguments["values"]
    effective_num_classes = int(resolved_config.dataset.effective_num_classes)
    subset_mapping = resolve_class_mapping(
        getattr(parsed_arguments, "class_subset", None),
        getattr(parsed_arguments, "num_classes_subset", None),
        total_num_classes=APPCLASSNET_NUM_CLASSES,
    )
    if subset_mapping is not None:
        raw_root, subset_manifest_path = materialize_npy_class_subset(
            raw_root,
            REPO_ROOT / RESULTS_ROOT / "batches" / "subsets",
            subset_mapping,
            dataset_id="appclassnet_top200",
            expected_num_features=APPCLASSNET_NUM_FEATURES,
            mmap_mode="r" if parsed_arguments.use_mmap else None,
        )
        logging.info(
            "Batches mode class subset materialized: subset_id=%s root=%s manifest=%s",
            subset_mapping.subset_id,
            raw_root,
            subset_manifest_path,
        )
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
        str(effective_num_classes),
        "--split_mode",
        "provided",
        "--number_k_folds",
        str(combination["effective_number_k_folds"]),
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
        "--batch_classifier_subset_size",
        str(parsed_arguments.batch_classifier_subset_size),
        "--min_samples_per_class_required",
        str(parsed_arguments.min_samples_per_class_required),
        "--generation_strategy",
        parsed_arguments.generation_strategy,
        "--classes_per_group",
        str(parsed_arguments.classes_per_group),
        "--source_profile",
        resolved_config.dataset.source_profile,
        "--feature_transform",
        resolved_config.transform.feature_transform,
        "--generator_transform",
        resolved_config.transform.generator_transform,
        "--classifier_transform",
        resolved_config.transform.classifier_transform,
        "--evaluation_space",
        resolved_config.transform.evaluation_space,
        "--evaluation_protocol",
        resolved_config.evaluation.evaluation_protocol,
        "--evaluation_mode",
        resolved_config.evaluation.evaluation_mode,
        "--random_state",
        str(resolved_config.run.random_state),
    ]

    command.extend(["--synthetic_control", getattr(parsed_arguments, "synthetic_control", "none")])
    if subset_mapping is None and getattr(parsed_arguments, "class_subset", None) is not None:
        command.extend(["--class_subset", parsed_arguments.class_subset])
    if subset_mapping is None and getattr(parsed_arguments, "num_classes_subset", None) is not None:
        command.extend(["--num_classes_subset", str(parsed_arguments.num_classes_subset)])

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
        command.extend([f"--{parameter}", str(sample_values[parameter])])
    for parameter in ("real_class_count_policy", "samples_per_class_scope"):
        command.extend([f"--{parameter}", str(real_count_values[parameter])])

    command.extend([
        "--sample_plan",
        "class_counts",
        "--number_samples_per_class",
        resolved_config.sample_plan.number_samples_per_class,
    ])

    if parsed_arguments.n_estimators is not None:
        command.extend(["--n_estimators", str(parsed_arguments.n_estimators)])

    if parsed_arguments.max_depth is not None:
        command.extend(["--max_depth", str(parsed_arguments.max_depth)])

    if parsed_arguments.max_samples is not None:
        command.extend(["--max_samples", str(parsed_arguments.max_samples)])

    if parsed_arguments.class_weight is not None:
        command.extend(["--class_weight", str(parsed_arguments.class_weight)])

    if getattr(parsed_arguments, "full", False) or getattr(parsed_arguments, "run_mode_effective", None) == "full":
        command.append("--strict_min_samples_per_class")

    if parsed_arguments.dry_run_memory:
        command.append("--dry_run_memory")
    if getattr(parsed_arguments, "run_tr_tr_effective", False):
        command.append("--run_tr_tr")

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
    else:
        command.append("--no-inverse_transform_synthetic")

    for parameter, value in combination.items():
        if parameter in INTERNAL_COMBINATION_PARAMETERS:
            continue
        if parameter == "number_k_folds":
            continue
        if parameter in {
            "autoencoder_number_classes",
            "variational_autoencoder_number_classes",
            "wasserstein_number_classes",
            "wasserstein_gp_number_classes",
            "quantized_vae_number_classes",
        }:
            value = effective_num_classes
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

    validate_command_before_subprocess(command)
    register_command_manifest(command, resolved_config, combination)
    return command


def build_plot_command(python_executable, dataset_path, output_dir_run, combination, plot_title, results_paths, final_plot):
    effective_folds = int(combination.get("effective_number_k_folds", combination["number_k_folds"]))
    command = [
        python_executable,
        str(REPO_ROOT / "plots.py"),
        "--results",
        ",".join(str(path) for path in results_paths) if final_plot else str(results_paths[-1]),
        "--title",
        *plot_title.split(),
        "--folds",
        str(effective_folds),
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
            for fold in range(effective_folds)
        ]
        command.extend(["--training", *map(str, training_files)])

    return command


def register_command_manifest(command, resolved_config, combination):
    COMMAND_MANIFESTS[tuple(map(str, command))] = {
        "command": [str(token) for token in command],
        "canonical_command": shlex.join(map(str, command)),
        "combination": {
            str(key): value
            for key, value in combination.items()
            if key not in K_FOLD_METADATA_PARAMETERS
        },
        "resolved_config": _resolved_config_payload(resolved_config),
    }


def _command_manifest_for(command):
    key = tuple(map(str, command))
    if key in COMMAND_MANIFESTS:
        return COMMAND_MANIFESTS[key]
    tokens = [str(token) for token in command]
    if len(tokens) > 2 and tokens[0] == "pipenv" and tokens[1] == "run":
        return COMMAND_MANIFESTS.get(tuple(tokens[2:]))
    return None


def _command_output_dir(command):
    argv = _main_command_argv(command)
    for index, token in enumerate(argv):
        if token == "--output_dir" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if token.startswith("--output_dir="):
            return Path(token.split("=", 1)[1])
    return None


def write_command_manifest(command):
    if not _is_main_command(command):
        return None
    output_dir = _command_output_dir(command)
    if output_dir is None:
        raise ValueError("InvalidMainCommand: --output_dir is required to write command_manifest.json.")
    manifest = _command_manifest_for(command) or {
        "command": [str(token) for token in command],
        "canonical_command": shlex.join(map(str, command)),
        "resolved_config": None,
    }
    manifest = dict(manifest)
    manifest["command"] = [str(token) for token in command]
    manifest["canonical_command"] = shlex.join(map(str, command))
    manifest["validated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["validation"] = {
        "duplicate_flags": [],
        "unknown_flags": [],
        "main_parser_valid": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "command_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return manifest_path


def run_cmd(command):
    validate_command_before_subprocess(command)
    canonical_command = shlex.join(map(str, command))
    logging.info("Command line: %s", canonical_command)
    if not arguments.dryrun:
        manifest_path = write_command_manifest(command)
        if manifest_path is not None:
            logging.info("Command manifest saved to %s", manifest_path)
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
        validate_command_before_subprocess(normal_command)
        subprocess.run(normal_command, check=True)

    logging.info("Compare modes: running batches mode.")
    if parsed_arguments.dryrun:
        logging.info("Command line: %s", shlex.join(map(str, batches_command)))
    else:
        validate_command_before_subprocess(batches_command)
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
    validate_command_before_subprocess(command)
    completed_process = subprocess.run(command, check=False)
    sys.exit(completed_process.returncode)


def configure_logging(output_dir, verbosity):
    logging_filename = output_dir / "evaluation_campaigns.log"
    logging_format = "%(asctime)s\t---\t%(message)s"
    if verbosity == logging.DEBUG:
        logging_format = "%(asctime)s\t---\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s"

    logging.basicConfig(format=logging_format, level=verbosity, force=True)
    rotating_file_handler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotating_file_handler.setLevel(verbosity)
    rotating_file_handler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotating_file_handler)


def build_output_directory(parsed_arguments):
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix_parts = [run_id]
    if parsed_arguments.output_suffix:
        suffix_parts.append(str(parsed_arguments.output_suffix))
    if parsed_arguments.execution_mode == "batches":
        suffix_parts.append("batches")
    output_dir = (
        REPO_ROOT
        / "outputs"
        / "appclassnet_top200"
        / str(parsed_arguments.run_mode_effective)
        / "_".join(suffix_parts)
    )
    parsed_arguments.run_id = output_dir.name
    return output_dir


def count_campaign_combinations(campaigns_chosen):
    total = 0
    for campaign_name in campaigns_chosen:
        campaign = campaigns_available[campaign_name]
        total += math.prod(len(values) for values in campaign.values())
    return total


def log_full_run_summary(parsed_arguments, campaigns_chosen, raw_root, output_dir):
    if parsed_arguments.run_mode_effective != "full":
        return
    effective_parameters = getattr(parsed_arguments, "_effective_parameters", {})
    vae_epochs = effective_parameters.get("vae_epochs", {}).get("effective_value", FULL_PROFILE.vae_epochs)
    gan_epochs = effective_parameters.get("gan_epochs", {}).get("effective_value", FULL_PROFILE.gan_epochs)
    planned_synthetic_per_class = (
        getattr(parsed_arguments, "generated_samples_per_class", None)
        if _argument_was_explicit(parsed_arguments, "generated_samples_per_class")
        else _profile_parameter_value(parsed_arguments, "generated_samples_per_class")
    ) or DEFAULT_SYNTHETIC_SAMPLES_PER_CLASS * 2
    try:
        import numpy

        real_rows = {
            split_name: int(numpy.load(validate_raw_split(raw_root, split_name)[1], mmap_mode="r").shape[0])
            for split_name in APPCLASSNET_SPLITS
        }
    except Exception as error:
        real_rows = {"unavailable": str(error)}
    logging.info("Full run plan:")
    logging.info("  models: %d", len(campaigns_chosen))
    logging.info("  combinations: %d", count_campaign_combinations(campaigns_chosen))
    logging.info("  classes: %d", APPCLASSNET_NUM_CLASSES)
    logging.info("  real samples: %s", real_rows)
    logging.info("  planned synthetic per class: %s", planned_synthetic_per_class)
    logging.info("  epochs: vae=%s gan=%s", vae_epochs, gan_epochs)
    logging.info("  expected space: data_space=source evaluation_space=%s", parsed_arguments.evaluation_space)
    logging.info("  output: %s", output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="MalDataGen AppClassNet top200 campaign runner")

    parser.add_argument(
        "--campaign",
        "-c",
        help=f"Campaign list, comma separated list, sf demo alias, or sf2 SDV alias. Default: {DEFAULT_CAMPAIGN}",
        default=None,
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--run_mode",
        "--mode",
        dest="run_mode",
        choices=["demo", "full"],
        default=None,
        help="execution profile: demo uses reduced AppClassNet settings; full preserves the complete run profile",
    )
    parser.add_argument(
        "--pipeline",
        choices=["tr_tr", "synthetic", "augmentation", "all"],
        default=None,
        help="pipeline selector: tr_tr, synthetic (TR-TS and TS-TR), augmentation (TR+TS-TR), or all",
    )
    parser.add_argument("--dryrun", "-d", help="show commands without running them", action="store_true")
    parser.add_argument("--pipenv", "-p", help="prefix subprocesses with pipenv run", action="store_true")
    parser.add_argument("--verbosity", "-v", default=DEFAULT_VERBOSITY_LEVEL, type=int)
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable),
        help="Python executable used to run main.py and plots.py",
    )
    parser.add_argument(
        "--skip_plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="skip plots.py after each campaign run",
    )
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
        help="deprecated; use --run_mode full",
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
    parser.add_argument("--prepare_max_samples", default=None, type=int, help="max rows written to the prepared CSV")
    parser.add_argument("--data_load_max_samples", default=None, type=int, help="max rows passed to main.py after CSV loading")
    parser.add_argument("--prepare_chunk_size", default=DEFAULT_CHUNK_SIZE, type=int, help="rows per CSV write chunk")
    parser.add_argument(
        "--execution_mode",
        choices=["normal", "batches"],
        default=None,
        help="normal preserves the current CSV flow; batches uses AppClassNet .npy files directly",
    )
    parser.add_argument("--batch_size", default=None, type=int, help="training batch size for batches mode")
    parser.add_argument(
        "--eval_batch_size",
        default=None,
        type=int,
        help="evaluation batch size for batches mode",
    )
    parser.add_argument(
        "--generation_batch_size",
        default=None,
        type=int,
        help="generation batch size for batches mode",
    )
    parser.add_argument(
        "--use_mmap",
        action=argparse.BooleanOptionalAction,
        default=None,
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
        "--experiment_budget_scenario",
        choices=sorted(EXPERIMENT_BUDGET_SCENARIOS),
        default=None,
        help="explicit AppClassNet budget scenario A-G; individual quota flags still override scenario values",
    )
    parser.add_argument(
        "--evaluation_protocol",
        choices=EVALUATION_PROTOCOL_CHOICES,
        default="appclassnet_strict",
        help="canonical protocol selector: tr_tr, tr_ts, ts_tr, tr_plus_ts_tr, all; legacy aliases remain valid",
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=["none", "tr_ts", "ts_tr", "tr_ts_tr", "both", "all"],
        default="both",
        help="synthetic evaluation mode; both preserves legacy TR-TS/TS-TR, all also runs TR+TS-TR",
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
        "--synthetic_control",
        choices=["none", "real_resample", "label_permutation"],
        default="none",
        help="diagnostic control written through the synthetic npy_batches path",
    )
    parser.add_argument(
        "--baseline_classifier",
        choices=["decision_tree", "extra_trees", "random_forest_light", "sgd"],
        default="decision_tree",
        help="real-real baseline classifier used by --baseline_real_only",
    )
    parser.add_argument(
        "--train_samples_per_class",
        default=None,
        type=int,
        help="stratified real train samples per class used by --baseline_real_only",
    )
    parser.add_argument(
        "--test_samples_per_class",
        default=None,
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
        "--generated_samples_per_class",
        default=None,
        type=int,
        help="per-class synthetic rows generated before splitting synthetic train/test quotas",
    )
    parser.add_argument(
        "--class_subset",
        default=None,
        help="comma-separated original class labels to keep and remap inside this diagnostic experiment",
    )
    parser.add_argument(
        "--num_classes_subset",
        default=None,
        type=int,
        help="keep original labels 0..N-1 and remap labels inside this diagnostic experiment",
    )
    parser.add_argument(
        "--real_class_count_policy",
        choices=["strict", "uniform_min", "available_cap"],
        default=None,
        help=("real split quota policy; default is strict for AppClassNet after campaign resolution. "
              "strict requires the request, uniform_min uses a common cap, available_cap caps per class."),
    )
    parser.add_argument(
        "--samples_per_class_scope",
        choices=["split", "fold"],
        default=None,
        help="scope for per-class real quotas; default is split for AppClassNet after campaign resolution",
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
        "--vae_epochs",
        default=None,
        type=int,
        help="diagnostic alias with precedence over campaign variational_autoencoder_number_epochs",
    )
    parser.add_argument(
        "--gan_epochs",
        default=None,
        type=int,
        help="diagnostic alias with precedence over campaign GAN epoch counts",
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
    apply_execution_profile(arguments)
    apply_experiment_budget_scenario(arguments)
    normalize_classifier_arguments(arguments)

    if arguments.list_campaigns:
        print("\n".join(sorted(campaigns_available)))
        return 0

    if arguments.classes_per_group <= 0:
        raise ValueError("--classes_per_group must be a positive integer.")
    if arguments.num_classes_subset is not None and arguments.num_classes_subset <= 0:
        raise ValueError("--num_classes_subset must be a positive integer.")

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

    campaigns_chosen = choose_campaigns(arguments.campaign, full=arguments.run_mode_effective == "full")
    normalize_preprocessing_arguments(arguments, campaigns_chosen)
    output_dir = build_output_directory(arguments)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, arguments.verbosity)
    logging.info(
        "Resolved AppClassNet run_mode=%s origin=%s pipeline=%s origin=%s",
        arguments.run_mode_effective,
        arguments.run_mode_origin,
        arguments.pipeline_effective,
        arguments.pipeline_origin,
    )
    if arguments.baseline_real_only:
        if arguments.evaluation_mode != "none":
            logging.warning(
                "--baseline_real_only executes only TR-TR; ignoring --evaluation_mode=%s and using none.",
                arguments.evaluation_mode,
            )
        arguments.evaluation_mode = effective_evaluation_mode(arguments)
        baseline_sample_arguments = resolve_effective_sample_arguments(arguments, {})
        for parameter, value in baseline_sample_arguments["values"].items():
            setattr(arguments, parameter, value)
        baseline_real_count_arguments = resolve_effective_real_class_count_arguments(arguments, {})
        for parameter, value in baseline_real_count_arguments["values"].items():
            setattr(arguments, parameter, value)
    print_all_settings(arguments)
    warn_prepare_limit_if_needed(arguments, campaigns_chosen)

    raw_root = resolve_project_path(arguments.raw_root)
    converted_root = resolve_project_path(arguments.converted_root)
    effective_raw_root = raw_root
    effective_converted_root = converted_root
    log_full_run_summary(arguments, campaigns_chosen, raw_root, output_dir)

    if arguments.diagnostic_only:
        diagnostics_path, diagnostics = run_input_diagnostics(arguments, raw_root, output_dir, campaigns_chosen)
        logging.info("Diagnostic-only mode completed. Diagnostics: %s", diagnostics_path)
        return 1 if diagnostics["errors"] else 0

    if arguments.baseline_real_only:
        metrics_path, metrics = run_real_real_baseline(arguments, raw_root, output_dir)
        write_result = write_baseline_batches_metrics(metrics, REPO_ROOT / RESULTS_ROOT / "batches" / "metrics.json")
        payload = write_result[1] if isinstance(write_result, tuple) else metrics
        write_run_results(output_dir, arguments, campaigns_chosen, payload, [])
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
            split_mode = "provided" if arguments.execution_mode == "batches" else "cross_validation"
            combination = with_k_fold_metadata(combination, split_mode)
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
    if arguments.dryrun:
        logging.info("Dry run completed; training, generation, evaluation and RunResults.json writing were skipped.")
        logging.info("Evaluation duration: %s", time_end_evaluation - time_start_evaluation)
        return 0
    if arguments.execution_mode == "batches":
        write_result = write_batches_metrics(
            results_grouping,
            REPO_ROOT / RESULTS_ROOT / "batches" / "metrics.json",
            arguments,
        )
        payload = write_result[1] if isinstance(write_result, tuple) else write_result
        print_results_payload(payload)
        write_run_results(output_dir, arguments, campaigns_chosen, payload, results_grouping)
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
    else:
        payload = collect_normal_metrics(results_grouping, arguments)
        write_run_results(output_dir, arguments, campaigns_chosen, payload, results_grouping)
    logging.info("Evaluation duration: %s", time_end_evaluation - time_start_evaluation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
