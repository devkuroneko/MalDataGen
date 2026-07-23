#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Kayuã Oleques Paim'
__email__ = 'kayuaolequesp@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Kayuã Oleques']


# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import gc
    import json
    import sys
    import time
    import numpy
    import pandas
    import logging
    import subprocess
    import tensorflow
    from pathlib import Path

    from sklearn.utils import shuffle

    from Engine.Metrics.Metrics import Metrics

    from Engine.DataIO.CSVLoader import autosave
    from Engine.DataIO.CSVLoader import autoload
    from Engine.DataIO.SamplePlanner import build_sample_plan_from_args
    from Engine.DataIO.SamplePlanner import sample_plan_to_legacy_metadata
    from Engine.DataIO.RealClassCountPolicy import build_split_sample_plan

    from Engine.Arguments.Arguments import Arguments
    from Engine.Arguments.Arguments import arguments

    from Engine.Metrics.Metrics import import_metrics

    from Engine.Evaluation.Evaluation import Evaluation
    from Engine.Evaluation.ExperimentProtocol import CANONICAL_PROTOCOLS
    from Engine.Evaluation.ExperimentProtocol import legacy_name_for_protocol_id
    from Engine.Evaluation.ExperimentProtocol import protocol_metadata
    from Engine.Evaluation.ExperimentProtocol import resolve_evaluation_protocol_plan
    from sklearn.model_selection import StratifiedKFold

    from Engine.DataIO.CSVLoader import CSVDataProcessor
    from Engine.DataIO.SyntheticBatchIO import SyntheticBatchWriter
    from Engine.DataIO.SyntheticBatchIO import SyntheticSplitBatchReaders
    from Engine.DataIO.DatasetContracts import AlignedDataset
    from Engine.DataIO.DatasetContracts import SplitData
    from Engine.DataIO.DatasetContracts import validate_xy_alignment
    from Engine.DataIO.SyntheticLabelAudit import SyntheticLabelGenerationAudit
    from Engine.DataIO.SyntheticLabelAudit import audit_synthetic_label_generation
    from Engine.DataIO.LabelUtils import to_one_hot_batch
    from Engine.DataIO.SyntheticQualityAudit import assert_label_permutation_control_metrics
    from Engine.DataIO.SyntheticQualityAudit import assert_real_resample_control_metrics
    from Engine.DataIO.SyntheticQualityAudit import run_synthetic_quality_audit
    from Engine.DataIO.SyntheticSanityChecks import run_synthetic_sanity_checks
    from Engine.DataIO.JsonIO import atomic_write_json
    from Engine.Utils.ResourceMonitor import get_current_memory_mb
    from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy
    from Engine.Preprocessing.FeatureTransformManager import ModelInputAdapter
    from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
    from Engine.Pipelines.Interfaces import partition_generation_classes
    from Engine.Evaluation.TrTrPipeline import run_tr_tr_pipeline
    from Engine.Evaluation.TrTrPipeline import tr_tr_config_from_namespace

    from Engine.Classifiers.Classifiers import Classifiers

    from Engine.Models.GenerativeModels import import_models

    from Engine.Models.GenerativeModels import GenerativeModels

    from Engine.Evaluation.CrossValidation import StratifiedData
    from Engine.Classifiers.Classifiers import import_classifiers
    from Engine.Support.HardwareManager import HardwareManager

except ImportError as error:
    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 -m venv ~/Python3venv/SyntheticOceanAI ")
    print("  source ~/Python3venv/SyntheticOceanAI/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)



DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"
PARTITIONED_GENERATION_SUPPORTED_MODELS = {
    "adversarial",
    "autoencoder",
    "variational",
    "wasserstein",
    "wasserstein_gp",
    "quantized",
    "denoising_diffusion",
}


class SyntheticControlSplitMismatchError(ValueError):
    """Raised when a synthetic control source split does not match its role."""


class AggregateDataUsedAsRawSamplesError(ValueError):
    """Raised when aggregate rows such as centroids are passed as raw control samples."""


def _dataset_bundle_from_evaluation_input(owner, evaluation_input):
    if hasattr(evaluation_input, "train") and hasattr(evaluation_input, "test"):
        return evaluation_input
    if isinstance(evaluation_input, dict):
        return evaluation_input.get("dataset_bundle") or getattr(owner, "_dataset_bundle", None)
    return getattr(owner, "_dataset_bundle", None)


def _legacy_dictionary_for_real_splits(real_train_data, real_test_data):
    return {
        "x_training_real": real_train_data.X,
        "y_training_real": real_train_data.y,
        "x_evaluation_real": real_test_data.X,
        "y_evaluation_real": real_test_data.y,
        "training_split_name": real_train_data.name,
        "evaluation_split_name": real_test_data.name,
        "real_train_split": real_train_data,
        "real_test_split": real_test_data,
        "split_metadata": {
            "train": {
                "name": real_train_data.name,
                "x_path": real_train_data.x_path,
                "y_path": real_train_data.y_path,
                "dataset_id": real_train_data.dataset_id,
                "num_samples": real_train_data.num_samples,
                "class_counts": {str(key): int(value) for key, value in real_train_data.class_counts.items()},
                "minimum_class_count": real_train_data.minimum_class_count,
                "shape": list(real_train_data.X.shape),
            },
            "test": {
                "name": real_test_data.name,
                "x_path": real_test_data.x_path,
                "y_path": real_test_data.y_path,
                "dataset_id": real_test_data.dataset_id,
                "num_samples": real_test_data.num_samples,
                "class_counts": {str(key): int(value) for key, value in real_test_data.class_counts.items()},
                "minimum_class_count": real_test_data.minimum_class_count,
                "shape": list(real_test_data.X.shape),
            },
        },
    }


def run_synthetic_evaluation_modes(owner, dictionary_data, evaluation_synthetic):
    protocol_plan = resolve_evaluation_protocol_plan(owner.arguments)
    evaluation_mode = getattr(owner.arguments, "evaluation_mode", protocol_plan.protocol)
    run_tr_ts = protocol_plan.run_tr_ts
    run_ts_tr = protocol_plan.run_ts_tr
    run_tr_ts_tr = protocol_plan.run_tr_plus_ts_tr
    synthetic_for_tr_ts = getattr(evaluation_synthetic, "test_reader", evaluation_synthetic)
    synthetic_for_ts_tr = getattr(evaluation_synthetic, "train_reader", evaluation_synthetic)
    dataset_bundle = _dataset_bundle_from_evaluation_input(owner, dictionary_data)
    use_provided_bundle = (
        dataset_bundle is not None
        and getattr(owner.arguments, "split_mode", "cross_validation") == "provided"
        and getattr(dataset_bundle, "test", None) is not None
    )
    if use_provided_bundle:
        real_train = dataset_bundle.train
        real_test = dataset_bundle.test
        guard_dictionary = _legacy_dictionary_for_real_splits(real_train, real_test)
    else:
        real_train = None
        real_test = None
        guard_dictionary = dictionary_data
    if run_tr_ts:
        owner._guard_current_evaluation_space(guard_dictionary, synthetic_for_tr_ts)
    if (run_ts_tr or run_tr_ts_tr) and synthetic_for_ts_tr is not synthetic_for_tr_ts:
        owner._guard_current_evaluation_space(guard_dictionary, synthetic_for_ts_tr)
    elif (run_ts_tr or run_tr_ts_tr) and not run_tr_ts:
        owner._guard_current_evaluation_space(guard_dictionary, synthetic_for_ts_tr)
    if run_tr_ts:
        if use_provided_bundle:
            owner.evaluation_TR_TS(real_train_data=real_train, synthetic_test_data=synthetic_for_tr_ts)
        else:
            owner.evaluation_TR_TS(dictionary_data, synthetic_for_tr_ts)
    else:
        owner.mark_evaluation_classifiers_not_applicable(
            "TR-TS",
            owner.fold_number + 1,
            f"TR-TS skipped because evaluation_mode={evaluation_mode}.",
        )
    if run_ts_tr:
        if use_provided_bundle:
            owner.evaluation_TS_TR(synthetic_train_data=synthetic_for_ts_tr, real_test_data=real_test)
        else:
            owner.evaluation_TS_TR(dictionary_data, synthetic_for_ts_tr)
    else:
        owner.mark_evaluation_classifiers_not_applicable(
            "TS-TR",
            owner.fold_number + 1,
            f"TS-TR skipped because evaluation_mode={evaluation_mode}.",
        )
    if run_tr_ts_tr:
        if use_provided_bundle:
            owner.evaluation_TR_TS_TR(
                real_train_data=real_train,
                synthetic_train_data=synthetic_for_ts_tr,
                real_test_data=real_test,
            )
        else:
            owner.evaluation_TR_TS_TR(dictionary_data, synthetic_for_ts_tr)


def _argument_value_or_fallback(arguments, argument_name, fallback_name):
    value = getattr(arguments, argument_name, None)
    if value is not None:
        return value
    return getattr(arguments, fallback_name, None)


def _json_ready(value):
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _protocol_selection_for_arguments(arguments):
    return list(resolve_evaluation_protocol_plan(arguments).active_protocol_ids)



class SynDataGen(Arguments, CSVDataProcessor, Metrics, GenerativeModels, Classifiers, Evaluation):
    """
    SYNTHETIC DATA GENERATION AND EVALUATION FRAMEWORK
    =================================================

    The Synthetic Ocean AI library is designed for the generation of tabular data using generative models.
    It provides comprehensive GPU support, allowing for efficient processing, and can be used either as a
    framework or as a Python library, offering flexibility depending on the user's requirements.

    The library supports data ingestion from various formats, including CSV and XLS, enabling seamless
    integration with existing datasets. It features a wide array of pre-implemented generative algorithms
    and includes pre-trained models for immediate use, reducing the need for extensive training.

    One of the key strengths of Synthetic Ocean AI is its ability to provide fine-grained model and
    algorithm parameterization, giving users control over hyperparameters, training configurations,
    and other aspects of the generative process. The library also includes built-in support for evaluation
    metrics, allowing users to assess the quality of generated data. Additionally, it offers tools for
    graph generation, enabling visual analysis of model performance and data generation processes.

    A comprehensive pipeline for generating and evaluating synthetic data using various generative models.
    The class combines data processing, model training, generation, and evaluation capabilities.

    Key Features:
    -------------
        @ Supports multiple generative models (adversarial, autoencoder, variational, etc.)
        @ Built-in stratified k-fold cross validation
        @ Multiple evaluation strategies (TS-TR, TR-TS)
        @ Automated metrics calculation and reporting
        @ Model persistence and data export capabilities

    # Version: 1.0.1
    # Last Updated: 2025-5-28
    # Author: Synthetic Ocean AI - Team
    # License: MIT

    Purpose:
    --------
      Provides an end-to-end pipeline for:
        - Generating synthetic datasets using state-of-the-art generative models
        - Evaluating synthetic data quality through multiple validation strategies
        - Comparing model performance across different architectures
        - Producing publication-ready metrics and visualizations

    Architecture Overview:
    ---------------------

                                +--------+-------+      +--------+-------+      +--------+-------+
                                |  Activations   +------+     Layers     +------+    Specials    |
                                +-------+--------+      +-------+--------+      +--------+-------+
                                                                |
                                                                |
                                +-----------------+     +-------+--------+    +--------+-------+
                                |    Arguments    |     |     Models     |    |      Loss      |
                                +--------+--------+     +--------+-------+    +--------+-------+
                                        |                        |                     |
        +---------------+       +-------+--------+               |            +--------+-------+         +--------+-------+
        | DataProcessor +-------+   Generative   +---------------@------------+   Algorithms   +---------+   Optimizers   |
        +---------------+       |     Models     |                            +----------------+         +----------------+
                                +-------+--------+
                                        |
        +-------v--------+      +-------v--------+               +----------------+
        |     Plotter    +------+     Metrics    +---------------+   Classifiers  |
        +-------+--------+      +-------+--------+               +----------------+
                                        |
                                +-------v--------+
                                |   SynDataGen   |
                                +----------------+


    Model Catalog:
    --------------
        1. Adversarial (GAN) [model_type='adversarial']

            Implements an adversarial training algorithm, typically used in Generative Adversarial Networks (GANs).

            This class performs adversarial training by utilizing a generator and a discriminator,
            optimizing the generator to produce realistic data while training the discriminator to differentiate
            between real and fake data.

        2. Autoencoder [model_type='autoencoder']

            Implements a  AutoEncoder model for generating synthetic data.

            This class implements an Autoencoder model by inheriting from the VanillaEncoder and VanillaDecoder classes.
            It constructs an autoencoder architecture by combining both an encoder and a decoder with customizable
            hyperparameters. The autoencoder is typically used for tasks such as dimensionality reduction, feature learning,
            and denoising.

        3. Variational Autoencoder [model_type='variational']

            Implements a Variational AutoEncoder (VAE) model for generating synthetic data.

            The model includes an encoder and a decoder for encoding input data and reconstructing
            it from a learned latent space. During training, it computes both the reconstruction loss
            and the KL divergence loss. The trained decoder can be used to generate synthetic data.

        4. Vector Quantized Variational Autoencoder [model_type='quantized']

            Implements a Vector Quantized Variational Autoencoder (VQ-VAE) model for generating synthetic data.

            This class implements a VQ-VAE by combining an encoder, a quantized latent space with a codebook,
            and a decoder. The model learns discrete latent representations by mapping encoder outputs to
            the nearest codebook vectors during training. The decoder then reconstructs the input from
            these quantized latent embeddings.

        5. Wasserstein GAN [model_type='wasserstein']

            A Wasserstein Generative Adversarial Network (Wasserstein GAN) model.

           This class represents a Wasserstein GAN consisting of a generator and discriminator (critic) model.
           It implements the Wasserstein loss to train the discriminator and generator, promoting more
           stable training compared to traditional GANs.

        6. Wasserstein GP GAN [model_type='wasserstein_gp']

            A Wasserstein GP Generative Adversarial Network (WassersteinGP GAN) model.

            This class represents a WassersteinGP GAN consisting of a generator and discriminator model.
            It implements the WassersteinGP loss with gradient penalty to improve the training of the
            discriminator and generator.

        7. Latent Diffusion Models [model_type='latent_diffusion']

            Implements a diffusion process using UNet architectures for generating synthetic data.

            This model integrates an autoencoder and a diffusion network, enabling both data
            reconstruction and controlled generative modeling through Gaussian diffusion.

        8. Denoising Diffusion [model_type='denoising_diffusion']

            Implements a diffusion process using UNet architectures for generating synthetic data.

            This model integrates an autoencoder and a diffusion network, enabling both data
            reconstruction and controlled generative modeling through Gaussian diffusion.

        9. Copy/Paste [model_type='copy']

            Copy is a naive machine learning model designed to generate synthetic data samples
            for specific classes based on provided real samples. This simple approach is primarily used
            for testing and comparison purposes, serving as a baseline method in experiments.


    Data Flow:
    ---------
        1. Input Data → 2. Preprocessing → 3. Stratified Splitting
        ↓                                    ↓
        7. Results Collection ← 6. Evaluation ← 5. Generation ← 4. Model Training

    Evaluation Strategies:
    --------------------
        A. TS-TR (Train Synthetic - Assess Real)
            - Trains: On generated synthetic data
            - Tests: On held-out real validation data
            - Measures: Generalization capability

        B. TR-TS (Train Real - Assess Synthetic)
            - Trains: On real training data
            - Tests: On generated synthetic data
            - Measures: Generation quality

    Metrics Tracked:
    ---------------
        Primary Metrics:
        - Accuracy, Precision, Recall, F1, ROC-AUC, FalseNegativeRate, MSE, MAE, TrueNegativeRate

        Secondary Metrics:
        - EuclideanDistance, HellingerDistance, LogLikelihood, ManhattanDistance

    Example Workflows:
    -----------------
        1. Basic Usage:
            >>> gen = SynDataGen()
            >>> gen.run_experiments()

        2. Custom Configuration:
            >>> gen = SynDataGen()
            >>> gen.arguments.model_type = 'variational'
            >>> gen.arguments.number_k_folds = 5
            >>> gen.run_experiments()

        3. Research Pipeline:
            >>> for model in ['adversarial', 'variational', 'latent_diffusion']:
            ...     gen = SynDataGen()
            ...     gen.arguments.model_type = model
            ...     gen.run_experiments()
            >>>     gen.save_comparison_report()

    """
    @arguments
    def __init__(self):
        """
        CONSTRUCTOR
        ==========
        Initializes the synthetic data generation pipeline with default parameters.

        Detailed Initialization Sequence:
        -------------------------------
        1. Parent Class Initialization:
           - Arguments: Loads CLI/config file parameters
           - CSVDataProcessor: Initializes data loading pipelines
           - Metrics: Sets up metric tracking structures
           - GenerativeModels: Prepares model architectures
           - Classifiers: Loads evaluation classifiers

        2. Instance Variable Setup:
           - fold_number: Initialized to None, tracks current CV fold [0, n_folds-1]
           - data_generated: Dictionary structure:
               {
                   class_0: np.ndarray (n_samples, n_features),
                   class_1: np.ndarray (n_samples, n_features),
                   ...
               }
           - generator_name: String identifier matching model_type
           - directory_output_data: Path object with structure:
               ./output/
                   ├── models/
                   ├── data/
                   │   ├── fold_1/
                   │   ├── ...
                   └── metrics/

        3. Filesystem Preparation:
           - Creates required directory structure
           - Initializes log files with timestamp
           - Validates write permissions

        """

        super().__init__()

        self.fold_number = None
        self.data_generated = None
        self.generator_name = None
        self.directory_output_data = None
        self.directory_output_data = self.get_data_generated_path()
        self._manager = HardwareManager(use_gpu=self.arguments.use_gpu)
        self._manager.configure()

        self._sdv = None 
        self._model_input_adapter = None
        self._current_real_source_metadata = None
        self._current_synthetic_metadata = None
        self._current_evaluation_source_x = None

    def _metric_block_for_protocol(self, legacy_name):
        for classifier_name in getattr(self, "_dictionary_classifiers_name", []):
            classifier_block = self._dictionary_metrics.get(legacy_name, {}).get(classifier_name, {})
            fold_block = classifier_block.get(f"{self.fold_number + 1 if self.fold_number is not None else 1}-Fold")
            if not isinstance(fold_block, dict):
                fold_block = next(
                    (value for key, value in classifier_block.items() if key.endswith("-Fold") and isinstance(value, dict)),
                    None,
                )
            if not isinstance(fold_block, dict):
                continue
            required = ("Accuracy", "BalancedAccuracy", "MacroF1", "WeightedF1")
            if all(isinstance(fold_block.get(metric), (int, float, numpy.integer, numpy.floating)) for metric in required):
                return classifier_name, fold_block
        return None, {}

    def _batch_metadata_for_protocol(self, legacy_name):
        batch_blocks = self._dictionary_metrics.get("BatchClassifier", {})
        for fold_block in batch_blocks.values():
            if isinstance(fold_block, dict) and isinstance(fold_block.get(legacy_name), dict):
                return dict(fold_block[legacy_name])
        return {}

    def _evaluation_metadata_for_protocol(self, legacy_name):
        metadata_blocks = self._dictionary_metrics.get("EvaluationMetadata", {})
        for fold_block in metadata_blocks.values():
            if isinstance(fold_block, dict) and isinstance(fold_block.get(legacy_name), dict):
                return dict(fold_block[legacy_name])
        return {}

    def _protocol_status_summary(self, protocol_id):
        legacy_name = legacy_name_for_protocol_id(protocol_id)
        requested = protocol_id in set(_protocol_selection_for_arguments(self.arguments))
        if not requested:
            return {"status": "not_run", "reason": "protocol was not requested"}

        classifier_name, metrics = self._metric_block_for_protocol(legacy_name)
        batch_metadata = self._batch_metadata_for_protocol(legacy_name)
        evaluation_metadata = self._evaluation_metadata_for_protocol(legacy_name)
        required = ("Accuracy", "BalancedAccuracy", "MacroF1", "WeightedF1")
        missing = [metric for metric in required if not isinstance(metrics.get(metric), (int, float, numpy.integer, numpy.floating))]
        status = "completed" if classifier_name and not missing else "failed"
        reason = None if status == "completed" else f"missing valid metric(s): {', '.join(missing) or 'all'}"
        classifier_payload = {
            "requested_classifier": batch_metadata.get(
                "requested_classifier",
                getattr(self.arguments, "normal_classifier", None)
                or getattr(self.arguments, "eval_classifier", None)
                or getattr(self.arguments, "classifier", None),
            ),
            "effective_classifier": batch_metadata.get(
                "effective_classifier",
                getattr(self.arguments, "eval_classifier", None) or classifier_name,
            ),
            "reported_classifier": classifier_name or batch_metadata.get("classifier"),
            "effective_fit_rows": batch_metadata.get("effective_fit_rows"),
            "quota_per_class": batch_metadata.get("subset_quota_per_class"),
            "global_limit": getattr(self.arguments, "batch_classifier_subset_size", None),
            "discarded_rows_for_fit": batch_metadata.get("discarded_rows_for_fit"),
            "discarded_rows": batch_metadata.get("discarded_rows"),
            "hyperparameters": {
                key: value
                for key, value in batch_metadata.items()
                if key in {
                    "n_estimators",
                    "max_depth",
                    "max_samples",
                    "class_weight",
                    "criterion",
                    "random_state",
                    "subset_quota_per_class",
                    "batch_classifier",
                    "eval_classifier",
                }
            },
        }
        return {
            "status": status,
            "reason": reason,
            "legacy_metric_name": legacy_name,
            "classifier": classifier_payload,
            "metrics": {metric: metrics.get(metric) for metric in required},
            "train_sources": protocol_metadata(protocol_id)["train_sources"],
            "test_source": protocol_metadata(protocol_id)["test_source"],
            "train_shape": evaluation_metadata.get("train_shape"),
            "test_shape": evaluation_metadata.get("test_shape"),
            "train_class_counts": evaluation_metadata.get("train_class_counts", {}),
            "test_class_counts": evaluation_metadata.get("test_class_counts", {}),
            "dataset_hashes": {
                "train_hash": evaluation_metadata.get("train_hash"),
                "test_hash": evaluation_metadata.get("test_hash"),
            },
            "batch_classifier_metadata": batch_metadata,
        }

    def _effective_protocol_budgets(self):
        generated_plan = getattr(self, "_number_samples_per_class", None)
        generated_classes = generated_plan.get("classes", {}) if isinstance(generated_plan, dict) else {}
        generator_training_real_samples_per_class = getattr(self.arguments, "train_samples_per_class", None)
        return {
            "generator_training_real_samples_per_class": generator_training_real_samples_per_class,
            "classifier_training_samples_per_class": {
                "real": getattr(self.arguments, "train_samples_per_class", None),
                "synthetic_train": getattr(self.arguments, "synthetic_train_samples_per_class", None),
                "synthetic_test": getattr(self.arguments, "synthetic_test_samples_per_class", None),
            },
            "test_samples_per_class": getattr(self.arguments, "test_samples_per_class", None),
            "generated_samples_per_class": (
                next(iter(generated_classes.values())) if generated_classes else None
            ),
            "limit_global": getattr(self.arguments, "batch_classifier_subset_size", None),
        }

    def _write_experiment_protocol_artifacts(self):
        output_dir = Path(self.current_subdir)
        protocol_plan = resolve_evaluation_protocol_plan(self.arguments)
        selected_protocols = list(protocol_plan.active_protocol_ids)
        protocol_id = (
            selected_protocols[0]
            if len(selected_protocols) == 1
            else ("ALL" if set(selected_protocols) == {item["protocol_id"] for item in CANONICAL_PROTOCOLS.values()} else "CUSTOM")
        )
        summaries = {
            protocol_key: self._protocol_status_summary(protocol_key)
            for protocol_key in ("TR_TR", "TR_TS", "TS_TR", "TR_PLUS_TS_TR")
        }
        first_metadata = next(
            (
                metadata
                for fold_block in self._dictionary_metrics.get("EvaluationMetadata", {}).values()
                if isinstance(fold_block, dict)
                for metadata in fold_block.values()
                if isinstance(metadata, dict)
            ),
            {},
        )
        budget_payload = self._effective_protocol_budgets()
        controls_payload = {
            "synthetic_control": getattr(self.arguments, "synthetic_control", "none"),
            "real_resample": getattr(self.arguments, "synthetic_control", "none") == "real_resample",
            "label_permutation": getattr(self.arguments, "synthetic_control", "none") == "label_permutation",
            "presented_as_generation": getattr(self.arguments, "synthetic_control", "none") == "none",
        }
        protocol_payload = {
            "protocol_id": protocol_id,
            "protocol_plan": protocol_plan.as_dict(),
            "requested_protocols": selected_protocols,
            "train_sources": {
                protocol_key: protocol_metadata(protocol_key)["train_sources"]
                for protocol_key in ("TR_TR", "TR_TS", "TS_TR", "TR_PLUS_TS_TR")
            },
            "test_source": {
                protocol_key: protocol_metadata(protocol_key)["test_source"]
                for protocol_key in ("TR_TR", "TR_TS", "TS_TR", "TR_PLUS_TS_TR")
            },
            "real_rows_per_class": {
                "train": first_metadata.get("train_class_counts", {}),
                "test": first_metadata.get("test_class_counts", {}),
            },
            "synthetic_rows_per_class": {
                key: value.get("batch_classifier_metadata", {}).get("synthetic_samples_used_by_class", {})
                for key, value in summaries.items()
            },
            "generator_training_budget": {
                "real_samples_per_class": budget_payload["generator_training_real_samples_per_class"],
            },
            "classifier_training_budget": budget_payload["classifier_training_samples_per_class"],
            "test_budget": {"real_samples_per_class": budget_payload["test_samples_per_class"]},
            "class_count": getattr(self.arguments, "num_classes", None) or getattr(self, "_synthetic_control_number_classes", lambda: None)(),
            "feature_count": self.get_number_columns(),
            "classifier": {
                "requested_classifier": getattr(self.arguments, "normal_classifier", None)
                or getattr(self.arguments, "eval_classifier", None)
                or getattr(self.arguments, "classifier", None),
                "effective_classifier": getattr(self.arguments, "eval_classifier", None)
                or getattr(self.arguments, "normal_classifier", None)
                or getattr(self.arguments, "classifier", None),
                "random_state": getattr(self.arguments, "random_state", None),
                "hyperparameters": {
                    "n_estimators": getattr(self.arguments, "n_estimators", None),
                    "max_depth": getattr(self.arguments, "max_depth", None),
                    "max_samples": getattr(self.arguments, "max_samples", None),
                    "class_weight": getattr(self.arguments, "class_weight", None),
                    "batch_classifier_subset_size": getattr(self.arguments, "batch_classifier_subset_size", None),
                },
            },
            "seeds": {"random_state": getattr(self.arguments, "random_state", None)},
            "transforms": {
                "feature_transform": getattr(self.arguments, "feature_transform", None),
                "generator_transform": getattr(self.arguments, "generator_transform", None),
                "classifier_transform": getattr(self.arguments, "classifier_transform", None),
                "evaluation_space": getattr(self.arguments, "evaluation_space", None),
            },
            "dataset_hashes": {
                protocol_key: value.get("dataset_hashes", {})
                for protocol_key, value in summaries.items()
            },
            "leakage_checks": {
                "ts_tr_test_source": summaries["TS_TR"].get("test_source") == "real_test",
                "tr_plus_ts_tr_test_source": summaries["TR_PLUS_TS_TR"].get("test_source") == "real_test",
                "synthetic_train_derives_from_test": False,
            },
            "run_id": output_dir.name,
        }
        protocol_path = output_dir / "experiment_protocol.json"
        atomic_write_json(
            protocol_payload,
            protocol_path,
            run_id=output_dir.name,
            protocol=protocol_plan.protocol,
            model=getattr(self.arguments, "model_type", None),
        )

        summary_payload = {
            **summaries,
            "controls": controls_payload,
            "budgets": budget_payload,
            "artifacts": {
                "experiment_protocol": str(protocol_path),
                "results_json": str(Path(self.get_evaluation_results_path()) / "Results.json"),
            },
        }
        summary_path = output_dir / "results_summary.json"
        atomic_write_json(
            summary_payload,
            summary_path,
            run_id=output_dir.name,
            protocol=protocol_plan.protocol,
            model=getattr(self.arguments, "model_type", None),
        )
        self._dictionary_metrics["ExperimentProtocol"] = protocol_payload
        self._dictionary_metrics["ResultsSummary"] = summary_payload
        logging.info("Experiment protocol manifest saved to %s", protocol_path)
        logging.info("Consolidated protocol summary saved to %s", summary_path)
        return protocol_path, summary_path

    def _run_explicit_tr_tr_pipeline(self):
        bundle = getattr(self, "_dataset_bundle", None)
        if bundle is None:
            raise ValueError(
                "pipeline=tr_tr requires data_format=npy_xy with a loaded DatasetBundle. "
                "Provide train_x/train_y and test_x/test_y paths."
            )
        if getattr(bundle, "test", None) is None:
            raise ValueError("pipeline=tr_tr requires a provided test split.")

        config = tr_tr_config_from_namespace(self.arguments)
        output_base = Path(self.current_subdir) / "tr_tr"
        model_params = {
            "classifier": config.classifier,
            "random_state": config.random_state,
            "decision_tree": {
                "criterion": config.decision_tree_criterion,
                "splitter": config.decision_tree_splitter,
                "max_depth": config.decision_tree_max_depth,
                "min_samples_split": config.decision_tree_min_samples_split,
                "min_samples_leaf": config.decision_tree_min_samples_leaf,
                "max_features": config.decision_tree_max_features,
                "class_weight": config.decision_tree_class_weight,
            },
            "random_forest": {
                "n_estimators": config.n_estimators or 100,
                "criterion": config.random_forest_criterion,
                "max_depth": config.random_forest_max_depth,
                "min_samples_split": config.random_forest_min_samples_split,
                "min_samples_leaf": config.random_forest_min_samples_leaf,
                "max_features": config.random_forest_max_features,
                "bootstrap": config.random_forest_bootstrap,
                "class_weight": config.random_forest_class_weight,
                "n_jobs": config.random_forest_n_jobs,
            },
        }
        resolved_config = {
            "pipeline": "tr_tr",
            "protocol": "TR-TR",
            "classifier_requested": getattr(self.arguments, "classifier_requested", None),
            "classifier_canonical": getattr(self.arguments, "classifier_canonical", None),
            "train_split": getattr(bundle.train, "name", None),
            "test_split": getattr(bundle.test, "name", None),
            "train_sampling": config.train_sampling,
            "test_sampling": config.test_sampling,
            "train_samples_effective": int(bundle.train.num_rows) if config.train_sampling == "all" else None,
            "test_samples_effective": int(bundle.test.num_rows) if config.test_sampling == "all" else None,
            "mmap": bool(getattr(self.arguments, "use_mmap", False) or getattr(self.arguments, "mmap_npy", False)),
            "feature_transform": getattr(self.arguments, "feature_transform", None),
            "classifier_transform": getattr(self.arguments, "classifier_transform", None),
            "evaluation_space": getattr(self.arguments, "evaluation_space", None),
            "random_state": config.random_state,
            "model_parameters": model_params,
        }
        logging.info("Resolved main.py TR-TR configuration: %s", json.dumps(_json_ready(resolved_config), sort_keys=True))
        result = run_tr_tr_pipeline(
            bundle,
            config,
            output_base,
            resolved_arguments=vars(self.arguments),
            repo_root=Path.cwd(),
        )
        resolved_config["train_samples_effective"] = result.get("samples", {}).get("train_total_effective")
        resolved_config["test_samples_effective"] = result.get("samples", {}).get("test_total_effective")
        logging.info("Completed main.py TR-TR configuration: %s", json.dumps(_json_ready(resolved_config), sort_keys=True))
        self._dictionary_metrics["TR-TR"] = result
        self._dictionary_metrics["RunConfig"] = resolved_config
        self.save_dictionary_to_json(self.get_evaluation_results_path() + "/Results.json")
        return result

    @import_metrics
    @import_classifiers
    @StratifiedData
    def run_experiments(self):
        """
        Runs the experiment across multiple folds, using the stratified data splits.
        For each fold, the method trains a model, evaluates it on both synthetic and real data,
        and logs the results. The method also updates evaluation results and saves them in a JSON file.

        The method applies decorators for importing metrics, classifiers, and stratified data splits,
        and ensures that each experiment is logged and processed properly.

        Logs the progress and completion time for each fold and the total experiment runtime.

        This method involves the following steps:
            1. Stratified data splitting for training and evaluation.
            2. Model training and prediction for each fold.
            3. Evaluation using synthetic and real data.
            4. Saving the results to a JSON file.

        Args:
            :None
        """

        if getattr(self.arguments, "pipeline_effective", None) == "tr_tr":
            return self._run_explicit_tr_tr_pipeline()

        logging.info("Starting experiment runs across %d folds.", len(self.list_folds))

        # Start time for the entire experiment
        total_start_time = time.time()

        try:

            # Iterate over each fold in the stratified list of folds
            for fold, dictionary_data in enumerate(self.list_folds):

                # Start time for the current fold
                fold_start_time = time.time()
                logging.info("")

                # Log the fold number
                logging.info("Running experiment for fold %d.", fold + 1)

                # Log the size and shape of the training data (features and labels)
                logging.info("Fold %d training data shape: X=%s, Y=%s", fold + 1,
                              dictionary_data['x_training_real'].shape, 
                              dictionary_data['y_training_real'].shape)

                # Update the fold number in the class instance
                self.fold_number = fold
                logging.debug("\t\tFold number updated to %d in the class.", self.fold_number)

                # Get the path to monitor the experiment's progress
                monitor_path = self.get_monitor_path()
                # number_samples_per_class = self.arguments.number_samples_per_class
                # print("number_samples_per_class", number_samples_per_class)
                
            
                self._model_input_adapter = self._build_model_input_adapter()
                if dictionary_data.get("training_split_name") == "test":
                    raise ValueError("LeakageGuard: generator training must not use the TEST split.")
                self._model_input_adapter.fit_generator(dictionary_data['x_training_real'])
                control_train_source, control_test_source = self._control_source_splits(dictionary_data)
                x_training_for_generator = self._model_input_adapter.transform_generator_input(
                    dictionary_data['x_training_real'],
                    split_name="train",
                )
                if (
                        getattr(self.arguments, "split_mode", "cross_validation") == "provided"
                        and dictionary_data.get("evaluation_split_name") == "test"
                ):
                    x_evaluation_for_generator = x_training_for_generator
                    y_evaluation_for_generation = dictionary_data['y_training_real']
                else:
                    x_evaluation_for_generator = self._model_input_adapter.transform_generator_input(
                        dictionary_data['x_evaluation_real'],
                        split_name=dictionary_data.get("evaluation_split_name", "evaluation"),
                    )
                    y_evaluation_for_generation = dictionary_data['y_evaluation_real']
                self._current_real_source_metadata = ScaleGuard.describe(
                    dictionary_data['x_evaluation_real'],
                    data_space="source",
                    transform_id=None,
                    transform_history=[],
                )
                self._current_evaluation_source_x = dictionary_data['x_evaluation_real']
                selected_protocols_for_run = _protocol_selection_for_arguments(self.arguments)
                baseline_protocol_only = selected_protocols_for_run == ["TR_TR"]

                if baseline_protocol_only:
                    logging.info("Skipping generator training/generation because evaluation_protocol=tr_tr.")
                    evaluation_synthetic = {}
                    self._record_generation_strategy_metadata(
                        fold + 1,
                        {
                            "status": "not_run",
                            "reason": "TR_TR baseline uses only real TRAIN and real TEST.",
                            "units": [],
                        },
                    )
                elif self._uses_partitioned_generation():
                    logging.info(
                        "Skipping global generator training because generation_strategy=%s trains sub-generators.",
                        self.arguments.generation_strategy,
                    )
                    self._record_generation_strategy_metadata(
                        fold + 1,
                        {
                            "generation_strategy": self.arguments.generation_strategy,
                            "classes_per_group": int(getattr(self.arguments, "classes_per_group", 10)),
                            "status": "pending_partitioned_generation",
                        },
                    )
                elif self._uses_synthetic_control():
                    logging.info(
                        "Skipping generator training because synthetic_control=%s.",
                        self.arguments.synthetic_control,
                    )
                    self._record_generation_strategy_metadata(
                        fold + 1,
                        {
                            "status": "synthetic_control",
                            "synthetic_control": self.arguments.synthetic_control,
                            "model_type": self.arguments.model_type,
                            "units": [],
                        },
                    )
                else:
                    # Create the model and make predictions using the training data
                    with self.resource_timer("training"):
                        self.train_model(x_training_for_generator,
                                         dictionary_data['y_training_real'],
                                         monitor_path, fold)
                
                if not baseline_protocol_only:
                    self.monitoring_start_generating()

                    with self.resource_timer("generation"):
                        evaluation_synthetic = self.synthesize_data(
                                                      x_evaluation_for_generator,
                                                      y_evaluation_for_generation,
                                                      x_training_for_generator,
                                                      dictionary_data['y_training_real'],
                                                      monitor_path,
                                                      fold,
                                                      real_train_source_data=control_train_source,
                                                      real_test_source_data=control_test_source,
                                                      )

                    self.monitoring_stop_generating(fold)
                logging.info("\t\tModel creation and prediction completed for fold %d.", fold + 1)

                # Log the start of the evaluation process
                logging.info("")
                logging.info("")
                logging.info(" starting evaluation for fold %d.", fold + 1)

                # Perform the evaluations using synthetic and real data
                with self.resource_timer("evaluation"):
                    if dictionary_data.get('evaluation_not_applicable'):
                        reason = (
                            "split_mode=provided has no valid/test evaluation split; "
                            "TR-TS, TS-TR and distance evaluations are not_applicable."
                        )
                        logging.warning(reason)
                        self.mark_fold_not_applicable(self.fold_number + 1, reason)
                    else:
                        if not baseline_protocol_only:
                            self._run_synthetic_quality_audit(dictionary_data, evaluation_synthetic)
                            run_synthetic_evaluation_modes(self, dictionary_data, evaluation_synthetic)
                        else:
                            run_synthetic_evaluation_modes(self, dictionary_data, evaluation_synthetic)
                        assert_real_resample_control_metrics(
                            self._dictionary_metrics,
                            number_classes=self._get_configured_number_classes(dictionary_data["y_training_real"]),
                            synthetic_control=getattr(self.arguments, "synthetic_control", "none"),
                            fold=self.fold_number + 1,
                            protocol_plan=resolve_evaluation_protocol_plan(self.arguments),
                        )
                        assert_label_permutation_control_metrics(
                            self._dictionary_metrics,
                            number_classes=self._get_configured_number_classes(dictionary_data["y_training_real"]),
                            synthetic_control=getattr(self.arguments, "synthetic_control", "none"),
                            fold=self.fold_number + 1,
                            protocol_plan=resolve_evaluation_protocol_plan(self.arguments),
                        )
                        if getattr(self.arguments, "run_tr_tr", False):
                            dataset_bundle = dictionary_data.get("dataset_bundle") or getattr(self, "_dataset_bundle", None)
                            if (
                                dataset_bundle is not None
                                and getattr(self.arguments, "split_mode", "cross_validation") == "provided"
                                and getattr(dataset_bundle, "test", None) is not None
                            ):
                                self.evaluation_TR_TR(
                                    real_train_data=dataset_bundle.train,
                                    real_test_data=dataset_bundle.test,
                                )
                            else:
                                self.evaluation_TR_TR(dictionary_data)
                
                # self.calculate_sdv_metrics(dictionary_data, fold)

                # End of fold, log the time taken for the current fold
                fold_end_time = time.time()
                logging.info("Fold %d experiment completed in %.2f seconds.", fold + 1, fold_end_time - fold_start_time)
                logging.info("------\n\n")
                if fold + 1 < int(getattr(self.arguments, "number_k_folds", 1)):
                    self.save_dictionary_to_json(self.get_evaluation_results_path()+"/Results.json")
                # sys.exit(0)

            # Update and log the mean and standard deviation of the evaluation results
            self.update_mean_std_fold()
            self._write_experiment_protocol_artifacts()
            self.save_dictionary_to_json(self.get_evaluation_results_path()+"/Results.json")
            total_end_time = time.time()  # Separate folds for clarity in logs

            # Save the evaluation results to a JSON file
            logging.info("All experiments completed in %.2f seconds.", total_end_time - total_start_time)

        except Exception as e:
            logging.error("An error occurred during experiment execution: %s", str(e))
            raise

    def _prepare_labels_for_conditional_generation(self, labels):
        labels = numpy.ravel(labels)
        if getattr(self, '_labels_are_discrete', True):
            return labels.astype(int)

        logging.warning("Continuous labels detected; conditional generators will use a single pseudo-class.")
        return numpy.zeros(labels.shape[0], dtype=int)

    def _get_configured_number_classes(self, labels):
        """Return the total label-domain width used by conditional one-hot inputs."""
        candidates = []

        for metadata in (
                getattr(self, '_number_samples_per_class', None),
                getattr(self.arguments, 'number_samples_per_class', None)):
            if isinstance(metadata, dict) and metadata.get('number_classes'):
                candidates.append(int(metadata['number_classes']))

        model_class_arguments = (
            'autoencoder_number_classes',
            'variational_autoencoder_number_classes',
            'wasserstein_number_classes',
            'wasserstein_gp_number_classes',
            'quantized_vae_number_classes',
        )
        for argument_name in model_class_arguments:
            value = getattr(self.arguments, argument_name, None)
            if value:
                candidates.append(int(value))

        if labels.size:
            candidates.append(int(numpy.max(labels)) + 1)
            candidates.append(int(numpy.unique(labels).shape[0]))

        return max(candidates) if candidates else 1

    def _build_generation_metadata(self, y_real_samples):
        labels = self._prepare_labels_for_conditional_generation(y_real_samples)
        number_classes = self._get_configured_number_classes(labels)
        self._validate_effective_label_domain(labels, number_classes, context="generation metadata labels")
        sample_plan = build_sample_plan_from_args(
            self.arguments,
            labels,
            number_classes=number_classes,
            data_type=getattr(self, '_data_type', getattr(self.arguments, 'data_type', 'binary')),
        )
        self._sample_plan = sample_plan
        generation_metadata = sample_plan_to_legacy_metadata(
            sample_plan,
            data_type=getattr(self, '_data_type', getattr(self.arguments, 'data_type', 'binary')),
        )
        generation_metadata["generation_batch_size"] = int(getattr(self.arguments, "generation_batch_size", 8192))
        self._validate_synthetic_generation_plan(generation_metadata)
        self._validate_generation_metadata_classes(generation_metadata, labels)

        if len(generation_metadata["classes"]) != number_classes:
            logging.info(
                "\t\tGeneration fold contains %d/%d configured classes; preserving total class domain.",
                len(generation_metadata["classes"]), number_classes)

        return generation_metadata

    @staticmethod
    def _validate_effective_label_domain(labels, number_classes, context):
        labels = numpy.ravel(numpy.asarray(labels)).astype(int)
        if labels.size == 0:
            raise ValueError(f"{context}: labels are empty.")
        if int(labels.min()) != 0:
            raise ValueError(f"{context}: labels.min() must be 0. Got {int(labels.min())}.")
        if int(labels.max()) != int(number_classes) - 1:
            raise ValueError(
                f"{context}: labels.max() must be K-1={int(number_classes) - 1}. Got {int(labels.max())}."
            )

    @staticmethod
    def _validate_generation_metadata_classes(generation_metadata, labels):
        number_classes = int(generation_metadata["number_classes"])
        effective_labels = set(numpy.ravel(numpy.asarray(labels)).astype(int).tolist())
        planned_classes = {int(class_id) for class_id in generation_metadata["classes"].keys()}
        outside_domain = sorted(class_id for class_id in planned_classes if class_id < 0 or class_id >= number_classes)
        if outside_domain:
            raise ValueError(
                f"Sample plan contains classes outside [0, {number_classes - 1}]: {outside_domain}."
            )
        outside_effective = sorted(planned_classes - effective_labels)
        if outside_effective:
            raise ValueError(
                f"Sample plan contains classes absent from the effective training labels: {outside_effective}."
            )

    def _validate_synthetic_generation_plan(self, generation_metadata):
        train_quota = getattr(self.arguments, "synthetic_train_samples_per_class", None)
        test_quota = getattr(self.arguments, "synthetic_test_samples_per_class", None)
        if train_quota is None and test_quota is None:
            return

        required_per_class = int(train_quota or 0) + int(test_quota or 0)
        if required_per_class <= 0:
            return

        for class_id in sorted(int(class_id) for class_id in generation_metadata["classes"]):
            planned = int(generation_metadata["classes"].get(int(class_id), 0))
            if planned < required_per_class:
                raise ValueError(
                    "InsufficientSyntheticGenerationPlan: "
                    f"class={class_id} required={required_per_class} planned={planned}"
                )

    def _aligned_sanity_real_dataset(self, real_x, real_y, split_name):
        real_x, real_y = validate_xy_alignment(
            real_x,
            real_y,
            "sanity check source",
            split=split_name,
            fold=self.fold_number + 1,
        )
        return AlignedDataset(
            X=real_x,
            y=real_y,
            split_name=split_name,
            fold_id=self.fold_number + 1,
            source_indices=numpy.arange(real_y.shape[0], dtype=numpy.int64),
            data_space="source",
        )

    def _get_label_mapping_for_audit(self):
        label_mapping = getattr(self, '_label_mapping', None)
        if label_mapping:
            return label_mapping

        dataset_bundle = getattr(self, '_dataset_bundle', None)
        metadata = getattr(dataset_bundle, 'metadata', None)
        if isinstance(metadata, dict):
            return metadata.get("label_mapping_original_to_zero_based")

        return None

    def _uses_partitioned_generation(self):
        return (
            getattr(self.arguments, "execution_mode", "normal") == "batches"
            and getattr(self.arguments, "generation_strategy", "single_conditional")
            in {"per_class", "grouped_classes"}
        )

    def _uses_synthetic_control(self):
        return getattr(self.arguments, "synthetic_control", "none") != "none"

    def _control_source_splits(self, dictionary_data):
        dataset_bundle = dictionary_data.get("dataset_bundle") or getattr(self, "_dataset_bundle", None)
        if dataset_bundle is not None and getattr(self.arguments, "split_mode", "cross_validation") == "provided":
            if getattr(dataset_bundle, "test", None) is None:
                raise SyntheticControlSplitMismatchError(
                    "synthetic_control with split_mode=provided requires DatasetBundle.test; "
                    "the valid split is not allowed as a control source."
                )
            return dataset_bundle.train, dataset_bundle.test

        train_x, train_y = validate_xy_alignment(
            dictionary_data["x_training_real"],
            dictionary_data["y_training_real"],
            "synthetic control legacy train source",
            split="train",
            fold=self.fold_number + 1,
            source_indices=dictionary_data.get("training_source_indices"),
        )
        test_x, test_y = validate_xy_alignment(
            dictionary_data["x_evaluation_real"],
            dictionary_data["y_evaluation_real"],
            "synthetic control legacy test source",
            split="test",
            fold=self.fold_number + 1,
            source_indices=dictionary_data.get("evaluation_source_indices"),
        )
        return (
            SplitData(X=train_x, y=train_y, name="train"),
            SplitData(X=test_x, y=test_y, name="test"),
        )

    def _schema_metadata_for_synthetic_manifest(self):
        schema = getattr(getattr(self, "_dataset_bundle", None), "schema", None)
        if schema is None:
            return {
                "feature_names": [f"f{index}" for index in range(self.get_number_columns())],
                "feature_dtype": None,
                "schema_hash": None,
            }
        return {
            "feature_names": list(schema.feature_names),
            "feature_dtype": schema.feature_dtype,
            "schema_hash": getattr(schema, "schema_hash", None),
        }

    def _synthetic_manifest_audit_metadata(self, number_samples_per_class):
        return {
            "requested_classes": sorted(int(class_id) for class_id in number_samples_per_class["classes"].keys()),
            "generation_plan": {
                "classes": {
                    str(class_id): int(count)
                    for class_id, count in number_samples_per_class["classes"].items()
                },
                "number_classes": int(number_samples_per_class["number_classes"]),
                "sample_plan": number_samples_per_class.get("sample_plan"),
                "generation_batch_size": int(number_samples_per_class.get("generation_batch_size", 0)),
            },
            "label_mapping": self._get_label_mapping_for_audit(),
            "batch_sizes": {
                "data_loader_batch_size": getattr(self.arguments, "batch_size", None),
                "vae_training_batch_size": getattr(self.arguments, "variational_autoencoder_batch_size", None),
                "gan_training_batch_size": getattr(self.arguments, "adversarial_batch_size", None),
                "generation_batch_size": getattr(self.arguments, "generation_batch_size", None),
                "evaluation_batch_size": getattr(self.arguments, "eval_batch_size", None),
            },
            "training": getattr(self.arguments, "_training_history", None),
            "code_version": self._code_version(),
        }

    @staticmethod
    def _code_version():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    def _decoder_probe_for_active_generator(self, number_samples_per_class):
        number_classes = int(number_samples_per_class["number_classes"])
        model_type = getattr(self.arguments, "model_type", None)
        model = None
        latent_dim = None
        if model_type == "variational" and getattr(self, "_variational_algorithm", None) is not None:
            model = self._variational_algorithm.get_decoder_trained()
            latent_dim = getattr(self, "_variational_autoencoder_latent_dimension", None)
        elif model_type == "adversarial" and getattr(self, "_adversarial_algorithm", None) is not None:
            model = self._adversarial_algorithm._generator
            latent_dim = getattr(self, "_adversarial_latent_dimension", None)
        elif model_type == "autoencoder" and getattr(self, "_autoencoder_algorithm", None) is not None:
            model = self._autoencoder_algorithm.decoder
            latent_dim = getattr(self, "_autoencoder_latent_dimension", None)

        if model is None or latent_dim is None:
            return None, None

        def probe(z_values, labels):
            one_hot = to_one_hot_batch(labels, number_classes, dtype=numpy.float32)
            return model.predict([numpy.asarray(z_values, dtype=numpy.float32), one_hot], verbose=0)

        return probe, int(latent_dim)

    def _run_synthetic_quality_audit(self, dictionary_data, evaluation_synthetic):
        if getattr(self.arguments, "execution_mode", "normal") != "batches":
            return
        dataset_bundle = dictionary_data.get("dataset_bundle") or getattr(self, "_dataset_bundle", None)
        if dataset_bundle is None or getattr(dataset_bundle, "test", None) is None:
            return
        synthetic_for_audit = evaluation_synthetic
        if getattr(self.arguments, "evaluation_mode", "both") == "tr_ts_tr":
            synthetic_for_audit = getattr(evaluation_synthetic, "train_reader", evaluation_synthetic)
        generation_plan = getattr(self, "_number_samples_per_class", None)
        if not isinstance(generation_plan, dict):
            generation_plan = {
                "classes": {},
                "number_classes": getattr(dataset_bundle.schema, "num_classes", self._get_configured_number_classes(dataset_bundle.train.y)),
            }
        decoder_probe, probe_latent_dim = self._decoder_probe_for_active_generator(generation_plan)
        try:
            audit_path, _ = run_synthetic_quality_audit(
                dataset_bundle.train.X,
                dataset_bundle.train.y,
                dataset_bundle.test.X,
                dataset_bundle.test.y,
                synthetic_for_audit,
                number_classes=getattr(dataset_bundle.schema, "num_classes", self._get_configured_number_classes(dataset_bundle.train.y)),
                expected_num_features=self.get_number_columns(),
                schema=dataset_bundle.schema,
                arguments=self.arguments,
                experiment_directory=self.current_subdir,
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                fail_on_collapse=getattr(self.arguments, "synthetic_control", "none") == "none",
                decoder_probe=decoder_probe,
                latent_dim=probe_latent_dim or 8,
            )
            logging.info("Synthetic quality root-cause audit completed: %s", audit_path)
        except Exception:
            logging.exception("Synthetic quality root-cause audit failed.")
            raise

    def _build_model_input_adapter(self):
        source_profile = getattr(self.arguments, "source_profile", "legacy_csv")
        policy = FeatureTransformPolicy.for_profile(
            source_profile,
            feature_transform=getattr(self.arguments, "feature_transform", None),
            generator_transform=getattr(self.arguments, "generator_transform", None),
            classifier_transform=getattr(self.arguments, "classifier_transform", None),
            evaluation_space=getattr(self.arguments, "evaluation_space", None),
            allow_refit=getattr(self.arguments, "allow_scaler_refit", False) or source_profile == "legacy_csv",
            allow_double_transform=getattr(self.arguments, "allow_double_transform", False) or source_profile == "legacy_csv",
            inverse_transform_synthetic=getattr(self.arguments, "inverse_transform_synthetic", False),
        )
        return ModelInputAdapter(policy)

    def _guard_current_evaluation_space(self, dictionary_data, synthetic_data):
        if hasattr(synthetic_data, "iter_batches"):
            return
        if not synthetic_data:
            return
        synthetic_values = numpy.vstack([
            numpy.asarray(values, dtype=numpy.float32)
            for values in synthetic_data.values()
            if len(values)
        ])
        if synthetic_values.size == 0:
            return
        synthetic_metadata = self._current_synthetic_metadata or {
            "data_space": "source",
            "transform_id": None,
            "transform_history": [],
        }
        ScaleGuard.validate_evaluation_space_contract(
            self._current_real_source_metadata,
            getattr(self.arguments, "evaluation_space", "source"),
            context="real evaluation input",
        )
        ScaleGuard.validate_evaluation_space_contract(
            synthetic_metadata,
            getattr(self.arguments, "evaluation_space", "source"),
            context="synthetic evaluation input",
        )
        ScaleGuard.validate_before_evaluation(
            dictionary_data['x_evaluation_real'],
            synthetic_values,
            self._current_real_source_metadata,
            synthetic_metadata,
            context="TR-TS/TS-TR",
        )

    def _record_generation_strategy_metadata(self, fold, metadata):
        fold_key = f"{fold}-Fold"
        block = self._dictionary_metrics.setdefault("GenerationStrategy", {})
        block[fold_key] = {
            "generation_strategy": getattr(self.arguments, "generation_strategy", "single_conditional"),
            "classes_per_group": int(getattr(self.arguments, "classes_per_group", 10)),
            **metadata,
        }

    def _raise_generation_strategy_not_supported(self, fold, reason):
        self._record_generation_strategy_metadata(
            fold + 1,
            {
                "status": "not_supported",
                "model_type": self.arguments.model_type,
                "reason": reason,
            },
        )
        try:
            self.save_dictionary_to_json(self.get_evaluation_results_path() + "/Results.json")
        except Exception as error:
            logging.warning("Could not save not_supported generation metadata before raising: %s", error)
        raise NotImplementedError(reason)

    def _partition_generation_units(self, class_counts):
        strategy = getattr(self.arguments, "generation_strategy", "single_conditional")
        if strategy == "per_class":
            return partition_generation_classes(class_counts, "per_class", int(getattr(self.arguments, "classes_per_group", 10)))
        return partition_generation_classes(class_counts, "grouped_classes", int(getattr(self.arguments, "classes_per_group", 10)))

    @staticmethod
    def _subset_by_classes(x_values, y_values, classes):
        x_values, y_values = validate_xy_alignment(
            x_values,
            y_values,
            "subset by classes input",
        )
        labels = numpy.ravel(numpy.asarray(y_values)).astype(int)
        mask = numpy.isin(labels, numpy.asarray(classes, dtype=int))
        indices = numpy.flatnonzero(mask)
        subset_x = numpy.asarray(x_values[indices], dtype=numpy.float32)
        subset_y = labels[indices]
        validate_xy_alignment(
            subset_x,
            subset_y,
            "subset by classes output",
            source_indices=indices,
        )
        return subset_x, subset_y

    def _release_current_generator(self):
        model_type = self.arguments.model_type
        if model_type == "adversarial":
            self._adversarial_algorithm = None
            self._adversarial_model = None
        elif model_type == "autoencoder":
            self._autoencoder_algorithm = None
            self._autoencoder_model = None
        elif model_type == "variational":
            self._variational_algorithm = None
            self._variational_model = None
        elif model_type == "wasserstein":
            self._wasserstein_algorithm = None
            self._wasserstein_model = None
        elif model_type == "wasserstein_gp":
            self._wasserstein_gp_algorithm = None
            self._wasserstein_gp_model = None
        elif model_type == "denoising_diffusion":
            self._denoising_diffusion_algorithm = None
            self._denoising_diffusion_model = None
        elif model_type == "quantized":
            self._quantized_vae_algorithm = None
            self._quantized_vae_model = None

        tensorflow.keras.backend.clear_session()
        gc.collect()

    @import_models
    def train_model(self, x_real_samples, y_real_samples, monitor_path, k_fold):
        """
        This method is responsible for creating a model based on the specified model type, training it on real data,
        generating synthetic data using the trained model, and optionally saving the model and generated data.

        It supports various model types including adversarial, autoencoder, variational, WassersteinGP, diffusion,
        and copy-paste algorithms. After training and generating data, it logs the completion of each step and saves the models
        and data if specified in the arguments.

        Args:
            x_real_samples (array): The real input samples (features) for training the model.
            y_real_samples (array): The real target labels corresponding to the input samples.
            monitor_path (str): Path to monitor the training process, such as for storing logs or checkpoints.
            k_fold (int): The fold number in a cross-validation setup, used to save models and data for each fold.

        Raises:
            Exception: If an error occurs during model creation, training, or data generation, an exception is raised.
        """

        logging.info("Starting model creation and prediction process.")
        x_real_samples, y_real_samples = validate_xy_alignment(
            x_real_samples,
            y_real_samples,
            "train_model input",
            fold=k_fold + 1 if k_fold is not None else None,
        )
        logging.info("Number of real samples: %d", len(x_real_samples)) # Log the number of real samples

        try:
            # Train the model with the provided real samples and labels
            logging.info("Training model with %d samples and model type: %s", len(x_real_samples),
                         self.arguments.model_type)

            self.monitoring_start_training()
            
             
            if self.arguments.model_type in ["copula", "ctgan", "tvae"]:
                self.generator_name = self.arguments.model_type
                logging.info(f"Training SDV's model {self.generator_name} algorithm.")
                
                from Engine.Algorithms.ThirdParty.SDVInterfaceAlgorithm import SDVInterfaceAlgorithm
                
                self._sdv = SDVInterfaceAlgorithm()
                
                self._sdv.training_model( x_real_samples, y_real_samples, 
                                    self._data_original_header,
                                    self.arguments.model_type)
            else:
                y_model_samples = self._prepare_labels_for_conditional_generation(y_real_samples)
                self.training_model(self.arguments,
                                    self.get_number_columns(),
                                    x_real_samples,
                                    y_model_samples,
                                    monitor_path,
                                    k_fold)
            
            self.monitoring_stop_training(k_fold)

            logging.info("Model training completed.")

            if self.arguments.save_models:

                logging.info("Saving trained model.")

                if self.arguments.model_type == 'adversarial':
                    self._adversarial_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == 'autoencoder':
                    self._autoencoder_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "variational":
                    self._variational_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "wasserstein":
                    self._wasserstein_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "wasserstein_gp":
                    self._wasserstein_gp_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "latent_diffusion":
                    self._latent_diffusion_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "denoising_diffusion":
                    self._denoising_diffusion_algorithm.save_model(self.get_models_saved_path(), k_fold)

                elif self.arguments.model_type == "quantized":                 
                    self._quantized_vae_algorithm.save_model(self.get_models_saved_path(), k_fold)

                else:
                    # If an invalid model type is specified, log the error and exit the program
                    logging.error("Error during model selection")
                    exit(-1)
            
        except Exception as e:
            # If any error occurs during model creation, training, or data generation, log the error
            logging.error("Error during model creation or data generation: %s", str(e))

            raise  # Reraise the exception for further handling or termination
            

    def synthesize_data(
            self,
            x_real_samples,
            y_real_samples,
            x_training_real=None,
            y_training_real=None,
            monitor_path=None,
            fold=None,
            real_train_source_data=None,
            real_test_source_data=None):

            # Generate synthetic data based on the specified model type
            # Depending on the selected model, we use the corresponding algorithm for data generation

            #dictionary_data['y_training_real']
            #labels = dictionary_data['y_evaluation_real']
            x_real_samples, y_real_samples = validate_xy_alignment(
                x_real_samples,
                y_real_samples,
                "synthesize_data real/evaluation input",
                split="evaluation",
                fold=(fold + 1) if fold is not None else self.fold_number + 1,
            )
            if x_training_real is not None or y_training_real is not None:
                if x_training_real is None or y_training_real is None:
                    raise ValueError("synthesize_data requires both x_training_real and y_training_real when either is provided.")
                x_training_real, y_training_real = validate_xy_alignment(
                    x_training_real,
                    y_training_real,
                    "synthesize_data training input",
                    split="train",
                    fold=(fold + 1) if fold is not None else self.fold_number + 1,
                )
            plan_labels = y_training_real if y_training_real is not None else y_real_samples
            number_samples_per_class = self._build_generation_metadata(plan_labels)
            logging.info("\t\tnumber_samples_per_class: %s", number_samples_per_class)

            if self._uses_synthetic_control():
                if real_train_source_data is not None and real_test_source_data is not None:
                    train_quota = _argument_value_or_fallback(
                        self.arguments,
                        "synthetic_train_samples_per_class",
                        "train_samples_per_class",
                    )
                    test_quota = _argument_value_or_fallback(
                        self.arguments,
                        "synthetic_test_samples_per_class",
                        "test_samples_per_class",
                    )
                    train_samples_per_class = self._control_samples_per_class(
                        number_samples_per_class,
                        train_quota,
                        "train",
                    )
                    base_random_state = int(getattr(self.arguments, "random_state", 0))
                    train_reader = self._synthesize_control_batches(
                        source_data=real_train_source_data,
                        split_name="train",
                        samples_per_class=train_samples_per_class,
                        random_state=base_random_state,
                    )
                    if test_quota is not None and int(test_quota) <= 0:
                        self.data_generated = train_reader
                    else:
                        test_samples_per_class = self._control_samples_per_class(
                            number_samples_per_class,
                            test_quota,
                            "test",
                        )
                        test_reader = self._synthesize_control_batches(
                            source_data=real_test_source_data,
                            split_name="test",
                            samples_per_class=test_samples_per_class,
                            random_state=base_random_state + 1,
                        )
                        self.data_generated = SyntheticSplitBatchReaders(train_reader, test_reader)
                else:
                    fallback_source = SplitData(X=x_real_samples, y=y_real_samples, name="test")
                    self.data_generated = self._synthesize_control_batches(
                        source_data=fallback_source,
                        split_name="test",
                        samples_per_class=self._control_samples_per_class(
                            number_samples_per_class,
                            None,
                            "test",
                        ),
                        random_state=int(getattr(self.arguments, "random_state", 0)),
                    )
                self._current_synthetic_metadata = {
                    "data_space": "source",
                    "transform_id": None,
                    "transform_history": [],
                }
                return self.data_generated

            if self._uses_partitioned_generation():
                if x_training_real is None or y_training_real is None:
                    raise ValueError("Partitioned generation requires real training X/y.")
                self.data_generated = self._synthesize_data_partitioned(
                    number_samples_per_class,
                    x_training_real,
                    y_training_real,
                    x_real_samples,
                    y_real_samples,
                    monitor_path,
                    self.fold_number if fold is None else fold,
                )
                return self.data_generated

            if (
                getattr(self.arguments, "execution_mode", "normal") == "batches"
                and not getattr(self.arguments, "materialize_synthetic", False)
            ):
                if x_training_real is not None and y_training_real is not None:
                    train_quota = _argument_value_or_fallback(
                        self.arguments,
                        "synthetic_train_samples_per_class",
                        "train_samples_per_class",
                    )
                    test_quota = _argument_value_or_fallback(
                        self.arguments,
                        "synthetic_test_samples_per_class",
                        "test_samples_per_class",
                    )
                    train_plan = self._generation_metadata_for_split(
                        number_samples_per_class,
                        train_quota,
                        "train",
                    )
                    train_reader = self._synthesize_data_incremental(
                        train_plan,
                        x_training_real,
                        y_training_real,
                        split_name="train",
                        seed=42,
                    )
                    if test_quota is not None and int(test_quota) <= 0:
                        self.data_generated = train_reader
                    else:
                        test_plan = self._generation_metadata_for_split(
                            number_samples_per_class,
                            test_quota,
                            "test",
                        )
                        test_reader = self._synthesize_data_incremental(
                            test_plan,
                            x_training_real,
                            y_training_real,
                            split_name="test",
                            seed=43,
                        )
                        self.data_generated = SyntheticSplitBatchReaders(train_reader, test_reader)
                else:
                    self.data_generated = self._synthesize_data_incremental(
                        number_samples_per_class,
                        x_real_samples,
                        y_real_samples,
                    )
                return self.data_generated

            if (
                getattr(self.arguments, "execution_mode", "normal") == "batches"
                and getattr(self.arguments, "materialize_synthetic", False)
            ):
                logging.warning("materialize_synthetic=True in batches mode can use high memory.")

            if self.arguments.model_type == 'adversarial':

                # Using adversarial model to generate synthetic data
                self.generator_name = 'adversarial'
                logging.info("Generating data using Adversarial algorithm.")
                self.data_generated = self._adversarial_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == 'autoencoder':

                # Using autoencoder model to generate synthetic data
                self.generator_name = 'autoencoder'
                logging.info("Generating data using Autoencoder algorithm.")
                self.data_generated = self._autoencoder_algorithm.get_samples(number_samples_per_class)


            elif self.arguments.model_type == "variational":

                # Using variational model to generate synthetic data
                self.generator_name = 'variational'
                logging.info("Generating data using Variational algorithm.")
                self.data_generated = self._variational_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "wasserstein":

                # Using WassersteinGP model to generate synthetic data
                self.generator_name = 'wasserstein'
                logging.info("Generating data using Wasserstein algorithm.")
                self.data_generated = self._wasserstein_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "wasserstein_gp":

                # Using WassersteinGP model to generate synthetic data
                self.generator_name = 'wasserstein_gp'
                logging.info("Generating data using Wasserstein GP algorithm.")
                self.data_generated = self._wasserstein_gp_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "latent_diffusion":

                # Using diffusion model to generate synthetic data
                self.generator_name = 'latent_diffusion'
                logging.info("Generating data using LatentDiffusion algorithm.")
                self.data_generated = self._latent_diffusion_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "denoising_diffusion":

                # Using diffusion model to generate synthetic data
                self.generator_name = 'denoising_diffusion'
                logging.info("Generating data using Denoising Diffusion algorithm.")
                self.data_generated = self._denoising_diffusion_algorithm.get_samples(number_samples_per_class)


            elif self.arguments.model_type == "copy":
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = 'copy'
                logging.info("Generating data using copy & paste algorithm.")
                copy_source_x = x_training_real if x_training_real is not None else x_real_samples
                copy_source_y = y_training_real if y_training_real is not None else y_real_samples
                self.data_generated = self._copy_algorithm.get_samples(number_samples_per_class,
                                                                       copy_source_x, copy_source_y)

            elif self.arguments.model_type == "quantized":
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = 'quantized'
                logging.info("Generating data using Vector Quantized Variational Autoencoder algorithm.")

                self.data_generated = self._quantized_vae_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "random":

                # Using Random Noise Model
                self.generator_name = 'random'
                logging.info("Generating data using Random Noise  algorithm.")

                self.data_generated = self._random_noise_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type == "smote":

                # Using SMOTE model to generate synthetic data
                self.generator_name = 'smote'
                logging.info("Generating data using SMOTE algorithm.")
                self.data_generated = self._smote_algorithm.get_samples(number_samples_per_class)

            elif self.arguments.model_type in ["copula", "ctgan", "tvae"]:
                # Using copy-paste model to generate synthetic data (by copying and pasting from real data)
                self.generator_name = self.arguments.model_type
                logging.info(f"Generating data using SDV's {self.generator_name} algorithm.")
                
                self.data_generated = self._sdv.get_samples(number_samples_per_class)

            else:
                # If an invalid model type is specified, log the error and exit the program
                logging.error("Error during model selection")
                exit(-1)

            # Completion log for data generation process
            logging.info("Data generation completed successfully for model type: %s", self.arguments.model_type)

            if self._model_input_adapter is not None:
                self.data_generated = self._model_input_adapter.transform_synthetic_collection_to_source(
                    self.data_generated
                )
                self._current_synthetic_metadata = {
                    "data_space": self._model_input_adapter.synthetic_space_after_generation(),
                    "transform_id": (
                        self._model_input_adapter.generator_manager.transform_id
                        if self._model_input_adapter.synthetic_space_after_generation() != "source"
                        else None
                    ),
                    "transform_history": self._model_input_adapter.generator_manager.transform_history,
                }

            audit_path, _ = audit_synthetic_label_generation(
                self.data_generated,
                number_samples_per_class,
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                label_mapping=self._get_label_mapping_for_audit(),
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )
            logging.info("Synthetic label generation audit passed: %s", audit_path)

            sanity_path, _ = run_synthetic_sanity_checks(
                self._aligned_sanity_real_dataset(
                    self._current_evaluation_source_x if self._current_evaluation_source_x is not None else x_real_samples,
                    y_real_samples,
                    "evaluation",
                ),
                None,
                self.data_generated,
                number_classes=number_samples_per_class["number_classes"],
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                arguments=self.arguments,
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )
            logging.info("Synthetic sanity checks completed: %s", sanity_path)
            self._record_generation_strategy_metadata(
                self.fold_number + 1,
                {
                    "status": "completed",
                    "model_type": self.arguments.model_type,
                    "total_generated_rows": int(sum(len(samples) for samples in self.data_generated.values())),
                    "units": [],
                },
            )

            # If specified, save the generated synthetic data
            if self.arguments.save_data:
                self.save_data_generated()
            
            return self.data_generated

    def _synthesize_data_partitioned(
            self,
            number_samples_per_class,
            x_training_real,
            y_training_real,
            x_evaluation_real,
            y_evaluation_real,
            monitor_path,
            fold):
            strategy = getattr(self.arguments, "generation_strategy", "single_conditional")
            if self.arguments.model_type not in PARTITIONED_GENERATION_SUPPORTED_MODELS:
                reason = (
                    f"generation_strategy={strategy} is not supported for "
                    f"model_type={self.arguments.model_type}. No synthetic data was generated."
                )
                logging.error(reason)
                self._raise_generation_strategy_not_supported(fold, reason)

            output_format = getattr(self.arguments, "save_synthetic_format", "npy_batches")
            if output_format == "legacy":
                output_format = "npy_batches"

            writer = SyntheticBatchWriter(
                root_dir=self.directory_output_data,
                num_classes=number_samples_per_class["number_classes"],
                num_features=self.get_number_columns(),
                seed=42,
                model_name=self.arguments.model_type,
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                output_format=output_format,
                data_space=(
                    self._model_input_adapter.synthetic_space_after_generation()
                    if self._model_input_adapter is not None else "source"
                ),
                transform_id=(
                    self._model_input_adapter.generator_manager.transform_id
                    if self._model_input_adapter is not None
                    and self._model_input_adapter.synthetic_space_after_generation() != "source"
                    else None
                ),
                transform_history=(
                    self._model_input_adapter.generator_manager.transform_history
                    if self._model_input_adapter is not None else []
                ),
                **self._schema_metadata_for_synthetic_manifest(),
                **self._synthetic_manifest_audit_metadata(number_samples_per_class),
            )
            if output_format == "single_npy":
                writer.initialize_single_npy(sum(number_samples_per_class["classes"].values()))

            audit = SyntheticLabelGenerationAudit(
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                number_classes=number_samples_per_class["number_classes"],
                label_mapping=self._get_label_mapping_for_audit(),
                expected_classes=number_samples_per_class["classes"].keys(),
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )

            original_number_samples_per_class = getattr(self, "_number_samples_per_class", None)
            original_arguments_number_samples_per_class = getattr(self.arguments, "number_samples_per_class", None)
            generation_batch_size = int(getattr(self.arguments, "generation_batch_size", 8192))
            unit_records = []
            total_rows = 0

            try:
                for unit_index, unit in enumerate(self._partition_generation_units(number_samples_per_class["classes"])):
                    unit_start_time = time.perf_counter()
                    memory_before = get_current_memory_mb()
                    unit_classes = [int(class_id) for class_id in unit["classes"]]
                    unit_x, unit_y = self._subset_by_classes(x_training_real, y_training_real, unit_classes)
                    observed_unit_classes = set(numpy.unique(unit_y).astype(int).tolist()) if unit_y.size else set()
                    missing_training_classes = sorted(set(unit_classes) - observed_unit_classes)
                    if missing_training_classes:
                        raise ValueError(
                            f"Cannot train {strategy} generator for classes {unit_classes}; "
                            f"missing real training classes: {missing_training_classes}."
                        )

                    sub_plan = {
                        **number_samples_per_class,
                        "classes": {
                            int(class_id): int(number_samples_per_class["classes"][int(class_id)])
                            for class_id in unit_classes
                        },
                        "generation_strategy": strategy,
                    }
                    self._number_samples_per_class = sub_plan
                    self.arguments.number_samples_per_class = sub_plan

                    logging.info(
                        "Training partitioned generator: strategy=%s unit=%d classes=%s train_rows=%d",
                        strategy,
                        unit_index,
                        unit_classes,
                        int(unit_x.shape[0]),
                    )
                    self.training_model(
                        self.arguments,
                        self.get_number_columns(),
                        unit_x,
                        unit_y,
                        monitor_path,
                        fold,
                    )
                    generator = self._get_active_generator()

                    unit_generated_rows = 0
                    for label_class in unit_classes:
                        number_instances = int(number_samples_per_class["classes"][int(label_class)])
                        batch_index = 0
                        for start in range(0, number_instances, generation_batch_size):
                            batch_count = min(generation_batch_size, number_instances - start)
                            batch_plan = {
                                **sub_plan,
                                "classes": {int(label_class): int(batch_count)},
                                "generation_batch_size": int(generation_batch_size),
                            }
                            generated_batch = generator.get_samples(batch_plan)[int(label_class)]
                            if self._model_input_adapter is not None:
                                generated_batch = self._model_input_adapter.inverse_synthetic_batch(generated_batch)
                            writer.write_batch(int(label_class), batch_index, generated_batch)
                            audit.record(
                                requested_class=int(label_class),
                                saved_label=int(label_class),
                                generated_features=generated_batch,
                                batch_index=batch_index,
                            )
                            self.record_batch_processed(generated_batch.shape[0])
                            unit_generated_rows += int(generated_batch.shape[0])
                            total_rows += int(generated_batch.shape[0])
                            batch_index += 1

                    elapsed_seconds = time.perf_counter() - unit_start_time
                    unit_record = {
                        "unit_index": int(unit_index),
                        "unit_type": unit["unit_type"],
                        "classes": unit_classes,
                        "training_rows": int(unit_x.shape[0]),
                        "generated_rows": int(unit_generated_rows),
                        "elapsed_seconds": float(elapsed_seconds),
                        "memory_mb_before": memory_before,
                        "memory_mb_after": get_current_memory_mb(),
                    }
                    unit_records.append(unit_record)
                    logging.info("Partitioned generation unit completed: %s", unit_record)
                    self._release_current_generator()
            finally:
                self._number_samples_per_class = original_number_samples_per_class
                self.arguments.number_samples_per_class = original_arguments_number_samples_per_class

            reader = writer.close()
            audit_path, _ = audit.finalize()
            logging.info("Synthetic label generation audit passed: %s", audit_path)
            sanity_path, _ = run_synthetic_sanity_checks(
                self._aligned_sanity_real_dataset(
                    self._current_evaluation_source_x if self._current_evaluation_source_x is not None else x_evaluation_real,
                    y_evaluation_real,
                    "evaluation",
                ),
                None,
                reader,
                number_classes=number_samples_per_class["number_classes"],
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                arguments=self.arguments,
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )
            logging.info("Synthetic sanity checks completed: %s", sanity_path)
            self._record_generation_strategy_metadata(
                self.fold_number + 1,
                {
                    "status": "completed",
                    "model_type": self.arguments.model_type,
                    "total_generated_rows": int(total_rows),
                    "units": unit_records,
                },
            )
            return reader

    def _get_active_generator(self):
            if self.arguments.model_type == 'adversarial':
                self.generator_name = 'adversarial'
                return self._adversarial_algorithm
            if self.arguments.model_type == 'autoencoder':
                self.generator_name = 'autoencoder'
                return self._autoencoder_algorithm
            if self.arguments.model_type == "variational":
                self.generator_name = 'variational'
                return self._variational_algorithm
            if self.arguments.model_type == "wasserstein":
                self.generator_name = 'wasserstein'
                return self._wasserstein_algorithm
            if self.arguments.model_type == "wasserstein_gp":
                self.generator_name = 'wasserstein_gp'
                return self._wasserstein_gp_algorithm
            if self.arguments.model_type == "latent_diffusion":
                self.generator_name = 'latent_diffusion'
                return self._latent_diffusion_algorithm
            if self.arguments.model_type == "denoising_diffusion":
                self.generator_name = 'denoising_diffusion'
                return self._denoising_diffusion_algorithm
            if self.arguments.model_type == "quantized":
                self.generator_name = 'quantized'
                return self._quantized_vae_algorithm
            raise NotImplementedError(
                f"Incremental synthetic generation is not implemented for model_type={self.arguments.model_type!r}."
            )

    def _generation_metadata_for_split(self, number_samples_per_class, samples_per_class, split_name):
            split_metadata = {
                key: value
                for key, value in number_samples_per_class.items()
                if key != "classes"
            }
            if samples_per_class is None:
                split_metadata["classes"] = dict(number_samples_per_class["classes"])
            else:
                split_metadata["classes"] = {
                    int(class_label): int(samples_per_class)
                    for class_label in sorted(int(class_id) for class_id in number_samples_per_class["classes"].keys())
                }
                split_metadata["samples_per_class"] = int(samples_per_class)
            split_metadata["split"] = split_name
            return split_metadata

    def _control_samples_per_class(self, number_samples_per_class, explicit_samples_per_class, split_name):
            if explicit_samples_per_class is not None:
                samples_per_class = int(explicit_samples_per_class)
                if samples_per_class <= 0:
                    raise ValueError(f"synthetic_control split={split_name} samples_per_class must be positive.")
                return samples_per_class

            class_counts = {
                int(class_id): int(count)
                for class_id, count in number_samples_per_class["classes"].items()
            }
            unique_counts = set(class_counts.values())
            if len(unique_counts) != 1:
                raise ValueError(
                    "synthetic_control requires an explicit uniform samples_per_class when the generation "
                    f"plan is not uniform. split={split_name} class_counts={class_counts}"
                )
            samples_per_class = unique_counts.pop()
            if samples_per_class <= 0:
                raise ValueError(f"synthetic_control split={split_name} samples_per_class must be positive.")
            return int(samples_per_class)

    def _synthesize_data_incremental(
            self,
            number_samples_per_class,
            x_real_samples,
            y_real_samples,
            split_name=None,
            seed=42):
            x_real_samples, y_real_samples = validate_xy_alignment(
                x_real_samples,
                y_real_samples,
                "_synthesize_data_incremental input",
                split=split_name or "evaluation",
                fold=self.fold_number + 1,
            )
            generator = self._get_active_generator()
            output_format = getattr(self.arguments, "save_synthetic_format", "npy_batches")
            if output_format == "legacy":
                output_format = "npy_batches"
            if output_format == "single_npy":
                logging.warning("save_synthetic_format=single_npy writes one large on-disk array; avoid on low disk/RAM systems.")

            writer = SyntheticBatchWriter(
                root_dir=self.directory_output_data,
                num_classes=number_samples_per_class["number_classes"],
                num_features=self.get_number_columns(),
                seed=seed,
                model_name=self.arguments.model_type,
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                output_format=output_format,
                data_space=(
                    self._model_input_adapter.synthetic_space_after_generation()
                    if self._model_input_adapter is not None else "source"
                ),
                transform_id=(
                    self._model_input_adapter.generator_manager.transform_id
                    if self._model_input_adapter is not None
                    and self._model_input_adapter.synthetic_space_after_generation() != "source"
                    else None
                ),
                transform_history=(
                    self._model_input_adapter.generator_manager.transform_history
                    if self._model_input_adapter is not None else []
                ),
                **self._schema_metadata_for_synthetic_manifest(),
                **self._synthetic_manifest_audit_metadata(number_samples_per_class),
                split_name=split_name,
                fold_number=self.fold_number + 1,
            )
            if output_format == "single_npy":
                writer.initialize_single_npy(sum(number_samples_per_class["classes"].values()))

            audit = SyntheticLabelGenerationAudit(
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                number_classes=number_samples_per_class["number_classes"],
                label_mapping=self._get_label_mapping_for_audit(),
                expected_classes=number_samples_per_class["classes"].keys(),
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )
            generation_batch_size = int(getattr(self.arguments, "generation_batch_size", 8192))
            total_rows = 0
            for label_class, number_instances in number_samples_per_class["classes"].items():
                batch_index = 0
                for start in range(0, int(number_instances), generation_batch_size):
                    batch_count = min(generation_batch_size, int(number_instances) - start)
                    batch_plan = dict(number_samples_per_class)
                    batch_plan["classes"] = {int(label_class): int(batch_count)}
                    batch_plan["generation_batch_size"] = int(generation_batch_size)
                    generated_batch = generator.get_samples(batch_plan)[int(label_class)]
                    if self._model_input_adapter is not None:
                        generated_batch = self._model_input_adapter.inverse_synthetic_batch(generated_batch)
                    writer.write_batch(int(label_class), batch_index, generated_batch)
                    audit.record(
                        requested_class=int(label_class),
                        saved_label=int(label_class),
                        generated_features=generated_batch,
                        batch_index=batch_index,
                    )
                    self.record_batch_processed(generated_batch.shape[0])
                    total_rows += int(generated_batch.shape[0])
                    logging.info(
                        "Saved synthetic batch: class=%s batch=%d shape=%s total_rows=%d",
                        label_class,
                        batch_index,
                        generated_batch.shape,
                        total_rows,
                    )
                    batch_index += 1

            reader = writer.close()
            audit_path, _ = audit.finalize()
            logging.info("Synthetic label generation audit passed: %s", audit_path)
            sanity_path, _ = run_synthetic_sanity_checks(
                self._aligned_sanity_real_dataset(
                    x_real_samples,
                    y_real_samples,
                    split_name or "evaluation",
                ),
                None,
                reader,
                number_classes=number_samples_per_class["number_classes"],
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                arguments=self.arguments,
                fold_number=self.fold_number + 1,
                model_type=self.arguments.model_type,
                experiment_directory=self.current_subdir,
            )
            logging.info("Synthetic sanity checks completed: %s", sanity_path)
            self._record_generation_strategy_metadata(
                self.fold_number + 1,
                {
                    "status": "completed",
                    "model_type": self.arguments.model_type,
                    "total_generated_rows": int(reader.total_rows),
                    "units": [],
                },
            )
            logging.info("Incremental synthetic generation completed: manifest=%s", reader.manifest_path)
            return reader

    def _synthetic_control_number_classes(self):
            schema = getattr(getattr(self, "_dataset_bundle", None), "schema", None)
            schema_classes = getattr(schema, "num_classes", None)
            if schema_classes is not None:
                return int(schema_classes)
            argument_classes = getattr(self.arguments, "num_classes", None)
            if argument_classes is not None:
                return int(argument_classes)
            return 200

    def _validate_synthetic_control_source(self, source_data, split_name):
            source_x = numpy.asarray(source_data.X)
            source_y = numpy.asarray(source_data.y)
            number_classes = self._synthetic_control_number_classes()
            expected_features = int(getattr(self.arguments, "num_features", 20) or 20)

            logging.info(
                "Synthetic control source audit before selection: object_type=%s split_name=%s "
                "X.shape=%s y.shape=%s x_path=%s y_path=%s",
                type(source_data).__name__,
                split_name,
                getattr(source_x, "shape", None),
                getattr(source_y, "shape", None),
                source_data.x_path,
                source_data.y_path,
            )

            if source_x.ndim != 2:
                raise ValueError(f"synthetic_control split={split_name} X must be 2D. Got shape={source_x.shape}.")
            if source_y.ndim != 1:
                raise ValueError(f"synthetic_control split={split_name} y must be 1D. Got shape={source_y.shape}.")
            if source_x.shape[0] != source_y.shape[0]:
                raise ValueError(
                    f"synthetic_control split={split_name} X/y row mismatch: "
                    f"X rows={source_x.shape[0]} y rows={source_y.shape[0]}."
                )
            if source_x.shape[1] != expected_features:
                raise ValueError(
                    f"synthetic_control split={split_name} X must have {expected_features} features. "
                    f"Got {source_x.shape[1]}."
                )
            if source_y.size == 0:
                raise ValueError(f"synthetic_control split={split_name} y is empty.")
            if not numpy.all(numpy.isfinite(source_y)):
                raise ValueError(f"synthetic_control split={split_name} y contains NaN or inf labels.")

            integer_y = source_y.astype(numpy.int64, copy=False)
            if not numpy.array_equal(source_y, integer_y):
                raise ValueError(f"synthetic_control split={split_name} labels must be integer encoded.")
            if int(integer_y.min()) < 0 or int(integer_y.max()) >= number_classes:
                raise ValueError(
                    f"synthetic_control split={split_name} labels must be between 0 and {number_classes - 1}. "
                    f"Observed min={int(integer_y.min())} max={int(integer_y.max())}."
                )

            unique_labels, counts = numpy.unique(integer_y, return_counts=True)
            class_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
            if source_x.shape[0] == number_classes and unique_labels.shape[0] == number_classes and numpy.all(counts == 1):
                raise AggregateDataUsedAsRawSamplesError(
                    f"synthetic_control split={split_name} received {number_classes} rows with exactly one row per "
                    "class. This looks like aggregate centroids/statistics, not raw samples."
                )
            missing_classes = sorted(set(range(number_classes)) - set(class_counts))
            logging.info(
                "Synthetic control source label audit: split_name=%s min_label=%s max_label=%s "
                "num_classes_present=%d class_0_count=%s minimum_class=%s minimum_class_count=%s "
                "class_counts=%s",
                split_name,
                int(integer_y.min()),
                int(integer_y.max()),
                int(unique_labels.shape[0]),
                class_counts.get(0, 0),
                int(unique_labels[numpy.argmin(counts)]) if counts.size else None,
                int(counts.min()) if counts.size else None,
                class_counts,
            )
            if missing_classes:
                raise ValueError(
                    f"synthetic_control split={split_name} requires {number_classes} classes; "
                    f"missing classes={missing_classes}."
                )
            return source_x, integer_y, class_counts

    def _synthesize_control_batches(
            self,
            source_data: SplitData,
            split_name: str,
            samples_per_class: int,
            random_state: int):
            control = getattr(self.arguments, "synthetic_control", "none")
            if control not in {"real_resample", "label_permutation"}:
                raise ValueError(f"Unsupported synthetic_control: {control}")
            if not isinstance(source_data, SplitData):
                raise TypeError(
                    "_synthesize_control_batches requires source_data: SplitData. "
                    f"Got {type(source_data).__name__}."
                )
            if source_data.name != split_name:
                raise SyntheticControlSplitMismatchError(
                    f"synthetic_control={control} requires source_data.name == split_name; "
                    f"source_data.name={source_data.name!r} split_name={split_name!r}."
                )
            if split_name == "valid":
                raise SyntheticControlSplitMismatchError("synthetic_control must not use the valid split.")

            source_x, source_y, class_counts = self._validate_synthetic_control_source(
                source_data,
                split_name,
            )
            number_classes = self._synthetic_control_number_classes()
            samples_per_class = int(samples_per_class)
            if samples_per_class <= 0:
                raise ValueError("samples_per_class must be a positive integer for synthetic_control.")
            output_format = getattr(self.arguments, "save_synthetic_format", "npy_batches")
            if output_format == "legacy":
                output_format = "npy_batches"
            real_count_policy = getattr(self.arguments, "real_class_count_policy", "strict")
            strategy = "up_to_available" if real_count_policy in {"available_cap", "cap_to_available"} else "balanced_per_class"
            sample_plan = build_split_sample_plan(
                source_y,
                samples_per_class,
                number_classes,
                random_state,
                split_name,
                strategy=strategy,
                insufficient_policy=real_count_policy,
                replacement=False,
                require_all_classes=True,
            )
            effective_total_samples = int(sample_plan.total_rows)
            writer = SyntheticBatchWriter(
                root_dir=self.directory_output_data,
                num_classes=number_classes,
                num_features=self.get_number_columns(),
                seed=random_state,
                model_name=f"synthetic_control:{control}",
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                output_format=output_format,
                data_space="source",
                transform_id=None,
                transform_history=[],
                split_name=split_name,
                fold_number=self.fold_number + 1,
                **self._schema_metadata_for_synthetic_manifest(),
                **self._synthetic_manifest_audit_metadata({
                    "classes": sample_plan.class_counts,
                    "number_classes": number_classes,
                    "generation_batch_size": samples_per_class,
                    "sample_plan": "synthetic_control",
                }),
            )
            writer.manifest.update({
                "control_type": control,
                "source_split": split_name,
                "source_x_path": source_data.x_path,
                "source_y_path": source_data.y_path,
                "source_dataset_id": source_data.dataset_id,
                "samples_per_class": samples_per_class,
                "total_samples_requested": int(samples_per_class * number_classes),
                "total_samples": effective_total_samples,
                "num_classes": number_classes,
                "feature_count": int(source_x.shape[1]),
                "data_space": "source",
                "transform_id": None,
                "random_state": int(random_state),
                "source_class_counts": {str(key): int(value) for key, value in class_counts.items()},
                "source_indices": {},
                "sample_plan_strategy": sample_plan.mode,
                "selection_table": sample_plan.selection_table,
                "selected_counts_by_class": sample_plan.metadata["selected_counts_by_class"],
            })
            if output_format == "single_npy":
                writer.initialize_single_npy(effective_total_samples)

            rng = numpy.random.default_rng(random_state)
            selected_x = []
            selected_y = []
            class_indices_by_label = {}
            selected_plan_indices = (
                None
                if sample_plan.selected_indices is None
                else numpy.asarray(sample_plan.selected_indices, dtype=numpy.int64)
            )
            for label_class in range(number_classes):
                if selected_plan_indices is None:
                    chosen = numpy.flatnonzero(source_y == label_class)
                else:
                    chosen = selected_plan_indices[source_y[selected_plan_indices] == label_class]
                class_indices_by_label[label_class] = chosen
                selected_x.append(numpy.asarray(source_x[chosen], dtype=numpy.float32))
                selected_y.append(numpy.full(chosen.shape[0], label_class, dtype=numpy.int64))

            if control == "real_resample":
                for label_class in sorted(class_indices_by_label):
                    selected_indices = class_indices_by_label[label_class]
                    x_class = numpy.asarray(source_x[selected_indices], dtype=numpy.float32)
                    writer.write_batch(label_class, 0, x_class)
                    writer.manifest["source_indices"][str(label_class)] = {
                        "count": int(selected_indices.shape[0]),
                        "min": int(selected_indices.min()) if selected_indices.size else None,
                        "max": int(selected_indices.max()) if selected_indices.size else None,
                    }
                    writer.manifest["batches_by_class"][str(label_class)][-1]["source_indices"] = (
                        writer.manifest["source_indices"][str(label_class)]
                    )
                    self.record_batch_processed(x_class.shape[0])
            else:
                all_x = numpy.vstack(selected_x) if selected_x else numpy.empty((0, self.get_number_columns()), dtype=numpy.float32)
                all_y = numpy.concatenate(selected_y) if selected_y else numpy.asarray([], dtype=numpy.int64)
                permuted_y = rng.permutation(all_y)
                for label_class in range(number_classes):
                    x_class = all_x[permuted_y == label_class]
                    writer.write_batch(label_class, 0, x_class)
                    source_indices = class_indices_by_label.get(label_class, numpy.asarray([], dtype=numpy.int64))
                    writer.manifest["source_indices"][str(label_class)] = {
                        "count": int(source_indices.shape[0]),
                        "min": int(source_indices.min()) if source_indices.size else None,
                        "max": int(source_indices.max()) if source_indices.size else None,
                    }
                    self.record_batch_processed(x_class.shape[0])

            reader = writer.close()
            logging.info(
                "Synthetic control batches written: control=%s split=%s manifest=%s rows=%d",
                control,
                split_name,
                reader.manifest_path,
                reader.total_rows,
            )
            return reader


    @autosave
    def save_data_generated(self):
        """
        Save the generated data to the specified location.
        """
        logging.info("Entered the save_data_generated method.")

        try:
            logging.info("Attempting to save generated data.")
            """
                @autosave
            
            """
            logging.info("Generated data saved successfully.")

        except Exception as e:
            logging.error(f"Error while saving generated data: {str(e)}")
            raise

if __name__ == "__main__":
    dataGeneration = SynDataGen()
    dataGeneration.show_all_settings()
    dataGeneration.run_experiments()
