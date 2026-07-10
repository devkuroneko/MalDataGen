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
    import sys
    import time
    import numpy
    import pandas
    import logging
    import tensorflow

    from sklearn.utils import shuffle

    from Engine.Metrics.Metrics import Metrics

    from Engine.DataIO.CSVLoader import autosave
    from Engine.DataIO.CSVLoader import autoload
    from Engine.DataIO.SamplePlanner import build_sample_plan_from_args
    from Engine.DataIO.SamplePlanner import sample_plan_to_legacy_metadata

    from Engine.Arguments.Arguments import Arguments
    from Engine.Arguments.Arguments import arguments

    from Engine.Metrics.Metrics import import_metrics

    from Engine.Evaluation.Evaluation import Evaluation
    from sklearn.model_selection import StratifiedKFold

    from Engine.DataIO.CSVLoader import CSVDataProcessor
    from Engine.DataIO.SyntheticBatchIO import SyntheticBatchWriter
    from Engine.DataIO.SyntheticLabelAudit import SyntheticLabelGenerationAudit
    from Engine.DataIO.SyntheticLabelAudit import audit_synthetic_label_generation
    from Engine.DataIO.SyntheticSanityChecks import run_synthetic_sanity_checks
    from Engine.Utils.ResourceMonitor import get_current_memory_mb

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
                
            
                if self._uses_partitioned_generation():
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
                else:
                    # Create the model and make predictions using the training data
                    with self.resource_timer("training"):
                        self.train_model(dictionary_data['x_training_real'],
                                         dictionary_data['y_training_real'],
                                         monitor_path, fold)
                
                self.monitoring_start_generating()

                with self.resource_timer("generation"):
                    evaluation_synthetic = self.synthesize_data(
                                                  dictionary_data['x_evaluation_real'],
                                                  dictionary_data['y_evaluation_real'],
                                                  dictionary_data['x_training_real'],
                                                  dictionary_data['y_training_real'],
                                                  monitor_path,
                                                  fold,
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
                        self.evaluation_TR_TS(dictionary_data, evaluation_synthetic)
                        self.evaluation_TS_TR(dictionary_data, evaluation_synthetic)
                
                #self.evaluation_TR_TR(dictionary_data)
                # self.calculate_sdv_metrics(dictionary_data, fold)

                # End of fold, log the time taken for the current fold
                fold_end_time = time.time()
                logging.info("Fold %d experiment completed in %.2f seconds.", fold + 1, fold_end_time - fold_start_time)
                logging.info("------\n\n")
                self.save_dictionary_to_json(self.get_evaluation_results_path()+"/Results.json")
                # sys.exit(0)

            # Update and log the mean and standard deviation of the evaluation results
            self.update_mean_std_fold()
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

        if len(generation_metadata["classes"]) != number_classes:
            logging.info(
                "\t\tGeneration fold contains %d/%d configured classes; preserving total class domain.",
                len(generation_metadata["classes"]), number_classes)

        return generation_metadata

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
        classes = sorted(int(class_id) for class_id, count in class_counts.items() if int(count) > 0)
        strategy = getattr(self.arguments, "generation_strategy", "single_conditional")
        if strategy == "per_class":
            return [{"unit_type": "class", "classes": [class_id]} for class_id in classes]

        classes_per_group = int(getattr(self.arguments, "classes_per_group", 10))
        if classes_per_group <= 0:
            raise ValueError("--classes_per_group must be a positive integer.")
        return [
            {"unit_type": "group", "classes": classes[start:start + classes_per_group]}
            for start in range(0, len(classes), classes_per_group)
        ]

    @staticmethod
    def _subset_by_classes(x_values, y_values, classes):
        labels = numpy.ravel(numpy.asarray(y_values)).astype(int)
        mask = numpy.isin(labels, numpy.asarray(classes, dtype=int))
        return numpy.asarray(x_values[mask], dtype=numpy.float32), labels[mask]

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
                    self._latent_variational_algorithm_diffusion.save_model(self.get_models_saved_path(), k_fold)

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
            fold=None):

            # Generate synthetic data based on the specified model type
            # Depending on the selected model, we use the corresponding algorithm for data generation

            #dictionary_data['y_training_real']
            #labels = dictionary_data['y_evaluation_real']
            number_samples_per_class = self._build_generation_metadata(y_real_samples)
            logging.info("\t\tnumber_samples_per_class: %s", number_samples_per_class)

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
                
                self.data_generated = self._copy_algorithm.get_samples(number_samples_per_class,
                                                                       x_real_samples, y_real_samples)

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
                x_real_samples,
                y_real_samples,
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
            )
            if output_format == "single_npy":
                writer.initialize_single_npy(sum(number_samples_per_class["classes"].values()))

            audit = SyntheticLabelGenerationAudit(
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                number_classes=number_samples_per_class["number_classes"],
                label_mapping=self._get_label_mapping_for_audit(),
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
                x_evaluation_real,
                y_evaluation_real,
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

    def _synthesize_data_incremental(self, number_samples_per_class, x_real_samples, y_real_samples):
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
                seed=42,
                model_name=self.arguments.model_type,
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                output_format=output_format,
            )
            if output_format == "single_npy":
                writer.initialize_single_npy(sum(number_samples_per_class["classes"].values()))

            audit = SyntheticLabelGenerationAudit(
                execution_mode=getattr(self.arguments, "execution_mode", "normal"),
                number_classes=number_samples_per_class["number_classes"],
                label_mapping=self._get_label_mapping_for_audit(),
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
                x_real_samples,
                y_real_samples,
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
