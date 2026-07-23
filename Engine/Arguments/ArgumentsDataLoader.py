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

DEFAULT_DATA_LOAD_LABEL_COLUMN = -1
DEFAULT_DATA_LOAD_MAX_SAMPLES = -1
DEFAULT_DATA_LOAD_MAX_COLUMNS = -1
DEFAULT_DATA_LOAD_START_COLUMN = 0
DEFAULT_DATA_LOAD_END_COLUMN = -1
DEFAULT_DATA_LOAD_PATH_FILE_INPUT = 'Datasets/converted/train_x.csv'
DEFAULT_DATA_LOAD_PATH_FILE_OUTPUT = 'OutputDir'
DEFAULT_DATA_LOAD_EXCLUDE_COLUMNS = -1
DEFAULT_DATA_TYPE = 'binary'
DEFAULT_DATA_FORMAT = 'csv'
DEFAULT_SPLIT_MODE = 'cross_validation'
DEFAULT_TARGET_TYPE = 'auto'
DEFAULT_FEATURE_TYPE = 'auto'
DEFAULT_NUM_CLASSES = None
DEFAULT_EXECUTION_MODE = 'normal'
DEFAULT_BATCH_SIZE = 8192
DEFAULT_EVAL_BATCH_SIZE = 16384
DEFAULT_GENERATION_BATCH_SIZE = 8192
DEFAULT_SAVE_SYNTHETIC_FORMAT = 'legacy'
DEFAULT_SCALER = 'none'
DEFAULT_SOURCE_PROFILE = 'legacy_csv'
DEFAULT_FEATURE_TRANSFORM = 'preserve'
DEFAULT_GENERATOR_TRANSFORM = 'preserve'
DEFAULT_CLASSIFIER_TRANSFORM = 'preserve'
DEFAULT_EVALUATION_SPACE = 'source'
DEFAULT_MIN_SAMPLES_PER_CLASS_REQUIRED = 1
DEFAULT_LEGACY_NUMBER_SAMPLES_PER_CLASS = "1:256,2:256"
DEFAULT_REAL_CLASS_COUNT_POLICY = "strict"
DEFAULT_SAMPLES_PER_CLASS_SCOPE = "split"
DEFAULT_APPCLASSNET_TOP200_ROOT = 'Datasets/raw/AppClassNet/top200'
MODEL_CLASS_COUNT_ARGUMENTS = (
    'autoencoder_number_classes',
    'variational_autoencoder_number_classes',
    'quantized_vae_number_classes',
    'wasserstein_number_classes',
    'wasserstein_gp_number_classes',
)


def validate_data_load_arguments(arguments):
    arguments.use_mmap = bool(getattr(arguments, 'mmap_npy', False))
    if (
        getattr(arguments, 'pipeline_effective', None) == 'tr_tr'
        and getattr(arguments, 'source_profile', None) == 'appclassnet_top200'
    ):
        arguments.data_format = 'npy_xy'
        arguments.split_mode = 'provided'
        if not getattr(arguments, 'train_x_path', None):
            arguments.train_x_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/train_x.npy"
        if not getattr(arguments, 'train_y_path', None):
            arguments.train_y_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/train_y.npy"
        if not getattr(arguments, 'valid_x_path', None):
            arguments.valid_x_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/valid_x.npy"
        if not getattr(arguments, 'valid_y_path', None):
            arguments.valid_y_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/valid_y.npy"
        if not getattr(arguments, 'test_x_path', None):
            arguments.test_x_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/test_x.npy"
        if not getattr(arguments, 'test_y_path', None):
            arguments.test_y_path = f"{DEFAULT_APPCLASSNET_TOP200_ROOT}/test_y.npy"
        if getattr(arguments, 'target_type', 'auto') == 'auto':
            arguments.target_type = 'multiclass'
        if getattr(arguments, 'feature_type', 'auto') == 'auto':
            arguments.feature_type = 'continuous'
    arguments._legacy_number_samples_per_class_explicit = bool(
        getattr(arguments, '_legacy_number_samples_per_class_explicit', False)
    )
    if getattr(arguments, 'min_samples_per_class_required', 1) < 0:
        raise ValueError("--min_samples_per_class_required must be non-negative.")
    if getattr(arguments, 'real_class_count_policy', DEFAULT_REAL_CLASS_COUNT_POLICY) not in {
        'strict',
        'uniform_min',
        'available_cap',
        'cap_to_available',
    }:
        raise ValueError("--real_class_count_policy must be one of: strict, uniform_min, available_cap, cap_to_available.")
    if getattr(arguments, 'samples_per_class_scope', DEFAULT_SAMPLES_PER_CLASS_SCOPE) not in {'split', 'fold'}:
        raise ValueError("--samples_per_class_scope must be one of: split, fold.")

    if arguments.data_format == 'csv':
        return arguments

    missing_paths = []
    if not arguments.train_x_path:
        missing_paths.append('--train_x_path')
    if not arguments.train_y_path:
        missing_paths.append('--train_y_path')

    if missing_paths:
        raise ValueError(
            "data_format=npy_xy requires {}.".format(" and ".join(missing_paths))
        )

    if (arguments.valid_x_path is None) != (arguments.valid_y_path is None):
        raise ValueError("--valid_x_path and --valid_y_path must be provided together.")

    if (arguments.test_x_path is None) != (arguments.test_y_path is None):
        raise ValueError("--test_x_path and --test_y_path must be provided together.")

    if arguments.split_mode == 'provided':
        if getattr(arguments, 'number_k_folds', 1) != 1:
            import logging
            logging.warning(
                "split_mode=provided uses the provided train/valid/test split contract and ignores "
                "cross-validation folds. Setting number_k_folds=1 for this run."
            )
            arguments.number_k_folds = 1

        has_valid_split = arguments.valid_x_path and arguments.valid_y_path
        has_test_split = arguments.test_x_path and arguments.test_y_path
        if not has_valid_split and not has_test_split:
            import logging
            logging.warning(
                "split_mode=provided was selected for data_format=npy_xy, but no valid/test split was "
                "provided. The loader will continue with the train split only."
            )

    if arguments.target_type == 'multiclass' and arguments.num_classes is not None:
        import logging
        for class_count_argument in MODEL_CLASS_COUNT_ARGUMENTS:
            if hasattr(arguments, class_count_argument):
                current_value = getattr(arguments, class_count_argument)
                if current_value != arguments.num_classes:
                    logging.info(
                        "target_type=multiclass uses --num_classes=%s for %s instead of legacy value %s.",
                        arguments.num_classes,
                        class_count_argument,
                        current_value,
                    )
                    setattr(arguments, class_count_argument, arguments.num_classes)

    sample_plan = getattr(arguments, 'sample_plan', 'legacy')
    has_explicit_legacy_counts = getattr(arguments, '_legacy_number_samples_per_class_explicit', False)
    if getattr(arguments, 'pipeline_effective', None) == 'tr_tr':
        return arguments

    if sample_plan == 'legacy' and not has_explicit_legacy_counts:
        raise ValueError(
            "data_format=npy_xy requires an explicit sampling plan. Use --sample_plan balanced_per_class "
            "with --samples_per_class, --sample_plan match_train_distribution or total_rows with "
            "--total_synthetic_rows, or pass --number_samples_per_class explicitly."
        )

    if sample_plan == 'balanced_per_class' and getattr(arguments, 'samples_per_class', None) is None:
        raise ValueError("--sample_plan balanced_per_class requires --samples_per_class.")

    if sample_plan == 'total_rows' and getattr(arguments, 'total_synthetic_rows', None) is None:
        raise ValueError("--sample_plan total_rows requires --total_synthetic_rows.")

    if sample_plan == 'class_counts' and not has_explicit_legacy_counts:
        raise ValueError("--sample_plan class_counts requires --number_samples_per_class.")

    return arguments

def add_argument_data_load(parser):

    parser.add_argument('-i', '--data_load_path_file_input', type=str, default=DEFAULT_DATA_LOAD_PATH_FILE_INPUT,
                       help='Path to the input CSV file.')


    parser.add_argument('--data_load_label_column', type=int, default=DEFAULT_DATA_LOAD_LABEL_COLUMN,
                        help='Index of the column to be used as the label.')

    parser.add_argument('--data_load_max_samples', type=int, default=DEFAULT_DATA_LOAD_MAX_SAMPLES,
                        help='Maximum number of samples to be loaded.')

    parser.add_argument('--data_load_max_columns', type=int, default=DEFAULT_DATA_LOAD_MAX_COLUMNS,
                        help='Maximum number of columns to be considered.')

    parser.add_argument('--data_load_start_column', type=int, default=DEFAULT_DATA_LOAD_START_COLUMN,
                        help='Index of the first column to be loaded.')

    parser.add_argument('--data_load_end_column', type=int, default=DEFAULT_DATA_LOAD_END_COLUMN,
                        help='Index of the last column to be loaded.')

    parser.add_argument('--data_load_path_file_output', type=str, default=DEFAULT_DATA_LOAD_PATH_FILE_OUTPUT,
                        help='Path to the output CSV file.')

    parser.add_argument('--data_load_exclude_columns', type=int, default=DEFAULT_DATA_LOAD_EXCLUDE_COLUMNS,
                        help='Columns to exclude from processing.')

    parser.add_argument('--data_type', type=str, default=DEFAULT_DATA_TYPE,
                        choices=['binary', 'multiclass', 'continuous'],
                        help='Dataset mode used by loading, generation post-processing and metrics.')

    parser.add_argument('--data_format', type=str, default=DEFAULT_DATA_FORMAT,
                        choices=['csv', 'npy_xy'],
                        help=("Input data format. Default 'csv' preserves the legacy single-CSV flow. "
                              "Use 'npy_xy' only for explicit X/y NumPy files, for example: "
                              "--data_format npy_xy --train_x_path train_x.npy --train_y_path train_y.npy."))

    parser.add_argument('--train_x_path', type=str, default=None,
                        help='Path to train_x.npy when --data_format npy_xy is selected.')

    parser.add_argument('--train_y_path', type=str, default=None,
                        help='Path to train_y.npy when --data_format npy_xy is selected.')

    parser.add_argument('--valid_x_path', type=str, default=None,
                        help='Optional path to valid_x.npy for --data_format npy_xy --split_mode provided.')

    parser.add_argument('--valid_y_path', type=str, default=None,
                        help='Optional path to valid_y.npy for --data_format npy_xy --split_mode provided.')

    parser.add_argument('--test_x_path', type=str, default=None,
                        help='Optional path to test_x.npy for --data_format npy_xy --split_mode provided.')

    parser.add_argument('--test_y_path', type=str, default=None,
                        help='Optional path to test_y.npy for --data_format npy_xy --split_mode provided.')

    parser.add_argument('--split_mode', type=str, default=DEFAULT_SPLIT_MODE,
                        choices=['cross_validation', 'provided'],
                        help=("Split strategy. Default 'cross_validation' preserves the current K-fold behavior. "
                              "Use 'provided' with npy_xy to consume train/valid/test files when supplied."))

    parser.add_argument('--target_type', type=str, default=DEFAULT_TARGET_TYPE,
                        choices=['auto', 'binary', 'multiclass', 'regression', 'none'],
                        help=("Target semantics for new loaders. Default 'auto' preserves existing behavior. "
                              "Example for AppClassNet: --target_type multiclass."))

    parser.add_argument('--feature_type', type=str, default=DEFAULT_FEATURE_TYPE,
                        choices=['auto', 'binary', 'continuous', 'mixed'],
                        help=("Feature semantics for new loaders. Default 'auto' preserves existing behavior. "
                              "Example for AppClassNet: --feature_type continuous."))

    parser.add_argument('--scaler', type=str, default=DEFAULT_SCALER,
                        choices=['none', 'minmax', 'standard'],
                        help=("Feature scaler. Default 'none' preserves the legacy CSV behavior; AppClassNet "
                              "runner defaults to preserve/source and only scales when explicitly requested."))

    parser.add_argument('--source_profile', type=str, default=DEFAULT_SOURCE_PROFILE,
                        choices=['legacy_csv', 'appclassnet_top200', 'custom'],
                        help='Dataset preprocessing profile. Default legacy_csv preserves the original CSV flow.')

    parser.add_argument('--feature_transform', type=str, default=DEFAULT_FEATURE_TRANSFORM,
                        choices=['preserve', 'auto', 'minmax', 'standard'],
                        help='Centralized feature transform applied before pipeline ingestion.')

    parser.add_argument('--generator_transform', type=str, default=DEFAULT_GENERATOR_TRANSFORM,
                        choices=['preserve', 'auto', 'minmax', 'standard'],
                        help='Internal generator input transform. Synthetic data can be inverse-transformed after generation.')

    parser.add_argument('--classifier_transform', type=str, default=DEFAULT_CLASSIFIER_TRANSFORM,
                        choices=['preserve', 'auto', 'minmax', 'standard'],
                        help='Classifier input transform. Tree classifiers should normally preserve AppClassNet scale.')

    parser.add_argument('--evaluation_space', type=str, default=DEFAULT_EVALUATION_SPACE,
                        choices=['source', 'transformed'],
                        help='Feature space used by evaluations.')

    parser.add_argument('--allow_double_transform', action='store_true', default=False,
                        help='Allow applying an equivalent feature transform more than once.')

    parser.add_argument('--allow_scaler_refit', action='store_true', default=False,
                        help='Allow refitting a previously fitted preprocessing scaler.')

    parser.add_argument('--inverse_transform_synthetic', action='store_true', default=False,
                        help='Inverse-transform synthetic samples back to source space after generator-space generation.')

    parser.add_argument('--no-inverse_transform_synthetic', action='store_false',
                        dest='inverse_transform_synthetic',
                        help='Diagnostic override: keep synthetic data in generator space after generation.')

    parser.add_argument('--synthetic_control', type=str, default='none',
                        choices=['none', 'real_resample', 'label_permutation'],
                        help='Diagnostic synthetic path control written as npy_batches.')

    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES,
                        help='Optional class-domain size for new loaders, for example --num_classes 200.')

    parser.add_argument('--class_subset', type=str, default=None,
                        help='Comma-separated original class labels to keep and remap inside this experiment.')

    parser.add_argument('--num_classes_subset', type=int, default=None,
                        help='Keep original labels 0..N-1 and remap inside this diagnostic experiment.')

    parser.add_argument('--remap_labels_to_zero_based', action='store_true', default=False,
                        help=("Explicitly remap 1-based labels to zero-based labels in new loaders. "
                              "Disabled by default to avoid silent label changes."))

    parser.add_argument('--mmap_npy', action='store_true', default=False,
                        help=("Use numpy memory mapping for npy_xy files. Default is False at CLI level to keep "
                              "new behavior opt-in; CSV mode ignores this flag."))

    parser.add_argument('--use_mmap', action='store_true', dest='mmap_npy',
                        help='Alias for --mmap_npy.')

    parser.add_argument('--execution_mode', type=str, default=DEFAULT_EXECUTION_MODE,
                        choices=['normal', 'batches'],
                        help=("Execution mode. Default 'normal' preserves the current full-matrix flow. "
                              "'batches' enables the lower-memory npy_xy path used by AppClassNet."))

    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Generic training batch size used by batches mode.')

    parser.add_argument('--eval_batch_size', type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                        help='Generic evaluation batch size used by batches mode.')

    parser.add_argument('--generation_batch_size', type=int, default=DEFAULT_GENERATION_BATCH_SIZE,
                        help='Generic generation batch size used by batches mode.')

    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optional maximum number of training rows used by batches mode.')

    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Optional maximum number of training rows per class used by batches mode.')

    parser.add_argument('--min_samples_per_class_required', type=int,
                        default=DEFAULT_MIN_SAMPLES_PER_CLASS_REQUIRED,
                        help=('Minimum rows required per class after batches stratified selection. '
                              'Default 1 is intended for debug; use 500 or 1000 for AppClassNet experiments.'))

    parser.add_argument('--strict_min_samples_per_class', action='store_true', default=False,
                        help='Raise an error instead of warning when batches stratified selection is below the minimum.')

    parser.add_argument('--real_class_count_policy', type=str,
                        default=DEFAULT_REAL_CLASS_COUNT_POLICY,
                        choices=['strict', 'uniform_min', 'available_cap', 'cap_to_available'],
                        help=('Policy applied after selecting the real split for per-class real evaluation quotas. '
                              'strict requires the requested count; uniform_min uses a common capped count; '
                              'available_cap/cap_to_available caps each class independently.'))

    parser.add_argument('--samples_per_class_scope', type=str,
                        default=DEFAULT_SAMPLES_PER_CLASS_SCOPE,
                        choices=['split', 'fold'],
                        help='Scope used when interpreting per-class real quotas. AppClassNet provided splits use split.')

    parser.add_argument('--dry_run_memory', action='store_true', default=False,
                        help='Load metadata, log shapes/memory estimates, and skip experiment execution.')

    parser.add_argument('--save_synthetic_format', type=str, default=DEFAULT_SAVE_SYNTHETIC_FORMAT,
                        choices=['legacy', 'npy_batches', 'csv_batches', 'single_npy'],
                        help=("Synthetic persistence format. Default 'legacy' preserves existing behavior. "
                              "Batches mode should use npy_batches unless explicit materialization is required."))

    parser.add_argument('--materialize_synthetic', action='store_true', default=False,
                        help='Materialize synthetic data in memory in batches mode. This can use substantial RAM.')

    parser.add_argument('--vae_epochs', type=int, default=None,
                        help='Diagnostic alias for variational_autoencoder_number_epochs.')

    parser.add_argument('--gan_epochs', type=int, default=None,
                        help='Diagnostic alias for adversarial/wasserstein GAN epoch counts.')

    return parser
