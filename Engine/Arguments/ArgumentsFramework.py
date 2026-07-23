#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

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


# try:
import sys
import datetime
import argparse
import logging


# except ImportError as error:
#
#     print(error)
#     print()
#     print("1. (optional) Setup a virtual environment: ")
#     print("  python3 - m venv ~/Python3env/DroidAugmentor ")
#     print("  source ~/Python3env/DroidAugmentor/bin/activate ")
#     print()
#     print("2. Install requirements:")
#     print("  pip3 install --upgrade pip")
#     print("  pip3 install -r requirements.txt ")
#     print()
#     sys.exit(-1)

from ..Classifiers.Classifiers import Classifiers
from Engine.Evaluation.ExperimentProtocol import EVALUATION_PROTOCOL_CHOICES
DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"

DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 100
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_SAVE_MODELS = False
DEFAULT_SAVE_DATA = False
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "training_curve"
DEFAULT_CLASSIFIER_LIST = ["RandomForest", "KNN", "DecisionTree"]
DEFAULT_EVALUATION_METHOD = ["TrAs", "TsAr"]
DEFAULT_SAMPLE_PLAN = "legacy"
CLASSIFIER_ALIASES = {
    "decisiontree": "DecisionTree",
    "decision_tree": "DecisionTree",
    "randomforest": "RandomForest",
    "random_forest": "RandomForest",
    "decisiontreesubset": "DecisionTreeSubset",
    "decision_tree_subset": "DecisionTreeSubset",
    "randomforestsubset": "RandomForestSubset",
    "random_forest_subset": "RandomForestSubset",
    "randomforestlight": "RandomForestLight",
    "random_forest_light": "RandomForestLight",
}
BATCH_CLASSIFIER_CHOICES = [
    "sgd",
    "passive_aggressive",
    "naive_bayes",
    "mlp_small",
    "decision_tree_subset",
    "extra_trees_subset",
    "random_forest_light",
    "random_forest_subset",
]
EVAL_CLASSIFIER_CHOICES = [
    "decision_tree_subset",
    "extra_trees_subset",
    "random_forest_light",
    "sgd",
]
NORMAL_CLASSIFIER_CHOICES = [
    "decision_tree",
    "random_forest",
    "decision_tree_subset",
    "random_forest_subset",
]
GENERATION_STRATEGY_CHOICES = [
    "single_conditional",
    "per_class",
    "grouped_classes",
]

DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

LOGGING_FILE_NAME = "logging.log"
 

def parse_number_samples(samples_str):

    samples = samples_str.split(',')
    parsed_samples = {}
    #max_class_id = 0

    for sample in samples:
        class_id, num_samples = sample.split(':')
        #max_class_id = max(max_class_id, int(class_id))
        parsed_samples[int(class_id)] = int(num_samples)

    return {"classes": parsed_samples,
            "number_classes": len(parsed_samples)}


class NumberSamplesPerClassAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, parse_number_samples(values))
        setattr(namespace, "_legacy_number_samples_per_class_explicit", True)

def add_argument_framework():

    parser = argparse.ArgumentParser(description='SynDataGen Data Generator')

    
    parser.set_defaults(_legacy_number_samples_per_class_explicit=False)

    parser.add_argument('--number_samples_per_class', action=NumberSamplesPerClassAction,
                        default="1:256,2:256",
                        help="Class and number of samples in the format class1:num1,class2:num2,...")

    parser.add_argument('--sample_plan', type=str, default=DEFAULT_SAMPLE_PLAN,
                        choices=['legacy', 'class_counts', 'total_rows', 'match_train_distribution',
                                 'balanced_per_class'],
                        help=("Synthetic sampling plan. Default 'legacy' preserves number_samples_per_class behavior. "
                              "Use balanced_per_class with --samples_per_class or match_train_distribution/total_rows "
                              "with --total_synthetic_rows for provided X/y datasets."))

    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Number of synthetic samples per class for --sample_plan balanced_per_class.')

    parser.add_argument('--total_synthetic_rows', type=int, default=None,
                        help='Total synthetic rows for --sample_plan total_rows or match_train_distribution.')

    parser.add_argument('-c', '--classifier', type=str, default=DEFAULT_CLASSIFIER_LIST, nargs="+",
                        choices=sorted(set(Classifiers.dictionary_classifiers_name) | set(CLASSIFIER_ALIASES)),
                        help="Classifier (or list of classifiers separated by empty space) default: {} availabe: {}.".format(
                            DEFAULT_CLASSIFIER_LIST,
                            sorted(set(Classifiers.dictionary_classifiers_name) | set(CLASSIFIER_ALIASES))))

    parser.add_argument('--pipeline', type=str, default=None,
                        choices=['tr_tr', 'synthetic', 'synthetic_eval', 'augmentation', 'all'],
                        help='Pipeline selector. tr_tr executes real train -> real test; synthetic keeps legacy generation/evaluation.')

    parser.add_argument('--baseline_real_only', action='store_true', default=False,
                        help='Legacy alias for --pipeline tr_tr.')

    parser.add_argument('--batch_classifier', type=str, default=None,
                        choices=BATCH_CLASSIFIER_CHOICES,
                        help='Legacy alias for --eval_classifier in execution_mode=batches.')

    parser.add_argument('--eval_classifier', type=str, default="decision_tree_subset",
                        choices=EVAL_CLASSIFIER_CHOICES,
                        help=('Classifier used by execution_mode=batches. Default: decision_tree_subset. '
                              'SGDClassifier remains available for low-RAM checks but can severely underestimate '
                              'AppClassNet synthetic quality.'))

    parser.add_argument('--evaluation_mode', type=str, default="both",
                        choices=["none", "tr_ts", "ts_tr", "tr_ts_tr", "both", "all"],
                        help=('Synthetic evaluation mode: none, TR-TS, TS-TR, TR+TS-TR, both (TR-TS and TS-TR), '
                              'or all. Default: both.'))

    parser.add_argument('--run_tr_tr', action='store_true', default=False,
                        help=('Also execute the pipeline TR-TR evaluator. Default false preserves the original '
                              'synthetic pipeline output; use --baseline_real_only for the AppClassNet golden.'))

    parser.add_argument('--evaluation_protocol', type=str, default="legacy",
                        choices=EVALUATION_PROTOCOL_CHOICES,
                        help=('Evaluation protocol. Canonical AppClassNet selectors: tr_tr, tr_ts, ts_tr, '
                              'tr_plus_ts_tr, all. legacy/appclassnet_strict are preserved for compatibility.'))

    parser.add_argument('--batch_classifier_subset_size', type=int, default=100000,
                        help='Maximum rows loaded for *_subset batch classifiers. Default: 100000.')

    parser.add_argument('--train_samples_per_class', type=int, default=None,
                        help='Per-class reservoir quota for subset eval classifiers.')

    parser.add_argument('--test_samples_per_class', type=int, default=None,
                        help='Per-class evaluation cap used by batch evaluators when applicable.')

    parser.add_argument('--train_sampling', type=str, default=None,
                        choices=['all', 'balanced_per_class', 'up_to_available'],
                        help='TR-TR train sampling strategy. all uses the full provided train split.')

    parser.add_argument('--test_sampling', type=str, default=None,
                        choices=['all', 'balanced_per_class', 'up_to_available'],
                        help='TR-TR test sampling strategy. all evaluates the full provided test split.')

    parser.add_argument('--synthetic_train_samples_per_class', type=int, default=None,
                        help='Explicit per-class synthetic quota used to train TS-TR classifiers.')

    parser.add_argument('--synthetic_test_samples_per_class', type=int, default=None,
                        help='Explicit per-class synthetic cap used to evaluate TR-TS classifiers.')

    parser.add_argument('--n_estimators', type=int, default=None,
                        help='Estimator count for extra_trees_subset and random_forest_light.')

    parser.add_argument('--rf_n_estimators', dest='random_forest_n_estimators', type=int, default=100,
                        help='TR-TR RandomForestClassifier n_estimators.')

    parser.add_argument('--rf_n_jobs', dest='random_forest_n_jobs', type=int, default=1,
                        help='TR-TR RandomForestClassifier n_jobs.')

    parser.add_argument('--rf_criterion', dest='random_forest_criterion', type=str, default='gini',
                        choices=['gini', 'entropy', 'log_loss'],
                        help='TR-TR RandomForestClassifier criterion.')

    parser.add_argument('--rf_max_depth', dest='random_forest_max_depth', type=int, default=None,
                        help='TR-TR RandomForestClassifier max_depth.')

    parser.add_argument('--rf_min_samples_split', dest='random_forest_min_samples_split', default=2,
                        help='TR-TR RandomForestClassifier min_samples_split.')

    parser.add_argument('--rf_min_samples_leaf', dest='random_forest_min_samples_leaf', default=1,
                        help='TR-TR RandomForestClassifier min_samples_leaf.')

    parser.add_argument('--rf_max_features', dest='random_forest_max_features', default='sqrt',
                        help='TR-TR RandomForestClassifier max_features.')

    parser.add_argument('--rf_bootstrap', dest='random_forest_bootstrap', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='TR-TR RandomForestClassifier bootstrap.')

    parser.add_argument('--rf_class_weight', dest='random_forest_class_weight', default=None,
                        choices=[None, 'balanced', 'balanced_subsample'],
                        help='TR-TR RandomForestClassifier class_weight.')

    parser.add_argument('--max_depth', type=int, default=None,
                        help='Tree max_depth for subset eval classifiers.')

    parser.add_argument('--max_samples', type=float, default=None,
                        help='Optional RandomForest max_samples for random_forest_light.')

    parser.add_argument('--class_weight', type=str, default=None,
                        choices=[None, 'balanced', 'balanced_subsample'],
                        help='Optional class_weight for tree ensemble eval classifiers.')

    parser.add_argument('--normal_classifier', type=str, default=None,
                        choices=NORMAL_CLASSIFIER_CHOICES,
                        help=('Optional classifier override for normal mode. By default the legacy classifier list '
                              'is preserved. Subset variants train DecisionTree/RandomForest on a limited subset.'))

    parser.add_argument('--generation_strategy', type=str, default="single_conditional",
                        choices=GENERATION_STRATEGY_CHOICES,
                        help=('Generation strategy for batches mode. single_conditional preserves the existing '
                              'conditional generator; per_class and grouped_classes train separate generators when '
                              'the selected algorithm supports it.'))

    parser.add_argument('--classes_per_group', type=int, default=10,
                        help='Number of classes per generator for --generation_strategy grouped_classes.')

    parser.add_argument('--evaluation', type=str, default=DEFAULT_EVALUATION_METHOD, nargs="+",
                        help="List Evaluation Methods ['TrAs', 'TsAr'}")

    parser.add_argument('-o', '--output_dir', type=str,
                        default=f'outputs/out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Directory for saving output files. The final experiment directory is kept under outputs/.')

    parser.add_argument('--number_k_folds', type=int,
                        default=DEFAULT_NUMBER_STRATIFICATION_FOLD,
                        help='Number of folds for cross-validation.')

    parser.add_argument('--random_state', type=int,
                        default=0,
                        help='Random seed used by deterministic evaluators and subset sampling.')

    parser.add_argument('--use_gpu', action='store_true',
                        default=False,
                        help='Enable GPU processing if available.')

    parser.add_argument("--verbosity", type=int,
                        help='Verbosity (Default {})'.format(DEFAULT_VERBOSITY),
                        default=DEFAULT_VERBOSITY)

    parser.add_argument("--save_models", type=bool,
                        help='Save trained models (Default {})'.format(DEFAULT_SAVE_MODELS),
                        default=DEFAULT_SAVE_MODELS)
    
    parser.add_argument("--save_data", type=bool,
                        help='Save generated data (Default {})'.format(DEFAULT_SAVE_DATA),
                        default=DEFAULT_SAVE_DATA)

    parser.add_argument("--path_confusion_matrix", type=str,
                        help='Output directory for confusion matrices',
                        default=DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX)

    parser.add_argument("--path_curve_loss", type=str,
                        help='Output directory for training curve plots',
                        default=DEFAULT_OUTPUT_PATH_TRAINING_CURVE)

    parser.add_argument('--model_type',
                        choices=['smote',
                                 'random',
                                 'adversarial',
                                 'latent_diffusion',
                                 'denoising_diffusion',
                                 'wasserstein',
                                 'wasserstein_gp',
                                 'variational',
                                 'autoencoder',
                                 'quantized',
                                 'diffusion_kernel',
                                 'copy',
                                 'copula',
                                 'ctgan',
                                 'tvae'],
                        default='adversarial', help='Select the model type')

    return parser
