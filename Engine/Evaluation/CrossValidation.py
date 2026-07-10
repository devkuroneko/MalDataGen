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


try:

    import sys
    import time

    import numpy
    import pandas
    import logging

    from sklearn.utils import shuffle

    from Engine.DataIO.CSVLoader import CSVDataProcessor
    from Engine.DataIO.LabelUtils import build_class_metadata
    from Engine.DataIO.LabelUtils import validate_zero_based_labels
    from Engine.DataIO.NpyXYLoader import NpyXYLoader

    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

except ImportError as error:
    print(error)

    sys.exit(-1)

NOT_APPLICABLE = "not_applicable"

def _save_data_to_csv(directory_output_data, data, labels, filename_prefix, fold):
        """Helper function to save data and labels to CSV."""
        # Concatenate data and labels
        data_with_labels = numpy.column_stack((data, labels))
        
        # Create column names
        columns = [f"f{i}" for i in range(data.shape[1])] + ["label"]
        
        # Create and save DataFrame
        data_frame = pandas.DataFrame(data_with_labels, columns=columns)
        csv_filename = f"{directory_output_data}/{filename_prefix}_fold_{fold + 1}.csv"
        data_frame.to_csv(csv_filename, index=False)

def _build_class_metadata(labels):
        """Build class counts using zero-based labels already prepared by the loader."""
        return build_class_metadata(labels)

def _target_type_for_npy(arguments):
        target_type = getattr(arguments, 'target_type', 'auto')
        return 'multiclass' if target_type == 'auto' else target_type

def _feature_type_for_npy(arguments):
        feature_type = getattr(arguments, 'feature_type', 'auto')
        return 'continuous' if feature_type == 'auto' else feature_type

def _mmap_mode_for_npy(arguments):
        return 'r' if getattr(arguments, 'mmap_npy', False) else None

def _log_bundle_summary(bundle):
        schema = bundle.schema
        logging.info("Dataset format selected: npy_xy")
        logging.info("Dataset target_type: %s", schema.target_type)
        logging.info("Dataset num_classes: %s", schema.num_classes)
        logging.info("Dataset feature_type: %s", schema.feature_type)
        logging.info("Dataset mmap_mode: %s", bundle.metadata.get("mmap_mode"))

        for split_name, split in bundle.splits.items():
            y_shape = None if split.y is None else split.y.shape
            logging.info("Dataset split %s: X=%s, y=%s", split_name, split.X.shape, y_shape)

def load_dataset_from_args(arguments, owner=None):
        """Load the configured dataset without changing the legacy CSV default.

        For CSV, this performs the same initialization and load sequence used by
        the legacy autoload decorator. For npy_xy, it returns a DatasetBundle for
        the caller to adapt into the current fold contract.
        """
        data_format = getattr(arguments, 'data_format', 'csv')

        if data_format == 'csv':
            logging.info("Dataset format selected: csv")
            if owner is None:
                return None
            CSVDataProcessor.__init__(owner, arguments)
            owner.load_csv()
            return None

        if data_format != 'npy_xy':
            raise ValueError(f"Unsupported data_format: {data_format}")

        loader = NpyXYLoader(
            train_x_path=arguments.train_x_path,
            train_y_path=arguments.train_y_path,
            valid_x_path=getattr(arguments, 'valid_x_path', None),
            valid_y_path=getattr(arguments, 'valid_y_path', None),
            test_x_path=getattr(arguments, 'test_x_path', None),
            test_y_path=getattr(arguments, 'test_y_path', None),
            mmap_mode=_mmap_mode_for_npy(arguments),
            target_type=_target_type_for_npy(arguments),
            feature_type=_feature_type_for_npy(arguments),
            num_classes=getattr(arguments, 'num_classes', None),
            remap_labels_to_zero_based=getattr(arguments, 'remap_labels_to_zero_based', False),
        )
        bundle = loader.load()
        _log_bundle_summary(bundle)
        return bundle

def _apply_bundle_to_owner(owner, bundle):
        owner._dataset_bundle = bundle
        owner._target_type = bundle.schema.target_type
        owner._data_type = bundle.schema.feature_type
        owner._data_loaded = numpy.asarray(bundle.train.X, dtype=numpy.float32)
        owner._data_loaded_labels = validate_zero_based_labels(
            bundle.train.y,
            num_classes=bundle.schema.num_classes,
            context="train y",
        )
        owner._data_loaded_header = list(bundle.schema.feature_names) + [bundle.schema.target_name or "label"]
        owner._data_original_header = list(owner._data_loaded_header)
        owner._label_mapping = None
        owner._label_inverse_mapping = None
        owner._labels_are_discrete = bundle.schema.target_type in ('binary', 'multiclass')
        owner.list_folds = []
        owner._number_samples_per_class = _number_samples_per_class_from_schema(bundle.schema, owner._data_loaded_labels)
        owner.arguments.number_samples_per_class = owner._number_samples_per_class

def _number_samples_per_class_from_schema(schema, labels):
        metadata = build_class_metadata(labels, num_classes=schema.num_classes)
        if schema.num_classes is not None:
            metadata["number_classes"] = int(schema.num_classes)
        metadata["data_type"] = getattr(schema, "feature_type", "unknown")
        metadata["target_type"] = getattr(schema, "target_type", "auto")
        return metadata

def _create_fold(training_split, evaluation_split, evaluation_name=None, evaluation_not_applicable=False):
        evaluation_source = evaluation_split if evaluation_split is not None else training_split
        return {
            'x_training_real': numpy.asarray(training_split.X, dtype=numpy.float32),
            'y_training_real': validate_zero_based_labels(training_split.y, context=f"{training_split.name} y"),

            'x_evaluation_real': numpy.asarray(evaluation_source.X, dtype=numpy.float32),
            'y_evaluation_real': validate_zero_based_labels(evaluation_source.y, context=f"{evaluation_source.name} y"),

            'x_training_synthetic': None,
            'y_training_synthetic': None,

            'x_evaluation_synthetic': None,
            'y_evaluation_synthetic': None,

            'evaluation_split_name': evaluation_name or evaluation_source.name,
            'evaluation_not_applicable': evaluation_not_applicable,
        }

def _mark_test_not_applicable(owner, reason):
        metrics = getattr(owner, '_dictionary_metrics', None)
        if metrics is None:
            return
        metrics.setdefault("ProvidedSplits", {})["test"] = {
            "status": NOT_APPLICABLE,
            "reason": reason,
        }

def _build_provided_split_folds(owner, bundle):
        evaluation_split = bundle.valid or bundle.test
        evaluation_name = None if evaluation_split is None else evaluation_split.name
        evaluation_not_applicable = evaluation_split is None

        if bundle.valid is not None and bundle.test is not None:
            reason = "Final test split evaluation is not supported by the current TR-TS/TS-TR pipeline."
            logging.warning("%s Marking test split as %s.", reason, NOT_APPLICABLE)
            _mark_test_not_applicable(owner, reason)

        if evaluation_not_applicable:
            logging.warning(
                "split_mode=provided has no valid/test split. Training can proceed from train, but predictive "
                "and distance evaluations are %s for this run.",
                NOT_APPLICABLE,
            )

        owner.arguments.number_k_folds = 1
        owner.list_folds.append(_create_fold(
            bundle.train,
            evaluation_split,
            evaluation_name=evaluation_name,
            evaluation_not_applicable=evaluation_not_applicable,
        ))

        logging.info(
            "Provided split fold created: train=%s, evaluation=%s, evaluation_not_applicable=%s",
            bundle.train.name,
            evaluation_name,
            evaluation_not_applicable,
        )

def StratifiedData(function):

    

    """
    A decorator that performs stratified K-Fold cross-validation on the provided dataset.
    This method shuffles the dataset, splits it into `k` stratified folds, and saves
    the corresponding training and validation datasets as CSV files. It also logs detailed
    information about the process and each fold.

    Args:
        function (Callable): The function to be wrapped. After completing the stratified K-Fold
                              cross-validation process, this function will be called with the
                              provided arguments.

    Returns:
        Callable: The wrapped function with stratified K-Fold processing applied.
    """

    def wrapper(self, *args, **kwargs):
        dataset_bundle = load_dataset_from_args(self.arguments, owner=self)

        if dataset_bundle is not None:
            _apply_bundle_to_owner(self, dataset_bundle)
            if getattr(self.arguments, 'split_mode', 'cross_validation') == 'provided':
                _build_provided_split_folds(self, dataset_bundle)
                return function(self, *args, **kwargs)

            if dataset_bundle.valid is not None or dataset_bundle.test is not None:
                logging.warning(
                    "data_format=npy_xy with split_mode=cross_validation uses only the train split for K-fold. "
                    "Provided valid/test splits are not concatenated or used."
                )

        # Track the start time of the entire stratified K-fold process
        start_time = time.time()

        data_type = getattr(self, '_data_type', getattr(self.arguments, 'data_type', 'binary'))
        labels_are_discrete = getattr(self, '_labels_are_discrete', True)
        use_stratified_split = labels_are_discrete
        split_name = "StratifiedKFold" if use_stratified_split else "KFold"

        logging.info(
            "%s initialization started. Data type: %s. Number of splits: %d, Shuffle: True, Random state: 42.",
            split_name, data_type, self.arguments.number_k_folds)

        try:
            # Shuffle the data before performing stratified splitting
            shuffled_data, shuffled_labels = shuffle(self._data_loaded, self._data_loaded_labels, random_state=42)

            if labels_are_discrete:
                self.arguments.number_samples_per_class = build_class_metadata(
                    shuffled_labels,
                    num_classes=getattr(getattr(getattr(self, '_dataset_bundle', None), 'schema', None), 'num_classes', None),
                )
                dataset_bundle = getattr(self, '_dataset_bundle', None)
                schema_num_classes = getattr(getattr(dataset_bundle, 'schema', None), 'num_classes', None)
                if schema_num_classes is not None:
                    self.arguments.number_samples_per_class["number_classes"] = int(schema_num_classes)
                self.arguments.number_samples_per_class["data_type"] = data_type
                self.arguments.number_samples_per_class["target_type"] = getattr(
                    self,
                    '_target_type',
                    getattr(self.arguments, 'target_type', 'auto'),
                )
                self._number_samples_per_class = self.arguments.number_samples_per_class
                logging.info("Class metadata inferred from dataset labels: %s", self._number_samples_per_class)
            else:
                self.arguments.number_samples_per_class = {
                    "classes": {0: int(len(shuffled_labels))},
                    "number_classes": 1,
                    "data_type": data_type,
                }
                self._number_samples_per_class = self.arguments.number_samples_per_class
                logging.info("Non-discrete labels detected; class metadata was not inferred.")

            if use_stratified_split:
                splitter = StratifiedKFold(n_splits=self.arguments.number_k_folds,
                                           shuffle=True,
                                           random_state=42)
                split_iterator = splitter.split(shuffled_data, numpy.ravel(shuffled_labels))
            else:
                splitter = KFold(n_splits=self.arguments.number_k_folds,
                                 shuffle=True,
                                 random_state=42)
                split_iterator = splitter.split(shuffled_data)

            logging.info("Data loaded for stratification. Total samples: %d", len(self._data_loaded))

            # Loop through each fold and process the stratified splits
            for fold, (train_index, val_index) in enumerate(split_iterator):
                # Track the start time of processing a specific fold
                fold_start_time = time.time()

                logging.info("Processing fold %d of %d.", fold + 1, self.arguments.number_k_folds)

                
                
                logging.info("Training   indices (size %d): %s", len(train_index), train_index)
                logging.info("Evaluating indices (size %d): %s", len(val_index), val_index)

                 

                # # Concatenate the data and labels for the shuffled dataset
                # data_with_labels = numpy.column_stack((shuffled_data, shuffled_labels))

                # # Create a DataFrame for saving the shuffled dataset as CSV or XLS
                # columns = [f"feature_{i}" for i in range(shuffled_data.shape[1])] + ["label"]
                # data_frame = pandas.DataFrame(data_with_labels, columns=columns)

                # # Save the shuffled data to CSV
                # csv_filename = "{}/data_shuffled_dataset.csv".format(self.directory_output_data)
                # data_frame.to_csv(csv_filename, index=False)

                # Dentro do seu loop for fold:
                # Save training data
                _save_data_to_csv(
                    self.directory_output_data,
                    self._data_loaded[train_index],
                    self._data_loaded_labels[train_index],
                    "data_training",
                    fold
                )
 

                # Save validation data
                _save_data_to_csv(
                    self.directory_output_data,
                    self._data_loaded[val_index],
                    self._data_loaded_labels[val_index],
                    "data_evaluation",
                    fold
                ) 

                # Shuffle the training and evaluation data
                training_shuffled_data, training_shuffled_labels = shuffle(self._data_loaded[train_index],
                                                                           self._data_loaded_labels[train_index],
                                                                           random_state=42)
                
                

                evaluation_shuffled_data, evaluation_shuffled_labels = shuffle(self._data_loaded[val_index],
                                                                               self._data_loaded_labels[val_index],
                                                                               random_state=42)

                # Store the training and evaluation data for later use
                self.list_folds.append({
                    'x_training_real': training_shuffled_data,
                    'y_training_real': training_shuffled_labels,

                    'x_evaluation_real': evaluation_shuffled_data,
                    'y_evaluation_real': evaluation_shuffled_labels,
                    
                    'x_training_synthetic': None,
                    'y_training_synthetic': None,

                    'x_evaluation_synthetic': None,
                    'y_evaluation_synthetic': None
                })

                # Track the end time for this fold
                fold_end_time = time.time()

                logging.info("Fold %d processed successfully in %.2f seconds.", fold + 1,
                             fold_end_time - fold_start_time)

            # Track the end time for the entire K-Fold process
            end_time = time.time()

            logging.info("All %d folds successfully created using %s in %.2f seconds.",
                         self.arguments.number_k_folds, split_name, end_time - start_time)

            # Call the original function passed as a decorator argument
            return function(self, *args, **kwargs)

        except Exception as e:
            logging.error("An error occurred during fold processing: %s", str(e))
            raise  # Re-raise the exception after logging the error

    return wrapper
