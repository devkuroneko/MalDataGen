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
    from pathlib import Path

    import numpy
    import pandas
    import logging

    from sklearn.utils import shuffle

    from Engine.DataIO.CSVLoader import CSVDataProcessor
    from Engine.DataIO.LabelUtils import build_class_metadata
    from Engine.DataIO.LabelUtils import validate_zero_based_labels
    from Engine.DataIO.DatasetContracts import validate_xy_alignment
    from Engine.DataIO.NpyXYLoader import NpyXYLoader
    from Engine.DataIO.StratifiedNpySelection import build_minimum_coverage_report
    from Engine.DataIO.StratifiedNpySelection import get_last_stratified_selection_report
    from Engine.DataIO.StratifiedNpySelection import save_stratified_selection_report
    from Engine.DataIO.StratifiedNpySelection import select_stratified_indices_from_npy
    from Engine.Utils.ResourceMonitor import Timer

    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import train_test_split

except ImportError as error:
    print(error)

    sys.exit(-1)

NOT_APPLICABLE = "not_applicable"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPCLASSNET_SELECTION_ROOT = PROJECT_ROOT / "results" / "appclassnet_top200" / "batches" / "subsets"

def _format_bytes(num_bytes):
        value = float(num_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.2f} {unit}"
            value /= 1024
        return f"{value:.2f} TiB"

def _array_nbytes(array):
        try:
            return int(numpy.prod(array.shape) * numpy.dtype(array.dtype).itemsize)
        except Exception:
            return None

def _log_array_memory(name, array):
        nbytes = _array_nbytes(array)
        logging.info(
            "%s shape=%s dtype=%s approx_ram=%s",
            name,
            getattr(array, "shape", None),
            getattr(array, "dtype", None),
            _format_bytes(nbytes) if nbytes is not None else "unknown",
        )

def _log_execution_memory_settings(arguments):
        logging.info("Execution mode: %s", getattr(arguments, "execution_mode", "normal"))
        logging.info("Batch size: %s", getattr(arguments, "batch_size", None))
        logging.info("Eval batch size: %s", getattr(arguments, "eval_batch_size", None))
        logging.info("Generation batch size: %s", getattr(arguments, "generation_batch_size", None))
        logging.info("Use mmap: %s", getattr(arguments, "mmap_npy", False))
        logging.info("Max train samples: %s", getattr(arguments, "max_train_samples", None))
        logging.info("Max samples per class: %s", getattr(arguments, "max_samples_per_class", None))

def _apply_batch_limits(owner, bundle):
        if getattr(owner.arguments, "execution_mode", "normal") != "batches":
            return

        _apply_stratified_split_selection(
            owner,
            bundle.train,
            getattr(owner.arguments, "train_y_path"),
            split_name="train",
            samples_per_class=_batch_samples_per_class(owner.arguments, "train"),
        )
        if bundle.valid is not None:
            _apply_stratified_split_selection(
                owner,
                bundle.valid,
                getattr(owner.arguments, "valid_y_path"),
                split_name="valid",
                samples_per_class=_batch_samples_per_class(owner.arguments, "valid"),
            )
        if bundle.test is not None:
            _apply_stratified_split_selection(
                owner,
                bundle.test,
                getattr(owner.arguments, "test_y_path"),
                split_name="test",
                samples_per_class=_batch_samples_per_class(owner.arguments, "test"),
            )

def _batch_samples_per_class(arguments, split_role):
        min_required = int(getattr(arguments, "min_samples_per_class_required", 1))
        if split_role == "train":
            if getattr(arguments, "max_samples_per_class", None) is not None:
                return max(min_required, int(arguments.max_samples_per_class))
            if getattr(arguments, "train_samples_per_class", None) is not None:
                return max(min_required, int(arguments.train_samples_per_class))
            if getattr(arguments, "max_train_samples", None) is not None:
                num_classes = int(getattr(arguments, "num_classes", 1) or 1)
                quota = int(numpy.ceil(int(arguments.max_train_samples) / max(1, num_classes)))
                return max(min_required, quota)
            return max(1, min_required)

        if split_role == "test" and getattr(arguments, "test_samples_per_class", None) is not None:
            return max(min_required, int(arguments.test_samples_per_class))
        return max(1, min_required)

def _apply_stratified_split_selection(owner, split, y_path, split_name, samples_per_class):
        if y_path is None:
            logging.warning("Batches stratified selection skipped for %s: no y_path is available.", split_name)
            return

        if getattr(owner.arguments, "max_train_samples", None) is not None and split_name == "train":
            logging.warning(
                "max_train_samples=%s will not be applied before per-class coverage. "
                "Using samples_per_class=%s for stratified train selection.",
                owner.arguments.max_train_samples,
                samples_per_class,
            )

        num_classes = int(getattr(owner.arguments, "num_classes", None) or getattr(owner._dataset_bundle.schema, "num_classes", 0))
        indices = select_stratified_indices_from_npy(
            y_path,
            int(samples_per_class),
            num_classes,
            seed=42,
            mmap_mode=_mmap_mode_for_npy(owner.arguments) or "r",
        )
        selection_report = get_last_stratified_selection_report() or {}
        coverage_report = build_minimum_coverage_report(
            selection_report,
            int(getattr(owner.arguments, "min_samples_per_class_required", 1)),
        )
        selection_report.update({
            "split_name": split_name,
            "samples_per_class_used_for_selection": int(samples_per_class),
            "minimum_coverage": coverage_report,
        })

        report_path = APPCLASSNET_SELECTION_ROOT / f"{split_name}_stratified_selection.json"
        save_stratified_selection_report(selection_report, report_path)
        output_report_path = Path(owner.current_subdir) / "SelectionReports" / f"{split_name}_stratified_selection.json"
        save_stratified_selection_report(selection_report, output_report_path)

        if not coverage_report["passes_minimum"]:
            message = (
                f"Batches {split_name} split has class(es) below "
                f"min_samples_per_class_required={coverage_report['min_samples_per_class_required']}: "
                f"{coverage_report['classes_below_minimum']}."
            )
            if getattr(owner.arguments, "strict_min_samples_per_class", False):
                raise ValueError(message)
            logging.warning(message)

        logging.info(
            "Batches mode: stratified %s selection rows=%d samples_per_class=%d report=%s",
            split_name,
            int(indices.shape[0]),
            int(samples_per_class),
            report_path,
        )
        split.X, split.y = validate_xy_alignment(
            split.X,
            split.y,
            "batches split before stratified selection",
            split=split_name,
        )
        scanned_rows = int(selection_report.get("total_rows_scanned", -1))
        if scanned_rows != int(split.X.shape[0]):
            raise ValueError(
                f"Batches stratified selection y_path row mismatch for split={split_name}: "
                f"y_path rows={scanned_rows}, split X rows={split.X.shape[0]}, "
                f"split y rows={split.y.shape[0]}, y_path={y_path}."
            )
        if indices.size:
            max_selected_index = int(indices.max())
            if max_selected_index >= split.X.shape[0]:
                raise RuntimeError(
                    f"Batches stratified selection produced index {max_selected_index} outside "
                    f"{split_name} X rows={split.X.shape[0]}; y_path={y_path} does not match this split X."
                )
        split.X = numpy.asarray(split.X[indices], dtype=numpy.float32)
        split.y = numpy.asarray(split.y[indices])
        split.refresh_metadata()
        validate_xy_alignment(
            split.X,
            split.y,
            "batches split after stratified selection",
            split=split_name,
            source_indices=indices,
        )
        _log_array_memory(f"Limited {split_name} X", split.X)
        _log_array_memory(f"Limited {split_name} y", split.y)

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
            logging.info(
                "Dataset split %s: X=%s, y=%s x_path=%s y_path=%s min_class_count=%s dataset_id=%s",
                split_name,
                split.X.shape,
                y_shape,
                split.x_path,
                split.y_path,
                split.minimum_class_count,
                split.dataset_id,
            )

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
            source_profile=getattr(arguments, 'source_profile', 'unknown'),
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
        owner._label_mapping = bundle.metadata.get("label_mapping_original_to_zero_based")
        if owner._label_mapping:
            owner._label_inverse_mapping = {
                zero_based: original
                for original, zero_based in owner._label_mapping.items()
            }
        else:
            owner._label_inverse_mapping = None
        owner._labels_are_discrete = bundle.schema.target_type in ('binary', 'multiclass')
        owner.list_folds = []
        owner._number_samples_per_class = _number_samples_per_class_from_schema(bundle.schema, owner._data_loaded_labels)
        owner.arguments.number_samples_per_class = owner._number_samples_per_class
        _log_array_memory("Loaded train X", owner._data_loaded)
        _log_array_memory("Loaded train y", owner._data_loaded_labels)
        _apply_batch_limits(owner, bundle)
        owner._data_loaded = numpy.asarray(bundle.train.X, dtype=numpy.float32)
        owner._data_loaded_labels = validate_zero_based_labels(
            bundle.train.y,
            num_classes=bundle.schema.num_classes,
            context="train y",
        )
        owner._number_samples_per_class = _number_samples_per_class_from_schema(bundle.schema, owner._data_loaded_labels)
        owner.arguments.number_samples_per_class = owner._number_samples_per_class

def _number_samples_per_class_from_schema(schema, labels):
        metadata = build_class_metadata(labels, num_classes=schema.num_classes)
        if schema.num_classes is not None:
            metadata["number_classes"] = int(schema.num_classes)
        metadata["data_type"] = getattr(schema, "feature_type", "unknown")
        metadata["target_type"] = getattr(schema, "target_type", "auto")
        return metadata

def _split_metadata(split):
        if split is None:
            return None
        return {
            "name": split.name,
            "x_path": split.x_path,
            "y_path": split.y_path,
            "dataset_id": split.dataset_id,
            "num_samples": split.num_samples,
            "class_counts": {str(key): int(value) for key, value in split.class_counts.items()},
            "minimum_class_count": split.minimum_class_count,
            "shape": list(split.X.shape),
        }

def _create_fold(
        training_split,
        evaluation_split,
        evaluation_name=None,
        evaluation_not_applicable=False,
        valid_split=None,
        test_split=None,
        dataset_bundle=None):
        evaluation_source = evaluation_split if evaluation_split is not None else training_split
        train_x, train_y = validate_xy_alignment(
            training_split.X,
            training_split.y,
            "fold training",
            split=training_split.name,
        )
        evaluation_x, evaluation_y = validate_xy_alignment(
            evaluation_source.X,
            evaluation_source.y,
            "fold evaluation",
            split=evaluation_source.name,
        )
        train_source_indices = numpy.arange(train_y.shape[0], dtype=numpy.int64)
        evaluation_source_indices = numpy.arange(evaluation_y.shape[0], dtype=numpy.int64)
        return {
            'x_training_real': numpy.asarray(train_x, dtype=numpy.float32),
            'y_training_real': validate_zero_based_labels(train_y, context=f"{training_split.name} y"),
            'training_source_indices': train_source_indices,

            'x_evaluation_real': numpy.asarray(evaluation_x, dtype=numpy.float32),
            'y_evaluation_real': validate_zero_based_labels(evaluation_y, context=f"{evaluation_source.name} y"),
            'evaluation_source_indices': evaluation_source_indices,

            'x_training_synthetic': None,
            'y_training_synthetic': None,

            'x_evaluation_synthetic': None,
            'y_evaluation_synthetic': None,

            'evaluation_split_name': evaluation_name or evaluation_source.name,
            'training_split_name': training_split.name,
            'validation_split_name': None if valid_split is None else valid_split.name,
            'evaluation_not_applicable': evaluation_not_applicable,
            'real_train_split': training_split,
            'real_valid_split': valid_split,
            'real_test_split': test_split,
            'dataset_bundle': dataset_bundle,
            'split_metadata': {
                "train": _split_metadata(training_split),
                "valid": _split_metadata(valid_split),
                "test": _split_metadata(test_split),
                "evaluation": _split_metadata(evaluation_source),
            },
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
        evaluation_not_applicable = bundle.test is None

        if evaluation_not_applicable:
            logging.warning(
                "split_mode=provided has no test split. Training/validation can proceed, but final predictive "
                "evaluations are %s for this run.",
                NOT_APPLICABLE,
            )

        owner.arguments.number_k_folds = 1
        owner.list_folds.append(_create_fold(
            bundle.train,
            evaluation_split,
            evaluation_name=evaluation_name,
            evaluation_not_applicable=evaluation_not_applicable,
            valid_split=bundle.valid,
            test_split=bundle.test,
            dataset_bundle=bundle,
        ))

        logging.info(
            "Provided split fold created: train=%s, validation=%s, final_test=%s, evaluation_not_applicable=%s",
            bundle.train.name,
            None if bundle.valid is None else bundle.valid.name,
            None if bundle.test is None else bundle.test.name,
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
        _log_execution_memory_settings(self.arguments)
        resource_timer = getattr(self, "resource_timer", None)
        record_resource_usage = getattr(self, "record_resource_usage", None)

        if resource_timer is None:
            dataset_bundle = load_dataset_from_args(self.arguments, owner=self)
        else:
            with resource_timer("loading"):
                dataset_bundle = load_dataset_from_args(self.arguments, owner=self)

        preprocessing_timer = Timer("preprocessing")
        preprocessing_timer.__enter__()

        if dataset_bundle is not None:
            _apply_bundle_to_owner(self, dataset_bundle)
            if getattr(self.arguments, 'dry_run_memory', False):
                logging.info("dry_run_memory enabled; skipping fold materialization and experiment execution.")
                preprocessing_timer.__exit__(None, None, None)
                if record_resource_usage is not None:
                    record_resource_usage("preprocessing", preprocessing_timer.elapsed_seconds)
                return None
            if getattr(self.arguments, 'split_mode', 'cross_validation') == 'provided':
                _build_provided_split_folds(self, dataset_bundle)
                preprocessing_timer.__exit__(None, None, None)
                if record_resource_usage is not None:
                    record_resource_usage("preprocessing", preprocessing_timer.elapsed_seconds)
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
            if getattr(self.arguments, 'dry_run_memory', False):
                _log_array_memory("Loaded CSV X", self._data_loaded)
                _log_array_memory("Loaded CSV y", self._data_loaded_labels)
                logging.info("dry_run_memory enabled; skipping fold materialization and experiment execution.")
                preprocessing_timer.__exit__(None, None, None)
                if record_resource_usage is not None:
                    record_resource_usage("preprocessing", preprocessing_timer.elapsed_seconds)
                return None

            # Shuffle the data before performing stratified splitting
            self._data_loaded, self._data_loaded_labels = validate_xy_alignment(
                self._data_loaded,
                self._data_loaded_labels,
                "CrossValidation loaded dataset",
                split="loaded",
            )
            loaded_source_indices = numpy.arange(self._data_loaded_labels.shape[0], dtype=numpy.int64)
            shuffled_data, shuffled_labels, shuffled_source_indices = shuffle(
                self._data_loaded,
                self._data_loaded_labels,
                loaded_source_indices,
                random_state=42,
            )
            validate_xy_alignment(
                shuffled_data,
                shuffled_labels,
                "CrossValidation shuffled dataset",
                split="shuffled",
                source_indices=shuffled_source_indices,
            )

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

                 

                if getattr(self.arguments, "execution_mode", "normal") == "batches":
                    logging.info("Batches mode: skipping intermediate fold CSV materialization.")
                else:
                    _save_data_to_csv(
                        self.directory_output_data,
                        shuffled_data[train_index],
                        shuffled_labels[train_index],
                        "data_training",
                        fold
                    )

                    _save_data_to_csv(
                        self.directory_output_data,
                        shuffled_data[val_index],
                        shuffled_labels[val_index],
                        "data_evaluation",
                        fold
                    )

                # Shuffle the training and evaluation data
                training_shuffled_data, training_shuffled_labels, training_source_indices = shuffle(
                    shuffled_data[train_index],
                    shuffled_labels[train_index],
                    shuffled_source_indices[train_index],
                    random_state=42,
                )
                
                

                evaluation_shuffled_data, evaluation_shuffled_labels, evaluation_source_indices = shuffle(
                    shuffled_data[val_index],
                    shuffled_labels[val_index],
                    shuffled_source_indices[val_index],
                    random_state=42,
                )
                validate_xy_alignment(
                    training_shuffled_data,
                    training_shuffled_labels,
                    "CrossValidation fold train",
                    split=split_name,
                    fold=fold + 1,
                    source_indices=training_source_indices,
                )
                validate_xy_alignment(
                    evaluation_shuffled_data,
                    evaluation_shuffled_labels,
                    "CrossValidation fold evaluation",
                    split=split_name,
                    fold=fold + 1,
                    source_indices=evaluation_source_indices,
                )

                # Store the training and evaluation data for later use
                self.list_folds.append({
                    'x_training_real': training_shuffled_data,
                    'y_training_real': training_shuffled_labels,
                    'training_source_indices': training_source_indices,

                    'x_evaluation_real': evaluation_shuffled_data,
                    'y_evaluation_real': evaluation_shuffled_labels,
                    'evaluation_source_indices': evaluation_source_indices,
                    
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
            preprocessing_timer.__exit__(None, None, None)
            if record_resource_usage is not None:
                record_resource_usage("preprocessing", preprocessing_timer.elapsed_seconds)
            return function(self, *args, **kwargs)

        except Exception as e:
            preprocessing_timer.__exit__(*sys.exc_info())
            if record_resource_usage is not None:
                record_resource_usage("preprocessing", preprocessing_timer.elapsed_seconds)
            logging.error("An error occurred during fold processing: %s", str(e))
            raise  # Re-raise the exception after logging the error

    return wrapper
