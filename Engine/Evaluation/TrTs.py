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
    import numpy
    import logging

    from Engine.DataIO.LabelUtils import labels_to_1d_integer
    from Engine.DataIO.DatasetContracts import validate_xy_alignment
    from Engine.Classifiers.BatchClassifiers import get_batch_classifier_display_name
    from Engine.Classifiers.BatchClassifiers import iter_array_batches
    from Engine.Classifiers.BatchClassifiers import predict_synthetic_batches
    from Engine.Classifiers.BatchClassifiers import train_batch_classifier
    from Engine.Classifiers.BatchClassifiers import validate_real_array_for_batch_evaluation
    from Engine.Classifiers.BatchClassifiers import validate_synthetic_batches_for_evaluation
    from Engine.Evaluation.EvaluationRunner import EvaluationMode
    from Engine.Evaluation.EvaluationRunner import EvaluationRunner
    from Engine.Evaluation.EvaluationRunner import require_split_name
    from Engine.Evaluation.EvaluationRunner import split_to_dictionary
    from Engine.Evaluation.EvaluationRunner import uses_strict_appclassnet_protocol
    from Engine.Preprocessing.FeatureTransformManager import ScaleGuard

    from sklearn.metrics.pairwise import euclidean_distances

except ImportError as error:
    print(error)
    sys.exit(-1)

class TrTs:

    def evaluation_TR_TS(self, dictionary_data=None, synthetic_data=None, *, real_train_data=None, synthetic_test_data=None):
        """
        Evaluates the performance of classifiers trained on real data and evaluated on synthetic data.
        This method trains classifiers using real data and evaluates them using synthetic data. The binary
        classification metrics for each classifier are calculated and logged for each fold.

        Args:
            dictionary_data (dict): A dictionary containing real training data. The key 'x_training_real' holds
                                    the features for training, and the key 'y_training_real' holds the true labels.
        """

        # Logging the evaluation strategy
        logging.info(f"")
        logging.info(f"")
        logging.info(f"")
        logging.info(f"#################################################################################")
        logging.info(f"\tTR-TS: train on real, test on synthetic")
        if real_train_data is not None:
            require_split_name("TR-TS", real_train_data.name, "train")
            dictionary_data = split_to_dictionary(real_train_data=real_train_data, real_test_data=real_train_data)
        if synthetic_test_data is not None:
            synthetic_data = synthetic_test_data
        if dictionary_data is None:
            raise ValueError("TR-TS requires real_train_data or dictionary_data.")
        logging.info(
            "TR-TS routing: train_split=%s test_split=synthetic_test real_train_path=%s real_train_y_path=%s "
            "real_train_minimum_class_count=%s requested_train_samples_per_class=%s",
            dictionary_data.get("training_split_name"),
            dictionary_data.get("split_metadata", {}).get("train", {}).get("x_path"),
            dictionary_data.get("split_metadata", {}).get("train", {}).get("y_path"),
            dictionary_data.get("split_metadata", {}).get("train", {}).get("minimum_class_count"),
            getattr(getattr(self, "arguments", None), "train_samples_per_class", None),
        )

        arguments = getattr(self, "arguments", None)
        if getattr(arguments, "execution_mode", "normal") == "batches":
            train_x_key = 'x_training_real' if uses_strict_appclassnet_protocol(arguments) else 'x_evaluation_real'
            train_y_key = 'y_training_real' if uses_strict_appclassnet_protocol(arguments) else 'y_evaluation_real'
            if hasattr(synthetic_data, "manifest"):
                ScaleGuard.validate_compatible_metadata(
                    ScaleGuard.describe(dictionary_data[train_x_key], data_space="source"),
                    {
                        "data_space": synthetic_data.manifest.get("data_space", "source"),
                        "transform_id": synthetic_data.manifest.get("transform_id"),
                        "transform_history": synthetic_data.manifest.get("transform_history", []),
                    },
                    context="TR-TS",
                )
            if not getattr(self, '_labels_are_discrete', True):
                reason = "TR-TS predictive evaluation skipped because labels are not discrete."
                logging.warning("\t\t%s", reason)
                self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
                return

            classifier_key = getattr(
                self.arguments,
                "eval_classifier",
                getattr(self.arguments, "batch_classifier", "decision_tree_subset"),
            )
            classifier_name = get_batch_classifier_display_name(classifier_key)
            train_labels = labels_to_1d_integer(
                dictionary_data[train_y_key],
                context="TR-TS training labels",
            )
            validate_xy_alignment(
                dictionary_data[train_x_key],
                train_labels,
                f"TR-TS training split={train_x_key}",
            )
            expected_classes = self._get_configured_number_classes(train_labels)
            validate_real_array_for_batch_evaluation(
                dictionary_data[train_x_key],
                train_labels,
                "TR-TS",
                expected_num_classes=expected_classes,
                samples_per_class=getattr(self.arguments, "train_samples_per_class", None),
                real_class_count_policy=getattr(self.arguments, "real_class_count_policy", "strict"),
                samples_per_class_scope=getattr(self.arguments, "samples_per_class_scope", "split"),
            )
            validate_synthetic_batches_for_evaluation(
                synthetic_data,
                "TR-TS",
                expected_num_classes=expected_classes,
                samples_per_class=(
                    getattr(self.arguments, "synthetic_test_samples_per_class", None)
                    or getattr(self.arguments, "test_samples_per_class", None)
                ),
                expected_num_features=dictionary_data[train_x_key].shape[1],
                expected_data_space="source",
                expected_schema_hash=getattr(getattr(getattr(self, "_dataset_bundle", None), "schema", None), "schema_hash", None),
            )
            train_batches = iter_array_batches(
                dictionary_data[train_x_key],
                train_labels,
                getattr(self.arguments, "batch_size", 8192),
            )
            classifier_instance, metadata = train_batch_classifier(
                classifier_key,
                train_batches,
                expected_classes,
                self.arguments,
                batch_recorder=self.record_batch_processed,
            )
            labels, predictions, evaluation_time = predict_synthetic_batches(
                classifier_instance,
                synthetic_data,
                batch_recorder=self.record_batch_processed,
                max_samples_per_class=(
                    getattr(self.arguments, "synthetic_test_samples_per_class", None)
                    or getattr(self.arguments, "test_samples_per_class", None)
                ),
            )
            metadata["evaluation_time_seconds"] = float(evaluation_time)
            metadata["evaluation_time"] = float(evaluation_time)
            metadata["real_samples_used_by_class"] = metadata.get("train_class_counts", {})
            metadata["synthetic_samples_used_by_class"] = {
                str(int(label)): int(count)
                for label, count in zip(*numpy.unique(labels, return_counts=True))
            } if labels.size else {}
            self.record_batch_classifier_metadata("TR-TS", classifier_name, self.fold_number + 1, metadata)

            if labels.size == 0:
                reason = "TR-TS predictive evaluation skipped because no synthetic batch rows were generated."
                logging.warning("\t\t%s", reason)
                self.mark_classifier_metrics_not_applicable("TR-TS", classifier_name, self.fold_number + 1, reason)
                return

            logging.info("")
            logging.info(f"\t\tTR-TS {classifier_name}")
            self.get_task_metrics(labels, predictions, "TR-TS", classifier_name, self.fold_number + 1)
            return

        runner = EvaluationRunner(self, EvaluationMode.TR_TS)
        try:
            evaluation_dataset = runner.build_evaluation_dataset(dictionary_data, synthetic_data)
            runner.validate(evaluation_dataset)
            runner.save_results(evaluation_dataset, self.fold_number + 1)
        except ValueError as error:
            reason = f"TR-TS predictive evaluation skipped: {error}"
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
            return

        # Logging the total number of generated synthetic samples to be processed
        total_samples = len(evaluation_dataset.y_test)
        logging.info(f"\t\tTR-TS: Total number of samples to be saved: {total_samples}")

        if not getattr(self, '_labels_are_discrete', True):
            reason = "TR-TS predictive evaluation skipped because labels are not discrete."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
            return

        if total_samples == 0:
            reason = "TR-TS predictive evaluation skipped because no synthetic samples were generated."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
            return

        classifiers = runner.fit_classifier(evaluation_dataset)

        # Evaluate the classifiers on synthetic data for each classifier instance
        for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
            # Predict the labels using the trained classifier on the synthetic data
            label_predicted = runner.predict(classifier_instances, evaluation_dataset)
            logging.info("")
            logging.info(f"\t\tTR-TS {classifier_name}")
            # Calculate and log the binary classification metrics (such as accuracy, precision, recall, etc.)
            runner.calculate_metrics(
                evaluation_dataset,
                label_predicted,
                classifier_name,
                self.fold_number + 1,
            )
        
        
        
    
       
 
