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
    from Engine.Classifiers.BatchClassifiers import get_batch_classifier_display_name
    from Engine.Classifiers.BatchClassifiers import iter_synthetic_labeled_batches
    from Engine.Classifiers.BatchClassifiers import predict_array_batches
    from Engine.Classifiers.BatchClassifiers import train_batch_classifier
    from sklearn.utils import shuffle
except ImportError as error:
    print(error)
    sys.exit(-1)


def _counts_by_class(labels):
    unique_labels, counts = numpy.unique(numpy.asarray(labels, dtype=numpy.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}


def _select_stratified_array_subset(x_values, y_values, samples_per_class, seed=42):
    if samples_per_class is None:
        return x_values, y_values

    y_values = numpy.asarray(y_values, dtype=numpy.int64)
    random_generator = numpy.random.default_rng(seed)
    selected = []
    for label in numpy.unique(y_values):
        label_indices = numpy.flatnonzero(y_values == label)
        take = min(int(samples_per_class), int(label_indices.shape[0]))
        if take > 0:
            selected.append(random_generator.choice(label_indices, size=take, replace=False))
    if not selected:
        return x_values[:0], y_values[:0]
    indices = numpy.concatenate(selected).astype(numpy.int64, copy=False)
    random_generator.shuffle(indices)
    return x_values[indices], y_values[indices]


class TsTr:

    def evaluation_TS_TR(self, dictionary_data, synthetic_data):

        """
        Evaluates the performance of classifiers trained on synthetic data and evaluated on real data.
        This method trains classifiers using synthetic data (generated data) and evaluates them using
        real evaluation data. The binary classification metrics for each classifier are calculated and
        logged for each fold.

        Args:
            dictionary_data (dict): A dictionary containing real evaluation data. The key 'x_evaluation_real'
                                    holds the features for evaluation, and the key 'y_evaluation_real' holds the true labels.
        """
        # Logging the evaluation strategy
        logging.info(f"")
        logging.info(f"")
        logging.info(f"")
        logging.info(f"#################################################################################")
        logging.info(f"\tTS-TR: train on synthetic, test on real")

        arguments = getattr(self, "arguments", None)
        if getattr(arguments, "execution_mode", "normal") == "batches":
            if not getattr(self, '_labels_are_discrete', True):
                reason = "TS-TR predictive evaluation skipped because labels are not discrete."
                logging.warning("\t\t%s", reason)
                self.mark_evaluation_classifiers_not_applicable("TS-TR", self.fold_number + 1, reason)
                return

            classifier_key = getattr(
                self.arguments,
                "eval_classifier",
                getattr(self.arguments, "batch_classifier", "decision_tree_subset"),
            )
            classifier_name = get_batch_classifier_display_name(classifier_key)
            train_batches = iter_synthetic_labeled_batches(synthetic_data)
            classifier_instance, metadata = train_batch_classifier(
                classifier_key,
                train_batches,
                self._get_configured_number_classes(dictionary_data['y_evaluation_real']),
                self.arguments,
                batch_recorder=self.record_batch_processed,
            )
            evaluation_x, evaluation_y = _select_stratified_array_subset(
                dictionary_data['x_evaluation_real'],
                labels_to_1d_integer(dictionary_data['y_evaluation_real'], context="TS-TR evaluation labels"),
                getattr(self.arguments, "test_samples_per_class", None),
            )
            real_labels, predicted_labels, evaluation_time = predict_array_batches(
                classifier_instance,
                evaluation_x,
                evaluation_y,
                getattr(self.arguments, "eval_batch_size", 16384),
                batch_recorder=self.record_batch_processed,
            )
            metadata["evaluation_time_seconds"] = float(evaluation_time)
            metadata["evaluation_time"] = float(evaluation_time)
            metadata["synthetic_samples_used_by_class"] = metadata.get("train_class_counts", {})
            metadata["real_samples_used_by_class"] = _counts_by_class(real_labels) if real_labels.size else {}
            self.record_batch_classifier_metadata("TS-TR", classifier_name, self.fold_number + 1, metadata)

            if real_labels.size == 0:
                reason = "TS-TR predictive evaluation skipped because no real evaluation rows were available."
                logging.warning("\t\t%s", reason)
                self.mark_classifier_metrics_not_applicable("TS-TR", classifier_name, self.fold_number + 1, reason)
            else:
                logging.info("")
                logging.info(f"\t\tTS-TR {classifier_name}")
                self.get_task_metrics(real_labels, predicted_labels, "TS-TR", classifier_name, self.fold_number + 1)

            reason = (
                "R-S distance evaluation is skipped in batches mode to avoid materializing all synthetic rows."
            )
            logging.warning("\t\t%s", reason)
            self.mark_distance_metrics_not_applicable("R-S", self.fold_number + 1, reason)
            return

        # Initialize empty lists for labels and data
        labels, data = [], []

        # Logging the total number of generated synthetic samples to be processed
        total_samples = sum(len(samples) for samples in self.data_generated.values())
        logging.info(f"\t\tTS-TR: Total number of samples to be saved: {total_samples}")

        # Iterate through each class and its corresponding generated synthetic samples
        for label_class, generated_samples in synthetic_data.items():
            
            logging.info(f"\t\tTS-TR: Processing {len(generated_samples)} samples for label class {label_class}.")

            # Add the label (class) for each generated sample
            labels.extend([label_class] * len(generated_samples))

            # Add generated samples to the data list
            data.extend(generated_samples)

        if getattr(self, '_labels_are_discrete', True) and data:
            shuffled_data, shuffled_labels = shuffle(data, numpy.array(labels), random_state=42)
            # Train classifiers using the generated synthetic data and corresponding labels
            classifiers = self.get_trained_classifiers(shuffled_data, shuffled_labels, numpy.float32, self.get_number_columns())

            # Evaluate the classifiers on real data for each classifier instance
            for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
                # Predict the labels using the trained classifier on the real evaluation data
                label_predicted = classifier_instances.predict(dictionary_data['x_evaluation_real'])

                logging.info("")
                logging.info(f"\t\tTS-TR {classifier_name}")
                # Calculate and log metrics selected by data_type.
                self.get_task_metrics(labels_to_1d_integer(
                                            dictionary_data['y_evaluation_real'],
                                            context="TS-TR evaluation labels")[:total_samples],
                                        numpy.array(label_predicted)[:total_samples],
                                        "TS-TR", classifier_name, self.fold_number + 1)
        else:
            if not getattr(self, '_labels_are_discrete', True):
                reason = "TS-TR predictive evaluation skipped because labels are not discrete."
            else:
                reason = "TS-TR predictive evaluation skipped because no synthetic samples were generated."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TS-TR", self.fold_number + 1, reason)
            
        
        data_real = numpy.array(dictionary_data['x_training_real'])
        n_real, m_real = data_real.shape
        logging.info(f"x_real_eva size:{data_real.size} shape: {n_real}, {m_real}")

        data_synthetic = numpy.array(data)
        if data_synthetic.size == 0:
            reason = "R-S distance evaluation skipped because no synthetic samples were generated."
            logging.warning("\t\t%s", reason)
            self.mark_distance_metrics_not_applicable("R-S", self.fold_number + 1, reason)
            return
        n_synt, m_synt = data_synthetic.shape
        logging.info(f"x_synt size:{data_synthetic.size} shape: {n_synt}, {m_synt}")

        assert m_synt == m_real, \
            f"Synthetic data has {m_synt} columns != real columns {m_real}"

        if n_real > n_synt:
            data_real = data_real[:n_synt]
        elif n_real < n_synt:
            data_synthetic = data_synthetic[:n_real]
        else:
            pass

        self.get_distance_metrics(data_real, data_synthetic, "R-S", self.fold_number + 1)
