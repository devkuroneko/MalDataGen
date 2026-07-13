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
    from Engine.Classifiers.BatchClassifiers import iter_array_batches
    from Engine.Classifiers.BatchClassifiers import predict_synthetic_batches
    from Engine.Classifiers.BatchClassifiers import train_batch_classifier
    from Engine.Preprocessing.FeatureTransformManager import ScaleGuard

    from sklearn.metrics.pairwise import euclidean_distances

except ImportError as error:
    print(error)
    sys.exit(-1)

class TrTs:

    def evaluation_TR_TS(self, dictionary_data, synthetic_data):
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

        arguments = getattr(self, "arguments", None)
        if getattr(arguments, "execution_mode", "normal") == "batches":
            if hasattr(synthetic_data, "manifest"):
                ScaleGuard.validate_compatible_metadata(
                    ScaleGuard.describe(dictionary_data['x_evaluation_real'], data_space="source"),
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
                dictionary_data['y_evaluation_real'],
                context="TR-TS evaluation labels",
            )
            train_batches = iter_array_batches(
                dictionary_data['x_evaluation_real'],
                train_labels,
                getattr(self.arguments, "batch_size", 8192),
            )
            classifier_instance, metadata = train_batch_classifier(
                classifier_key,
                train_batches,
                self._get_configured_number_classes(train_labels),
                self.arguments,
                batch_recorder=self.record_batch_processed,
            )
            labels, predictions, evaluation_time = predict_synthetic_batches(
                classifier_instance,
                synthetic_data,
                batch_recorder=self.record_batch_processed,
                max_samples_per_class=getattr(self.arguments, "test_samples_per_class", None),
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

        # Initialize empty lists for labels and data
        labels, data = [], []

        # Logging the total number of generated synthetic samples to be processed
        total_samples = sum(len(samples) for samples in self.data_generated.values())
        logging.info(f"\t\tTR-TS: Total number of samples to be saved: {total_samples}")

        # Iterate through each class and its corresponding generated synthetic samples
        for label_class, generated_samples in synthetic_data.items():
            
            logging.info(f"\t\tTR-TS: Processing {len(generated_samples)} samples for label class {label_class}.")

            # Add the label (class) for each generated sample
            labels.extend([label_class] * len(generated_samples))

            # Add generated samples to the data list
            data.extend(generated_samples)

        if not getattr(self, '_labels_are_discrete', True):
            reason = "TR-TS predictive evaluation skipped because labels are not discrete."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
            return

        if not data:
            reason = "TR-TS predictive evaluation skipped because no synthetic samples were generated."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR-TS", self.fold_number + 1, reason)
            return

        synthetic_array = numpy.asarray(data, dtype=numpy.float32)
        synthetic_metadata = getattr(self, "_current_synthetic_metadata", None) or {
            "data_space": "source",
            "transform_id": None,
            "transform_history": [],
        }
        real_metadata = getattr(self, "_current_real_source_metadata", None) or ScaleGuard.describe(
            dictionary_data['x_evaluation_real'],
            data_space="source",
            transform_id=None,
            transform_history=[],
        )
        ScaleGuard.validate_before_evaluation(
            dictionary_data['x_evaluation_real'],
            synthetic_array,
            real_metadata,
            synthetic_metadata,
            context="TR-TS",
        )

        # Train classifiers using the real training data and corresponding labels
        classifiers = self.get_trained_classifiers(dictionary_data['x_evaluation_real'],
                                                    labels_to_1d_integer(
                                                        dictionary_data['y_evaluation_real'],
                                                        context="TR-TS evaluation labels"),
                                                    numpy.float32, self.get_number_columns())

        # Evaluate the classifiers on synthetic data for each classifier instance
        for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
            # Predict the labels using the trained classifier on the synthetic data
            label_predicted = classifier_instances.predict(synthetic_array)
            logging.info("")
            logging.info(f"\t\tTR-TS {classifier_name}")
            # Calculate and log the binary classification metrics (such as accuracy, precision, recall, etc.)
            self.get_task_metrics(numpy.array(labels), numpy.array(label_predicted),
                                    "TR-TS", classifier_name, self.fold_number + 1)
        
        
        
    
       
 
