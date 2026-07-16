#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import itertools
    import logging
    import sys

    import numpy

    from Engine.Classifiers.BatchClassifiers import get_batch_classifier_display_name
    from Engine.Classifiers.BatchClassifiers import iter_array_batches
    from Engine.Classifiers.BatchClassifiers import iter_synthetic_labeled_batches
    from Engine.Classifiers.BatchClassifiers import predict_array_batches
    from Engine.Classifiers.BatchClassifiers import train_batch_classifier
    from Engine.Classifiers.BatchClassifiers import validate_real_array_for_batch_evaluation
    from Engine.Classifiers.BatchClassifiers import validate_synthetic_batches_for_evaluation
    from Engine.DataIO.LabelUtils import labels_to_1d_integer
    from Engine.Evaluation.EvaluationRunner import EvaluationMode
    from Engine.Evaluation.EvaluationRunner import EvaluationRunner
    from Engine.Evaluation.EvaluationRunner import require_split_name
    from Engine.Evaluation.EvaluationRunner import split_to_dictionary
    from Engine.Evaluation.TsTr import _counts_by_class
    from Engine.Evaluation.TsTr import _select_stratified_array_subset
    from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
except ImportError as error:
    print(error)
    sys.exit(-1)


def _argument_value_or_fallback(arguments, argument_name, fallback_name):
    value = getattr(arguments, argument_name, None)
    if value is not None:
        return value
    return getattr(arguments, fallback_name, None)


class TrTsTr:

    def evaluation_TR_TS_TR(
            self,
            dictionary_data=None,
            synthetic_data=None,
            *,
            real_train_data=None,
            synthetic_train_data=None,
            real_test_data=None):
        logging.info("")
        logging.info("")
        logging.info("")
        logging.info("#################################################################################")
        logging.info("\tTR+TS-TR: train on real plus synthetic, test on real")

        if real_train_data is not None or real_test_data is not None:
            require_split_name("TR+TS-TR", real_train_data.name, "train")
            require_split_name("TR+TS-TR", real_test_data.name, "test")
            dictionary_data = split_to_dictionary(real_train_data=real_train_data, real_test_data=real_test_data)
        if synthetic_train_data is not None:
            synthetic_data = synthetic_train_data
        if dictionary_data is None:
            raise ValueError("TR+TS-TR requires real_train_data/real_test_data or dictionary_data.")
        if synthetic_data is None:
            raise ValueError("TR+TS-TR requires synthetic training data.")

        split_metadata = dictionary_data.get("split_metadata", {})
        logging.info(
            "TR+TS-TR routing: real_train_split=%s synthetic_split=train real_test_split=%s "
            "real_train_path=%s real_test_path=%s requested_train_samples_per_class=%s "
            "requested_test_samples_per_class=%s synthetic_train_samples_per_class=%s",
            split_metadata.get("train", {}).get("name", dictionary_data.get("training_split_name")),
            split_metadata.get("test", {}).get("name", dictionary_data.get("evaluation_split_name")),
            split_metadata.get("train", {}).get("x_path"),
            split_metadata.get("test", {}).get("x_path"),
            getattr(getattr(self, "arguments", None), "train_samples_per_class", None),
            getattr(getattr(self, "arguments", None), "test_samples_per_class", None),
            getattr(getattr(self, "arguments", None), "synthetic_train_samples_per_class", None),
        )

        arguments = getattr(self, "arguments", None)
        if getattr(arguments, "execution_mode", "normal") == "batches":
            self._evaluation_TR_TS_TR_batches(dictionary_data, synthetic_data)
            return

        runner = EvaluationRunner(self, EvaluationMode.TR_TS_TR)
        try:
            evaluation_dataset = runner.build_evaluation_dataset(dictionary_data, synthetic_data)
            runner.validate(evaluation_dataset)
            runner.save_results(evaluation_dataset, self.fold_number + 1)
        except ValueError as error:
            reason = f"TR+TS-TR predictive evaluation skipped: {error}"
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR+TS-TR", self.fold_number + 1, reason)
            return

        if not getattr(self, '_labels_are_discrete', True):
            reason = "TR+TS-TR predictive evaluation skipped because labels are not discrete."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR+TS-TR", self.fold_number + 1, reason)
            return

        classifiers = runner.fit_classifier(evaluation_dataset)
        for classifier_name, classifier_instances in zip(self._dictionary_classifiers_name, classifiers):
            label_predicted = runner.predict(classifier_instances, evaluation_dataset)
            logging.info("")
            logging.info("\t\tTR+TS-TR %s", classifier_name)
            runner.calculate_metrics(
                evaluation_dataset,
                label_predicted,
                classifier_name,
                self.fold_number + 1,
            )

    def _evaluation_TR_TS_TR_batches(self, dictionary_data, synthetic_data):
        if hasattr(synthetic_data, "manifest"):
            ScaleGuard.validate_compatible_metadata(
                ScaleGuard.describe(dictionary_data["x_training_real"], data_space="source"),
                {
                    "data_space": synthetic_data.manifest.get("data_space", "source"),
                    "transform_id": synthetic_data.manifest.get("transform_id"),
                    "transform_history": synthetic_data.manifest.get("transform_history", []),
                },
                context="TR+TS-TR",
            )
        if not getattr(self, '_labels_are_discrete', True):
            reason = "TR+TS-TR predictive evaluation skipped because labels are not discrete."
            logging.warning("\t\t%s", reason)
            self.mark_evaluation_classifiers_not_applicable("TR+TS-TR", self.fold_number + 1, reason)
            return

        classifier_key = getattr(
            self.arguments,
            "eval_classifier",
            getattr(self.arguments, "batch_classifier", "decision_tree_subset"),
        )
        classifier_name = get_batch_classifier_display_name(classifier_key)
        train_labels = labels_to_1d_integer(
            dictionary_data["y_training_real"],
            context="TR+TS-TR real training labels",
        )
        evaluation_labels = labels_to_1d_integer(
            dictionary_data["y_evaluation_real"],
            context="TR+TS-TR evaluation labels",
        )
        expected_classes = self._get_configured_number_classes(train_labels)
        validate_real_array_for_batch_evaluation(
            dictionary_data["x_training_real"],
            train_labels,
            "TR+TS-TR train",
            expected_num_classes=expected_classes,
            samples_per_class=getattr(self.arguments, "train_samples_per_class", None),
            real_class_count_policy=getattr(self.arguments, "real_class_count_policy", "strict"),
            samples_per_class_scope=getattr(self.arguments, "samples_per_class_scope", "split"),
        )
        validate_real_array_for_batch_evaluation(
            dictionary_data["x_evaluation_real"],
            evaluation_labels,
            "TR+TS-TR test",
            expected_num_classes=expected_classes,
            samples_per_class=getattr(self.arguments, "test_samples_per_class", None),
            real_class_count_policy=getattr(self.arguments, "real_class_count_policy", "strict"),
            samples_per_class_scope=getattr(self.arguments, "samples_per_class_scope", "split"),
        )
        synthetic_counts = validate_synthetic_batches_for_evaluation(
            synthetic_data,
            "TR+TS-TR",
            expected_num_classes=expected_classes,
            samples_per_class=_argument_value_or_fallback(
                self.arguments,
                "synthetic_train_samples_per_class",
                "train_samples_per_class",
            ),
            expected_num_features=dictionary_data["x_training_real"].shape[1],
            expected_data_space="source",
            expected_schema_hash=getattr(getattr(getattr(self, "_dataset_bundle", None), "schema", None), "schema_hash", None),
        )

        split_metadata = dictionary_data.get("split_metadata", {}).get("test", {})
        evaluation_x, evaluation_y, real_subset_report = _select_stratified_array_subset(
            dictionary_data["x_evaluation_real"],
            evaluation_labels,
            getattr(self.arguments, "test_samples_per_class", None),
            num_classes=expected_classes,
            split_name=split_metadata.get("name", dictionary_data.get("evaluation_split_name", "test")),
            real_class_count_policy=getattr(self.arguments, "real_class_count_policy", "strict"),
        )
        real_train_batches = iter_array_batches(
            dictionary_data["x_training_real"],
            train_labels,
            getattr(self.arguments, "batch_size", 8192),
        )
        synthetic_train_batches = iter_synthetic_labeled_batches(synthetic_data)
        classifier_instance, metadata = train_batch_classifier(
            classifier_key,
            itertools.chain(real_train_batches, synthetic_train_batches),
            expected_classes,
            self.arguments,
            batch_recorder=self.record_batch_processed,
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
        metadata["augmented_training"] = True
        metadata["real_train_samples_requested_per_class"] = getattr(self.arguments, "train_samples_per_class", None)
        metadata["synthetic_train_samples_requested_per_class"] = getattr(
            self.arguments,
            "synthetic_train_samples_per_class",
            None,
        )
        metadata["real_samples_used_by_class"] = _counts_by_class(train_labels)
        metadata["synthetic_samples_used_by_class"] = synthetic_counts
        metadata["augmented_samples_used_by_class"] = metadata.get("train_class_counts", {})
        metadata["test_real_samples_used_by_class"] = _counts_by_class(real_labels) if real_labels.size else {}
        metadata["real_class_count_policy"] = getattr(self.arguments, "real_class_count_policy", "strict")
        metadata["samples_per_class_scope"] = getattr(self.arguments, "samples_per_class_scope", "split")
        metadata["real_test_split"] = split_metadata.get("name", dictionary_data.get("evaluation_split_name"))
        metadata["requested_samples_per_class"] = real_subset_report.get("requested_samples_per_class")
        metadata["minimum_available_per_class"] = real_subset_report.get("minimum_available_per_class")
        metadata["effective_samples_per_class"] = real_subset_report.get("effective_samples_per_class")
        self.record_batch_classifier_metadata("TR+TS-TR", classifier_name, self.fold_number + 1, metadata)

        if real_labels.size == 0:
            reason = "TR+TS-TR predictive evaluation skipped because no real evaluation rows were available."
            logging.warning("\t\t%s", reason)
            self.mark_classifier_metrics_not_applicable("TR+TS-TR", classifier_name, self.fold_number + 1, reason)
            return

        logging.info("")
        logging.info("\t\tTR+TS-TR %s", classifier_name)
        self.get_task_metrics(real_labels, predicted_labels, "TR+TS-TR", classifier_name, self.fold_number + 1)
