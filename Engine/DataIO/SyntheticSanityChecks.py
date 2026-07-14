#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sanity checks for class signal and scale in generated synthetic samples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy
from sklearn.tree import DecisionTreeClassifier

from Engine.Classifiers.BatchClassifiers import iter_synthetic_labeled_batches
from Engine.DataIO.DatasetContracts import AlignedDataset
from Engine.DataIO.DatasetContracts import validate_xy_alignment
from Engine.DataIO.LabelUtils import labels_to_1d_integer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPCLASSNET_SANITY_ROOT = PROJECT_ROOT / "results" / "appclassnet_top200"
NON_CLASS_CONDITIONAL_WARNING = "Synthetic samples appear non-class-conditional or labels may be inconsistent."
SCALE_WARNING = "More than 20% of synthetic feature values are outside the real feature range."


def run_synthetic_sanity_checks(
        real_x,
        real_y,
        synthetic_data,
        number_classes,
        execution_mode,
        arguments=None,
        fold_number=None,
        model_type=None,
        experiment_directory=None):
    if isinstance(real_x, AlignedDataset):
        aligned_real = real_x
        real_x = aligned_real.X
        real_y = aligned_real.y
    else:
        aligned_real = AlignedDataset(
            X=real_x,
            y=real_y,
            split_name="synthetic_sanity_real",
            fold_id=fold_number,
        )
        real_x = aligned_real.X
        real_y = aligned_real.y
    checker = SyntheticSanityChecker(
        real_x=real_x,
        real_y=real_y,
        synthetic_data=synthetic_data,
        number_classes=number_classes,
        execution_mode=execution_mode,
        arguments=arguments,
        fold_number=fold_number,
        model_type=model_type,
        experiment_directory=experiment_directory,
    )
    return checker.run()


class SyntheticSanityChecker:

    def __init__(
            self,
            real_x,
            real_y,
            synthetic_data,
            number_classes,
            execution_mode,
        arguments=None,
        fold_number=None,
        model_type=None,
        experiment_directory=None):
        self.real_x, aligned_real_y = validate_xy_alignment(
            real_x,
            real_y,
            f"SyntheticSanityChecks real split fold={fold_number} experiment={experiment_directory}",
        )
        self.real_y = labels_to_1d_integer(aligned_real_y, context="synthetic sanity real labels")
        self.real_x, self.real_y = validate_xy_alignment(
            self.real_x,
            self.real_y,
            f"SyntheticSanityChecks normalized real split fold={fold_number} experiment={experiment_directory}",
        )
        self.synthetic_data = synthetic_data
        self.number_classes = int(number_classes)
        self.execution_mode = execution_mode or "normal"
        self.arguments = arguments
        self.fold_number = None if fold_number is None else int(fold_number)
        self.model_type = model_type
        self.experiment_directory = experiment_directory
        self.output_path = (
            APPCLASSNET_SANITY_ROOT
            / self.execution_mode
            / "synthetic"
            / "synthetic_sanity_checks.json"
        )
        self.num_features = int(self.real_x.shape[1])
        self.expected_classes = set(range(self.number_classes))
        self.warnings = []

    def run(self):
        self._log_preflight_shapes()
        real_stats = self._real_feature_and_class_stats()
        classifier, classifier_metadata = self._train_real_decision_tree_subset(real_stats["feature_mean"])
        synthetic_stats = self._synthetic_stats_and_predictions(classifier, real_stats)
        distance_report = self._class_mean_distances(real_stats, synthetic_stats)
        classifier_report = self._classifier_report(synthetic_stats, classifier_metadata)
        distribution_report = self._distribution_report(synthetic_stats)
        scale_report = self._scale_report(real_stats, synthetic_stats)

        report = {
            "status": "completed",
            "execution_mode": self.execution_mode,
            "number_classes": self.number_classes,
            "fold_number": self.fold_number,
            "model_type": self.model_type,
            "experiment_directory": self.experiment_directory,
            "warnings": self.warnings,
            "distribution_by_class": distribution_report,
            "scale": scale_report,
            "class_mean_distance": distance_report,
            "real_to_synthetic_classifier": classifier_report,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as output_file:
            json.dump(report, output_file, indent=2)
        return self.output_path, report

    def _log_preflight_shapes(self):
        synthetic_rows = 0
        synthetic_shape = None
        synthetic_y_rows = 0
        for x_batch, y_batch in iter_synthetic_labeled_batches(self.synthetic_data):
            synthetic_rows += int(x_batch.shape[0])
            synthetic_y_rows += int(y_batch.shape[0])
            if synthetic_shape is None:
                synthetic_shape = (0, int(x_batch.shape[1])) if x_batch.ndim == 2 else x_batch.shape
        if synthetic_shape is None:
            synthetic_shape = (0, self.num_features)
        else:
            synthetic_shape = (synthetic_rows, synthetic_shape[1])
        logging.info(
            "SyntheticSanityChecks preflight: fold=%s split=%s real_x.shape=%s real_y.shape=%s "
            "synthetic_x.shape=%s synthetic_y.shape=%s",
            self.fold_number,
            "evaluation",
            self.real_x.shape,
            self.real_y.shape,
            synthetic_shape,
            (synthetic_y_rows,),
        )

    def _real_feature_and_class_stats(self):
        real_x = self.real_x.astype(numpy.float64, copy=False)
        real_feature_min = numpy.nanmin(real_x, axis=0)
        real_feature_max = numpy.nanmax(real_x, axis=0)
        real_feature_mean = numpy.nanmean(real_x, axis=0)
        real_feature_mean = numpy.where(numpy.isfinite(real_feature_mean), real_feature_mean, 0.0)
        real_counts = numpy.zeros(self.number_classes, dtype=numpy.int64)
        real_sums = numpy.zeros((self.number_classes, self.num_features), dtype=numpy.float64)

        for row, label in zip(real_x, self.real_y):
            label = int(label)
            if 0 <= label < self.number_classes:
                real_counts[label] += 1
                real_sums[label] += row

        real_means = _safe_class_means(real_sums, real_counts)
        return {
            "feature_min": real_feature_min,
            "feature_max": real_feature_max,
            "feature_mean": real_feature_mean,
            "class_counts": real_counts,
            "class_sums": real_sums,
            "class_means": real_means,
        }

    def _train_real_decision_tree_subset(self, real_feature_mean=None):
        subset_x, subset_y, subset_counts = _stratified_real_subset(
            self.real_x,
            self.real_y,
            self.number_classes,
            _subset_quota(self.arguments, self.number_classes),
        )
        if real_feature_mean is not None:
            subset_x = _sanitize_for_classifier(subset_x, real_feature_mean)
        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(subset_x, subset_y)
        metadata = {
            "classifier": "DecisionTreeClassifier",
            "random_state": 0,
            "real_subset_total_rows": int(subset_y.shape[0]),
            "real_subset_counts_by_class": _counts_dict_from_array(subset_counts),
        }
        return classifier, metadata

    def _synthetic_stats_and_predictions(self, classifier, real_stats):
        synthetic_counts = numpy.zeros(self.number_classes, dtype=numpy.int64)
        synthetic_sums = numpy.zeros((self.number_classes, self.num_features), dtype=numpy.float64)
        synthetic_min = numpy.full(self.num_features, numpy.inf, dtype=numpy.float64)
        synthetic_max = numpy.full(self.num_features, -numpy.inf, dtype=numpy.float64)
        out_of_range_counts = numpy.zeros(self.num_features, dtype=numpy.int64)
        confusion = numpy.zeros((self.number_classes, self.number_classes), dtype=numpy.int64)
        predicted_counts = numpy.zeros(self.number_classes, dtype=numpy.int64)
        extra_labels = {}
        total_rows = 0

        real_min = real_stats["feature_min"]
        real_max = real_stats["feature_max"]
        for x_batch, y_batch in iter_synthetic_labeled_batches(self.synthetic_data):
            x_batch = numpy.asarray(x_batch, dtype=numpy.float64)
            y_batch = numpy.asarray(y_batch, dtype=numpy.int64)
            if x_batch.shape[0] == 0:
                continue

            total_rows += int(x_batch.shape[0])
            synthetic_min = numpy.minimum(synthetic_min, numpy.nanmin(x_batch, axis=0))
            synthetic_max = numpy.maximum(synthetic_max, numpy.nanmax(x_batch, axis=0))
            out_of_range_counts += numpy.sum((x_batch < real_min) | (x_batch > real_max), axis=0).astype(numpy.int64)
            prediction_batch = _sanitize_for_classifier(x_batch, real_stats["feature_mean"])
            predictions = classifier.predict(prediction_batch).astype(numpy.int64)

            for label, prediction, row in zip(y_batch, predictions, x_batch):
                label = int(label)
                prediction = int(prediction)
                if 0 <= prediction < self.number_classes:
                    predicted_counts[prediction] += 1
                if 0 <= label < self.number_classes:
                    synthetic_counts[label] += 1
                    synthetic_sums[label] += row
                    if 0 <= prediction < self.number_classes:
                        confusion[label, prediction] += 1
                else:
                    extra_labels[str(label)] = int(extra_labels.get(str(label), 0)) + 1

        synthetic_means = _safe_class_means(synthetic_sums, synthetic_counts)
        if total_rows == 0:
            synthetic_min = numpy.full(self.num_features, numpy.nan, dtype=numpy.float64)
            synthetic_max = numpy.full(self.num_features, numpy.nan, dtype=numpy.float64)

        return {
            "total_rows": int(total_rows),
            "class_counts": synthetic_counts,
            "class_sums": synthetic_sums,
            "class_means": synthetic_means,
            "feature_min": synthetic_min,
            "feature_max": synthetic_max,
            "out_of_range_counts": out_of_range_counts,
            "extra_label_counts": extra_labels,
            "confusion_matrix": confusion,
            "predicted_counts": predicted_counts,
        }

    def _distribution_report(self, synthetic_stats):
        observed_classes = {
            class_id
            for class_id, count in enumerate(synthetic_stats["class_counts"])
            if int(count) > 0
        }
        missing_classes = sorted(self.expected_classes - observed_classes)
        extra_classes = sorted(int(label) for label in synthetic_stats["extra_label_counts"])
        return {
            "synthetic_counts_by_class": _counts_dict_from_array(synthetic_stats["class_counts"]),
            "missing_classes": missing_classes,
            "extra_classes": extra_classes,
            "extra_label_counts": synthetic_stats["extra_label_counts"],
        }

    def _scale_report(self, real_stats, synthetic_stats):
        total_rows = max(1, int(synthetic_stats["total_rows"]))
        out_counts = synthetic_stats["out_of_range_counts"]
        out_fraction_by_feature = out_counts.astype(numpy.float64) / float(total_rows)
        global_out_fraction = float(out_counts.sum() / max(1, total_rows * self.num_features))
        if global_out_fraction > 0.20:
            self.warnings.append(SCALE_WARNING)
        return {
            "real_feature_min": _float_list(real_stats["feature_min"]),
            "real_feature_max": _float_list(real_stats["feature_max"]),
            "synthetic_feature_min": _float_list(synthetic_stats["feature_min"]),
            "synthetic_feature_max": _float_list(synthetic_stats["feature_max"]),
            "synthetic_outside_real_range_count_by_feature": _int_list(out_counts),
            "synthetic_outside_real_range_percent_by_feature": _percent_list(out_fraction_by_feature),
            "synthetic_outside_real_range_global_percent": float(global_out_fraction * 100.0),
        }

    def _class_mean_distances(self, real_stats, synthetic_stats):
        distances = []
        for class_id in range(self.number_classes):
            real_count = int(real_stats["class_counts"][class_id])
            synthetic_count = int(synthetic_stats["class_counts"][class_id])
            if real_count == 0 or synthetic_count == 0:
                distance = None
            else:
                distance = float(numpy.linalg.norm(
                    real_stats["class_means"][class_id] - synthetic_stats["class_means"][class_id]
                ))
            distances.append({
                "class": int(class_id),
                "real_count": real_count,
                "synthetic_count": synthetic_count,
                "real_feature_mean": (
                    _float_list(real_stats["class_means"][class_id])
                    if real_count > 0 else None
                ),
                "synthetic_feature_mean": (
                    _float_list(synthetic_stats["class_means"][class_id])
                    if synthetic_count > 0 else None
                ),
                "l2_distance_between_means": distance,
            })

        worst = sorted(
            (entry for entry in distances if entry["l2_distance_between_means"] is not None),
            key=lambda entry: entry["l2_distance_between_means"],
            reverse=True,
        )
        return {
            "per_class": distances,
            "worst_classes": worst[:20],
        }

    def _classifier_report(self, synthetic_stats, classifier_metadata):
        confusion = synthetic_stats["confusion_matrix"]
        total_rows = int(confusion.sum())
        correct = int(numpy.trace(confusion))
        accuracy = float(correct / total_rows) if total_rows else 0.0
        predicted_counts = synthetic_stats["predicted_counts"]
        top_predicted = [
            {"class": int(class_id), "count": int(count), "percent": float(count / max(1, total_rows) * 100.0)}
            for class_id, count in sorted(
                enumerate(predicted_counts),
                key=lambda item: int(item[1]),
                reverse=True,
            )
            if int(count) > 0
        ][:20]
        dominant_fraction = float(predicted_counts.max() / max(1, total_rows)) if predicted_counts.size else 0.0
        if total_rows and dominant_fraction >= 0.90:
            self.warnings.append(NON_CLASS_CONDITIONAL_WARNING)
        return {
            **classifier_metadata,
            "top1_accuracy": accuracy,
            "correct_predictions": correct,
            "evaluated_synthetic_rows": total_rows,
            "confusion_matrix": confusion.astype(int).tolist(),
            "top_predicted_classes": top_predicted,
            "dominant_predicted_class_fraction": dominant_fraction,
        }


def _subset_quota(arguments, number_classes):
    if arguments is not None:
        samples_per_class = getattr(arguments, "train_samples_per_class", None)
        if samples_per_class is not None:
            return int(samples_per_class)
        subset_size = getattr(arguments, "batch_classifier_subset_size", None)
        if subset_size is not None:
            return max(1, int(subset_size) // max(1, int(number_classes)))
    return 1000


def _stratified_real_subset(real_x, real_y, number_classes, quota_per_class):
    real_x, real_y = validate_xy_alignment(real_x, real_y, "SyntheticSanityChecks stratified real subset")
    random_generator = numpy.random.default_rng(42)
    selected_indices = []
    subset_counts = numpy.zeros(int(number_classes), dtype=numpy.int64)
    real_y = numpy.asarray(real_y, dtype=numpy.int64)
    for class_id in range(int(number_classes)):
        class_indices = numpy.flatnonzero(real_y == class_id)
        if class_indices.size == 0:
            continue
        take = min(int(quota_per_class), int(class_indices.size))
        chosen = random_generator.choice(class_indices, size=take, replace=False)
        selected_indices.append(chosen)
        subset_counts[class_id] = int(take)

    if not selected_indices:
        raise ValueError("Synthetic sanity check cannot train DecisionTree: no real labels were available.")

    indices = numpy.concatenate(selected_indices).astype(numpy.int64, copy=False)
    random_generator.shuffle(indices)
    logging.info(
        "SyntheticSanityChecks stratified subset: real_x.shape=%s real_y.shape=%s max_selected_index=%s",
        real_x.shape,
        real_y.shape,
        int(indices.max()) if indices.size else None,
    )
    return (
        numpy.asarray(real_x[indices], dtype=numpy.float32),
        real_y[indices],
        subset_counts,
    )


def _safe_class_means(class_sums, class_counts):
    means = numpy.full(class_sums.shape, numpy.nan, dtype=numpy.float64)
    nonzero = class_counts > 0
    means[nonzero] = class_sums[nonzero] / class_counts[nonzero, None]
    return means


def _counts_dict_from_array(counts):
    return {
        str(int(class_id)): int(count)
        for class_id, count in enumerate(counts)
        if int(count) > 0
    }


def _float_list(values):
    result = []
    for value in numpy.asarray(values).tolist():
        if value is None or not numpy.isfinite(value):
            result.append(None)
        else:
            result.append(float(value))
    return result


def _int_list(values):
    return [int(value) for value in numpy.asarray(values).tolist()]


def _percent_list(fractions):
    return [float(value * 100.0) for value in numpy.asarray(fractions).tolist()]


def _sanitize_for_classifier(values, replacement_values):
    values = numpy.asarray(values, dtype=numpy.float32).copy()
    replacement_values = numpy.asarray(replacement_values, dtype=numpy.float32)
    if values.ndim != 2:
        return values
    nonfinite_mask = ~numpy.isfinite(values)
    if not nonfinite_mask.any():
        return values
    row_indices, column_indices = numpy.where(nonfinite_mask)
    values[row_indices, column_indices] = replacement_values[column_indices]
    return values
