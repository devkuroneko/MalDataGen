#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch-wise multiclass confusion matrix and metrics."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy

from Engine.DataIO.JsonIO import atomic_write_json


class MulticlassConfusionAccumulator:
    def __init__(self, class_labels, *, zero_division=0):
        self.class_labels = tuple(int(label) for label in class_labels)
        if not self.class_labels:
            raise ValueError("Multiclass metrics require at least one class label.")
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError(f"Class labels must be unique. Got {self.class_labels!r}.")
        self.num_classes = len(self.class_labels)
        self.zero_division = int(zero_division)
        if self.zero_division != 0:
            raise ValueError("Multiclass batch metrics only supports zero_division=0.")
        self.confusion_matrix = numpy.zeros((self.num_classes, self.num_classes), dtype=numpy.int64)
        self.batch_count = 0
        self.total_predictions = 0
        self._label_to_index = {label: index for index, label in enumerate(self.class_labels)}
        self._fast_zero_based = self.class_labels == tuple(range(self.num_classes))

    def update(self, y_true, y_pred, *, split_name="test"):
        y_true = numpy.asarray(y_true).reshape(-1)
        y_pred = numpy.asarray(y_pred).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Multiclass batch metrics length mismatch for {split_name}: "
                f"y_true={int(y_true.shape[0])} y_pred={int(y_pred.shape[0])}."
            )
        if y_true.size == 0:
            return

        true_index = self._labels_to_indices(y_true, "y_true", split_name)
        pred_index = self._labels_to_indices(y_pred, "y_pred", split_name)
        encoded = true_index * self.num_classes + pred_index
        self.confusion_matrix += numpy.bincount(
            encoded,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)
        self.batch_count += 1
        self.total_predictions += int(y_true.shape[0])

    def metrics(self, *, batch_size=None, inference_time_seconds=None):
        total = int(self.confusion_matrix.sum())
        if total <= 0:
            raise ValueError("Cannot compute multiclass metrics for an empty evaluation split.")

        true_positive = numpy.diag(self.confusion_matrix).astype(numpy.float64)
        support = self.confusion_matrix.sum(axis=1).astype(numpy.float64)
        predicted_count = self.confusion_matrix.sum(axis=0).astype(numpy.float64)
        precision = numpy.divide(
            true_positive,
            predicted_count,
            out=numpy.zeros_like(true_positive),
            where=predicted_count > 0,
        )
        recall = numpy.divide(
            true_positive,
            support,
            out=numpy.zeros_like(true_positive),
            where=support > 0,
        )
        f1 = numpy.divide(
            2 * precision * recall,
            precision + recall,
            out=numpy.zeros_like(recall),
            where=(precision + recall) > 0,
        )

        support_total = float(support.sum())
        present_classes = support > 0
        inference_time = None if inference_time_seconds is None else float(inference_time_seconds)
        throughput = None
        if inference_time is not None and inference_time > 0:
            throughput = float(total / inference_time)

        metrics = {
            "accuracy": float(true_positive.sum() / total),
            "balanced_accuracy": float(recall[present_classes].mean()) if bool(present_classes.any()) else 0.0,
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_precision": float((precision * support).sum() / support_total) if support_total > 0 else 0.0,
            "weighted_recall": float((recall * support).sum() / support_total) if support_total > 0 else 0.0,
            "weighted_f1": float((f1 * support).sum() / support_total) if support_total > 0 else 0.0,
            "support_by_class": {
                str(label): int(value)
                for label, value in zip(self.class_labels, support.astype(numpy.int64))
            },
            "predicted_count_by_class": {
                str(label): int(value)
                for label, value in zip(self.class_labels, predicted_count.astype(numpy.int64))
            },
            "classes_without_predictions": [
                int(label) for label, value in zip(self.class_labels, predicted_count) if int(value) == 0
            ],
            "classes_without_support": [
                int(label) for label, value in zip(self.class_labels, support) if int(value) == 0
            ],
            "zero_division": 0,
            "batch_count": int(self.batch_count),
            "batch_size": None if batch_size is None else int(batch_size),
            "total_predictions": int(total),
            "inference_time_seconds": inference_time,
            "throughput_samples_per_second": throughput,
            "primary_metrics": ["macro_f1", "balanced_accuracy"],
            "Accuracy": float(true_positive.sum() / total),
            "BalancedAccuracy": float(recall[present_classes].mean()) if bool(present_classes.any()) else 0.0,
            "MacroF1": float(f1.mean()),
            "WeightedF1": float((f1 * support).sum() / support_total) if support_total > 0 else 0.0,
        }
        return metrics

    def per_class_rows(self):
        true_positive = numpy.diag(self.confusion_matrix).astype(numpy.float64)
        support = self.confusion_matrix.sum(axis=1).astype(numpy.float64)
        predicted_count = self.confusion_matrix.sum(axis=0).astype(numpy.float64)
        precision = numpy.divide(
            true_positive,
            predicted_count,
            out=numpy.zeros_like(true_positive),
            where=predicted_count > 0,
        )
        recall = numpy.divide(
            true_positive,
            support,
            out=numpy.zeros_like(true_positive),
            where=support > 0,
        )
        f1 = numpy.divide(
            2 * precision * recall,
            precision + recall,
            out=numpy.zeros_like(recall),
            where=(precision + recall) > 0,
        )
        rows = []
        for index, label in enumerate(self.class_labels):
            rows.append({
                "class_id": int(label),
                "support": int(support[index]),
                "predicted_count": int(predicted_count[index]),
                "true_positive": int(true_positive[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            })
        return rows

    def save_artifacts(self, output_dir, metrics_payload, *, metrics_filename="metrics.json"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / metrics_filename
        confusion_path = output_dir / "confusion_matrix.npy"
        per_class_path = output_dir / "per_class_metrics.csv"

        numpy.save(confusion_path, self.confusion_matrix)
        self._write_per_class_csv(per_class_path)
        atomic_write_json(metrics_payload, metrics_path)
        return {
            "metrics": str(metrics_path),
            "confusion_matrix": str(confusion_path),
            "per_class_metrics": str(per_class_path),
        }

    def _labels_to_indices(self, labels, role, split_name):
        if self._fast_zero_based and numpy.issubdtype(labels.dtype, numpy.integer):
            indices = labels.astype(numpy.int64, copy=False)
            valid = (indices >= 0) & (indices < self.num_classes)
            if bool(valid.all()):
                return indices
            invalid = labels[~valid]
        else:
            flat_labels = labels.tolist()
            try:
                return numpy.fromiter(
                    (self._label_to_index[int(label)] for label in flat_labels),
                    dtype=numpy.int64,
                    count=len(flat_labels),
                )
            except KeyError:
                invalid = numpy.asarray([
                    int(label) for label in flat_labels if int(label) not in self._label_to_index
                ])
        preview = invalid[:10].tolist()
        raise ValueError(
            f"Multiclass batch metrics found unknown labels in {role} for {split_name}: "
            f"allowed={list(self.class_labels)[:10]}... invalid={preview}."
        )

    def _write_per_class_csv(self, path):
        columns = ["class_id", "support", "predicted_count", "true_positive", "precision", "recall", "f1"]
        with Path(path).open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.per_class_rows())


def evaluate_classifier_batchwise(
        classifier,
        x_values,
        y_values,
        *,
        class_labels,
        batch_size,
        transform_batch=None,
        split_name="test") -> tuple[MulticlassConfusionAccumulator, dict[str, Any]]:
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be a positive integer. Got {batch_size!r}.")
    y_values = numpy.asanyarray(y_values).reshape(-1)
    if int(x_values.shape[0]) != int(y_values.shape[0]):
        raise ValueError(
            f"Batch-wise evaluation shape mismatch for {split_name}: "
            f"X rows={int(x_values.shape[0])} y rows={int(y_values.shape[0])}."
        )
    accumulator = MulticlassConfusionAccumulator(class_labels, zero_division=0)
    inference_start = time.perf_counter()
    for start in range(0, int(y_values.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(y_values.shape[0]))
        batch_x = x_values[start:end]
        if transform_batch is not None:
            batch_x = transform_batch(batch_x)
        batch_predictions = classifier.predict(batch_x)
        accumulator.update(y_values[start:end], batch_predictions, split_name=split_name)
    inference_time_seconds = time.perf_counter() - inference_start
    stats = {
        "batch_count": int(accumulator.batch_count),
        "batch_size": int(batch_size),
        "total_predictions": int(accumulator.total_predictions),
        "inference_time_seconds": float(inference_time_seconds),
        "throughput_samples_per_second": (
            float(accumulator.total_predictions / inference_time_seconds)
            if inference_time_seconds > 0 else None
        ),
    }
    return accumulator, stats
