#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared evaluation helpers for TR-TR, TR-TS and TS-TR.

The classes in this module are intentionally small. They provide a common
dataset contract and metadata recording without replacing the legacy evaluator
classes in one large refactor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy

from Engine.DataIO.LabelUtils import labels_to_1d_integer
from Engine.DataIO.DatasetContracts import validate_xy_alignment
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard


class EvaluationMode(str, Enum):
    TR_TR = "TR-TR"
    TR_TS = "TR-TS"
    TS_TR = "TS-TR"


@dataclass
class EvaluationDataset:
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    train_origin: str
    test_origin: str
    data_space: str = "source"
    transform_id: str | None = None
    class_labels: Any = None
    train_metadata: dict[str, Any] | None = None
    test_metadata: dict[str, Any] | None = None


def evaluation_protocol(arguments) -> str:
    return getattr(arguments, "evaluation_protocol", "legacy")


def uses_strict_appclassnet_protocol(arguments) -> bool:
    if evaluation_protocol(arguments) == "appclassnet_strict":
        return True
    return getattr(arguments, "source_profile", "legacy_csv") == "appclassnet_top200"


def materialize_synthetic_dict(synthetic_data):
    labels = []
    data = []
    for label_class, generated_samples in synthetic_data.items():
        labels.extend([label_class] * len(generated_samples))
        data.extend(generated_samples)
    if not data:
        return numpy.empty((0, 0), dtype=numpy.float32), numpy.asarray([], dtype=numpy.int64)
    return numpy.asarray(data, dtype=numpy.float32), labels_to_1d_integer(
        labels,
        context="synthetic evaluation labels",
    )


def class_counts(labels):
    labels = labels_to_1d_integer(labels, context="evaluation metadata labels")
    unique_labels, counts = numpy.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}


def dataset_hash(values):
    values = numpy.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    if values.size:
        digest.update(numpy.ascontiguousarray(values).view(numpy.uint8).tobytes())
    return digest.hexdigest()


class EvaluationRunner:
    def __init__(self, owner, mode: EvaluationMode):
        self.owner = owner
        self.mode = mode

    def build_evaluation_dataset(self, dictionary_data, synthetic_data=None) -> EvaluationDataset:
        strict = uses_strict_appclassnet_protocol(getattr(self.owner, "arguments", None))

        if self.mode == EvaluationMode.TR_TR:
            return EvaluationDataset(
                X_train=dictionary_data["x_training_real"],
                y_train=labels_to_1d_integer(dictionary_data["y_training_real"], context="TR-TR training labels"),
                X_test=dictionary_data["x_evaluation_real"],
                y_test=labels_to_1d_integer(dictionary_data["y_evaluation_real"], context="TR-TR evaluation labels"),
                train_origin="real:training",
                test_origin="real:evaluation",
            )

        if self.mode == EvaluationMode.TR_TS:
            synthetic_x, synthetic_y = materialize_synthetic_dict(synthetic_data)
            if strict:
                train_x = dictionary_data["x_training_real"]
                train_y = dictionary_data["y_training_real"]
                train_origin = "real:training"
            else:
                train_x = dictionary_data["x_evaluation_real"]
                train_y = dictionary_data["y_evaluation_real"]
                train_origin = "real:evaluation:legacy_tr_ts"
            return EvaluationDataset(
                X_train=train_x,
                y_train=labels_to_1d_integer(train_y, context="TR-TS training labels"),
                X_test=synthetic_x,
                y_test=synthetic_y,
                train_origin=train_origin,
                test_origin="synthetic:evaluation",
                train_metadata=ScaleGuard.describe(train_x, data_space="source"),
                test_metadata=getattr(self.owner, "_current_synthetic_metadata", None),
            )

        if self.mode == EvaluationMode.TS_TR:
            synthetic_x, synthetic_y = materialize_synthetic_dict(synthetic_data)
            return EvaluationDataset(
                X_train=synthetic_x,
                y_train=synthetic_y,
                X_test=dictionary_data["x_evaluation_real"],
                y_test=labels_to_1d_integer(dictionary_data["y_evaluation_real"], context="TS-TR evaluation labels"),
                train_origin="synthetic:training",
                test_origin="real:evaluation",
                train_metadata=getattr(self.owner, "_current_synthetic_metadata", None),
                test_metadata=getattr(self.owner, "_current_real_source_metadata", None),
            )

        raise ValueError(f"Unsupported evaluation mode: {self.mode}")

    def validate(self, dataset: EvaluationDataset):
        dataset.X_train, dataset.y_train = validate_xy_alignment(
            dataset.X_train,
            dataset.y_train,
            f"{self.mode.value} train origin={dataset.train_origin}",
        )
        dataset.X_test, dataset.y_test = validate_xy_alignment(
            dataset.X_test,
            dataset.y_test,
            f"{self.mode.value} test origin={dataset.test_origin}",
        )
        if len(dataset.X_train) == 0 or len(dataset.X_test) == 0:
            raise ValueError(f"{self.mode.value} requires non-empty train and test datasets.")

        train_x = numpy.asarray(dataset.X_train)
        test_x = numpy.asarray(dataset.X_test)
        if train_x.ndim != 2 or test_x.ndim != 2:
            raise ValueError(f"{self.mode.value} requires 2D train and test feature arrays.")
        if train_x.shape[1] != test_x.shape[1]:
            raise ValueError(
                f"{self.mode.value} feature mismatch: {train_x.shape[1]} != {test_x.shape[1]}"
            )

        if dataset.train_metadata and dataset.test_metadata:
            ScaleGuard.validate_before_evaluation(
                train_x,
                test_x,
                dataset.train_metadata,
                dataset.test_metadata,
                context=self.mode.value,
            )

    def fit_classifier(self, dataset: EvaluationDataset):
        return self.owner.get_trained_classifiers(
            dataset.X_train,
            dataset.y_train,
            numpy.float32,
            self.owner.get_number_columns(),
        )

    @staticmethod
    def predict(classifier, dataset: EvaluationDataset):
        return classifier.predict(dataset.X_test)

    def calculate_metrics(self, dataset: EvaluationDataset, predictions, classifier_name, fold):
        self.owner.get_task_metrics(
            dataset.y_test,
            numpy.asarray(predictions),
            self.mode.value,
            classifier_name,
            fold,
        )

    def save_results(self, dataset: EvaluationDataset, fold):
        fold_key = f"{fold}-Fold"
        metrics = self.owner._dictionary_metrics.setdefault("EvaluationMetadata", {})
        mode_block = metrics.setdefault(fold_key, {})
        mode_block[self.mode.value] = {
            "train_origin": dataset.train_origin,
            "test_origin": dataset.test_origin,
            "data_space": dataset.data_space,
            "transform_id": dataset.transform_id,
            "train_shape": list(numpy.asarray(dataset.X_train).shape),
            "test_shape": list(numpy.asarray(dataset.X_test).shape),
            "train_hash": dataset_hash(dataset.X_train),
            "test_hash": dataset_hash(dataset.X_test),
            "train_class_counts": class_counts(dataset.y_train),
            "test_class_counts": class_counts(dataset.y_test),
            "class_labels": (
                [int(label) for label in dataset.class_labels]
                if dataset.class_labels is not None
                else sorted({int(label) for label in numpy.unique(dataset.y_train)} |
                            {int(label) for label in numpy.unique(dataset.y_test)})
            ),
            "schema_version": "evaluation_dataset/v1",
        }
