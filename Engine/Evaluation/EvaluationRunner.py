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
from Engine.Evaluation.ExperimentProtocol import LEGACY_TO_CANONICAL
from Engine.Evaluation.ExperimentProtocol import is_canonical_protocol_selector
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard


class EvaluationMode(str, Enum):
    TR_TR = "TR-TR"
    TR_TS = "TR-TS"
    TS_TR = "TS-TR"
    TR_TS_TR = "TR+TS-TR"


class EvaluationSplitMismatchError(ValueError):
    """Raised when a final evaluation receives the wrong real split."""


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
    if is_canonical_protocol_selector(evaluation_protocol(arguments)):
        return True
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


def minimum_class_count(labels):
    counts = class_counts(labels)
    if not counts:
        return 0
    return int(min(counts.values()))


def split_metadata_from_mapping(dictionary_data, role):
    metadata = dictionary_data.get("split_metadata", {})
    if role in metadata and metadata[role] is not None:
        return dict(metadata[role])
    name = dictionary_data.get(f"{role}_split_name")
    return {"name": name} if name is not None else {}


def split_to_dictionary(real_train_data=None, real_test_data=None):
    if real_train_data is None:
        raise ValueError("real_train_data is required.")
    test_split = real_test_data if real_test_data is not None else real_train_data
    return {
        "x_training_real": real_train_data.X,
        "y_training_real": real_train_data.y,
        "x_evaluation_real": test_split.X,
        "y_evaluation_real": test_split.y,
        "training_split_name": real_train_data.name,
        "evaluation_split_name": test_split.name,
        "real_train_split": real_train_data,
        "real_test_split": test_split,
        "split_metadata": {
            "train": split_metadata_from_split(real_train_data),
            "test": split_metadata_from_split(test_split),
            "evaluation": split_metadata_from_split(test_split),
        },
    }


def split_metadata_from_split(split):
    if split is None:
        return None
    return {
        "name": split.name,
        "x_path": getattr(split, "x_path", None),
        "y_path": getattr(split, "y_path", None),
        "dataset_id": getattr(split, "dataset_id", None),
        "num_samples": int(getattr(split, "num_samples", len(split.X))),
        "class_counts": {
            str(key): int(value)
            for key, value in getattr(split, "class_counts", {}).items()
        },
        "minimum_class_count": getattr(split, "minimum_class_count", None),
        "shape": list(numpy.asarray(split.X).shape),
    }


def require_split_name(evaluation_name, actual_name, expected_name):
    if actual_name != expected_name:
        raise EvaluationSplitMismatchError(
            f"{evaluation_name} requires real split '{expected_name}', but received '{actual_name}'."
        )


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
            train_metadata = split_metadata_from_mapping(dictionary_data, "train")
            test_metadata = split_metadata_from_mapping(dictionary_data, "test")
            if strict and train_metadata.get("name") is not None:
                require_split_name("TR-TR", train_metadata.get("name"), "train")
            if strict and test_metadata.get("name") is not None:
                require_split_name("TR-TR", test_metadata.get("name"), "test")
            return EvaluationDataset(
                X_train=dictionary_data["x_training_real"],
                y_train=labels_to_1d_integer(dictionary_data["y_training_real"], context="TR-TR training labels"),
                X_test=dictionary_data["x_evaluation_real"],
                y_test=labels_to_1d_integer(dictionary_data["y_evaluation_real"], context="TR-TR evaluation labels"),
                train_origin=f"real:{train_metadata.get('name', 'training')}",
                test_origin=f"real:{test_metadata.get('name', 'evaluation')}",
                train_metadata=train_metadata,
                test_metadata=test_metadata,
            )

        if self.mode == EvaluationMode.TR_TS:
            synthetic_x, synthetic_y = materialize_synthetic_dict(synthetic_data)
            if strict:
                train_x = dictionary_data["x_training_real"]
                train_y = dictionary_data["y_training_real"]
                train_metadata = split_metadata_from_mapping(dictionary_data, "train")
                if train_metadata.get("name") is not None:
                    require_split_name("TR-TS", train_metadata.get("name"), "train")
                train_origin = "real:train" if train_metadata.get("name") == "train" else "real:training"
            else:
                train_x = dictionary_data["x_evaluation_real"]
                train_y = dictionary_data["y_evaluation_real"]
                train_metadata = split_metadata_from_mapping(dictionary_data, "evaluation")
                train_origin = "real:evaluation:legacy_tr_ts"
            return EvaluationDataset(
                X_train=train_x,
                y_train=labels_to_1d_integer(train_y, context="TR-TS training labels"),
                X_test=synthetic_x,
                y_test=synthetic_y,
                train_origin=train_origin,
                test_origin="synthetic:evaluation",
                train_metadata={**ScaleGuard.describe(train_x, data_space="source"), **train_metadata},
                test_metadata=getattr(self.owner, "_current_synthetic_metadata", None),
            )

        if self.mode == EvaluationMode.TS_TR:
            synthetic_x, synthetic_y = materialize_synthetic_dict(synthetic_data)
            test_metadata = split_metadata_from_mapping(dictionary_data, "test")
            if strict and test_metadata.get("name") is not None:
                require_split_name("TS-TR", test_metadata.get("name"), "test")
            return EvaluationDataset(
                X_train=synthetic_x,
                y_train=synthetic_y,
                X_test=dictionary_data["x_evaluation_real"],
                y_test=labels_to_1d_integer(dictionary_data["y_evaluation_real"], context="TS-TR evaluation labels"),
                train_origin="synthetic:training",
                test_origin=f"real:{test_metadata.get('name', 'evaluation')}",
                train_metadata=getattr(self.owner, "_current_synthetic_metadata", None),
                test_metadata={**(getattr(self.owner, "_current_real_source_metadata", None) or {}), **test_metadata},
            )

        if self.mode == EvaluationMode.TR_TS_TR:
            synthetic_x, synthetic_y = materialize_synthetic_dict(synthetic_data)
            train_metadata = split_metadata_from_mapping(dictionary_data, "train")
            test_metadata = split_metadata_from_mapping(dictionary_data, "test")
            if strict and train_metadata.get("name") is not None:
                require_split_name("TR+TS-TR", train_metadata.get("name"), "train")
            if strict and test_metadata.get("name") is not None:
                require_split_name("TR+TS-TR", test_metadata.get("name"), "test")
            real_train_x = dictionary_data["x_training_real"]
            real_train_y = labels_to_1d_integer(
                dictionary_data["y_training_real"],
                context="TR+TS-TR real training labels",
            )
            return EvaluationDataset(
                X_train=numpy.vstack([
                    numpy.asarray(real_train_x, dtype=numpy.float32),
                    synthetic_x,
                ]),
                y_train=numpy.concatenate([real_train_y, synthetic_y]),
                X_test=dictionary_data["x_evaluation_real"],
                y_test=labels_to_1d_integer(
                    dictionary_data["y_evaluation_real"],
                    context="TR+TS-TR evaluation labels",
                ),
                train_origin="real:train+synthetic:training",
                test_origin=f"real:{test_metadata.get('name', 'evaluation')}",
                train_metadata={**ScaleGuard.describe(real_train_x, data_space="source"), **train_metadata},
                test_metadata={
                    **ScaleGuard.describe(dictionary_data["x_evaluation_real"], data_space="source"),
                    **(getattr(self.owner, "_current_real_source_metadata", None) or {}),
                    **test_metadata,
                },
            )

        raise ValueError(f"Unsupported evaluation mode: {self.mode}")

    def validate(self, dataset: EvaluationDataset):
        try:
            dataset.X_train, dataset.y_train = validate_xy_alignment(
                dataset.X_train,
                dataset.y_train,
                f"{self.mode.value} train origin={dataset.train_origin}",
            )
        except ValueError as error:
            prefix = (
                "train X/y length mismatch"
                if "X has" in str(error) and "y has" in str(error)
                else "train X/y alignment error"
            )
            raise ValueError(f"{prefix}: {error}") from error
        try:
            dataset.X_test, dataset.y_test = validate_xy_alignment(
                dataset.X_test,
                dataset.y_test,
                f"{self.mode.value} test origin={dataset.test_origin}",
            )
        except ValueError as error:
            prefix = (
                "test X/y length mismatch"
                if "X has" in str(error) and "y has" in str(error)
                else "test X/y alignment error"
            )
            raise ValueError(f"{prefix}: {error}") from error
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
            "protocol_id": LEGACY_TO_CANONICAL.get(self.mode.value, self.mode.value),
            "train_origin": dataset.train_origin,
            "test_origin": dataset.test_origin,
            "data_space": dataset.data_space,
            "transform_id": dataset.transform_id,
            "train_shape": list(numpy.asarray(dataset.X_train).shape),
            "test_shape": list(numpy.asarray(dataset.X_test).shape),
            "train_split_name": (dataset.train_metadata or {}).get("name"),
            "test_split_name": (dataset.test_metadata or {}).get("name"),
            "train_x_path": (dataset.train_metadata or {}).get("x_path"),
            "train_y_path": (dataset.train_metadata or {}).get("y_path"),
            "test_x_path": (dataset.test_metadata or {}).get("x_path"),
            "test_y_path": (dataset.test_metadata or {}).get("y_path"),
            "train_dataset_id": (dataset.train_metadata or {}).get("dataset_id"),
            "test_dataset_id": (dataset.test_metadata or {}).get("dataset_id"),
            "train_minimum_class_count": minimum_class_count(dataset.y_train),
            "test_minimum_class_count": minimum_class_count(dataset.y_test),
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
