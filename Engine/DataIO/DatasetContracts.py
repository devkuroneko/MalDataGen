#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Internal dataset contracts for loaders that work with explicit X/y splits.

These dataclasses are intentionally independent from the legacy CSV pipeline.
They provide a small shared representation that new loaders can use before
adapting data into the existing MalDataGen fold contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    import numpy
except ImportError as error:  # pragma: no cover - mirrors project import style
    print(error)
    raise


FEATURE_TYPES = {"binary", "continuous", "mixed", "unknown"}
TARGET_TYPES = {"binary", "multiclass", "regression", "none", "auto"}
SOURCE_FORMATS = {"csv", "npy_xy", "unknown"}


def _validate_choice(value: str, allowed_values: set[str], field_name: str) -> str:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{field_name} must be one of: {allowed}. Got {value!r}.")
    return value


def _normalize_optional_labels(class_labels: Iterable[Any] | None) -> tuple[Any, ...] | None:
    if class_labels is None:
        return None
    return tuple(class_labels)


@dataclass(slots=True)
class DatasetSchema:
    """Schema metadata shared by dataset loaders.

    The schema describes the already-loaded arrays; it does not perform feature
    transformation and does not change the legacy CSV loader behavior.
    """

    feature_names: list[str]
    feature_type: str = "unknown"
    target_name: str | None = "label"
    target_type: str = "auto"
    num_classes: int | None = None
    class_labels: Iterable[Any] | None = None
    source_format: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.feature_names, list):
            self.feature_names = list(self.feature_names)

        if not self.feature_names:
            raise ValueError("feature_names must contain at least one feature name.")

        if not all(isinstance(name, str) and name for name in self.feature_names):
            raise ValueError("feature_names must contain only non-empty strings.")

        self.feature_type = _validate_choice(self.feature_type, FEATURE_TYPES, "feature_type")
        self.target_type = _validate_choice(self.target_type, TARGET_TYPES, "target_type")
        self.source_format = _validate_choice(self.source_format, SOURCE_FORMATS, "source_format")
        self.class_labels = _normalize_optional_labels(self.class_labels)

        if self.target_type == "none":
            self.target_name = None

        if self.target_type == "multiclass":
            if self.num_classes is not None:
                if not isinstance(self.num_classes, int) or self.num_classes < 2:
                    raise ValueError("num_classes must be an integer >= 2 for target_type='multiclass'.")
            if self.class_labels is not None:
                if len(self.class_labels) < 2:
                    raise ValueError("class_labels must contain at least two labels for multiclass targets.")
                if self.num_classes is not None and len(self.class_labels) != self.num_classes:
                    raise ValueError("class_labels length must match num_classes when both are provided.")
        elif self.num_classes is not None and (not isinstance(self.num_classes, int) or self.num_classes < 1):
            raise ValueError("num_classes must be a positive integer when provided.")


@dataclass(slots=True)
class SplitData:
    """A named dataset split with explicit feature matrix and optional target."""

    X: Any
    y: Any | None = None
    name: str = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")

        self.X = numpy.asarray(self.X)
        if self.X.ndim != 2:
            raise ValueError(f"SplitData.X for split {self.name!r} must be 2D. Got shape {self.X.shape}.")

        if self.y is None:
            return

        self.y = self._normalize_y(self.y)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"SplitData X/y row mismatch for split {self.name!r}: "
                f"X has {self.X.shape[0]} rows, y has {self.y.shape[0]} rows."
            )

    @staticmethod
    def _normalize_y(y: Any) -> numpy.ndarray:
        y_array = numpy.asarray(y)
        if y_array.ndim == 0:
            raise ValueError("SplitData.y must be 1D or safely convertible to 1D.")
        if y_array.ndim == 1:
            return y_array

        squeezed = numpy.squeeze(y_array)
        if squeezed.ndim == 1:
            return squeezed

        raise ValueError(f"SplitData.y must be 1D or safely convertible to 1D. Got shape {y_array.shape}.")

    @property
    def num_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])


@dataclass(slots=True)
class SamplePlan:
    """Optional generation request shared by future loaders and samplers."""

    mode: str = "legacy"
    total_rows: int | None = None
    class_counts: dict[Any, int] | None = None
    samples_per_class: int | None = None
    number_classes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_modes = {
            "legacy",
            "class_counts",
            "total_rows",
            "match_train_distribution",
            "balanced_per_class",
        }
        if self.mode not in allowed_modes:
            raise ValueError(f"SamplePlan.mode must be one of: {', '.join(sorted(allowed_modes))}.")

        if self.total_rows is not None:
            if not isinstance(self.total_rows, int) or self.total_rows < 0:
                raise ValueError("total_rows must be a non-negative integer when provided.")

        if self.samples_per_class is not None:
            if not isinstance(self.samples_per_class, int) or self.samples_per_class < 0:
                raise ValueError("samples_per_class must be a non-negative integer when provided.")

        if self.number_classes is not None:
            if not isinstance(self.number_classes, int) or self.number_classes < 1:
                raise ValueError("number_classes must be a positive integer when provided.")

        if self.class_counts is not None:
            if not isinstance(self.class_counts, dict):
                raise ValueError("class_counts must be a dictionary when provided.")
            normalized_counts = {}
            for label, count in self.class_counts.items():
                if not isinstance(count, int) or count < 0:
                    raise ValueError("class_counts values must be non-negative integers.")
                normalized_counts[label] = count
            self.class_counts = normalized_counts

        if self.total_rows is None and self.class_counts is not None:
            self.total_rows = sum(self.class_counts.values())


@dataclass(slots=True)
class DatasetBundle:
    """A dataset represented as train plus optional validation/test splits."""

    train: SplitData
    schema: DatasetSchema
    valid: SplitData | None = None
    test: SplitData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.train, SplitData):
            raise ValueError("train must be a SplitData instance.")
        if not isinstance(self.schema, DatasetSchema):
            raise ValueError("schema must be a DatasetSchema instance.")

        self._validate_split(self.train)
        if self.valid is not None:
            if not isinstance(self.valid, SplitData):
                raise ValueError("valid must be a SplitData instance when provided.")
            self._validate_split(self.valid)
        if self.test is not None:
            if not isinstance(self.test, SplitData):
                raise ValueError("test must be a SplitData instance when provided.")
            self._validate_split(self.test)

    def _validate_split(self, split: SplitData) -> None:
        if split.num_features != len(self.schema.feature_names):
            raise ValueError(
                f"Split {split.name!r} has {split.num_features} features, "
                f"but schema defines {len(self.schema.feature_names)} feature names."
            )

        if self.schema.target_type == "none":
            if split.y is not None:
                raise ValueError(f"Split {split.name!r} has y but schema target_type is 'none'.")
            return

        if split.y is None:
            raise ValueError(f"Split {split.name!r} must include y for target_type={self.schema.target_type!r}.")

        if self.schema.target_type == "multiclass" and self.schema.num_classes is not None:
            labels = numpy.asarray(split.y)
            if labels.size == 0:
                return

            if self.schema.class_labels is not None:
                valid_labels = set(self.schema.class_labels)
                unknown_labels = set(labels.tolist()) - valid_labels
                if unknown_labels:
                    raise ValueError(f"Split {split.name!r} contains labels outside class_labels.")
                return

            try:
                numeric_labels = labels.astype(int)
            except (TypeError, ValueError):
                return

            if not numpy.allclose(labels, numeric_labels):
                raise ValueError(f"Split {split.name!r} contains non-integer labels for multiclass target.")
            if numeric_labels.min() < 0 or numeric_labels.max() >= self.schema.num_classes:
                raise ValueError(
                    f"Split {split.name!r} contains labels outside [0, {self.schema.num_classes - 1}]."
                )

    @property
    def splits(self) -> dict[str, SplitData]:
        available_splits = {"train": self.train}
        if self.valid is not None:
            available_splits["valid"] = self.valid
        if self.test is not None:
            available_splits["test"] = self.test
        return available_splits
