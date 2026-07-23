#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Internal dataset contracts for loaders that work with explicit X/y splits.

These dataclasses are intentionally independent from the legacy CSV pipeline.
They provide a small shared representation that new loaders can use before
adapting data into the existing MalDataGen fold contract.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy
except ImportError as error:  # pragma: no cover - mirrors project import style
    print(error)
    raise


FEATURE_TYPES = {"binary", "continuous", "mixed", "unknown"}
TARGET_TYPES = {"binary", "multiclass", "regression", "none", "auto"}
SOURCE_FORMATS = {"csv", "npy_xy", "unknown"}
SOURCE_PROFILES = {"legacy_csv", "appclassnet_top200", "custom", "unknown"}
DATA_SPACES = {"source", "generator", "classifier", "unknown", "transformed"}
APPCLASSNET_FEATURE_COUNT = 20


def _validate_choice(value: str, allowed_values: set[str], field_name: str) -> str:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"{field_name} must be one of: {allowed}. Got {value!r}.")
    return value


def _normalize_optional_labels(class_labels: Iterable[Any] | None) -> tuple[Any, ...] | None:
    if class_labels is None:
        return None
    return tuple(class_labels)


def _schema_hash_payload(schema: "DatasetSchema") -> dict[str, Any]:
    return {
        "feature_names": list(schema.feature_names),
        "num_features": schema.num_features,
        "feature_dtype": schema.feature_dtype,
        "target_dtype": schema.target_dtype,
        "feature_type": schema.feature_type,
        "target_type": schema.target_type,
        "num_classes": schema.num_classes,
        "classes": list(schema.classes) if schema.classes is not None else None,
        "class_labels": list(schema.class_labels) if schema.class_labels is not None else None,
        "source_format": schema.source_format,
        "data_format": schema.data_format,
        "split_mode": schema.split_mode,
        "train_feature_min": schema.train_feature_min,
        "train_feature_max": schema.train_feature_max,
        "source_profile": schema.source_profile,
        "data_space": schema.data_space,
        "transform_id": schema.transform_id,
    }


def compute_schema_hash(schema: "DatasetSchema") -> str:
    payload = _schema_hash_payload(schema)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shape_of(values):
    return getattr(values, "shape", None)


def _object_id(values):
    return hex(id(values))


def _label_summary(y_array):
    if y_array.size == 0:
        return "classes=0 min=None max=None"
    try:
        unique_labels = numpy.unique(y_array)
        min_label = unique_labels.min().item() if hasattr(unique_labels.min(), "item") else unique_labels.min()
        max_label = unique_labels.max().item() if hasattr(unique_labels.max(), "item") else unique_labels.max()
    except Exception:
        return f"classes=unknown min=unknown max=unknown"
    return (
        f"classes={int(unique_labels.shape[0])} "
        f"min={min_label} "
        f"max={max_label}"
    )


def _source_indices_summary(source_indices):
    if source_indices is None:
        return "source_indices=None"
    indices = numpy.asarray(source_indices).reshape(-1)
    if indices.size == 0:
        return "source_indices=size=0 min=None max=None"
    return (
        f"source_indices=size={int(indices.size)} "
        f"min={int(indices.min())} max={int(indices.max())}"
    )


def validate_xy_alignment(x, y, dataset_name: str, *, split=None, fold=None, source_indices=None):
    """Validate that a feature matrix and label vector describe the same rows."""
    x_array = numpy.asanyarray(x)
    y_array = numpy.asanyarray(y).reshape(-1)
    context = (
        f"{dataset_name} split={split} fold={fold} "
        f"X_id={_object_id(x)} y_id={_object_id(y)} "
        f"original_X_shape={_shape_of(x)} original_y_shape={_shape_of(y)} "
        f"{_source_indices_summary(source_indices)} {_label_summary(y_array)}"
    )
    if x_array.ndim != 2:
        raise ValueError(
            f"{context} X/y alignment error: X must be 2D; got X shape={x_array.shape}."
        )
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            f"{context} X/y alignment error: X has {x_array.shape[0]} rows, "
            f"y has {y_array.shape[0]} rows. X shape={x_array.shape}, y shape={y_array.shape}."
        )
    try:
        labels_are_finite = bool(numpy.all(numpy.isfinite(y_array)))
    except TypeError:
        labels_are_finite = False
    if not labels_are_finite:
        raise ValueError(f"{context} X/y alignment error: y contains NaN or inf labels.")
    return x_array, y_array


@dataclass(slots=True)
class AlignedDataset:
    X: Any
    y: Any
    split_name: str
    fold_id: int | None = None
    source_indices: Any | None = None
    data_space: str = "source"
    transform_id: str | None = None

    def __post_init__(self) -> None:
        self.X, self.y = validate_xy_alignment(
            self.X,
            self.y,
            "AlignedDataset",
            split=self.split_name,
            fold=self.fold_id,
            source_indices=self.source_indices,
        )


@dataclass(slots=True)
class DatasetSchema:
    """Schema metadata shared by dataset loaders.

    The schema describes the already-loaded arrays; it does not perform feature
    transformation and does not change the legacy CSV loader behavior.
    """

    feature_names: list[str]
    num_features: int | None = None
    feature_type: str = "unknown"
    target_name: str | None = "label"
    target_type: str = "auto"
    num_classes: int | None = None
    classes: Iterable[Any] | None = None
    class_labels: Iterable[Any] | None = None
    source_format: str = "unknown"
    data_format: str | None = None
    split_mode: str | None = None
    source_profile: str = "unknown"
    source_feature_range: tuple[float, float] | None = None
    current_feature_range: tuple[float, float] | None = None
    train_feature_min: float | None = None
    train_feature_max: float | None = None
    already_normalized: bool = False
    normalization_range: tuple[float, float] | None = None
    transform_history: list[dict[str, Any]] = field(default_factory=list)
    transform_id: str | None = None
    feature_dtype: str | None = None
    target_dtype: str | None = None
    data_space: str = "unknown"
    schema_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_names, list):
            self.feature_names = list(self.feature_names)

        if not self.feature_names:
            raise ValueError("feature_names must contain at least one feature name.")

        if not all(isinstance(name, str) and name for name in self.feature_names):
            raise ValueError("feature_names must contain only non-empty strings.")

        if self.num_features is None:
            self.num_features = len(self.feature_names)
        elif int(self.num_features) != len(self.feature_names):
            raise ValueError(
                f"num_features={self.num_features} must match feature_names length={len(self.feature_names)}."
            )
        else:
            self.num_features = int(self.num_features)

        if self.data_format is None:
            self.data_format = self.source_format
        elif self.source_format == "unknown":
            self.source_format = self.data_format
        elif self.data_format != self.source_format:
            raise ValueError(
                f"data_format={self.data_format!r} must match source_format={self.source_format!r}."
            )

        self.feature_type = _validate_choice(self.feature_type, FEATURE_TYPES, "feature_type")
        self.target_type = _validate_choice(self.target_type, TARGET_TYPES, "target_type")
        self.source_format = _validate_choice(self.source_format, SOURCE_FORMATS, "source_format")
        self.data_format = _validate_choice(self.data_format, SOURCE_FORMATS, "data_format")
        self.source_profile = _validate_choice(self.source_profile, SOURCE_PROFILES, "source_profile")
        self.data_space = _validate_choice(self.data_space, DATA_SPACES, "data_space")
        self.class_labels = _normalize_optional_labels(self.class_labels)
        self.classes = _normalize_optional_labels(self.classes)
        if self.classes is None and self.class_labels is not None:
            self.classes = tuple(self.class_labels)
        self.source_feature_range = self._normalize_range(self.source_feature_range, "source_feature_range")
        self.current_feature_range = self._normalize_range(self.current_feature_range, "current_feature_range")
        self.normalization_range = self._normalize_range(self.normalization_range, "normalization_range")
        self.train_feature_min = None if self.train_feature_min is None else float(self.train_feature_min)
        self.train_feature_max = None if self.train_feature_max is None else float(self.train_feature_max)
        self.transform_history = list(self.transform_history or [])

        if self.target_type == "none":
            self.target_name = None

        if self.target_type == "multiclass":
            if self.num_classes is not None:
                if not isinstance(self.num_classes, int) or self.num_classes < 2:
                    raise ValueError("num_classes must be an integer >= 2 for target_type='multiclass'.")
            if self.classes is not None:
                if len(self.classes) < 2:
                    raise ValueError("classes must contain at least two labels for multiclass targets.")
                if self.num_classes is not None and len(self.classes) > self.num_classes:
                    raise ValueError("classes length cannot exceed num_classes when both are provided.")
            if self.class_labels is not None:
                if len(self.class_labels) < 2:
                    raise ValueError("class_labels must contain at least two labels for multiclass targets.")
                if self.num_classes is not None and len(self.class_labels) != self.num_classes:
                    raise ValueError("class_labels length must match num_classes when both are provided.")
        elif self.num_classes is not None and (not isinstance(self.num_classes, int) or self.num_classes < 1):
            raise ValueError("num_classes must be a positive integer when provided.")

        if self.schema_hash is None:
            self.schema_hash = compute_schema_hash(self)

    @staticmethod
    def _normalize_range(value, field_name):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"{field_name} must contain exactly two values.")
        return float(value[0]), float(value[1])


@dataclass(slots=True)
class ClassMapping:
    """Resolved original-to-local class mapping for an experiment subset."""

    selected_original_classes: list[int]
    original_to_local_mapping: dict[int, int] = field(init=False)
    local_to_original_mapping: dict[int, int] = field(init=False)
    effective_num_classes: int = field(init=False)
    subset_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.selected_original_classes = [int(label) for label in self.selected_original_classes]
        if not self.selected_original_classes:
            raise ValueError("selected_original_classes must not be empty.")
        if len(set(self.selected_original_classes)) != len(self.selected_original_classes):
            raise ValueError("selected_original_classes contains duplicates.")
        if any(label < 0 for label in self.selected_original_classes):
            raise ValueError("selected_original_classes cannot contain negative labels.")
        self.original_to_local_mapping = {
            int(original): int(index)
            for index, original in enumerate(self.selected_original_classes)
        }
        self.local_to_original_mapping = {
            int(local): int(original)
            for original, local in self.original_to_local_mapping.items()
        }
        self.effective_num_classes = len(self.selected_original_classes)
        payload = {
            "selected_original_classes": self.selected_original_classes,
            "original_to_local_mapping": self.original_to_local_mapping,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.subset_id = hashlib.sha256(encoded).hexdigest()[:16]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "selected_original_classes": list(self.selected_original_classes),
            "original_to_local_mapping": {
                str(key): int(value)
                for key, value in self.original_to_local_mapping.items()
            },
            "local_to_original_mapping": {
                str(key): int(value)
                for key, value in self.local_to_original_mapping.items()
            },
            "effective_num_classes": int(self.effective_num_classes),
            "subset_id": self.subset_id,
        }


@dataclass(slots=True)
class SubsetManifest:
    """Manifest for a materialized class subset."""

    subset_id: str
    dataset_id: str
    selected_original_classes: list[int]
    original_to_local_mapping: dict[int, int]
    local_to_original_mapping: dict[int, int]
    effective_num_classes: int
    schema_hash: str
    data_space: str
    transform_id: str | None
    splits: dict[str, dict[str, Any]]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["original_to_local_mapping"] = {
            str(key): int(value)
            for key, value in self.original_to_local_mapping.items()
        }
        payload["local_to_original_mapping"] = {
            str(key): int(value)
            for key, value in self.local_to_original_mapping.items()
        }
        return payload

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as manifest_file:
            json.dump(self.to_json_dict(), manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        return path

@dataclass(slots=True)
class SplitData:
    """A named dataset split with explicit feature matrix and optional target."""

    X: Any | None = None
    y: Any | None = None
    name: str = "train"
    reader: Any | None = None
    y_reader: Any | None = None
    x_path: str | None = None
    y_path: str | None = None
    validate_paths: bool = False
    row_count: int | None = None
    feature_count: int | None = None
    dataset_id: str | None = None
    source_indices: Any | None = None
    data_space: str = "source"
    transform_id: str | None = None
    schema_hash: str | None = None
    subset_id: str | None = None
    label_mapping: dict[Any, Any] | None = None
    num_samples: int = 0
    class_counts: dict[Any, int] = field(default_factory=dict)
    minimum_class_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string.")

        self.data_space = _validate_choice(self.data_space, DATA_SPACES, "data_space")

        if self.X is None:
            if self.reader is None:
                raise ValueError("SplitData requires X or reader.")
            self.X = self.reader

        self.X = numpy.asanyarray(self.X)
        if self.X.ndim != 2:
            raise ValueError(f"SplitData.X for split {self.name!r} must be 2D. Got shape {self.X.shape}.")

        if self.source_indices is not None:
            self.source_indices = numpy.asarray(self.source_indices, dtype=numpy.int64).reshape(-1)
            if self.source_indices.shape[0] != self.X.shape[0]:
                raise ValueError(
                    f"SplitData source_indices for split {self.name!r} has {self.source_indices.shape[0]} rows; "
                    f"expected {self.X.shape[0]}."
                )

        if self.y is None and self.y_reader is not None:
            self.y = self.y_reader

        if self.y is None:
            self.refresh_metadata()
            return

        self.y = self._normalize_y(self.y)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"SplitData X/y row mismatch for split {self.name!r}: "
                f"X has {self.X.shape[0]} rows, y has {self.y.shape[0]} rows."
            )
        self.refresh_metadata()

    @staticmethod
    def _normalize_y(y: Any) -> numpy.ndarray:
        y_array = numpy.asanyarray(y)
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
        return int(self.row_count if self.row_count is not None else self.X.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.feature_count if self.feature_count is not None else self.X.shape[1])

    def refresh_metadata(self) -> None:
        self.row_count = int(self.X.shape[0])
        self.feature_count = int(self.X.shape[1])
        self.num_samples = int(self.row_count)
        if self.y is None:
            self.class_counts = {}
            self.minimum_class_count = None
            return
        labels = numpy.asanyarray(self.y).reshape(-1)
        if labels.size == 0:
            self.class_counts = {}
            self.minimum_class_count = 0
            return
        unique_labels, counts = numpy.unique(labels, return_counts=True)
        self.class_counts = {}
        for label, count in zip(unique_labels, counts):
            if numpy.issubdtype(unique_labels.dtype, numpy.integer):
                key = int(label)
            elif hasattr(label, "item"):
                key = label.item()
            else:
                key = label
            self.class_counts[key] = int(count)
        self.minimum_class_count = int(counts.min()) if counts.size else 0


@dataclass(slots=True)
class SamplePlan:
    """Optional generation request shared by future loaders and samplers."""

    mode: str = "legacy"
    total_rows: int | None = None
    class_counts: dict[Any, int] | None = None
    samples_per_class: int | None = None
    number_classes: int | None = None
    split_name: str | None = None
    split_size: int | None = None
    random_state: int | None = None
    insufficient_policy: str | None = None
    replacement: bool = False
    selected_indices: Any | None = None
    selection_table: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_modes = {
            "legacy",
            "class_counts",
            "total_rows",
            "match_train_distribution",
            "balanced_per_class",
            "all",
            "up_to_available",
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

        if self.split_size is not None:
            if not isinstance(self.split_size, int) or self.split_size < 0:
                raise ValueError("split_size must be a non-negative integer when provided.")

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

        if self.selection_table is None:
            self.selection_table = []


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
        validate_dataset_bundle_integrity(self)

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
            labels = numpy.asanyarray(split.y)
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


def resolve_class_mapping(
        class_subset: str | Iterable[int] | None = None,
        num_classes_subset: int | None = None,
        *,
        total_num_classes: int | None = None) -> ClassMapping | None:
    """Resolve CLI subset options before any data filtering occurs."""
    if class_subset is not None and num_classes_subset is not None:
        raise ValueError("Use either --class_subset or --num_classes_subset, not both.")
    if class_subset is None and num_classes_subset is None:
        return None
    if class_subset is not None:
        if isinstance(class_subset, str):
            selected = [
                int(value.strip())
                for value in class_subset.replace(" ", ",").split(",")
                if value.strip()
            ]
        else:
            selected = [int(value) for value in class_subset]
    else:
        if int(num_classes_subset) <= 0:
            raise ValueError("--num_classes_subset must be a positive integer.")
        selected = list(range(int(num_classes_subset)))

    if total_num_classes is not None:
        outside = [label for label in selected if label >= int(total_num_classes)]
        if outside:
            raise ValueError(
                f"Class subset contains labels outside configured class domain 0..{int(total_num_classes) - 1}: "
                f"{outside}."
            )
    return ClassMapping(selected)


def _path_shape(path, *, mmap_mode="r"):
    values = numpy.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    return values.shape, values.dtype


def _validate_split_paths(split: SplitData) -> None:
    if split.x_path is None:
        return
    x_path = Path(split.x_path)
    if not x_path.is_file():
        if not split.validate_paths:
            return
        raise FileNotFoundError(f"Split {split.name!r} x_path does not exist: {x_path}")
    x_shape, _ = _path_shape(x_path)
    if tuple(x_shape) != tuple(split.X.shape):
        raise ValueError(
            f"Split {split.name!r} x_path shape mismatch: path has {tuple(x_shape)}, "
            f"in-memory X has {tuple(split.X.shape)}."
        )
    if split.y_path is None:
        return
    y_path = Path(split.y_path)
    if not y_path.is_file():
        raise FileNotFoundError(f"Split {split.name!r} y_path does not exist: {y_path}")
    y_shape, _ = _path_shape(y_path)
    declared_y_shape = tuple(numpy.asanyarray(split.y).shape)
    if tuple(y_shape) != declared_y_shape:
        loaded_y = numpy.load(y_path, mmap_mode="r", allow_pickle=False)
        if tuple(numpy.asanyarray(loaded_y).reshape(-1).shape) != declared_y_shape:
            raise ValueError(
                f"Split {split.name!r} y_path shape mismatch: path has {tuple(y_shape)}, "
                f"in-memory y has {declared_y_shape}."
            )


def _validate_integer_label_range(split: SplitData, schema: DatasetSchema) -> None:
    if split.y is None or schema.target_type != "multiclass":
        return
    labels = numpy.asanyarray(split.y)
    if labels.size == 0:
        raise ValueError(f"Split {split.name!r} has no labels.")
    if schema.class_labels is not None:
        valid_labels = set(schema.class_labels)
        unknown_labels = set(labels.tolist()) - valid_labels
        if unknown_labels:
            raise ValueError(f"Split {split.name!r} contains labels outside class_labels: {unknown_labels}.")
        return
    try:
        integer_labels = labels.astype(numpy.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Split {split.name!r} labels must be integer encoded.") from error
    if not numpy.array_equal(labels, integer_labels):
        raise ValueError(f"Split {split.name!r} labels must be integer encoded.")
    if schema.num_classes is not None:
        if int(integer_labels.min()) < 0 or int(integer_labels.max()) >= int(schema.num_classes):
            raise ValueError(
                f"Split {split.name!r} labels must be in [0, {int(schema.num_classes) - 1}]. "
                f"Observed min={int(integer_labels.min())} max={int(integer_labels.max())}."
            )


def validate_dataset_bundle_integrity(bundle: DatasetBundle) -> None:
    """Validate that arrays, paths, indices, labels, schema and spaces agree."""
    expected_schema_hash = compute_schema_hash(bundle.schema)
    if bundle.schema.schema_hash != expected_schema_hash:
        raise ValueError("Dataset schema_hash is inconsistent with DatasetSchema contents.")

    source_index_sets: dict[tuple[str | None, str | None], dict[str, set[int]]] = {}
    for split in bundle.splits.values():
        if bundle.schema.target_type == "none":
            if split.X.ndim != 2:
                raise ValueError(f"Split {split.name!r} X must be 2D.")
        elif bundle.schema.class_labels is not None and not numpy.issubdtype(numpy.asanyarray(split.y).dtype, numpy.number):
            split.X = numpy.asanyarray(split.X)
            split.y = numpy.asanyarray(split.y).reshape(-1)
            if split.X.ndim != 2:
                raise ValueError(f"Split {split.name!r} X must be 2D.")
            if split.X.shape[0] != split.y.shape[0]:
                raise ValueError(
                    f"Split {split.name!r} X/y row mismatch: X has {split.X.shape[0]} rows, "
                    f"y has {split.y.shape[0]} rows."
                )
        else:
            validate_xy_alignment(
                split.X,
                split.y,
                "DatasetBundle integrity",
                split=split.name,
                source_indices=split.source_indices,
            )
        if split.num_features != len(bundle.schema.feature_names):
            raise ValueError(
                f"Split {split.name!r} feature_count={split.num_features} does not match schema "
                f"feature count={len(bundle.schema.feature_names)}."
            )
        if (
                bundle.schema.source_profile == "appclassnet_top200"
                and bundle.metadata.get("expected_num_features") is not None
                and split.num_features != int(bundle.metadata["expected_num_features"])):
            raise ValueError(
                f"Split {split.name!r} must have {int(bundle.metadata['expected_num_features'])} "
                f"AppClassNet features. Got {split.num_features}."
            )
        _validate_choice(split.data_space, DATA_SPACES, "data_space")
        if split.schema_hash is None:
            split.schema_hash = bundle.schema.schema_hash
        if split.schema_hash != bundle.schema.schema_hash:
            raise ValueError(
                f"Split {split.name!r} schema_hash={split.schema_hash} does not match bundle "
                f"schema_hash={bundle.schema.schema_hash}."
            )
        _validate_integer_label_range(split, bundle.schema)
        _validate_split_paths(split)
        if split.source_indices is None and split.x_path is None:
            split.source_indices = numpy.arange(split.num_rows, dtype=numpy.int64)
        if split.source_indices is not None:
            indices = numpy.asarray(split.source_indices, dtype=numpy.int64).reshape(-1)
            if indices.shape[0] != split.num_rows:
                raise ValueError(f"Split {split.name!r} source_indices length does not match row_count.")
            if indices.size and int(indices.min()) < 0:
                raise ValueError(f"Split {split.name!r} source_indices contain negative values.")
            limit = split.num_rows
            if split.x_path is not None and Path(split.x_path).is_file():
                path_shape, _ = _path_shape(Path(split.x_path))
                limit = int(path_shape[0])
            if indices.size and int(indices.max()) >= limit and split.subset_id is None:
                raise ValueError(
                    f"Split {split.name!r} source_indices contain values outside source row limit={limit}."
                )
            source_key = (split.dataset_id, None if split.subset_id else split.x_path)
            source_index_sets.setdefault(source_key, {})[split.name] = set(int(value) for value in indices.tolist())

    for source_key, split_sets in source_index_sets.items():
        if source_key[1] is None:
            continue
        names = sorted(split_sets)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1:]:
                overlap = split_sets[left_name].intersection(split_sets[right_name])
                if overlap:
                    raise ValueError(
                        f"Splits {left_name!r} and {right_name!r} share source_indices for source={source_key}: "
                        f"{sorted(overlap)[:10]}."
                    )


def apply_class_subset_to_bundle(
        bundle: DatasetBundle,
        class_mapping: ClassMapping,
        *,
        materialize_dir: str | Path | None = None,
        mmap_mode: str | None = "r") -> SubsetManifest | None:
    """Apply a resolved class subset and optionally materialize new split files."""
    split_entries = {}
    if materialize_dir is not None:
        materialize_dir = Path(materialize_dir) / class_mapping.subset_id
        materialize_dir.mkdir(parents=True, exist_ok=True)

    for split in bundle.splits.values():
        labels = numpy.asarray(split.y, dtype=numpy.int64).reshape(-1)
        original_indices = (
            numpy.asarray(split.source_indices, dtype=numpy.int64).reshape(-1)
            if split.source_indices is not None
            else numpy.arange(labels.shape[0], dtype=numpy.int64)
        )
        mask = numpy.isin(labels, numpy.asarray(class_mapping.selected_original_classes, dtype=numpy.int64))
        if not numpy.any(mask):
            raise ValueError(f"class subset removed all rows from split {split.name!r}.")
        selected_x = numpy.asarray(split.X[mask], dtype=numpy.float32)
        selected_original_y = labels[mask]
        selected_y = numpy.asarray(
            [class_mapping.original_to_local_mapping[int(label)] for label in selected_original_y],
            dtype=numpy.int64,
        )
        selected_source_indices = numpy.asarray(original_indices[mask], dtype=numpy.int64)

        if materialize_dir is not None:
            x_path = materialize_dir / f"{split.name}_x.npy"
            y_path = materialize_dir / f"{split.name}_y.npy"
            numpy.save(x_path, selected_x)
            numpy.save(y_path, selected_y)
            split.X = numpy.load(x_path, mmap_mode=mmap_mode, allow_pickle=False)
            split.y = numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)
            split.x_path = str(x_path)
            split.y_path = str(y_path)
        else:
            split.X = selected_x
            split.y = selected_y

        split.source_indices = selected_source_indices
        split.subset_id = class_mapping.subset_id
        split.label_mapping = dict(class_mapping.original_to_local_mapping)
        split.dataset_id = split.dataset_id or bundle.metadata.get("dataset_id")
        split.refresh_metadata()

        split_entries[split.name] = {
            "x_path": split.x_path,
            "y_path": split.y_path,
            "row_count": int(split.row_count),
            "feature_count": int(split.feature_count),
            "class_counts": {str(key): int(value) for key, value in split.class_counts.items()},
            "source_indices_min": int(selected_source_indices.min()) if selected_source_indices.size else None,
            "source_indices_max": int(selected_source_indices.max()) if selected_source_indices.size else None,
        }

    bundle.schema.num_classes = int(class_mapping.effective_num_classes)
    bundle.schema.class_labels = tuple(range(class_mapping.effective_num_classes))
    bundle.schema.schema_hash = None
    bundle.schema.schema_hash = compute_schema_hash(bundle.schema)
    for split in bundle.splits.values():
        split.schema_hash = bundle.schema.schema_hash
        split.dataset_id = split.dataset_id or bundle.schema.source_profile

    bundle.metadata["subset_id"] = class_mapping.subset_id
    bundle.metadata["selected_original_classes"] = list(class_mapping.selected_original_classes)
    bundle.metadata["original_to_local_mapping"] = {
        str(key): int(value) for key, value in class_mapping.original_to_local_mapping.items()
    }
    bundle.metadata["local_to_original_mapping"] = {
        str(key): int(value) for key, value in class_mapping.local_to_original_mapping.items()
    }
    bundle.metadata["effective_num_classes"] = int(class_mapping.effective_num_classes)

    validate_dataset_bundle_integrity(bundle)

    if materialize_dir is None:
        return None

    manifest = SubsetManifest(
        subset_id=class_mapping.subset_id,
        dataset_id=bundle.metadata.get("dataset_id", bundle.schema.source_profile),
        selected_original_classes=list(class_mapping.selected_original_classes),
        original_to_local_mapping=dict(class_mapping.original_to_local_mapping),
        local_to_original_mapping=dict(class_mapping.local_to_original_mapping),
        effective_num_classes=int(class_mapping.effective_num_classes),
        schema_hash=bundle.schema.schema_hash,
        data_space=bundle.schema.data_space,
        transform_id=bundle.schema.transform_id,
        splits=split_entries,
    )
    manifest.save(Path(materialize_dir) / "subset_manifest.json")
    return manifest


def materialize_npy_class_subset(
        raw_root,
        output_root,
        class_mapping: ClassMapping,
        *,
        split_names: Iterable[str] = ("train", "valid", "test"),
        dataset_id: str = "appclassnet_top200",
        expected_num_features: int = APPCLASSNET_FEATURE_COUNT,
        mmap_mode: str | None = "r") -> tuple[Path, Path]:
    """Materialize train/valid/test .npy files for a class subset."""
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    subset_root = output_root / class_mapping.subset_id
    manifest_path = subset_root / "subset_manifest.json"
    expected_files = [
        subset_root / f"{split_name}_{axis}.npy"
        for split_name in split_names
        for axis in ("x", "y")
    ]
    if manifest_path.is_file() and all(path.is_file() for path in expected_files):
        return subset_root, manifest_path

    feature_names = [f"f{index}" for index in range(expected_num_features)]
    schema = DatasetSchema(
        feature_names=feature_names,
        feature_type="continuous",
        target_name="label",
        target_type="multiclass",
        num_classes=200 if dataset_id == "appclassnet_top200" else None,
        source_format="npy_xy",
        source_profile=dataset_id if dataset_id in SOURCE_PROFILES else "custom",
        source_feature_range=(-0.5, 0.5) if dataset_id == "appclassnet_top200" else None,
        current_feature_range=(-0.5, 0.5) if dataset_id == "appclassnet_top200" else None,
        already_normalized=dataset_id == "appclassnet_top200",
        normalization_range=(-0.5, 0.5) if dataset_id == "appclassnet_top200" else None,
        feature_dtype="float32",
        data_space="source",
    )
    splits = {}
    for split_name in split_names:
        x_path = raw_root / f"{split_name}_x.npy"
        y_path = raw_root / f"{split_name}_y.npy"
        x_values = numpy.load(x_path, mmap_mode=mmap_mode, allow_pickle=False)
        y_values = numpy.asarray(numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)).reshape(-1)
        splits[split_name] = SplitData(
            X=x_values,
            y=y_values,
            name=split_name,
            x_path=str(x_path),
            y_path=str(y_path),
            validate_paths=True,
            dataset_id=dataset_id,
            source_indices=numpy.arange(y_values.shape[0], dtype=numpy.int64),
            data_space="source",
            schema_hash=schema.schema_hash,
        )
    bundle = DatasetBundle(
        train=splits["train"],
        valid=splits.get("valid"),
        test=splits.get("test"),
        schema=schema,
        metadata={"dataset_id": dataset_id},
    )
    apply_class_subset_to_bundle(bundle, class_mapping, materialize_dir=output_root, mmap_mode=mmap_mode)
    return subset_root, manifest_path
