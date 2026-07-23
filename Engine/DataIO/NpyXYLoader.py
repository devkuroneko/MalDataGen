#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Loader for datasets stored as separated X/y NumPy arrays.

This module is intentionally not wired into the legacy CSV pipeline. Callers
must instantiate it explicitly when they want to load `npy_xy` datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import numpy
except ImportError as error:  # pragma: no cover - mirrors project import style
    print(error)
    raise

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from Engine.DataIO.LabelUtils import labels_to_1d_integer


class NpyXYLoader:
    """Load train/valid/test splits from separated `.npy` X/y files."""

    APPCLASSNET_TOP200_EXPECTED_CLASSES = 200

    def __init__(
            self,
            train_x_path,
            train_y_path,
            valid_x_path=None,
            valid_y_path=None,
            test_x_path=None,
            test_y_path=None,
            mmap_mode="r",
            dtype=None,
            target_type="multiclass",
            feature_type="continuous",
            num_classes=None,
            feature_names=None,
            target_name="label",
            remap_labels_to_zero_based=False,
            source_profile="unknown",
            split_mode="provided",
            expected_num_classes=None,
            expected_num_features=None,
            validation_batch_size=65536,
            metadata=None):
        self.train_x_path = self._normalize_path(train_x_path, "train_x_path")
        self.train_y_path = self._normalize_path(train_y_path, "train_y_path")
        self.valid_x_path = self._normalize_optional_path(valid_x_path)
        self.valid_y_path = self._normalize_optional_path(valid_y_path)
        self.test_x_path = self._normalize_optional_path(test_x_path)
        self.test_y_path = self._normalize_optional_path(test_y_path)
        self.mmap_mode = mmap_mode
        self.dtype = dtype
        self.target_type = target_type
        self.feature_type = feature_type
        self.num_classes = num_classes
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.target_name = target_name
        self.remap_labels_to_zero_based = remap_labels_to_zero_based
        self.source_profile = source_profile
        self.split_mode = split_mode
        self.expected_num_classes = expected_num_classes
        self.expected_num_features = expected_num_features
        self.validation_batch_size = int(validation_batch_size)
        self.metadata = dict(metadata or {})
        self.label_mapping_original_to_zero_based = None

        self._validate_optional_pair(self.valid_x_path, self.valid_y_path, "valid")
        self._validate_optional_pair(self.test_x_path, self.test_y_path, "test")
        if self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be a positive integer.")

    @staticmethod
    def _normalize_path(path_value, field_name):
        if path_value is None:
            raise ValueError(f"{field_name} is required.")
        return Path(path_value)

    @staticmethod
    def _normalize_optional_path(path_value):
        if path_value is None:
            return None
        return Path(path_value)

    @staticmethod
    def _validate_optional_pair(x_path, y_path, split_name):
        if (x_path is None) != (y_path is None):
            raise ValueError(f"{split_name}_x_path and {split_name}_y_path must be provided together.")

    def load(self) -> DatasetBundle:
        train = self._load_split("train", self.train_x_path, self.train_y_path)
        valid = None
        test = None

        if self.valid_x_path is not None:
            valid = self._load_split("valid", self.valid_x_path, self.valid_y_path)
        if self.test_x_path is not None:
            test = self._load_split("test", self.test_x_path, self.test_y_path)

        original_labels = self._collect_labels(train, valid, test)
        self.label_mapping_original_to_zero_based = self._build_label_mapping(original_labels)

        if self.target_type == "multiclass":
            self._handle_multiclass_label_base(train, valid, test)

        self._validate_feature_widths(train, valid, test)
        feature_names = self._get_feature_names(train.num_features)
        observed_classes, class_labels, num_classes = self._get_class_metadata(train, valid, test)
        train_feature_min, train_feature_max = self._feature_range(train.X, train.name, train.x_path)
        expected_num_features = self._expected_num_features()
        if expected_num_features is not None and train.num_features != expected_num_features:
            raise ValueError(
                f"Split 'train' has {train.num_features} features; expected {expected_num_features} "
                f"for source_profile={self.source_profile!r}. X path={train.x_path} shape={train.X.shape}."
            )

        schema = DatasetSchema(
            feature_names=feature_names,
            num_features=train.num_features,
            feature_type=self.feature_type,
            target_name=self.target_name,
            target_type=self.target_type,
            num_classes=num_classes,
            classes=observed_classes,
            class_labels=class_labels,
            source_format="npy_xy",
            data_format="npy_xy",
            split_mode=self.split_mode,
            source_profile=self.source_profile,
            source_feature_range=(train_feature_min, train_feature_max),
            current_feature_range=(train_feature_min, train_feature_max),
            train_feature_min=train_feature_min,
            train_feature_max=train_feature_max,
            already_normalized=self.source_profile == "appclassnet_top200",
            normalization_range=self._source_feature_range() if self.source_profile == "appclassnet_top200" else None,
            transform_history=[],
            transform_id=None,
            feature_dtype=str(train.X.dtype),
            target_dtype=str(train.y.dtype),
            data_space="source",
        )

        metadata = {
            "mmap_mode": self.mmap_mode,
            "dtype": str(numpy.dtype(self.dtype)) if self.dtype is not None else None,
            "label_mapping_original_to_zero_based": self.label_mapping_original_to_zero_based,
            "split_mode": self.split_mode,
            "data_format": "npy_xy",
            "expected_num_classes": self._expected_num_classes(),
            "expected_num_features": expected_num_features,
            "observed_classes": [int(label) for label in observed_classes],
            "train_feature_min": train_feature_min,
            "train_feature_max": train_feature_max,
            **self.metadata,
        }

        return DatasetBundle(
            train=train,
            valid=valid,
            test=test,
            schema=schema,
            metadata=metadata,
        )

    def _source_feature_range(self):
        if self.source_profile == "appclassnet_top200":
            return -0.5, 0.5
        return None

    def _expected_num_classes(self):
        if self.expected_num_classes is not None:
            return int(self.expected_num_classes)
        if self.source_profile == "appclassnet_top200":
            return self.APPCLASSNET_TOP200_EXPECTED_CLASSES
        return self.num_classes

    def _expected_num_features(self):
        if self.expected_num_features is not None:
            return int(self.expected_num_features)
        return None

    def _load_split(self, split_name: str, x_path: Path, y_path: Path) -> SplitData:
        x_values = self._load_array(x_path)
        y_values = self._load_array(y_path)

        if self.dtype is not None:
            x_values = x_values.astype(self.dtype, copy=False)

        if x_values.ndim != 2:
            raise ValueError(
                f"Split {split_name!r} X must be a 2D matrix. Got shape={x_values.shape}; "
                f"expected shape=(n_rows, n_features). X path={x_path}."
            )

        y_values = self._normalize_y(split_name, y_values, y_path)

        if x_values.shape[0] != y_values.shape[0]:
            raise ValueError(
                f"Split {split_name!r} X/y row mismatch: X shape={x_values.shape}, y shape={y_values.shape}; "
                f"expected y rows={x_values.shape[0]}. X path={x_path}; y path={y_path}."
            )

        self._validate_finite_features(x_values, split_name, x_path)
        self._validate_finite_labels(y_values, split_name, y_path)

        if self.target_type == "multiclass":
            y_values = self._validate_multiclass_labels(split_name, y_values, y_path)

        return SplitData(
            X=x_values,
            y=y_values,
            name=split_name,
            x_path=str(x_path),
            y_path=str(y_path),
            validate_paths=True,
            dataset_id=self.source_profile,
            source_indices=None,
            data_space="source",
        )

    def _load_array(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"NumPy dataset file not found: {path}")

        try:
            return numpy.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        except ValueError:
            if self.mmap_mode is None:
                raise
            logging.warning("Could not memory-map %s; retrying numpy.load without mmap.", path)
            return numpy.load(path, mmap_mode=None, allow_pickle=False)

    @staticmethod
    def _normalize_y(split_name: str, y_values, y_path: Path) -> numpy.ndarray:
        y_array = numpy.asanyarray(y_values)
        if y_array.ndim == 1:
            return y_array
        raise ValueError(
            f"Split {split_name!r} y must be 1D. Got shape={y_array.shape}; "
            f"expected shape=(n_rows,). y path={y_path}."
        )

    def _validate_multiclass_labels(self, split_name: str, labels, y_path: Path) -> numpy.ndarray:
        labels_array = numpy.asanyarray(labels)
        if numpy.issubdtype(labels_array.dtype, numpy.integer):
            if labels_array.size and int(labels_array.min()) < 0:
                raise ValueError(
                    f"Split {split_name!r} contains negative labels. "
                    f"Observed min={int(labels_array.min())}; y path={y_path}; shape={labels_array.shape}."
                )
            return labels_array
        try:
            integer_labels = labels_to_1d_integer(labels, context=f"split {split_name!r} y path={y_path}")
        except ValueError as error:
            raise ValueError(
                f"Split {split_name!r} labels are not compatible with multiclass classification. "
                f"y path={y_path}; shape={getattr(labels, 'shape', None)}. {error}"
            ) from error
        if integer_labels.size and int(integer_labels.min()) < 0:
            raise ValueError(
                f"Split {split_name!r} contains negative labels. "
                f"Observed min={int(integer_labels.min())}; y path={y_path}; shape={integer_labels.shape}."
            )
        return integer_labels

    def _validate_finite_features(self, x_values, split_name: str, x_path: Path) -> None:
        for start in range(0, x_values.shape[0], self.validation_batch_size):
            end = min(start + self.validation_batch_size, x_values.shape[0])
            chunk = numpy.asanyarray(x_values[start:end])
            if not numpy.all(numpy.isfinite(chunk)):
                raise ValueError(
                    f"Split {split_name!r} X contains NaN or inf values in rows [{start}, {end}). "
                    f"X path={x_path}; shape={x_values.shape}; dtype={x_values.dtype}."
                )

    def _validate_finite_labels(self, y_values, split_name: str, y_path: Path) -> None:
        for start in range(0, y_values.shape[0], self.validation_batch_size):
            end = min(start + self.validation_batch_size, y_values.shape[0])
            chunk = numpy.asanyarray(y_values[start:end])
            if not numpy.all(numpy.isfinite(chunk)):
                raise ValueError(
                    f"Split {split_name!r} y contains NaN or inf values in rows [{start}, {end}). "
                    f"y path={y_path}; shape={y_values.shape}; dtype={y_values.dtype}."
                )

    def _feature_range(self, x_values, split_name: str, x_path: str | None) -> tuple[float, float]:
        feature_min = None
        feature_max = None
        for start in range(0, x_values.shape[0], self.validation_batch_size):
            end = min(start + self.validation_batch_size, x_values.shape[0])
            chunk = numpy.asanyarray(x_values[start:end])
            if chunk.size == 0:
                continue
            chunk_min = numpy.nanmin(chunk)
            chunk_max = numpy.nanmax(chunk)
            feature_min = chunk_min if feature_min is None else min(feature_min, chunk_min)
            feature_max = chunk_max if feature_max is None else max(feature_max, chunk_max)
        if feature_min is None or feature_max is None:
            raise ValueError(
                f"Split {split_name!r} X is empty; cannot compute train feature range. "
                f"X path={x_path}; shape={x_values.shape}."
            )
        return float(feature_min), float(feature_max)

    def _handle_multiclass_label_base(self, *splits) -> None:
        labels = self._collect_labels(*splits)
        if labels.size == 0:
            return

        min_label = int(labels.min())
        if min_label < 0:
            raise ValueError(f"Multiclass labels cannot be negative. Observed min={min_label}.")

        if min_label != 1:
            return

        if self.remap_labels_to_zero_based:
            logging.warning("Remapping labels from 1-based to zero-based.")
            for split in splits:
                if split is not None and split.y is not None:
                    split.y = numpy.asarray(split.y, dtype=numpy.int64) - 1
                    split.refresh_metadata()
            return

        logging.warning(
            "Labels appear to be 1-based. Pass remap_labels_to_zero_based=True to remap explicitly."
        )

    def _build_label_mapping(self, labels) -> dict[int, int] | None:
        labels = numpy.asarray(labels, dtype=numpy.int64)
        if labels.size == 0:
            return None

        unique_labels = sorted(int(label) for label in numpy.unique(labels))
        min_label = unique_labels[0]
        if min_label == 1 and self.remap_labels_to_zero_based:
            return {label: label - 1 for label in unique_labels}

        return {label: label for label in unique_labels}

    @staticmethod
    def _validate_feature_widths(*splits):
        expected_width = None
        expected_split = None
        for split in splits:
            if split is None:
                continue
            if expected_width is None:
                expected_width = split.num_features
                expected_split = split
            elif split.num_features != expected_width:
                raise ValueError(
                    f"Split {split.name!r} has {split.num_features} features; expected {expected_width} "
                    f"from split {expected_split.name!r}. X path={split.x_path}; shape={split.X.shape}; "
                    f"expected X shape=(n_rows, {expected_width})."
                )

    def _get_feature_names(self, num_features: int) -> list[str]:
        if self.feature_names is None:
            return [f"f{index}" for index in range(num_features)]

        if len(self.feature_names) != num_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} names, but data has {num_features} features."
            )
        return list(self.feature_names)

    def _get_class_metadata(self, *splits) -> tuple[tuple[Any, ...], tuple[Any, ...] | None, int | None]:
        if self.target_type != "multiclass":
            return tuple(), None, self.num_classes

        labels = self._collect_labels(*splits)
        if labels.size == 0:
            return tuple(), None, self.num_classes

        observed_classes = tuple(int(label) for label in numpy.unique(labels))
        min_label = int(labels.min())
        max_label = int(labels.max())
        labels_are_one_based = min_label == 1 and not self.remap_labels_to_zero_based

        inferred_num_classes = max_label if labels_are_one_based else max_label + 1
        expected_num_classes = self._expected_num_classes()
        num_classes = expected_num_classes if expected_num_classes is not None else inferred_num_classes

        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2 for multiclass targets.")

        if labels_are_one_based:
            if max_label > num_classes:
                raise ValueError(f"Labels contain class {max_label}, outside configured num_classes={num_classes}.")
            return observed_classes, tuple(range(1, num_classes + 1)), num_classes

        if max_label >= num_classes:
            raise ValueError(f"Labels contain class {max_label}, outside configured num_classes={num_classes}.")

        if self.source_profile == "appclassnet_top200" and num_classes != self.APPCLASSNET_TOP200_EXPECTED_CLASSES:
            raise ValueError(
                f"source_profile='appclassnet_top200' expects {self.APPCLASSNET_TOP200_EXPECTED_CLASSES} classes; "
                f"configured/inferred num_classes={num_classes}."
            )

        return observed_classes, None, num_classes

    @staticmethod
    def _collect_labels(*splits) -> numpy.ndarray:
        arrays = [numpy.asanyarray(split.y) for split in splits if split is not None and split.y is not None]
        if not arrays:
            return numpy.array([], dtype=numpy.int64)
        return numpy.concatenate(arrays).astype(numpy.int64, copy=False)
