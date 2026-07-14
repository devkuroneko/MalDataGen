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
            metadata=None):
        self.train_x_path = self._normalize_path(train_x_path, "train_x_.npy")
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
        self.metadata = dict(metadata or {})
        self.label_mapping_original_to_zero_based = None

        self._validate_optional_pair(self.valid_x_path, self.valid_y_path, "valid")
        self._validate_optional_pair(self.test_x_path, self.test_y_path, "test")

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
        class_labels, num_classes = self._get_class_metadata(train, valid, test)

        schema = DatasetSchema(
            feature_names=feature_names,
            feature_type=self.feature_type,
            target_name=self.target_name,
            target_type=self.target_type,
            num_classes=num_classes,
            class_labels=class_labels,
            source_format="npy_xy",
            source_profile=self.source_profile,
            source_feature_range=self._source_feature_range(),
            current_feature_range=self._source_feature_range(),
            already_normalized=self.source_profile == "appclassnet_top200",
            normalization_range=self._source_feature_range() if self.source_profile == "appclassnet_top200" else None,
            transform_history=[],
            transform_id=None,
            feature_dtype=str(train.X.dtype),
            data_space="source",
        )

        metadata = {
            "mmap_mode": self.mmap_mode,
            "dtype": str(numpy.dtype(self.dtype)) if self.dtype is not None else None,
            "label_mapping_original_to_zero_based": self.label_mapping_original_to_zero_based,
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

    def _load_split(self, split_name: str, x_path: Path, y_path: Path) -> SplitData:
        x_values = self._load_array(x_path)
        y_values = self._load_array(y_path)

        if self.dtype is not None:
            x_values = x_values.astype(self.dtype, copy=False)

        if x_values.ndim != 2:
            raise ValueError(f"{split_name} X must be a 2D matrix. Got shape {x_values.shape}.")

        y_values = self._normalize_y(split_name, y_values)

        if self.target_type == "multiclass":
            y_values = self._validate_multiclass_labels(split_name, y_values)

        return SplitData(
            X=x_values,
            y=y_values,
            name=split_name,
            x_path=str(x_path),
            y_path=str(y_path),
            dataset_id=self.source_profile,
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
    def _normalize_y(split_name: str, y_values) -> numpy.ndarray:
        y_array = numpy.asarray(y_values)
        if y_array.ndim == 1:
            return y_array

        squeezed = numpy.squeeze(y_array)
        if squeezed.ndim == 1:
            return squeezed

        raise ValueError(f"{split_name} y must be 1D or safely convertible to 1D. Got shape {y_array.shape}.")

    def _validate_multiclass_labels(self, split_name: str, labels) -> numpy.ndarray:
        return labels_to_1d_integer(labels, context=f"{split_name} y")

    def _handle_multiclass_label_base(self, *splits) -> None:
        labels = self._collect_labels(*splits)
        if labels.size == 0:
            return

        min_label = int(labels.min())
        if min_label < 0:
            raise ValueError("Multiclass labels cannot be negative.")

        if min_label != 1:
            return

        if self.remap_labels_to_zero_based:
            logging.warning("Remapping labels from 1-based to zero-based.")
            for split in splits:
                if split is not None and split.y is not None:
                    split.y = numpy.asarray(split.y, dtype=numpy.int64) - 1
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
        for split in splits:
            if split is None:
                continue
            if expected_width is None:
                expected_width = split.num_features
            elif split.num_features != expected_width:
                raise ValueError(
                    f"Split {split.name!r} has {split.num_features} features; expected {expected_width}."
                )

    def _get_feature_names(self, num_features: int) -> list[str]:
        if self.feature_names is None:
            return [f"f{index}" for index in range(num_features)]

        if len(self.feature_names) != num_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} names, but data has {num_features} features."
            )
        return list(self.feature_names)

    def _get_class_metadata(self, *splits) -> tuple[tuple[Any, ...] | None, int | None]:
        if self.target_type != "multiclass":
            return None, self.num_classes

        labels = self._collect_labels(*splits)
        if labels.size == 0:
            return None, self.num_classes

        min_label = int(labels.min())
        max_label = int(labels.max())
        labels_are_one_based = min_label == 1 and not self.remap_labels_to_zero_based

        inferred_num_classes = max_label if labels_are_one_based else max_label + 1
        num_classes = self.num_classes if self.num_classes is not None else inferred_num_classes

        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2 for multiclass targets.")

        if labels_are_one_based:
            if max_label > num_classes:
                raise ValueError(f"Labels contain class {max_label}, outside configured num_classes={num_classes}.")
            return tuple(range(1, num_classes + 1)), num_classes

        if max_label >= num_classes:
            raise ValueError(f"Labels contain class {max_label}, outside configured num_classes={num_classes}.")

        return None, num_classes

    @staticmethod
    def _collect_labels(*splits) -> numpy.ndarray:
        arrays = [numpy.asarray(split.y) for split in splits if split is not None and split.y is not None]
        if not arrays:
            return numpy.array([], dtype=numpy.int64)
        return numpy.concatenate(arrays).astype(numpy.int64, copy=False)
