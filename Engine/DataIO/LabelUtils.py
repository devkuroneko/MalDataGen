#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for class-label validation and encoding."""

from __future__ import annotations

import logging
from typing import Any

import numpy


def labels_to_1d_integer(labels: Any, context: str = "labels") -> numpy.ndarray:
    labels_array = numpy.asarray(labels)
    if labels_array.ndim == 0:
        raise ValueError(f"{context} must be a 1D integer vector.")

    if labels_array.ndim != 1:
        labels_array = numpy.squeeze(labels_array)
        if labels_array.ndim == 0 and numpy.asarray(labels).size == 1:
            labels_array = labels_array.reshape(1)
        if labels_array.ndim != 1:
            raise ValueError(f"{context} must be a 1D integer vector. Got shape {labels_array.shape}.")

    try:
        integer_labels = labels_array.astype(numpy.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must contain integer labels.") from error

    if not numpy.allclose(labels_array, integer_labels):
        raise ValueError(f"{context} must contain integer labels.")

    return integer_labels


def infer_num_classes(labels: Any, num_classes: int | None = None, context: str = "labels") -> int:
    integer_labels = labels_to_1d_integer(labels, context=context)
    if integer_labels.size == 0:
        if num_classes is None:
            raise ValueError(f"{context} is empty and num_classes was not provided.")
        return _validate_num_classes(num_classes)

    min_label = int(integer_labels.min())
    max_label = int(integer_labels.max())
    if min_label < 0:
        raise ValueError(f"{context} contains negative labels.")

    inferred = max_label + 1
    if num_classes is None:
        return _validate_num_classes(inferred)

    num_classes = _validate_num_classes(num_classes)
    if max_label >= num_classes:
        raise ValueError(
            f"{context} contains label {max_label}, but num_classes={num_classes}. "
            "Use zero-based labels or pass --remap_labels_to_zero_based when supported."
        )
    return num_classes


def build_class_metadata(labels: Any, num_classes: int | None = None, data_type: str | None = None) -> dict:
    integer_labels = labels_to_1d_integer(labels)
    unique_classes, counts = numpy.unique(integer_labels, return_counts=True)
    metadata = {
        "classes": {int(label): int(count) for label, count in zip(unique_classes, counts)},
        "number_classes": infer_num_classes(integer_labels, num_classes=num_classes),
    }
    if data_type is not None:
        metadata["data_type"] = data_type
    return metadata


def validate_zero_based_labels(labels: Any, num_classes: int | None = None, context: str = "labels") -> numpy.ndarray:
    integer_labels = labels_to_1d_integer(labels, context=context)
    if integer_labels.size == 0:
        return integer_labels

    min_label = int(integer_labels.min())
    max_label = int(integer_labels.max())
    if min_label < 0:
        raise ValueError(f"{context} contains negative labels.")

    if min_label > 0:
        logging.warning(
            "%s labels are not zero-based: min=%d, max=%d. Use --remap_labels_to_zero_based when supported.",
            context,
            min_label,
            max_label,
        )

    if num_classes is not None and max_label >= int(num_classes):
        raise ValueError(f"{context} contains label {max_label}, but num_classes={num_classes}.")

    return integer_labels


def one_hot_encode_labels(labels: Any, num_classes: int, context: str = "labels") -> numpy.ndarray:
    integer_labels = validate_zero_based_labels(labels, num_classes=num_classes, context=context)
    encoded = numpy.zeros((integer_labels.shape[0], int(num_classes)), dtype=numpy.float32)
    if integer_labels.size:
        encoded[numpy.arange(integer_labels.shape[0]), integer_labels] = 1.0
    return encoded


def to_one_hot_batch(labels: Any, num_classes: int, dtype=numpy.float32) -> numpy.ndarray:
    integer_labels = validate_zero_based_labels(labels, num_classes=num_classes, context="one-hot batch labels")
    num_classes = _validate_num_classes(int(num_classes))

    if integer_labels.size:
        min_label = int(integer_labels.min())
        max_label = int(integer_labels.max())
        if min_label < 0 or max_label >= num_classes:
            raise ValueError(
                f"one-hot batch labels must be in [0, {num_classes - 1}]. "
                f"Got min={min_label}, max={max_label}."
            )

    logging.info(
        "Using batch-wise one-hot encoding: labels_shape=%s num_classes=%d",
        integer_labels.shape,
        num_classes,
    )
    encoded = numpy.zeros((integer_labels.shape[0], num_classes), dtype=dtype)
    if integer_labels.size:
        encoded[numpy.arange(integer_labels.shape[0]), integer_labels] = 1.0
    logging.info("Using batch-wise one-hot encoding: output_shape=%s dtype=%s", encoded.shape, encoded.dtype)
    return encoded


def _validate_num_classes(num_classes: int) -> int:
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    return num_classes
