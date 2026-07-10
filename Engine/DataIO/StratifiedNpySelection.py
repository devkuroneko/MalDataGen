#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproducible stratified index selection for NumPy label arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy


_LAST_SELECTION_REPORT = None


def select_stratified_indices_from_npy(
        y_path,
        samples_per_class,
        num_classes,
        seed,
        mmap_mode="r"):
    """Select up to samples_per_class indices per class by scanning y once.

    The function memory-maps y, scans the whole label file, and performs
    per-class reservoir sampling. Scanning the whole file avoids selecting only
    early blocks when the file is class-ordered or otherwise structured.
    """
    global _LAST_SELECTION_REPORT

    samples_per_class = _validate_positive_int(samples_per_class, "samples_per_class")
    num_classes = _validate_positive_int(num_classes, "num_classes")
    labels = _load_labels(y_path, mmap_mode=mmap_mode)
    random_generator = numpy.random.default_rng(seed)

    reservoirs = {class_id: [] for class_id in range(num_classes)}
    observed_counts = numpy.zeros(num_classes, dtype=numpy.int64)
    ignored_label_counts = {}

    for index, raw_label in enumerate(labels):
        label = _coerce_label(raw_label)
        if label is None or label < 0 or label >= num_classes:
            key = str(raw_label.item() if hasattr(raw_label, "item") else raw_label)
            ignored_label_counts[key] = int(ignored_label_counts.get(key, 0)) + 1
            continue

        observed_counts[label] += 1
        seen_count = int(observed_counts[label])
        reservoir = reservoirs[label]
        if len(reservoir) < samples_per_class:
            reservoir.append(int(index))
            continue

        replacement_index = int(random_generator.integers(0, seen_count))
        if replacement_index < samples_per_class:
            reservoir[replacement_index] = int(index)

    selected_parts = []
    selected_counts = numpy.zeros(num_classes, dtype=numpy.int64)
    for class_id in range(num_classes):
        class_indices = reservoirs[class_id]
        selected_counts[class_id] = int(len(class_indices))
        if class_indices:
            selected_parts.append(numpy.asarray(class_indices, dtype=numpy.int64))

    if selected_parts:
        indices = numpy.concatenate(selected_parts).astype(numpy.int64, copy=False)
        random_generator.shuffle(indices)
    else:
        indices = numpy.array([], dtype=numpy.int64)

    classes_below_limit = [
        int(class_id)
        for class_id, count in enumerate(selected_counts)
        if int(count) < samples_per_class
    ]
    classes_absent = [
        int(class_id)
        for class_id, count in enumerate(selected_counts)
        if int(count) == 0
    ]
    _LAST_SELECTION_REPORT = {
        "y_path": str(Path(y_path)),
        "mmap_mode": mmap_mode,
        "seed": int(seed),
        "samples_per_class_requested": int(samples_per_class),
        "num_classes": int(num_classes),
        "total_rows_scanned": int(labels.shape[0]),
        "selected_total": int(indices.shape[0]),
        "selected_counts_by_class": _counts_dict(selected_counts),
        "observed_counts_by_class": _counts_dict(observed_counts),
        "classes_below_limit": classes_below_limit,
        "classes_absent": classes_absent,
        "ignored_label_counts": ignored_label_counts,
    }
    return indices


def get_last_stratified_selection_report():
    if _LAST_SELECTION_REPORT is None:
        return None
    return dict(_LAST_SELECTION_REPORT)


def save_stratified_selection_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)
    return output_path


def build_minimum_coverage_report(selection_report, min_samples_per_class_required):
    min_required = _validate_non_negative_int(
        min_samples_per_class_required,
        "min_samples_per_class_required",
    )
    counts = {
        int(class_id): int(count)
        for class_id, count in selection_report.get("selected_counts_by_class", {}).items()
    }
    num_classes = int(selection_report["num_classes"])
    below_minimum = [
        class_id
        for class_id in range(num_classes)
        if int(counts.get(class_id, 0)) < min_required
    ]
    absent = [
        class_id
        for class_id in range(num_classes)
        if int(counts.get(class_id, 0)) == 0
    ]
    return {
        "min_samples_per_class_required": int(min_required),
        "classes_below_minimum": below_minimum,
        "classes_absent": absent,
        "passes_minimum": not below_minimum,
    }


def _load_labels(y_path, mmap_mode):
    labels = numpy.load(y_path, mmap_mode=mmap_mode, allow_pickle=False)
    labels = labels.reshape(-1)
    return labels


def _coerce_label(raw_label):
    value = raw_label.item() if hasattr(raw_label, "item") else raw_label
    try:
        integer_value = int(value)
    except (TypeError, ValueError):
        return None
    try:
        if not numpy.isclose(float(value), integer_value):
            return None
    except (TypeError, ValueError):
        return None
    return integer_value


def _counts_dict(counts):
    return {
        str(int(class_id)): int(count)
        for class_id, count in enumerate(counts)
    }


def _validate_positive_int(value, field_name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _validate_non_negative_int(value, field_name):
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return int(value)
