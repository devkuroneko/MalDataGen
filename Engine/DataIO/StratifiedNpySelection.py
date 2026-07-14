#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproducible stratified index selection for NumPy label arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy

from Engine.DataIO.RealClassCountPolicy import coerce_label
from Engine.DataIO.RealClassCountPolicy import observed_counts_by_class
from Engine.DataIO.RealClassCountPolicy import resolve_effective_class_counts


_LAST_SELECTION_REPORT = None


def select_stratified_indices_from_npy(
        y_path,
        samples_per_class,
        num_classes,
        seed,
        mmap_mode="r",
        real_class_count_policy="available_cap",
        samples_per_class_scope="split",
        require_all_classes=False):
    """Select up to samples_per_class indices per class by scanning y once.

    The function memory-maps y, scans the whole label file, and performs
    per-class reservoir sampling. Scanning the whole file avoids selecting only
    early blocks when the file is class-ordered or otherwise structured.
    """
    global _LAST_SELECTION_REPORT

    samples_per_class = _validate_positive_int(samples_per_class, "samples_per_class")
    num_classes = _validate_positive_int(num_classes, "num_classes")
    labels = _load_labels(y_path, mmap_mode=mmap_mode)
    observed_counts, ignored_label_counts = observed_counts_by_class(labels, num_classes)
    resolution = resolve_effective_class_counts(
        observed_counts,
        samples_per_class,
        real_class_count_policy,
        Path(y_path).stem.replace("_y", ""),
        require_all_classes=require_all_classes,
    )
    effective_counts = numpy.asarray(resolution["effective_counts"], dtype=numpy.int64)
    random_generator = numpy.random.default_rng(seed)
    reservoirs = {class_id: [] for class_id in range(num_classes)}
    seen_counts = numpy.zeros(num_classes, dtype=numpy.int64)

    for index, raw_label in enumerate(labels):
        label = coerce_label(raw_label)
        if label is None or label < 0 or label >= num_classes:
            continue

        quota = int(effective_counts[label])
        if quota <= 0:
            continue

        seen_counts[label] += 1
        seen_count = int(seen_counts[label])
        reservoir = reservoirs[label]
        if len(reservoir) < quota:
            reservoir.append(int(index))
            continue

        replacement_index = int(random_generator.integers(0, seen_count))
        if replacement_index < quota:
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
        "requested_samples_per_class": resolution["requested_samples_per_class"],
        "minimum_available_per_class": resolution["minimum_available_per_class"],
        "effective_samples_per_class": resolution["effective_samples_per_class"],
        "real_class_count_policy": real_class_count_policy,
        "samples_per_class_scope": samples_per_class_scope,
        "num_classes": int(num_classes),
        "total_rows_scanned": int(labels.shape[0]),
        "selected_total": int(indices.shape[0]),
        "selected_counts_by_class": _counts_dict(selected_counts),
        "observed_counts_by_class": _counts_dict(observed_counts),
        "classes_below_limit": classes_below_limit,
        "classes_absent": classes_absent,
        "ignored_label_counts": ignored_label_counts,
        "classes_below_requested": resolution["classes_below_requested"],
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
