#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Translate sampling arguments into the legacy generation metadata contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy

from Engine.DataIO.DatasetContracts import SamplePlan
from Engine.DataIO.LabelUtils import build_class_metadata
from Engine.DataIO.LabelUtils import labels_to_1d_integer


SAMPLE_PLAN_MODES = {
    "legacy",
    "class_counts",
    "total_rows",
    "match_train_distribution",
    "balanced_per_class",
}


def build_sample_plan_from_args(arguments, y_train, number_classes: int, data_type: str = "binary") -> SamplePlan:
    mode = getattr(arguments, "sample_plan", "legacy")
    if mode not in SAMPLE_PLAN_MODES:
        raise ValueError(f"Unsupported sample_plan={mode!r}.")

    labels = labels_to_1d_integer(y_train, context="sample plan labels")
    train_metadata = build_class_metadata(labels, num_classes=number_classes, data_type=data_type)
    train_counts = train_metadata["classes"]

    explicit_legacy = _get_explicit_legacy_counts(arguments)

    if mode == "legacy":
        if explicit_legacy is not None:
            return _plan_from_counts("legacy", explicit_legacy, number_classes, data_type, source="number_samples_per_class")

        if getattr(arguments, "data_format", "csv") == "npy_xy":
            raise ValueError(
                "data_format=npy_xy requires an explicit sampling plan. Use --samples_per_class with "
                "--sample_plan balanced_per_class, --total_synthetic_rows with --sample_plan total_rows or "
                "match_train_distribution, or pass --number_samples_per_class explicitly."
            )

        return _plan_from_counts("legacy", train_counts, number_classes, data_type, source="train_distribution")

    if mode == "class_counts":
        if explicit_legacy is None:
            raise ValueError("--sample_plan class_counts requires --number_samples_per_class.")
        return _plan_from_counts("class_counts", explicit_legacy, number_classes, data_type, source="number_samples_per_class")

    if mode == "balanced_per_class":
        samples_per_class = getattr(arguments, "samples_per_class", None)
        if samples_per_class is None:
            raise ValueError("--sample_plan balanced_per_class requires --samples_per_class.")
        class_counts = {class_label: int(samples_per_class) for class_label in _class_domain(train_counts, number_classes)}
        return _plan_from_counts(
            "balanced_per_class",
            class_counts,
            number_classes,
            data_type,
            samples_per_class=int(samples_per_class),
        )

    if mode == "total_rows":
        total_rows = getattr(arguments, "total_synthetic_rows", None)
        if total_rows is None:
            raise ValueError("--sample_plan total_rows requires --total_synthetic_rows.")
        class_counts = _allocate_balanced_total(_class_domain(train_counts, number_classes), int(total_rows))
        return _plan_from_counts("total_rows", class_counts, number_classes, data_type, total_rows=int(total_rows))

    if mode == "match_train_distribution":
        total_rows = getattr(arguments, "total_synthetic_rows", None)
        if total_rows is None:
            total_rows = int(sum(train_counts.values()))
        class_counts = _allocate_proportional(train_counts, int(total_rows))
        return _plan_from_counts(
            "match_train_distribution",
            class_counts,
            number_classes,
            data_type,
            total_rows=int(total_rows),
        )

    raise ValueError(f"Unsupported sample_plan={mode!r}.")


def sample_plan_to_legacy_metadata(sample_plan: SamplePlan, data_type: str = "binary") -> dict:
    if sample_plan.class_counts is None:
        raise ValueError("SamplePlan must include class_counts before generation.")

    metadata = {
        "classes": {int(label): int(count) for label, count in sample_plan.class_counts.items()},
        "number_classes": int(sample_plan.number_classes),
        "data_type": data_type,
        "sample_plan": sample_plan.mode,
    }
    metadata.update(sample_plan.metadata)
    return metadata


def _get_explicit_legacy_counts(arguments) -> dict[int, int] | None:
    if hasattr(arguments, "_legacy_number_samples_per_class_explicit"):
        if not getattr(arguments, "_legacy_number_samples_per_class_explicit"):
            return None

    value = getattr(arguments, "number_samples_per_class", None)
    if not isinstance(value, dict):
        return None
    classes = value.get("classes")
    if not isinstance(classes, dict):
        return None
    return {int(label): int(count) for label, count in classes.items()}


def _plan_from_counts(
        mode: str,
        class_counts: dict[Any, int],
        number_classes: int,
        data_type: str,
        source: str | None = None,
        total_rows: int | None = None,
        samples_per_class: int | None = None) -> SamplePlan:
    normalized_counts = {int(label): int(count) for label, count in class_counts.items()}
    _validate_counts(normalized_counts, number_classes)
    metadata = {"data_type": data_type}
    if source is not None:
        metadata["source"] = source
    sample_plan = SamplePlan(
        mode=mode,
        class_counts=normalized_counts,
        total_rows=total_rows,
        samples_per_class=samples_per_class,
        number_classes=int(number_classes),
        metadata=metadata,
    )
    logging.info("Synthetic sample plan selected: %s", sample_plan.mode)
    logging.info("Synthetic sample plan class counts: %s", sample_plan.class_counts)
    return sample_plan


def _class_domain(train_counts: dict[int, int], number_classes: int) -> list[int]:
    if number_classes:
        return list(range(int(number_classes)))
    return sorted(int(label) for label in train_counts)


def _allocate_balanced_total(class_labels: list[int], total_rows: int) -> dict[int, int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative.")
    if not class_labels:
        raise ValueError("Cannot allocate samples without class labels.")
    base = total_rows // len(class_labels)
    remainder = total_rows % len(class_labels)
    return {
        int(label): int(base + (1 if index < remainder else 0))
        for index, label in enumerate(class_labels)
    }


def _allocate_proportional(train_counts: dict[int, int], total_rows: int) -> dict[int, int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative.")
    total_train = int(sum(train_counts.values()))
    if total_train <= 0:
        raise ValueError("Cannot match train distribution with zero training labels.")

    labels = sorted(int(label) for label in train_counts)
    raw = numpy.array([train_counts[label] * total_rows / total_train for label in labels], dtype=float)
    floors = numpy.floor(raw).astype(int)
    remainder = int(total_rows - floors.sum())

    if remainder > 0:
        fractional_order = numpy.argsort(-(raw - floors))
        for index in fractional_order[:remainder]:
            floors[index] += 1

    return {label: int(count) for label, count in zip(labels, floors)}


def _validate_counts(class_counts: dict[int, int], number_classes: int) -> None:
    if number_classes < 1:
        raise ValueError("number_classes must be positive.")
    for label, count in class_counts.items():
        if count < 0:
            raise ValueError("Sample counts must be non-negative.")
        if label < 0 or label >= number_classes:
            raise ValueError(f"Sample plan contains class {label}, outside num_classes={number_classes}.")
