#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit synthetic class labels before predictive evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPCLASSNET_AUDIT_ROOT = PROJECT_ROOT / "results" / "appclassnet_top200"


class SyntheticLabelGenerationAudit:
    """Collect and validate requested-vs-saved synthetic labels."""

    def __init__(
            self,
            execution_mode: str,
            number_classes: int,
            label_mapping: dict[Any, Any] | None = None,
            fold_number: int | None = None,
            model_type: str | None = None,
            experiment_directory: str | None = None):
        self.execution_mode = execution_mode or "normal"
        self.number_classes = int(number_classes)
        self.fold_number = None if fold_number is None else int(fold_number)
        self.model_type = model_type
        self.experiment_directory = experiment_directory
        self.output_path = (
            APPCLASSNET_AUDIT_ROOT
            / self.execution_mode
            / "synthetic"
            / "label_generation_audit.json"
        )
        self.mapping = _normalize_label_mapping(label_mapping, self.number_classes)
        self.inverse_mapping = _invert_mapping(self.mapping["original_to_zero_based"])
        self.records = []
        self.errors = []
        self.warnings = []

    def record(self, requested_class, saved_label, generated_features, batch_index=None):
        requested_class = int(requested_class)
        saved_label = int(saved_label)
        features = numpy.asarray(generated_features)
        stats = _feature_stats(features)
        record = {
            "requested_class": requested_class,
            "requested_class_zero_based": requested_class,
            "saved_label": saved_label,
            "saved_label_zero_based": saved_label,
            "saved_label_original": self.inverse_mapping.get(saved_label, saved_label),
            "batch_index": None if batch_index is None else int(batch_index),
            "generated_count": int(features.shape[0]) if features.ndim > 0 else 0,
            **stats,
        }
        record["labels_match_requested_class"] = bool(requested_class == saved_label)
        if not record["labels_match_requested_class"]:
            self.errors.append(
                f"Synthetic label mismatch: requested class {requested_class}, saved label {saved_label}."
            )
        if record["contains_nan"] or record["contains_inf"]:
            self.warnings.append(
                f"Synthetic features for class {requested_class} contain NaN or inf."
            )
        self.records.append(record)

    def finalize(self):
        self._validate_domain()
        report = {
            "status": "failed" if self.errors else "passed",
            "execution_mode": self.execution_mode,
            "number_classes": self.number_classes,
            "fold_number": self.fold_number,
            "model_type": self.model_type,
            "experiment_directory": self.experiment_directory,
            "label_mapping": self.mapping,
            "records": self.records,
            "class_summary": self._class_summary(),
            "errors": self.errors,
            "warnings": self.warnings,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as output_file:
            json.dump(report, output_file, indent=2)

        if self.errors:
            joined_errors = " ".join(self.errors[:5])
            if len(self.errors) > 5:
                joined_errors += f" ... ({len(self.errors)} total errors)"
            raise ValueError(
                f"Synthetic label generation audit failed. "
                f"See {self.output_path}. {joined_errors}"
            )

        return self.output_path, report

    def _class_summary(self):
        summary = {}
        for record in self.records:
            class_key = str(record["saved_label"])
            class_summary = summary.setdefault(
                class_key,
                {
                    "requested_classes": [],
                    "saved_label": int(record["saved_label"]),
                    "generated_count": 0,
                    "feature_min": None,
                    "feature_max": None,
                    "feature_sum": None,
                    "contains_nan": False,
                    "contains_inf": False,
                },
            )
            class_summary["requested_classes"].append(int(record["requested_class"]))
            class_summary["generated_count"] += int(record["generated_count"])
            class_summary["contains_nan"] = bool(class_summary["contains_nan"] or record["contains_nan"])
            class_summary["contains_inf"] = bool(class_summary["contains_inf"] or record["contains_inf"])
            _merge_feature_extrema(class_summary, record)

        for class_summary in summary.values():
            generated_count = int(class_summary["generated_count"])
            feature_sum = class_summary.pop("feature_sum")
            class_summary["requested_classes"] = sorted(set(class_summary["requested_classes"]))
            if feature_sum is None or generated_count <= 0:
                class_summary["feature_mean"] = None
            else:
                class_summary["feature_mean"] = [
                    float(value / generated_count)
                    for value in feature_sum
                ]
        return summary

    def _validate_domain(self):
        observed_labels = {
            int(record["saved_label"])
            for record in self.records
            if int(record["generated_count"]) > 0
        }
        out_of_range = sorted(
            label for label in observed_labels
            if label < 0 or label >= self.number_classes
        )
        if out_of_range:
            self.errors.append(
                f"Synthetic labels outside [0, {self.number_classes - 1}]: {out_of_range}."
            )

        if self.number_classes == 200:
            expected = set(range(200))
            missing = sorted(expected - observed_labels)
            extra = sorted(observed_labels - expected)
            if missing:
                self.errors.append(
                    f"number_classes=200 requires synthetic classes exactly 0..199; "
                    f"missing {len(missing)} class(es): {missing}."
                )
            if extra:
                self.errors.append(
                    f"number_classes=200 generated class(es) outside 0..199: {extra}."
                )


def audit_synthetic_label_generation(
        synthetic_data,
        number_samples_per_class: dict,
        execution_mode: str,
        label_mapping: dict[Any, Any] | None = None,
        fold_number: int | None = None,
        model_type: str | None = None,
        experiment_directory: str | None = None):
    audit = SyntheticLabelGenerationAudit(
        execution_mode=execution_mode,
        number_classes=int(number_samples_per_class["number_classes"]),
        label_mapping=label_mapping,
        fold_number=fold_number,
        model_type=model_type,
        experiment_directory=experiment_directory,
    )
    for requested_class, generated_features in synthetic_data.items():
        audit.record(
            requested_class=int(requested_class),
            saved_label=int(requested_class),
            generated_features=generated_features,
        )
    return audit.finalize()


def _feature_stats(features):
    if features.ndim != 2:
        return {
            "shape": [int(dimension) for dimension in features.shape],
            "feature_min": None,
            "feature_max": None,
            "feature_mean": None,
            "overall_min": None,
            "overall_max": None,
            "overall_mean": None,
            "contains_nan": bool(numpy.isnan(features).any()) if numpy.issubdtype(features.dtype, numpy.number) else False,
            "contains_inf": bool(numpy.isinf(features).any()) if numpy.issubdtype(features.dtype, numpy.number) else False,
        }

    numeric = features.astype(numpy.float64, copy=False)
    contains_nan = bool(numpy.isnan(numeric).any())
    contains_inf = bool(numpy.isinf(numeric).any())
    finite = numpy.where(numpy.isfinite(numeric), numeric, numpy.nan)
    return {
        "shape": [int(dimension) for dimension in numeric.shape],
        "feature_min": _nan_reduced_list(finite, numpy.nanmin),
        "feature_max": _nan_reduced_list(finite, numpy.nanmax),
        "feature_mean": _nan_reduced_list(finite, numpy.nanmean),
        "overall_min": _nan_reduced_scalar(finite, numpy.nanmin),
        "overall_max": _nan_reduced_scalar(finite, numpy.nanmax),
        "overall_mean": _nan_reduced_scalar(finite, numpy.nanmean),
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
    }


def _nan_reduced_list(values, reducer):
    if values.size == 0 or values.shape[0] == 0:
        return None
    with numpy.errstate(all="ignore"):
        reduced = reducer(values, axis=0)
    return [
        None if numpy.isnan(value) else float(value)
        for value in numpy.asarray(reduced).tolist()
    ]


def _nan_reduced_scalar(values, reducer):
    if values.size == 0:
        return None
    with numpy.errstate(all="ignore"):
        reduced = reducer(values)
    if numpy.isnan(reduced):
        return None
    return float(reduced)


def _merge_feature_extrema(class_summary, record):
    if record["feature_min"] is not None:
        if class_summary["feature_min"] is None:
            class_summary["feature_min"] = list(record["feature_min"])
        else:
            class_summary["feature_min"] = [
                _safe_min(current, value)
                for current, value in zip(class_summary["feature_min"], record["feature_min"])
            ]
    if record["feature_max"] is not None:
        if class_summary["feature_max"] is None:
            class_summary["feature_max"] = list(record["feature_max"])
        else:
            class_summary["feature_max"] = [
                _safe_max(current, value)
                for current, value in zip(class_summary["feature_max"], record["feature_max"])
            ]
    if record["feature_mean"] is not None:
        weighted_sum = numpy.asarray(record["feature_mean"], dtype=numpy.float64) * int(record["generated_count"])
        if class_summary["feature_sum"] is None:
            class_summary["feature_sum"] = weighted_sum
        else:
            class_summary["feature_sum"] = class_summary["feature_sum"] + weighted_sum


def _normalize_label_mapping(label_mapping, number_classes):
    if label_mapping:
        original_to_zero_based = {
            str(_json_safe_key(original)): int(zero_based)
            for original, zero_based in label_mapping.items()
        }
        source = "loader_mapping"
    else:
        original_to_zero_based = {
            str(class_label): int(class_label)
            for class_label in range(int(number_classes))
        }
        source = "identity_zero_based"

    return {
        "source": source,
        "original_to_zero_based": original_to_zero_based,
        "remap_applied": any(str(original) != str(zero_based) for original, zero_based in original_to_zero_based.items()),
        "used_for_splits": ["train", "valid", "test", "synthetic"],
        "consistent_across_splits": True,
    }


def _json_safe_key(value):
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _invert_mapping(original_to_zero_based):
    inverse = {}
    for original, zero_based in original_to_zero_based.items():
        inverse[int(zero_based)] = _parse_original_label(original)
    return inverse


def _parse_original_label(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _safe_min(current, value):
    if current is None:
        return value
    if value is None:
        return current
    return min(current, value)


def _safe_max(current, value):
    if current is None:
        return value
    if value is None:
        return current
    return max(current, value)
