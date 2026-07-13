#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centralized feature transforms for tabular pipelines.

Loaders must only load data. This module owns scaler fitting, application,
inverse transforms, manifests and scale validation metadata.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import joblib
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


SOURCE_PROFILES = {"legacy_csv", "appclassnet_top200", "custom"}
TRANSFORMS = {"preserve", "auto", "minmax", "standard"}
EVALUATION_SPACES = {"source", "transformed"}
DATA_SPACES = {"source", "generator", "classifier", "transformed", "unknown"}


class FeatureTransformError(ValueError):
    """Base error for preprocessing contract violations."""


class PreprocessingSpaceMismatchError(FeatureTransformError):
    """Raised when real and synthetic data are compared in different spaces."""


class DoubleTransformError(FeatureTransformError):
    """Raised when an equivalent transform would be applied twice."""


class ScalerRefitError(FeatureTransformError):
    """Raised when a fitted manager is refit while refit is disabled."""


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}. Got {value!r}.")
    return value


def _range_to_list(value):
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _json_default(value):
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.floating, numpy.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _array_stats(values, chunk_size: int | None = None) -> dict[str, Any]:
    values = numpy.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix. Got shape {values.shape}.")

    chunk_size = int(chunk_size or values.shape[0] or 1)
    feature_min = None
    feature_max = None
    feature_sum = numpy.zeros(values.shape[1], dtype=numpy.float64)
    feature_sum_sq = numpy.zeros(values.shape[1], dtype=numpy.float64)
    total_rows = 0
    has_nan = False
    has_inf = False

    for start in range(0, values.shape[0], chunk_size):
        end = min(start + chunk_size, values.shape[0])
        chunk = numpy.asarray(values[start:end], dtype=numpy.float64)
        if chunk.size == 0:
            continue
        has_nan = has_nan or bool(numpy.isnan(chunk).any())
        has_inf = has_inf or bool(numpy.isinf(chunk).any())
        chunk_min = numpy.nanmin(chunk, axis=0)
        chunk_max = numpy.nanmax(chunk, axis=0)
        feature_min = chunk_min if feature_min is None else numpy.minimum(feature_min, chunk_min)
        feature_max = chunk_max if feature_max is None else numpy.maximum(feature_max, chunk_max)
        finite_chunk = numpy.where(numpy.isfinite(chunk), chunk, numpy.nan)
        feature_sum += numpy.nansum(finite_chunk, axis=0)
        feature_sum_sq += numpy.nansum(finite_chunk * finite_chunk, axis=0)
        total_rows += int(chunk.shape[0])

    if feature_min is None:
        feature_min = numpy.full(values.shape[1], numpy.nan, dtype=numpy.float64)
        feature_max = numpy.full(values.shape[1], numpy.nan, dtype=numpy.float64)

    denominator = max(1, total_rows)
    mean = feature_sum / denominator
    variance = numpy.maximum((feature_sum_sq / denominator) - (mean * mean), 0.0)
    std = numpy.sqrt(variance)
    return {
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "feature_min": feature_min.tolist(),
        "feature_max": feature_max.tolist(),
        "global_min": float(numpy.nanmin(feature_min)) if feature_min.size else None,
        "global_max": float(numpy.nanmax(feature_max)) if feature_max.size else None,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
    }


@dataclass(slots=True)
class FeatureTransformPolicy:
    source_profile: str = "legacy_csv"
    feature_transform: str = "preserve"
    generator_transform: str = "preserve"
    classifier_transform: str = "preserve"
    evaluation_space: str = "source"
    expected_source_min: float | None = None
    expected_source_max: float | None = None
    allow_refit: bool = True
    allow_double_transform: bool = True
    inverse_transform_synthetic: bool = False
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        self.source_profile = _validate_choice(self.source_profile, SOURCE_PROFILES, "source_profile")
        self.feature_transform = _validate_choice(self.feature_transform, TRANSFORMS, "feature_transform")
        self.generator_transform = _validate_choice(self.generator_transform, TRANSFORMS, "generator_transform")
        self.classifier_transform = _validate_choice(self.classifier_transform, TRANSFORMS, "classifier_transform")
        self.evaluation_space = _validate_choice(self.evaluation_space, EVALUATION_SPACES, "evaluation_space")

    @classmethod
    def for_profile(
            cls,
            source_profile: str,
            feature_transform: str | None = None,
            generator_transform: str | None = None,
            classifier_transform: str | None = None,
            evaluation_space: str | None = None,
            **overrides):
        if source_profile == "appclassnet_top200":
            defaults = {
                "source_profile": "appclassnet_top200",
                "feature_transform": "preserve",
                "generator_transform": "preserve",
                "classifier_transform": "preserve",
                "evaluation_space": "source",
                "expected_source_min": -0.5,
                "expected_source_max": 0.5,
                "allow_refit": False,
                "allow_double_transform": False,
                "inverse_transform_synthetic": True,
                "tolerance": 1e-6,
            }
        elif source_profile == "legacy_csv":
            defaults = {
                "source_profile": "legacy_csv",
                "feature_transform": "preserve",
                "generator_transform": "preserve",
                "classifier_transform": "preserve",
                "evaluation_space": "source",
                "expected_source_min": None,
                "expected_source_max": None,
                "allow_refit": True,
                "allow_double_transform": True,
                "inverse_transform_synthetic": False,
                "tolerance": 1e-6,
            }
        else:
            defaults = {
                "source_profile": "custom",
                "feature_transform": "preserve",
                "generator_transform": "preserve",
                "classifier_transform": "preserve",
                "evaluation_space": "source",
                "expected_source_min": None,
                "expected_source_max": None,
                "allow_refit": False,
                "allow_double_transform": False,
                "inverse_transform_synthetic": True,
                "tolerance": 1e-6,
            }

        explicit = {
            "feature_transform": feature_transform,
            "generator_transform": generator_transform,
            "classifier_transform": classifier_transform,
            "evaluation_space": evaluation_space,
        }
        defaults.update({key: value for key, value in explicit.items() if value is not None})
        defaults.update(overrides)
        return cls(**defaults)

    def resolve_transform(self, stage: str) -> str:
        if stage == "feature":
            transform = self.feature_transform
        elif stage == "generator":
            transform = self.generator_transform
        elif stage == "classifier":
            transform = self.classifier_transform
        else:
            raise ValueError(f"Unsupported transform stage: {stage}")

        if transform == "auto":
            if self.source_profile == "appclassnet_top200":
                return "preserve"
            return "preserve"
        return transform

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformManifest:
    source_profile: str
    paths: dict[str, Any] = field(default_factory=dict)
    original_ranges: dict[str, Any] = field(default_factory=dict)
    current_ranges: dict[str, Any] = field(default_factory=dict)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    transform_id: str | None = None
    train_fit: dict[str, Any] = field(default_factory=dict)
    split_usage: dict[str, Any] = field(default_factory=dict)
    generator_input_space: str = "source"
    synthetic_output_space: str = "source"
    evaluation_space: str = "source"
    classifier_input_space: str = "source"
    inverse_transform_synthetic: bool = False
    warnings: list[str] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as manifest_file:
            json.dump(self.to_dict(), manifest_file, indent=2, sort_keys=True, default=_json_default)
            manifest_file.write("\n")
        return path


class FeatureTransformManager:
    """Fit-on-train manager for a single feature-space transform."""

    def __init__(self, policy: FeatureTransformPolicy, stage: str = "feature", output_space: str | None = None):
        self.policy = policy
        self.stage = stage
        self.operation = policy.resolve_transform(stage)
        self.output_space = output_space or ("transformed" if stage == "feature" else stage)
        self.scaler = None
        self.transform_id = None
        self.fit_split = None
        self.input_range = None
        self.output_range = None
        self.transform_history: list[dict[str, Any]] = []
        self.fit_stats = None

    @property
    def is_identity(self) -> bool:
        return self.operation == "preserve"

    def _make_scaler(self):
        if self.operation == "preserve":
            return None
        if self.operation == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        if self.operation == "standard":
            return StandardScaler()
        raise ValueError(f"Unsupported transform operation: {self.operation}")

    def _build_transform_id(self) -> str:
        payload = {
            "source_profile": self.policy.source_profile,
            "stage": self.stage,
            "operation": self.operation,
            "fit_split": self.fit_split,
            "input_range": self.input_range,
            "output_range": self.output_range,
        }
        if self.scaler is not None:
            for attribute in ("data_min_", "data_max_", "mean_", "scale_", "var_"):
                if hasattr(self.scaler, attribute):
                    payload[attribute] = getattr(self.scaler, attribute).tolist()
        encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _check_can_fit(self) -> None:
        if self.scaler is not None and not self.policy.allow_refit:
            raise ScalerRefitError(
                f"{self.stage} scaler is already fitted on {self.fit_split}; allow_refit=false prevents refit."
            )

    def _check_double_transform(self, transform_history, input_space: str) -> None:
        if self.operation == "preserve" or self.policy.allow_double_transform:
            return
        for entry in transform_history or []:
            same_operation = entry.get("operation") == self.operation
            same_stage = entry.get("output_space") == self.output_space or entry.get("stage") == self.stage
            same_transform = self.transform_id is not None and entry.get("transform_id") == self.transform_id
            same_input = entry.get("input_space") == input_space
            if same_transform or (same_operation and (same_stage or same_input)):
                raise DoubleTransformError(
                    f"Refusing to apply duplicate {self.operation} transform in {self.output_space} space. "
                    "Existing transform_history already records an equivalent transformation."
                )

    def _record_transform(self, input_space: str, output_space: str, input_stats, output_stats) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "operation": self.operation,
            "fit_split": self.fit_split,
            "input_space": input_space,
            "output_space": output_space,
            "input_range": [input_stats["global_min"], input_stats["global_max"]],
            "output_range": [output_stats["global_min"], output_stats["global_max"]],
            "transform_id": self.transform_id,
            "timestamp": _now_iso(),
        }

    def fit(self, train_x, split_name: str = "train"):
        self._check_can_fit()
        self.fit_split = split_name
        self.fit_stats = _array_stats(train_x)
        self.input_range = [self.fit_stats["global_min"], self.fit_stats["global_max"]]
        if self.operation == "preserve":
            self.scaler = None
            self.output_range = self.input_range
            self.transform_id = None
            return self

        self.scaler = self._make_scaler()
        self.scaler.fit(numpy.asarray(train_x, dtype=numpy.float32))
        if self.operation == "minmax":
            self.output_range = [0.0, 1.0]
        else:
            self.output_range = None
        self.transform_id = self._build_transform_id()
        return self

    def partial_fit_batches(self, train_batches, split_name: str = "train"):
        self._check_can_fit()
        self.fit_split = split_name
        if self.operation == "preserve":
            first_batch = None
            feature_min = None
            feature_max = None
            for batch in train_batches:
                x_batch = batch[0] if isinstance(batch, tuple) else batch
                x_batch = numpy.asarray(x_batch, dtype=numpy.float32)
                if first_batch is None:
                    first_batch = x_batch
                batch_min = numpy.nanmin(x_batch, axis=0)
                batch_max = numpy.nanmax(x_batch, axis=0)
                feature_min = batch_min if feature_min is None else numpy.minimum(feature_min, batch_min)
                feature_max = batch_max if feature_max is None else numpy.maximum(feature_max, batch_max)
            self.input_range = [
                float(numpy.nanmin(feature_min)) if feature_min is not None else None,
                float(numpy.nanmax(feature_max)) if feature_max is not None else None,
            ]
            self.output_range = self.input_range
            self.transform_id = None
            return self

        self.scaler = self._make_scaler()
        feature_min = None
        feature_max = None
        for batch in train_batches:
            x_batch = batch[0] if isinstance(batch, tuple) else batch
            x_batch = numpy.asarray(x_batch, dtype=numpy.float32)
            if x_batch.shape[0] == 0:
                continue
            batch_min = numpy.nanmin(x_batch, axis=0)
            batch_max = numpy.nanmax(x_batch, axis=0)
            feature_min = batch_min if feature_min is None else numpy.minimum(feature_min, batch_min)
            feature_max = batch_max if feature_max is None else numpy.maximum(feature_max, batch_max)
            if hasattr(self.scaler, "partial_fit"):
                self.scaler.partial_fit(x_batch)

        if feature_min is None:
            raise ValueError("Cannot fit scaler from empty batches.")
        if self.operation == "minmax":
            self.scaler.fit(numpy.vstack([feature_min, feature_max]).astype(numpy.float32, copy=False))
        self.input_range = [float(numpy.nanmin(feature_min)), float(numpy.nanmax(feature_max))]
        self.output_range = [0.0, 1.0] if self.operation == "minmax" else None
        self.transform_id = self._build_transform_id()
        return self

    def transform(
            self,
            values,
            split_name: str | None = None,
            input_space: str = "source",
            output_space: str | None = None,
            transform_history: list[dict[str, Any]] | None = None,
            return_metadata: bool = False):
        values = numpy.asarray(values, dtype=numpy.float32)
        output_space = output_space or self.output_space
        transform_history = list(transform_history or [])

        if self.operation == "preserve":
            metadata = {
                "data_space": input_space,
                "transform_id": None,
                "transform_history": transform_history,
                "stats": _array_stats(values),
            }
            return (values, metadata) if return_metadata else values

        if self.scaler is None:
            raise FeatureTransformError(f"{self.stage} scaler must be fitted on train before transform().")
        if split_name in {"valid", "validation", "test", "synthetic"} and self.fit_split != "train":
            raise FeatureTransformError(
                f"{self.stage} scaler used for {split_name} was not fitted on train; fit_split={self.fit_split}."
            )
        self._check_double_transform(transform_history, input_space)
        input_stats = _array_stats(values)
        transformed = self.scaler.transform(values).astype(numpy.float32, copy=False)
        output_stats = _array_stats(transformed)
        entry = self._record_transform(input_space, output_space, input_stats, output_stats)
        self.transform_history.append(entry)
        metadata = {
            "data_space": output_space,
            "transform_id": self.transform_id,
            "transform_history": [*transform_history, entry],
            "stats": output_stats,
        }
        return (transformed, metadata) if return_metadata else transformed

    def fit_transform_train(self, train_x, input_space: str = "source", output_space: str | None = None):
        self.fit(train_x, split_name="train")
        return self.transform(train_x, split_name="train", input_space=input_space, output_space=output_space)

    def inverse_transform(self, values):
        values = numpy.asarray(values, dtype=numpy.float32)
        if self.operation == "preserve" or self.scaler is None:
            return values
        return self.scaler.inverse_transform(values).astype(numpy.float32, copy=False)

    def transform_batch(self, batch, **kwargs):
        return self.transform(batch, **kwargs)

    def inverse_transform_batch(self, batch):
        return self.inverse_transform(batch)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "policy": self.policy.to_dict(),
                "stage": self.stage,
                "operation": self.operation,
                "output_space": self.output_space,
                "scaler": self.scaler,
                "transform_id": self.transform_id,
                "fit_split": self.fit_split,
                "input_range": self.input_range,
                "output_range": self.output_range,
                "transform_history": self.transform_history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        manager = cls(
            FeatureTransformPolicy(**state["policy"]),
            stage=state["stage"],
            output_space=state["output_space"],
        )
        manager.operation = state["operation"]
        manager.scaler = state["scaler"]
        manager.transform_id = state["transform_id"]
        manager.fit_split = state["fit_split"]
        manager.input_range = state["input_range"]
        manager.output_range = state["output_range"]
        manager.transform_history = state.get("transform_history", [])
        return manager


class ScaleGuard:
    """Validation helpers for train/generation/evaluation scale contracts."""

    @staticmethod
    def describe(values, data_space: str = "unknown", transform_id: str | None = None, transform_history=None):
        _validate_choice(data_space, DATA_SPACES, "data_space")
        return {
            "data_space": data_space,
            "transform_id": transform_id,
            "transform_history": list(transform_history or []),
            "stats": _array_stats(values),
        }

    @staticmethod
    def validate_finite(values, context: str) -> dict[str, Any]:
        stats = _array_stats(values)
        if stats["has_nan"] or stats["has_inf"]:
            raise FeatureTransformError(f"{context} contains NaN or infinite values.")
        return stats

    @staticmethod
    def validate_compatible_metadata(real_metadata, synthetic_metadata, context: str = "evaluation") -> None:
        real_space = (real_metadata or {}).get("data_space", "unknown")
        synthetic_space = (synthetic_metadata or {}).get("data_space", "unknown")
        real_transform_id = (real_metadata or {}).get("transform_id")
        synthetic_transform_id = (synthetic_metadata or {}).get("transform_id")
        real_stats = (real_metadata or {}).get("stats", {})
        synthetic_stats = (synthetic_metadata or {}).get("stats", {})

        if real_space != synthetic_space:
            raise PreprocessingSpaceMismatchError(
                "PreprocessingSpaceMismatchError: "
                f"real data is in {real_space} space "
                f"[{real_stats.get('global_min')},{real_stats.get('global_max')}]; "
                f"synthetic data is in {synthetic_space} space "
                f"[{synthetic_stats.get('global_min')},{synthetic_stats.get('global_max')}]; "
                f"inverse_transform synthetic data before {context}"
            )
        if real_space != "source" and real_transform_id != synthetic_transform_id:
            raise PreprocessingSpaceMismatchError(
                "PreprocessingSpaceMismatchError: transformed real and synthetic data use different "
                f"transform_id values before {context}: real={real_transform_id}, synthetic={synthetic_transform_id}."
            )

    @staticmethod
    def validate_before_evaluation(real_values, synthetic_values, real_metadata, synthetic_metadata, context: str):
        real_stats = ScaleGuard.validate_finite(real_values, f"{context} real data")
        synthetic_stats = ScaleGuard.validate_finite(synthetic_values, f"{context} synthetic data")
        real_metadata = {**(real_metadata or {}), "stats": real_stats}
        synthetic_metadata = {**(synthetic_metadata or {}), "stats": synthetic_stats}
        ScaleGuard.validate_compatible_metadata(real_metadata, synthetic_metadata, context=context)
        return {
            "context": context,
            "timestamp": _now_iso(),
            "real": real_metadata,
            "synthetic": synthetic_metadata,
        }

    @staticmethod
    def validate_train_test_compatible(train_values, test_values, train_metadata, test_metadata, context: str):
        train_stats = ScaleGuard.validate_finite(train_values, f"{context} train data")
        test_stats = ScaleGuard.validate_finite(test_values, f"{context} test data")
        train_metadata = {**(train_metadata or {}), "stats": train_stats}
        test_metadata = {**(test_metadata or {}), "stats": test_stats}
        if train_metadata.get("data_space", "unknown") != test_metadata.get("data_space", "unknown"):
            raise PreprocessingSpaceMismatchError(
                f"Train/test data_space mismatch before {context}: "
                f"train={train_metadata.get('data_space')}, test={test_metadata.get('data_space')}."
            )
        if (
            train_metadata.get("data_space") != "source"
            and train_metadata.get("transform_id") != test_metadata.get("transform_id")
        ):
            raise PreprocessingSpaceMismatchError(
                f"Train/test transform_id mismatch before {context}: "
                f"train={train_metadata.get('transform_id')}, test={test_metadata.get('transform_id')}."
            )
        return {"context": context, "timestamp": _now_iso(), "train": train_metadata, "test": test_metadata}


class ModelInputAdapter:
    """Separate source, generator, classifier and evaluation feature spaces."""

    def __init__(self, policy: FeatureTransformPolicy):
        self.policy = policy
        self.generator_manager = FeatureTransformManager(policy, stage="generator", output_space="generator")
        self.classifier_manager = FeatureTransformManager(policy, stage="classifier", output_space="classifier")
        self.synthetic_metadata = None
        self.generator_input_metadata = None

    def fit_generator(self, train_x):
        self.generator_manager.fit(train_x, split_name="train")
        return self

    def transform_generator_input(self, values, split_name: str = "train"):
        transformed, metadata = self.generator_manager.transform(
            values,
            split_name=split_name,
            input_space="source",
            output_space="generator",
            return_metadata=True,
        )
        self.generator_input_metadata = metadata
        return transformed

    def inverse_generator_output(self, values):
        source_values = self.generator_manager.inverse_transform(values)
        self.synthetic_metadata = ScaleGuard.describe(
            source_values,
            data_space="source",
            transform_id=None,
            transform_history=copy.deepcopy(self.generator_manager.transform_history),
        )
        return source_values

    def transform_synthetic_collection_to_source(self, synthetic_data):
        if (
            self.generator_manager.operation == "preserve"
            or not self.policy.inverse_transform_synthetic
        ):
            data_space = "source" if self.generator_manager.operation == "preserve" else "generator"
            transform_id = None if data_space == "source" else self.generator_manager.transform_id
            self.synthetic_metadata = {
                "data_space": data_space,
                "transform_id": transform_id,
                "transform_history": copy.deepcopy(self.generator_manager.transform_history),
            }
            return synthetic_data

        if hasattr(synthetic_data, "iter_batches"):
            raise FeatureTransformError(
                "Batch synthetic inverse_transform must be applied before writing batches."
            )

        transformed = {}
        for label_class, samples in synthetic_data.items():
            transformed[int(label_class)] = self.inverse_generator_output(samples)
        self.synthetic_metadata = {"data_space": "source", "transform_id": None, "transform_history": []}
        return transformed

    def inverse_synthetic_batch(self, batch):
        if (
            self.generator_manager.operation == "preserve"
            or not self.policy.inverse_transform_synthetic
        ):
            return numpy.asarray(batch, dtype=numpy.float32)
        return self.generator_manager.inverse_transform_batch(batch)

    def synthetic_space_after_generation(self) -> str:
        if self.generator_manager.operation == "preserve":
            return "source"
        return "source" if self.policy.inverse_transform_synthetic else "generator"
