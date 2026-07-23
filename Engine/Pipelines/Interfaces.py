#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit interfaces shared by TR-TR, synthetic generation, and future protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import SamplePlan


class DatasetLoader(Protocol):
    def load(self) -> DatasetBundle:
        """Load a dataset into the shared DatasetBundle contract."""


class TransformPolicyProvider(Protocol):
    def transform(self, values, **kwargs):
        """Transform feature values for a specific pipeline stage."""

    def inverse_transform(self, values, **kwargs):
        """Map feature values back to source space when configured."""


class SamplePlanner(Protocol):
    def build_plan(self, labels, split_name: str) -> SamplePlan:
        """Build an independent sample plan for one split."""


class ClassifierFactory(Protocol):
    def build(self, config):
        """Build a classifier without importing synthetic generator code."""


class SyntheticGenerator(Protocol):
    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """Fit a generator using train and, optionally, validation data."""

    def generate(self, class_counts: dict[int, int], batch_size: int):
        """Generate synthetic samples by global class label."""


class EvaluationProtocol(Protocol):
    protocol: str

    def run(self, bundle: DatasetBundle, output_dir: Path) -> dict[str, Any]:
        """Run one explicit evaluation protocol."""


class ArtifactWriter(Protocol):
    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Write a JSON artifact and return its path."""

    def write_array(self, name: str, values) -> Path:
        """Write an array artifact and return its path."""


@dataclass(slots=True)
class SyntheticGenerationConfig:
    generator: str
    generation_strategy: str = "single_conditional"
    classes_per_group: int = 10
    generation_batch_size: int = 8192
    synthetic_samples_per_class: int | None = None
    generator_transform: str = "preserve"
    random_state: int = 0


@dataclass(slots=True)
class SyntheticGenerationRequest:
    train_split: Any
    valid_split: Any | None
    schema: Any
    config: SyntheticGenerationConfig
    sample_plan: SamplePlan | None = None

    @property
    def uses_test_split(self) -> bool:
        return False


def build_synthetic_generation_request(
        bundle: DatasetBundle,
        config: SyntheticGenerationConfig,
        sample_plan: SamplePlan | None = None) -> SyntheticGenerationRequest:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError(f"Synthetic generation requires DatasetBundle. Got {type(bundle).__name__}.")
    if getattr(bundle.train, "name", None) != "train":
        raise ValueError(f"Synthetic generation requires train split. Got {getattr(bundle.train, 'name', None)!r}.")
    if bundle.valid is not None and getattr(bundle.valid, "name", None) != "valid":
        raise ValueError(f"Synthetic generation optional validation split must be named valid. Got {bundle.valid.name!r}.")
    return SyntheticGenerationRequest(
        train_split=bundle.train,
        valid_split=bundle.valid,
        schema=bundle.schema,
        config=config,
        sample_plan=sample_plan,
    )


def partition_generation_classes(class_counts: dict[int, int], strategy: str, classes_per_group: int) -> list[dict[str, Any]]:
    active_classes = [
        int(class_id)
        for class_id, count in sorted(class_counts.items(), key=lambda item: int(item[0]))
        if int(count) > 0
    ]
    if strategy == "single_conditional":
        return [{"unit_type": "global", "classes": active_classes}]
    if strategy == "per_class":
        return [{"unit_type": "class", "classes": [class_id]} for class_id in active_classes]
    if strategy == "grouped_classes":
        if int(classes_per_group) <= 0:
            raise ValueError("classes_per_group must be a positive integer.")
        return [
            {"unit_type": "group", "classes": active_classes[start:start + int(classes_per_group)]}
            for start in range(0, len(active_classes), int(classes_per_group))
        ]
    raise ValueError(
        "generation_strategy must be one of: single_conditional, per_class, grouped_classes. "
        f"Got {strategy!r}."
    )
