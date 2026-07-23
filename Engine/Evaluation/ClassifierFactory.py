#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Central classifier factory for explicit evaluation pipelines."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


SUPPORTED_INTEGRAL_CLASSIFIERS = {"decision_tree", "random_forest"}
SUBSET_CLASSIFIERS = {"decision_tree_subset", "random_forest_subset"}


@dataclass(slots=True)
class ClassifierConfig:
    classifier: str = "decision_tree"
    random_state: int = 0
    decision_tree_criterion: str = "gini"
    decision_tree_splitter: str = "best"
    decision_tree_max_depth: int | None = None
    decision_tree_min_samples_split: int | float = 2
    decision_tree_min_samples_leaf: int | float = 1
    decision_tree_max_features: int | float | str | None = None
    decision_tree_class_weight: str | dict | None = None
    random_forest_n_estimators: int = 100
    random_forest_criterion: str = "gini"
    random_forest_max_depth: int | None = None
    random_forest_min_samples_split: int | float = 2
    random_forest_min_samples_leaf: int | float = 1
    random_forest_max_features: int | float | str | None = "sqrt"
    random_forest_bootstrap: bool = True
    random_forest_class_weight: str | dict | None = None
    random_forest_n_jobs: int = 1

    def __post_init__(self):
        if self.classifier in SUBSET_CLASSIFIERS:
            raise ValueError(
                f"{self.classifier} is a subset classifier and must not be identified as an integral TR-TR model."
            )
        if self.classifier not in SUPPORTED_INTEGRAL_CLASSIFIERS:
            allowed = ", ".join(sorted(SUPPORTED_INTEGRAL_CLASSIFIERS))
            raise ValueError(f"Unsupported classifier {self.classifier!r}; expected one of: {allowed}.")
        self.random_state = int(self.random_state)
        self.random_forest_n_estimators = _positive_int(
            self.random_forest_n_estimators,
            "random_forest_n_estimators",
        )
        self.random_forest_n_jobs = int(self.random_forest_n_jobs)
        if self.random_forest_n_jobs == 0:
            raise ValueError("random_forest_n_jobs must not be 0.")
        self.decision_tree_min_samples_split = _positive_int_or_float_fraction(
            self.decision_tree_min_samples_split,
            "decision_tree_min_samples_split",
        )
        self.decision_tree_min_samples_leaf = _positive_int_or_float_fraction(
            self.decision_tree_min_samples_leaf,
            "decision_tree_min_samples_leaf",
        )
        self.random_forest_min_samples_split = _positive_int_or_float_fraction(
            self.random_forest_min_samples_split,
            "random_forest_min_samples_split",
        )
        self.random_forest_min_samples_leaf = _positive_int_or_float_fraction(
            self.random_forest_min_samples_leaf,
            "random_forest_min_samples_leaf",
        )
        self.decision_tree_max_features = normalize_tree_parameter(
            self.decision_tree_max_features,
            "decision_tree_max_features",
            allow_none=True,
        )
        self.random_forest_max_features = normalize_tree_parameter(
            self.random_forest_max_features,
            "random_forest_max_features",
            allow_none=True,
        )
        self.decision_tree_max_depth = _optional_positive_int(
            self.decision_tree_max_depth,
            "decision_tree_max_depth",
        )
        self.random_forest_max_depth = _optional_positive_int(
            self.random_forest_max_depth,
            "random_forest_max_depth",
        )

    def to_dict(self):
        return asdict(self)


def build_classifier(config: ClassifierConfig):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("scikit-learn is required to build TR-TR classifiers.") from error

    if config.classifier == "decision_tree":
        return DecisionTreeClassifier(
            criterion=config.decision_tree_criterion,
            splitter=config.decision_tree_splitter,
            max_depth=config.decision_tree_max_depth,
            min_samples_split=config.decision_tree_min_samples_split,
            min_samples_leaf=config.decision_tree_min_samples_leaf,
            max_features=config.decision_tree_max_features,
            class_weight=config.decision_tree_class_weight,
            random_state=config.random_state,
        )
    if config.classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.random_forest_n_estimators,
            criterion=config.random_forest_criterion,
            max_depth=config.random_forest_max_depth,
            min_samples_split=config.random_forest_min_samples_split,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            max_features=config.random_forest_max_features,
            bootstrap=config.random_forest_bootstrap,
            class_weight=config.random_forest_class_weight,
            n_jobs=config.random_forest_n_jobs,
            random_state=config.random_state,
        )
    allowed = ", ".join(sorted(SUPPORTED_INTEGRAL_CLASSIFIERS))
    raise ValueError(f"Unsupported classifier {config.classifier!r}; expected one of: {allowed}.")


def sklearn_version():
    try:
        import sklearn
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for TR-TR.") from error
    return sklearn.__version__


def estimator_statistics(model, classifier_name: str):
    if classifier_name == "decision_tree":
        tree = getattr(model, "tree_", None)
        if tree is None:
            return {}
        return {
            "max_depth": int(model.get_depth()),
            "node_count": int(tree.node_count),
            "leaf_count": int(model.get_n_leaves()),
        }
    if classifier_name == "random_forest":
        estimators = list(getattr(model, "estimators_", []) or [])
        depths = [int(estimator.get_depth()) for estimator in estimators]
        node_counts = [int(estimator.tree_.node_count) for estimator in estimators]
        leaf_counts = [int(estimator.get_n_leaves()) for estimator in estimators]
        return {
            "n_estimators_trained": int(len(estimators)),
            "mean_depth": float(sum(depths) / len(depths)) if depths else 0.0,
            "max_depth": int(max(depths)) if depths else 0,
            "total_nodes": int(sum(node_counts)),
            "total_leaves": int(sum(leaf_counts)),
        }
    return {}


def normalize_tree_parameter(value: Any, field_name: str, *, allow_none: bool = True):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} cannot be None.")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"none", "null"}:
            if allow_none:
                return None
            raise ValueError(f"{field_name} cannot be None.")
        try:
            if "." in normalized:
                return float(normalized)
            return int(normalized)
        except ValueError:
            return normalized
    return value


def _positive_int(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _optional_positive_int(value, field_name):
    if value is None:
        return None
    return _positive_int(value, field_name)


def _positive_int_or_float_fraction(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer or float in (0, 1].")
    if isinstance(value, str):
        value = normalize_tree_parameter(value, field_name, allow_none=False)
    if isinstance(value, float):
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"{field_name} as float must be in (0, 1].")
        return value
    value = int(value)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value
