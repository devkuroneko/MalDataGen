#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch-friendly classifier training utilities for low-memory evaluation."""

from __future__ import annotations

import time

import numpy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Engine.DataIO.DatasetContracts import validate_xy_alignment
from Engine.DataIO.RealClassCountPolicy import resolve_effective_class_counts


BATCH_CLASSIFIER_DISPLAY_NAMES = {
    "sgd": "SGDClassifier",
    "passive_aggressive": "PassiveAggressiveClassifier",
    "naive_bayes": "GaussianNB",
    "mlp_small": "MLPClassifierSmall",
    "decision_tree_subset": "DecisionTreeSubset",
    "extra_trees_subset": "ExtraTreesSubset",
    "random_forest_light": "RandomForestLight",
    "random_forest_subset": "RandomForestLight",
}

PARTIAL_FIT_CLASSIFIERS = {"sgd", "passive_aggressive", "naive_bayes", "mlp_small"}
SUBSET_CLASSIFIERS = {"decision_tree_subset", "extra_trees_subset", "random_forest_light", "random_forest_subset"}


def _random_state(arguments):
    return int(getattr(arguments, "random_state", 0))


def get_batch_classifier_display_name(classifier_key):
    return BATCH_CLASSIFIER_DISPLAY_NAMES[classifier_key]


def make_batch_classifier(classifier_key, arguments):
    if classifier_key == "random_forest_subset":
        classifier_key = "random_forest_light"
    random_state = _random_state(arguments)
    if classifier_key == "sgd":
        return SGDClassifier(loss="log_loss", random_state=random_state)
    if classifier_key == "passive_aggressive":
        return PassiveAggressiveClassifier(random_state=random_state)
    if classifier_key == "naive_bayes":
        return GaussianNB()
    if classifier_key == "mlp_small":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=1, random_state=random_state)
    if classifier_key == "decision_tree_subset":
        return DecisionTreeClassifier(
            criterion=getattr(arguments, "decision_tree_criterion", "gini"),
            max_depth=getattr(arguments, "max_depth", getattr(arguments, "decision_tree_max_depth", None)),
            max_features=getattr(arguments, "decision_tree_max_features", None),
            max_leaf_nodes=getattr(arguments, "decision_tree_max_leaf_nodes", None),
            random_state=random_state,
        )
    if classifier_key == "extra_trees_subset":
        return ExtraTreesClassifier(
            n_estimators=int(getattr(arguments, "n_estimators", None) or 50),
            max_depth=getattr(arguments, "max_depth", None),
            class_weight=getattr(arguments, "class_weight", None),
            n_jobs=-1,
            random_state=random_state,
        )
    if classifier_key == "random_forest_light":
        return RandomForestClassifier(
            n_estimators=int(getattr(arguments, "n_estimators", None) or 30),
            max_depth=getattr(arguments, "max_depth", None) or 30,
            max_leaf_nodes=getattr(arguments, "random_forest_max_leaf_nodes", None),
            max_samples=getattr(arguments, "max_samples", None),
            class_weight=getattr(arguments, "class_weight", None),
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported batch classifier: {classifier_key}")


def iter_array_batches(x_values, y_values, batch_size):
    x_values, y_values = validate_xy_alignment(x_values, y_values, "array batch iterator")
    for start in range(0, y_values.shape[0], int(batch_size)):
        end = min(start + int(batch_size), y_values.shape[0])
        yield numpy.asarray(x_values[start:end], dtype=numpy.float32), y_values[start:end].astype(numpy.int64)


def iter_synthetic_labeled_batches(synthetic_data):
    if hasattr(synthetic_data, "iter_batches"):
        iterator = synthetic_data.iter_batches()
    else:
        iterator = synthetic_data.items()

    for label_class, x_batch in iterator:
        x_batch = numpy.asarray(x_batch, dtype=numpy.float32)
        if x_batch.shape[0] == 0:
            continue
        labels = numpy.full(x_batch.shape[0], int(label_class), dtype=numpy.int64)
        yield x_batch, labels


def _counts_by_class(labels):
    unique_labels, counts = numpy.unique(numpy.asarray(labels, dtype=numpy.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}


def _require_finite(name, values):
    if not numpy.all(numpy.isfinite(numpy.asarray(values))):
        raise ValueError(f"{name} contains NaN or inf values.")


def _validate_class_counts(
        name,
        counts_by_class,
        expected_num_classes=None,
        samples_per_class=None,
        real_class_count_policy="strict"):
    if expected_num_classes is not None:
        expected = {str(class_id) for class_id in range(int(expected_num_classes))}
        observed = set(counts_by_class)
        missing = sorted(int(class_id) for class_id in expected - observed)
        extra = sorted(int(class_id) for class_id in observed - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing classes={missing[:20]}")
            if extra:
                details.append(f"unexpected classes={extra[:20]}")
            raise ValueError(f"{name} must contain exactly {expected_num_classes} classes; " + ", ".join(details))

    if samples_per_class is not None and real_class_count_policy == "strict":
        short = {
            class_id: count
            for class_id, count in counts_by_class.items()
            if int(count) < int(samples_per_class)
        }
        if short:
            raise ValueError(
                f"{name} has fewer than requested {samples_per_class} samples per class: {short}."
            )
    elif samples_per_class is not None and expected_num_classes is not None:
        ordered_counts = numpy.asarray(
            [int(counts_by_class.get(str(class_id), 0)) for class_id in range(int(expected_num_classes))],
            dtype=numpy.int64,
        )
        resolve_effective_class_counts(
            ordered_counts,
            samples_per_class,
            real_class_count_policy,
            name,
            require_all_classes=True,
        )


def validate_real_array_for_batch_evaluation(
        x_values,
        y_values,
        context,
        expected_num_classes=None,
        samples_per_class=None,
        real_class_count_policy="strict",
        samples_per_class_scope="split"):
    if x_values is None or y_values is None:
        raise ValueError(f"{context} requires real X/y arrays.")
    x_values, y_values = validate_xy_alignment(x_values, y_values, f"{context} real data")
    if x_values.shape[0] == 0:
        raise ValueError(f"{context} real data is empty.")
    _require_finite(f"{context} real X", x_values)
    _require_finite(f"{context} real y", y_values)
    counts = _counts_by_class(y_values)
    _validate_class_counts(
        f"{context} real data",
        counts,
        expected_num_classes=expected_num_classes,
        samples_per_class=samples_per_class,
        real_class_count_policy=real_class_count_policy,
    )
    return counts


def validate_synthetic_batches_for_evaluation(
        synthetic_data,
        context,
        expected_num_classes=None,
        samples_per_class=None,
        expected_num_features=None,
        expected_data_space="source",
        expected_schema_hash=None):
    if synthetic_data is None:
        raise ValueError(f"{context} requires synthetic data.")

    if hasattr(synthetic_data, "manifest"):
        manifest = synthetic_data.manifest
        data_space = manifest.get("data_space", "source")
        if expected_data_space is not None and data_space != expected_data_space:
            raise ValueError(
                f"{context} real and synthetic data_space differ: real={expected_data_space} synthetic={data_space}."
            )
        if expected_num_classes is not None and int(manifest.get("num_classes", expected_num_classes)) != int(expected_num_classes):
            raise ValueError(
                f"{context} synthetic manifest num_classes={manifest.get('num_classes')} "
                f"does not match expected {expected_num_classes}."
            )
        synthetic_schema_hash = manifest.get("schema_hash")
        if expected_schema_hash is not None and synthetic_schema_hash is not None:
            if synthetic_schema_hash != expected_schema_hash:
                raise ValueError(
                    f"{context} schema hash mismatch: real={expected_schema_hash} synthetic={synthetic_schema_hash}."
                )

    counts = {}
    feature_count = expected_num_features
    total_rows = 0
    for x_batch, y_batch in iter_synthetic_labeled_batches(synthetic_data):
        if x_batch.ndim != 2:
            raise ValueError(f"{context} synthetic X must be two-dimensional; got shape {x_batch.shape}.")
        if y_batch.ndim != 1 or x_batch.shape[0] != y_batch.shape[0]:
            raise ValueError(f"{context} synthetic X/y are not aligned: X={x_batch.shape} y={y_batch.shape}.")
        if feature_count is None:
            feature_count = int(x_batch.shape[1])
        elif int(x_batch.shape[1]) != int(feature_count):
            raise ValueError(
                f"{context} synthetic feature count mismatch: expected {feature_count}, got {x_batch.shape[1]}."
            )
        _require_finite(f"{context} synthetic X", x_batch)
        _require_finite(f"{context} synthetic y", y_batch)
        total_rows += int(x_batch.shape[0])
        for label, count in _counts_by_class(y_batch).items():
            counts[label] = int(counts.get(label, 0)) + int(count)

    if total_rows == 0:
        raise ValueError(f"{context} synthetic data is empty.")
    _validate_class_counts(
        f"{context} synthetic data",
        counts,
        expected_num_classes=expected_num_classes,
        samples_per_class=samples_per_class,
    )
    return counts


def _subset_quota(arguments, num_classes):
    samples_per_class = getattr(arguments, "train_samples_per_class", None)
    if samples_per_class is not None:
        return int(samples_per_class)

    subset_size = int(getattr(arguments, "batch_classifier_subset_size", 100000))
    return max(1, subset_size // max(1, int(num_classes)))


def _collect_stratified_subset(train_batches, arguments, num_classes):
    quota_per_class = _subset_quota(arguments, num_classes)
    random_generator = numpy.random.default_rng(_random_state(arguments))
    reservoirs = {class_id: [] for class_id in range(int(num_classes))}
    seen_by_class = {class_id: 0 for class_id in range(int(num_classes))}
    train_batches_seen = 0
    train_samples_seen = 0

    for x_batch, y_batch in train_batches:
        train_batches_seen += 1
        if x_batch.shape[0] == 0:
            continue
        x_batch = numpy.asarray(x_batch, dtype=numpy.float32)
        y_batch = numpy.asarray(y_batch, dtype=numpy.int64)
        train_samples_seen += int(x_batch.shape[0])

        for row_index, raw_label in enumerate(y_batch):
            label = int(raw_label)
            if label not in reservoirs:
                reservoirs[label] = []
                seen_by_class[label] = 0
            seen_by_class[label] += 1
            seen_count = seen_by_class[label]
            reservoir = reservoirs[label]
            row = x_batch[row_index].copy()
            if len(reservoir) < quota_per_class:
                reservoir.append(row)
            else:
                replacement_index = random_generator.integers(0, seen_count)
                if replacement_index < quota_per_class:
                    reservoir[int(replacement_index)] = row

    x_parts = []
    y_parts = []
    for label, rows in reservoirs.items():
        if not rows:
            continue
        x_parts.append(numpy.asarray(rows, dtype=numpy.float32))
        y_parts.append(numpy.full(len(rows), int(label), dtype=numpy.int64))

    if not x_parts:
        raise ValueError("No rows available to train the subset classifier.")

    x_subset = numpy.vstack(x_parts)
    y_subset = numpy.concatenate(y_parts)
    permutation = random_generator.permutation(y_subset.shape[0])
    return (
        x_subset[permutation],
        y_subset[permutation],
        train_batches_seen,
        train_samples_seen,
        quota_per_class,
        _counts_by_class(y_subset),
    )


def train_batch_classifier(classifier_key, train_batches, num_classes, arguments, batch_recorder=None):
    requested_classifier_key = classifier_key
    if classifier_key == "random_forest_subset":
        classifier_key = "random_forest_light"
    classifier = make_batch_classifier(classifier_key, arguments)
    uses_partial_fit = classifier_key in PARTIAL_FIT_CLASSIFIERS
    train_batches_seen = 0
    train_samples_seen = 0
    largest_batch_processed = 0
    partial_fit_class_counts = {}
    start_time = time.perf_counter()

    if uses_partial_fit:
        classes = numpy.arange(int(num_classes), dtype=numpy.int64)
        first_batch = True
        for x_batch, y_batch in train_batches:
            if x_batch.shape[0] == 0:
                continue
            train_batches_seen += 1
            train_samples_seen += int(x_batch.shape[0])
            largest_batch_processed = max(largest_batch_processed, int(x_batch.shape[0]))
            for label, count in _counts_by_class(y_batch).items():
                partial_fit_class_counts[label] = int(partial_fit_class_counts.get(label, 0)) + int(count)
            if batch_recorder is not None:
                batch_recorder(int(x_batch.shape[0]))
            if first_batch:
                classifier.partial_fit(x_batch, y_batch, classes=classes)
                first_batch = False
            else:
                classifier.partial_fit(x_batch, y_batch)

        if first_batch:
            raise ValueError("No rows available to train the incremental classifier.")
    else:
        (
            x_subset,
            y_subset,
            train_batches_seen,
            train_samples_seen,
            quota_per_class,
            subset_class_counts,
        ) = _collect_stratified_subset(train_batches, arguments, num_classes)
        largest_batch_processed = int(x_subset.shape[0])
        if batch_recorder is not None:
            batch_recorder(int(x_subset.shape[0]))
        classifier.fit(x_subset, y_subset)

    train_time_seconds = time.perf_counter() - start_time
    return classifier, {
        "requested_classifier": requested_classifier_key,
        "effective_classifier": classifier_key,
        "batch_classifier": classifier_key,
        "eval_classifier": classifier_key,
        "random_state": _random_state(arguments),
        "classifier_name": get_batch_classifier_display_name(classifier_key),
        "used_partial_fit": uses_partial_fit,
        "partial_fit": uses_partial_fit,
        "n_estimators": getattr(classifier, "n_estimators", None),
        "max_depth": getattr(classifier, "max_depth", None),
        "max_samples": getattr(classifier, "max_samples", None),
        "class_weight": getattr(classifier, "class_weight", None),
        "subset_quota_per_class": int(quota_per_class) if not uses_partial_fit else None,
        "subset_class_counts": subset_class_counts if not uses_partial_fit else None,
        "train_class_counts": partial_fit_class_counts if uses_partial_fit else subset_class_counts,
        "train_batches": int(train_batches_seen),
        "train_batch_count": int(train_batches_seen),
        "train_samples_seen": int(train_samples_seen),
        "total_samples_seen": int(train_samples_seen),
        "effective_fit_rows": int(train_samples_seen) if uses_partial_fit else int(largest_batch_processed),
        "fit_rows_input": int(train_samples_seen),
        "discarded_rows": 0 if uses_partial_fit else max(0, int(train_samples_seen) - int(largest_batch_processed)),
        "discarded_rows_for_fit": False if uses_partial_fit else int(train_samples_seen) > int(largest_batch_processed),
        "global_limit": int(getattr(arguments, "batch_classifier_subset_size", 100000)),
        "largest_batch_processed": int(largest_batch_processed),
        "training_time_seconds": float(train_time_seconds),
        "training_time": float(train_time_seconds),
    }


def predict_array_batches(classifier, x_values, y_values, batch_size, batch_recorder=None):
    labels = []
    predictions = []
    start_time = time.perf_counter()
    for x_batch, y_batch in iter_array_batches(x_values, y_values, batch_size):
        if x_batch.shape[0] == 0:
            continue
        if batch_recorder is not None:
            batch_recorder(int(x_batch.shape[0]))
        predictions.extend(classifier.predict(x_batch))
        labels.extend(y_batch)
    return numpy.asarray(labels), numpy.asarray(predictions), time.perf_counter() - start_time


def predict_synthetic_batches(classifier, synthetic_data, batch_recorder=None, max_samples_per_class=None):
    labels = []
    predictions = []
    counts_by_class = {}
    start_time = time.perf_counter()
    for x_batch, y_batch in iter_synthetic_labeled_batches(synthetic_data):
        if max_samples_per_class is not None:
            label = int(y_batch[0])
            remaining = int(max_samples_per_class) - int(counts_by_class.get(str(label), 0))
            if remaining <= 0:
                continue
            take = min(remaining, int(x_batch.shape[0]))
            x_batch = x_batch[:take]
            y_batch = y_batch[:take]
        if batch_recorder is not None:
            batch_recorder(int(x_batch.shape[0]))
        predictions.extend(classifier.predict(x_batch))
        labels.extend(y_batch)
        for label, count in _counts_by_class(y_batch).items():
            counts_by_class[label] = int(counts_by_class.get(label, 0)) + int(count)
    return numpy.asarray(labels), numpy.asarray(predictions), time.perf_counter() - start_time
