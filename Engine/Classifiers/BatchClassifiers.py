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
    y_values = numpy.ravel(numpy.asarray(y_values))
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
        "batch_classifier": classifier_key,
        "eval_classifier": classifier_key,
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
