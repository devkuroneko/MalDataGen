"""Real split class-count policies for stratified evaluation subsets."""

from __future__ import annotations

import numpy


REAL_CLASS_COUNT_POLICIES = {"strict", "uniform_min", "available_cap"}
SAMPLES_PER_CLASS_SCOPES = {"split", "fold"}


def validate_real_class_count_policy(policy):
    if policy not in REAL_CLASS_COUNT_POLICIES:
        allowed = ", ".join(sorted(REAL_CLASS_COUNT_POLICIES))
        raise ValueError(f"real_class_count_policy must be one of: {allowed}. Got {policy!r}.")
    return policy


def validate_samples_per_class_scope(scope):
    if scope not in SAMPLES_PER_CLASS_SCOPES:
        allowed = ", ".join(sorted(SAMPLES_PER_CLASS_SCOPES))
        raise ValueError(f"samples_per_class_scope must be one of: {allowed}. Got {scope!r}.")
    return scope


def validate_requested_samples_per_class(value, field_name="samples_per_class"):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        integer_value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive integer.") from error
    if integer_value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return integer_value


def observed_counts_by_class(labels, num_classes):
    labels_array = numpy.asarray(labels).reshape(-1)
    counts = numpy.zeros(int(num_classes), dtype=numpy.int64)
    ignored_label_counts = {}
    for raw_label in labels_array:
        label = coerce_label(raw_label)
        if label is None or label < 0 or label >= int(num_classes):
            key = str(raw_label.item() if hasattr(raw_label, "item") else raw_label)
            ignored_label_counts[key] = int(ignored_label_counts.get(key, 0)) + 1
            continue
        counts[label] += 1
    return counts, ignored_label_counts


def coerce_label(raw_label):
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


def resolve_effective_class_counts(
        observed_counts,
        requested_samples_per_class,
        policy,
        split_name,
        *,
        require_all_classes=True):
    policy = validate_real_class_count_policy(policy)
    requested = validate_requested_samples_per_class(requested_samples_per_class)
    counts = numpy.asarray(observed_counts, dtype=numpy.int64)
    minimum_available = int(counts.min()) if counts.size else 0
    missing_classes = [
        int(class_id)
        for class_id, count in enumerate(counts)
        if int(count) == 0
    ]
    if require_all_classes and missing_classes:
        raise ValueError(f"{split_name} split is missing class(es): {missing_classes}.")

    if requested is None:
        effective = counts.astype(numpy.int64, copy=True)
    elif policy == "strict":
        short = {
            str(int(class_id)): int(count)
            for class_id, count in enumerate(counts)
            if int(count) < int(requested)
        }
        if short:
            raise ValueError(
                f"{split_name} split has fewer than requested {requested} samples per class "
                f"with real_class_count_policy=strict: {short}."
            )
        effective = numpy.full(counts.shape, int(requested), dtype=numpy.int64)
    elif policy == "uniform_min":
        effective_count = min(int(requested), int(minimum_available))
        effective = numpy.full(counts.shape, int(effective_count), dtype=numpy.int64)
    else:
        effective = numpy.minimum(counts, int(requested)).astype(numpy.int64, copy=False)

    return {
        "requested_samples_per_class": requested,
        "minimum_available_per_class": minimum_available,
        "effective_counts": effective,
        "effective_samples_per_class": _effective_samples_summary(effective),
        "real_class_count_policy": policy,
        "classes_below_requested": {
            str(int(class_id)): int(count)
            for class_id, count in enumerate(counts)
            if requested is not None and int(count) < int(requested)
        },
    }


def select_stratified_indices_from_labels(
        labels,
        requested_samples_per_class,
        num_classes,
        seed,
        split_name,
        policy,
        *,
        require_all_classes=True):
    labels_array = numpy.asarray(labels).reshape(-1)
    observed_counts, ignored_label_counts = observed_counts_by_class(labels_array, num_classes)
    resolution = resolve_effective_class_counts(
        observed_counts,
        requested_samples_per_class,
        policy,
        split_name,
        require_all_classes=require_all_classes,
    )
    quotas = numpy.asarray(resolution["effective_counts"], dtype=numpy.int64)
    random_generator = numpy.random.default_rng(seed)
    selected_parts = []
    counts_by_class = {}
    short_classes = {}

    for class_id in range(int(num_classes)):
        class_indices = numpy.flatnonzero(labels_array == class_id)
        take = int(quotas[class_id])
        if take <= 0:
            selected = numpy.array([], dtype=numpy.int64)
        else:
            selected = random_generator.choice(class_indices, size=take, replace=False)
        selected_parts.append(selected)
        counts_by_class[str(class_id)] = int(selected.shape[0])
        if (
            resolution["requested_samples_per_class"] is not None
            and int(class_indices.shape[0]) < int(resolution["requested_samples_per_class"])
        ):
            short_classes[class_id] = int(class_indices.shape[0])

    selected_indices = numpy.concatenate(selected_parts) if selected_parts else numpy.array([], dtype=numpy.int64)
    random_generator.shuffle(selected_indices)
    report = {
        **resolution,
        "observed_counts_by_class": _counts_dict(observed_counts),
        "selected_counts_by_class": counts_by_class,
        "ignored_label_counts": ignored_label_counts,
    }
    return selected_indices.astype(numpy.int64, copy=False), report, short_classes


def _effective_samples_summary(effective_counts):
    effective_counts = numpy.asarray(effective_counts, dtype=numpy.int64)
    if effective_counts.size == 0:
        return 0
    if numpy.all(effective_counts == effective_counts[0]):
        return int(effective_counts[0])
    return {
        str(int(class_id)): int(count)
        for class_id, count in enumerate(effective_counts)
    }


def _counts_dict(counts):
    return {
        str(int(class_id)): int(count)
        for class_id, count in enumerate(counts)
    }
