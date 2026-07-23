"""Real split class-count policies for stratified evaluation subsets."""

from __future__ import annotations

import numpy

from Engine.DataIO.DatasetContracts import SamplePlan

REAL_CLASS_COUNT_POLICIES = {"strict", "uniform_min", "available_cap", "cap_to_available"}
SAMPLES_PER_CLASS_SCOPES = {"split", "fold"}
SPLIT_SAMPLE_STRATEGIES = {"all", "balanced_per_class", "up_to_available"}


def validate_real_class_count_policy(policy):
    if policy not in REAL_CLASS_COUNT_POLICIES:
        allowed = ", ".join(sorted(REAL_CLASS_COUNT_POLICIES))
        raise ValueError(f"real_class_count_policy must be one of: {allowed}. Got {policy!r}.")
    if policy == "cap_to_available":
        return "available_cap"
    return policy


def validate_split_sample_strategy(strategy):
    if strategy not in SPLIT_SAMPLE_STRATEGIES:
        allowed = ", ".join(sorted(SPLIT_SAMPLE_STRATEGIES))
        raise ValueError(f"sample strategy must be one of: {allowed}. Got {strategy!r}.")
    return strategy


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
    try:
        if numpy.issubdtype(labels_array.dtype, numpy.integer):
            integer_labels = labels_array.astype(numpy.int64, copy=False)
            valid_mask = (integer_labels >= 0) & (integer_labels < int(num_classes))
        elif numpy.issubdtype(labels_array.dtype, numpy.floating):
            finite_mask = numpy.isfinite(labels_array)
            integer_labels = labels_array.astype(numpy.int64, copy=False)
            valid_mask = (
                finite_mask
                & numpy.isclose(labels_array, integer_labels)
                & (integer_labels >= 0)
                & (integer_labels < int(num_classes))
            )
        else:
            raise TypeError("non-numeric labels require fallback counting")

        if numpy.any(valid_mask):
            counts += numpy.bincount(
                integer_labels[valid_mask],
                minlength=int(num_classes),
            )[:int(num_classes)].astype(numpy.int64, copy=False)

        if numpy.any(~valid_mask):
            invalid_values, invalid_counts = numpy.unique(labels_array[~valid_mask], return_counts=True)
            for raw_label, count in zip(invalid_values, invalid_counts):
                key = str(raw_label.item() if hasattr(raw_label, "item") else raw_label)
                ignored_label_counts[key] = int(count)
    except (TypeError, ValueError):
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


def build_split_sample_plan(
        labels,
        requested_samples_per_class,
        num_classes,
        seed,
        split_name,
        strategy="balanced_per_class",
        insufficient_policy="strict",
        *,
        replacement=False,
        require_all_classes=True):
    labels_array = numpy.asarray(labels).reshape(-1)
    split_size = int(labels_array.shape[0])
    if split_size < 0:
        raise ValueError(f"{split_name} split size must be non-negative.")
    strategy = validate_split_sample_strategy(strategy)
    if insufficient_policy == "cap_to_available":
        insufficient_policy = "available_cap"
    if insufficient_policy not in {"strict", "available_cap", "uniform_min"}:
        raise ValueError(
            "insufficient_policy must be one of: strict, cap_to_available, available_cap, uniform_min. "
            f"Got {insufficient_policy!r}."
        )
    if replacement:
        raise ValueError("replacement=True is not supported for real split sampling yet.")

    observed_counts, ignored_label_counts = observed_counts_by_class(labels_array, num_classes)
    if strategy == "all":
        requested = None
        quotas = observed_counts.astype(numpy.int64, copy=True)
        minimum_available = int(observed_counts.min()) if observed_counts.size else 0
        classes_below_requested = {}
    else:
        requested = validate_requested_samples_per_class(requested_samples_per_class)
        if requested is None:
            raise ValueError(f"strategy={strategy} requires samples_per_class for split={split_name}.")
        policy = "available_cap" if strategy == "up_to_available" else insufficient_policy
        resolution = resolve_effective_class_counts(
            observed_counts,
            requested,
            policy,
            split_name,
            require_all_classes=require_all_classes,
        )
        quotas = numpy.asarray(resolution["effective_counts"], dtype=numpy.int64)
        minimum_available = resolution["minimum_available_per_class"]
        classes_below_requested = resolution["classes_below_requested"]

    missing_classes = [
        int(class_id)
        for class_id, count in enumerate(observed_counts)
        if int(count) == 0
    ]
    if require_all_classes and missing_classes:
        raise ValueError(f"{split_name} split is missing class(es): {missing_classes}.")

    random_generator = numpy.random.default_rng(seed)
    selected_parts = []
    counts_by_class = {}
    short_classes = {}
    selection_table = []
    sorted_labels = None
    sorted_indices = None
    if strategy != "all":
        sorted_indices = numpy.argsort(labels_array, kind="stable")
        sorted_labels = labels_array[sorted_indices]

    for class_id in range(int(num_classes)):
        available = int(observed_counts[class_id])
        take = int(quotas[class_id])
        if take < 0:
            raise ValueError(f"{split_name} split class={class_id} selected count cannot be negative.")
        if take > available and not replacement:
            raise ValueError(
                f"{split_name} split class={class_id} requested {take} samples without replacement "
                f"but only {available} are available."
            )
        if strategy == "all":
            selected_count = available
        elif take <= 0:
            selected = numpy.array([], dtype=numpy.int64)
            selected_count = 0
        else:
            left = int(numpy.searchsorted(sorted_labels, class_id, side="left"))
            right = int(numpy.searchsorted(sorted_labels, class_id, side="right"))
            class_indices = sorted_indices[left:right]
            selected = random_generator.choice(class_indices, size=take, replace=replacement)
            selected_count = int(selected.shape[0])
        if strategy != "all":
            selected_parts.append(selected)
        counts_by_class[str(class_id)] = int(selected_count)
        if (
            requested is not None
            and available < int(requested)
        ):
            short_classes[class_id] = available
        selection_table.append({
            "class_id": int(class_id),
            "available": available,
            "requested": None if requested is None else int(requested),
            "selected": int(selected_count),
            "replacement": bool(replacement),
            "split": split_name,
        })

    if strategy == "all":
        selected_indices = None
    else:
        selected_indices = numpy.concatenate(selected_parts) if selected_parts else numpy.array([], dtype=numpy.int64)
        random_generator.shuffle(selected_indices)
        _validate_indices_for_split(selected_indices, split_size, split_name)

    class_counts = {int(key): int(value) for key, value in enumerate(quotas.tolist())}
    metadata = {
        "strategy": strategy,
        "requested_samples_per_class": requested,
        "minimum_available_per_class": minimum_available,
        "effective_samples_per_class": _effective_samples_summary(quotas),
        "insufficient_policy": insufficient_policy,
        "classes_below_requested": classes_below_requested,
        "observed_counts_by_class": _counts_dict(observed_counts),
        "selected_counts_by_class": counts_by_class,
        "ignored_label_counts": ignored_label_counts,
        "selection_table": selection_table,
        "cache_key": {
            "split": split_name,
            "split_size": split_size,
            "strategy": strategy,
            "samples_per_class": requested,
            "random_state": int(seed),
            "insufficient_policy": insufficient_policy,
            "replacement": bool(replacement),
        },
    }
    return SamplePlan(
        mode=strategy,
        total_rows=split_size if strategy == "all" else int(sum(class_counts.values())),
        class_counts=class_counts,
        samples_per_class=requested,
        number_classes=int(num_classes),
        split_name=split_name,
        split_size=split_size,
        random_state=int(seed),
        insufficient_policy=insufficient_policy,
        replacement=bool(replacement),
        selected_indices=selected_indices,
        selection_table=selection_table,
        metadata=metadata,
    )


def select_stratified_indices_from_labels(
        labels,
        requested_samples_per_class,
        num_classes,
        seed,
        split_name,
        policy,
        *,
        require_all_classes=True):
    normalized_policy = validate_real_class_count_policy(policy)
    strategy = "up_to_available" if normalized_policy == "available_cap" else "balanced_per_class"
    plan = build_split_sample_plan(
        labels,
        requested_samples_per_class,
        num_classes,
        seed,
        split_name,
        strategy=strategy,
        insufficient_policy=normalized_policy,
        replacement=False,
        require_all_classes=require_all_classes,
    )
    report = dict(plan.metadata)
    return plan.selected_indices, report, {
        int(key): int(value)
        for key, value in report.get("classes_below_requested", {}).items()
    }


def _validate_indices_for_split(indices, split_size, split_name):
    indices = numpy.asarray(indices, dtype=numpy.int64)
    if indices.size == 0:
        return
    min_index = int(indices.min())
    max_index = int(indices.max())
    if min_index < 0 or max_index >= int(split_size):
        raise IndexError(
            f"{split_name} sample plan produced out-of-bounds indices: "
            f"min={min_index} max={max_index} split_size={int(split_size)}."
        )


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
