#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Artifact helpers for explicit TR-TR executions."""

from __future__ import annotations

import csv
import datetime
import fcntl
import hashlib
import os
import platform
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy

from Engine.DataIO.JsonIO import atomic_write_json
from Engine.DataIO.RealClassCountPolicy import observed_counts_by_class


SUMMARY_COLUMNS = [
    "run_id",
    "classifier",
    "seed",
    "train_strategy",
    "train_samples",
    "test_samples",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "training_time_seconds",
    "peak_memory_bytes",
    "status",
]


def create_tr_tr_run_directory(base_dir, config) -> tuple[str, Path]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_run_id = make_tr_tr_run_id(config, timestamp)
    for suffix in range(1000):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:03d}"
        run_dir = base_dir / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique TR-TR run directory under {base_dir}.")


def make_tr_tr_run_id(config, timestamp: str) -> str:
    sampling = f"train-{config.train_sampling}_test-{config.test_sampling}"
    raw = f"{timestamp}_TR-TR_{config.classifier}_seed{config.random_state}_{sampling}"
    return _safe_token(raw)


def write_status(path, *, run_id, status, stage, exception: BaseException | None = None, extra=None):
    payload = {
        "run_id": run_id,
        "protocol": "TR-TR",
        "status": status,
        "stage": stage,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if exception is not None:
        payload.update({
            "exception_type": type(exception).__name__,
            "message": str(exception),
            "traceback": summarize_traceback(exception),
        })
    if extra:
        payload.update(extra)
    atomic_write_json(payload, path, run_id=run_id, protocol="TR-TR")
    return payload


def summarize_traceback(exception: BaseException, *, limit=12) -> list[str]:
    lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    return [line.rstrip("\n") for line in lines[-limit:]]


def write_execution_log(path, events):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as log_file:
        for event in events:
            log_file.write(str(event).rstrip("\n") + "\n")
        log_file.flush()
        os.fsync(log_file.fileno())
    os.replace(tmp_path, path)
    return path


def event(message):
    return f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {message}"


def write_class_distribution_csv(path, selection_table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    fieldnames = ["class_id", "available", "requested", "selected", "replacement", "split"]
    with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in selection_table:
            writer.writerow({field: row.get(field) for field in fieldnames})
        csv_file.flush()
        os.fsync(csv_file.fileno())
    os.replace(tmp_path, path)
    return path


def build_dataset_manifest(bundle, train_plan, test_plan, class_labels):
    splits = {
        "train": bundle.train,
        "valid": bundle.valid,
        "test": bundle.test,
    }
    split_payload = {}
    for split_name, split in splits.items():
        if split is None:
            split_payload[split_name] = None
            continue
        split_payload[split_name] = {
            "x_path": split.x_path,
            "y_path": split.y_path,
            "x_shape": list(split.X.shape),
            "y_shape": list(numpy.asanyarray(split.y).shape),
            "x_dtype": str(getattr(split.X, "dtype", None)),
            "y_dtype": str(getattr(split.y, "dtype", None)),
            "num_rows": int(split.num_rows),
            "num_features": int(split.num_features),
            "class_distribution": class_counts(split.y, class_labels),
            "files": {
                "x": file_identity(split.x_path),
                "y": file_identity(split.y_path),
            },
        }
    schema = bundle.schema
    return {
        "protocol": "TR-TR",
        "source_train": "real",
        "source_test": "real",
        "split_mode": schema.split_mode,
        "data_format": getattr(schema, "data_format", getattr(schema, "source_format", None)),
        "source_profile": getattr(schema, "source_profile", None),
        "num_features": int(schema.num_features),
        "num_classes": int(schema.num_classes) if schema.num_classes is not None else len(class_labels),
        "classes": [int(label) for label in class_labels],
        "feature_dtype": schema.feature_dtype,
        "target_dtype": schema.target_dtype,
        "feature_type": schema.feature_type,
        "target_type": schema.target_type,
        "train_feature_min": schema.train_feature_min,
        "train_feature_max": schema.train_feature_max,
        "splits": split_payload,
        "sample_plans": {
            "train": {
                "strategy": train_plan.mode,
                "requested_samples_per_class": train_plan.samples_per_class,
                "selected_total": int(train_plan.total_rows),
                "selection_table": train_plan.selection_table,
            },
            "test": {
                "strategy": test_plan.mode,
                "requested_samples_per_class": test_plan.samples_per_class,
                "selected_total": int(test_plan.total_rows),
                "selection_table": test_plan.selection_table,
            },
        },
    }


def class_counts(labels, class_labels):
    labels_array = numpy.asanyarray(labels).reshape(-1)
    normalized_labels = [int(class_id) for class_id in class_labels]
    if not normalized_labels:
        return {}
    if normalized_labels == list(range(len(normalized_labels))):
        counts, _ = observed_counts_by_class(labels_array, len(normalized_labels))
        return {
            str(class_id): int(counts[class_id])
            for class_id in normalized_labels
        }

    unique_labels, unique_counts = numpy.unique(labels_array.astype(numpy.int64, copy=False), return_counts=True)
    lookup = {
        int(label): int(count)
        for label, count in zip(unique_labels, unique_counts)
    }
    return {
        str(class_id): int(lookup.get(class_id, 0))
        for class_id in normalized_labels
    }


def file_identity(path):
    if path is None:
        return None
    path = Path(path)
    payload = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists() or not path.is_file():
        return payload
    stat = path.stat()
    payload.update({
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_first_1mib": _sha256_prefix(path),
    })
    return payload


def collect_environment(repo_root=None):
    payload = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "numpy": numpy.__version__,
        "scikit_learn": _package_version("sklearn"),
        "pandas": _package_version("pandas", required=False),
        "joblib": _package_version("joblib", required=False),
        "git": git_state(repo_root),
    }
    return payload


def git_state(repo_root=None):
    root = Path(repo_root or Path.cwd())
    return {
        "commit": _git_output(root, ["git", "rev-parse", "HEAD"]),
        "dirty": bool(_git_output(root, ["git", "status", "--short"])),
        "status_short": _git_output(root, ["git", "status", "--short"]),
    }


def append_summary_csv(summary_path, row):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_name(summary_path.name + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        write_header = not summary_path.exists() or summary_path.stat().st_size == 0
        with summary_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})
            csv_file.flush()
            os.fsync(csv_file.fileno())
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return summary_path


def summary_row(result, resource_usage):
    metrics = result.get("metrics", {})
    sample_plans = result.get("sample_plans", {})
    return {
        "run_id": result.get("run_id"),
        "classifier": result.get("classifier"),
        "seed": result.get("random_state"),
        "train_strategy": sample_plans.get("train", {}).get("mode"),
        "train_samples": result.get("samples", {}).get("train_total_effective"),
        "test_samples": result.get("samples", {}).get("test_total_effective"),
        "accuracy": metrics.get("Accuracy", metrics.get("accuracy")),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "training_time_seconds": resource_usage.get("training_time_seconds"),
        "peak_memory_bytes": resource_usage.get("peak_memory_bytes"),
        "status": result.get("status"),
    }


def model_disk_size(path):
    path = Path(path)
    return int(path.stat().st_size) if path.is_file() else 0


def _safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _sha256_prefix(path, *, max_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        digest.update(input_file.read(max_bytes))
    return digest.hexdigest()


def _package_version(module_name, *, required=True):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except ImportError:
        if required:
            raise
        return None


def _git_output(repo_root, command):
    try:
        return subprocess.check_output(
            command,
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
