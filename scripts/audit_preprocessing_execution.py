#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit preprocessing artifacts for an AppClassNet execution.

The audit is intentionally evidence-driven. It reports unknown/insufficient
evidence instead of inferring a root cause from MinMaxScaler alone.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas


NUMERIC_SUFFIXES = {".npy", ".csv", ".txt"}
JSON_NAMES = {
    "preprocessing_manifest.json",
    "preprocessing_stats.json",
    "manifest.json",
    "Results.json",
    "metrics.json",
}
LABEL_COLUMNS = {"label", "class", "target", "y"}
APPCLASSNET_SOURCE_MIN = -0.5
APPCLASSNET_SOURCE_MAX = 0.5
GENERATOR_MIN = 0.0
GENERATOR_MAX = 1.0
RANGE_TOLERANCE = 1e-4


@dataclass
class RangeSummary:
    path: Path
    kind: str
    rows: int | None = None
    columns: int | None = None
    feature_min: float | None = None
    feature_max: float | None = None
    feature_mean: float | None = None
    feature_std: float | None = None
    has_nan: bool | None = None
    has_inf: bool | None = None
    label_min: float | None = None
    label_max: float | None = None
    label_unique_count: int | None = None
    data_space: str | None = None
    transform_id: str | None = None
    transform_history_count: int | None = None
    scope: str = "result"
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AppClassNet preprocessing execution artifacts.")
    parser.add_argument("--result_dir", required=True, help="Execution result directory to audit.")
    parser.add_argument(
        "--max_csv_rows",
        type=int,
        default=250_000,
        help="Maximum rows sampled from each CSV/TXT file for range summaries.",
    )
    parser.add_argument(
        "--max_manifest_batches",
        type=int,
        default=200,
        help="Maximum synthetic manifest batch files sampled for range summaries.",
    )
    parser.add_argument(
        "--raw_root",
        default="Datasets/raw/AppClassNet/top200",
        help="Optional AppClassNet raw root used as source-scale reference when present.",
    )
    parser.add_argument(
        "--no_related_artifacts",
        action="store_true",
        help="Audit only --result_dir; do not include sibling log/metrics or raw source arrays.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def is_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def in_range(summary: RangeSummary, lower: float, upper: float, tolerance: float = RANGE_TOLERANCE) -> bool:
    if summary.feature_min is None or summary.feature_max is None:
        return False
    return summary.feature_min >= lower - tolerance and summary.feature_max <= upper + tolerance


def path_role(path: Path) -> str:
    name = path.name.lower()
    parent = str(path.parent).lower()
    if "synthetic" in parent or "dataoutput" in name:
        return "synthetic_x"
    if "train" in name or "training" in name:
        return "train_x"
    if "test" in name:
        return "test_x"
    if "valid" in name or "evaluation" in name:
        return "evaluation_x"
    return "unknown"


def update_numeric_stats(values, state):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0:
        return state
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    finite = numpy.where(numpy.isfinite(values), values, numpy.nan)
    state["rows"] += int(values.shape[0])
    state["cols"] = int(values.shape[1])
    state["has_nan"] = state["has_nan"] or bool(numpy.isnan(values).any())
    state["has_inf"] = state["has_inf"] or bool(numpy.isinf(values).any())
    current_min = numpy.nanmin(finite, axis=0)
    current_max = numpy.nanmax(finite, axis=0)
    state["feature_min"] = current_min if state["feature_min"] is None else numpy.minimum(state["feature_min"], current_min)
    state["feature_max"] = current_max if state["feature_max"] is None else numpy.maximum(state["feature_max"], current_max)
    state["sum"] += numpy.nansum(finite, axis=0)
    state["sum_sq"] += numpy.nansum(finite * finite, axis=0)
    return state


def finalize_numeric_stats(path: Path, kind: str, state) -> RangeSummary:
    if state["rows"] == 0 or state["feature_min"] is None:
        return RangeSummary(path=path, kind=kind, error="no numeric feature rows found")
    denominator = max(1, state["rows"])
    mean_by_feature = state["sum"] / denominator
    variance_by_feature = numpy.maximum((state["sum_sq"] / denominator) - (mean_by_feature * mean_by_feature), 0.0)
    return RangeSummary(
        path=path,
        kind=kind,
        rows=state["rows"],
        columns=state["cols"],
        feature_min=float(numpy.nanmin(state["feature_min"])),
        feature_max=float(numpy.nanmax(state["feature_max"])),
        feature_mean=float(numpy.nanmean(mean_by_feature)),
        feature_std=float(numpy.nanmean(numpy.sqrt(variance_by_feature))),
        has_nan=state["has_nan"],
        has_inf=state["has_inf"],
    )


def new_stats_state(num_features: int | None = None):
    width = int(num_features or 0)
    return {
        "rows": 0,
        "cols": width if width else None,
        "feature_min": None,
        "feature_max": None,
        "sum": numpy.zeros(width, dtype=numpy.float64) if width else None,
        "sum_sq": numpy.zeros(width, dtype=numpy.float64) if width else None,
        "has_nan": False,
        "has_inf": False,
    }


def ensure_state_width(state, width: int):
    if state["sum"] is None:
        state["sum"] = numpy.zeros(width, dtype=numpy.float64)
        state["sum_sq"] = numpy.zeros(width, dtype=numpy.float64)
        state["cols"] = width
    return state


def summarize_npy(path: Path) -> RangeSummary:
    try:
        array = numpy.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 1 or path.name.lower().endswith("_y.npy"):
            feature_min = None
            feature_max = None
            unique_values = set()
            has_nan = False
            has_inf = False
            chunk_size = 500_000
            for start in range(0, array.shape[0], chunk_size):
                end = min(start + chunk_size, array.shape[0])
                values = numpy.asarray(array[start:end])
                if values.size == 0:
                    continue
                if numpy.issubdtype(values.dtype, numpy.number):
                    has_nan = has_nan or bool(numpy.isnan(values).any())
                    has_inf = has_inf or bool(numpy.isinf(values).any())
                current_min = float(numpy.nanmin(values)) if values.size else None
                current_max = float(numpy.nanmax(values)) if values.size else None
                feature_min = current_min if feature_min is None else min(feature_min, current_min)
                feature_max = current_max if feature_max is None else max(feature_max, current_max)
                if values.size <= 2_000_000 or numpy.issubdtype(values.dtype, numpy.integer):
                    unique_values.update(numpy.unique(values).tolist())
            note = ""
            if unique_values and all(is_number(value) and float(value).is_integer() for value in unique_values):
                integer_labels = {int(value) for value in unique_values}
                if integer_labels and min(integer_labels) >= 0 and max(integer_labels) <= 199:
                    missing = sorted(set(range(200)) - integer_labels)
                    note = f"missing_classes={missing}" if missing else "missing_classes=[]"
            return RangeSummary(
                path=path,
                kind="npy_labels" if path.name.lower().endswith("_y.npy") else "npy_vector",
                rows=int(array.shape[0]),
                columns=1,
                label_min=feature_min,
                label_max=feature_max,
                label_unique_count=int(len(unique_values)) if unique_values else None,
                has_nan=has_nan if numpy.issubdtype(array.dtype, numpy.number) else None,
                has_inf=has_inf if numpy.issubdtype(array.dtype, numpy.number) else None,
                data_space="source" if "appclassnet" in str(path).lower() else None,
                error=note,
            )
        state = new_stats_state(array.shape[1])
        chunk_size = 100_000
        for start in range(0, array.shape[0], chunk_size):
            end = min(start + chunk_size, array.shape[0])
            update_numeric_stats(array[start:end], state)
        summary = finalize_numeric_stats(path, "npy_features", state)
        if "appclassnet" in str(path).lower():
            summary.data_space = "source"
        return summary
    except Exception as error:
        return RangeSummary(path=path, kind="npy", error=str(error))


def summarize_csv(path: Path, max_rows: int) -> RangeSummary:
    try:
        rows_seen = 0
        state = new_stats_state()
        label_values = []
        for chunk in pandas.read_csv(path, chunksize=50_000):
            if rows_seen >= max_rows:
                break
            remaining = max_rows - rows_seen
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            rows_seen += int(len(chunk))

            numeric = chunk.select_dtypes(include=["number"])
            if numeric.empty:
                continue
            label_columns = [
                column for column in numeric.columns
                if str(column).lower() in LABEL_COLUMNS or str(column).lower().endswith("_label")
            ]
            for label_column in label_columns:
                label_values.extend(numeric[label_column].dropna().tolist())
            feature_columns = [
                column for column in numeric.columns
                if column not in label_columns and not str(column).lower().startswith("unnamed")
            ]
            if not feature_columns:
                feature_columns = [column for column in numeric.columns if column not in label_columns]
            features = numeric[feature_columns].to_numpy(dtype=numpy.float64, copy=False)
            ensure_state_width(state, features.shape[1])
            update_numeric_stats(features, state)
        summary = finalize_numeric_stats(path, "csv_features", state)
        if label_values:
            labels = numpy.asarray(label_values, dtype=numpy.float64)
            summary.label_min = float(numpy.nanmin(labels))
            summary.label_max = float(numpy.nanmax(labels))
            summary.label_unique_count = int(numpy.unique(labels).shape[0])
        if rows_seen >= max_rows:
            summary.error = f"sampled first {max_rows} rows"
        return summary
    except Exception as error:
        return RangeSummary(path=path, kind="csv", error=str(error))


def summarize_synthetic_manifest(path: Path, manifest: dict[str, Any], max_batches: int) -> RangeSummary | None:
    batches_by_class = manifest.get("batches_by_class")
    if not isinstance(batches_by_class, dict):
        return None
    state = new_stats_state()
    labels = []
    batches_seen = 0
    for class_label, batches in batches_by_class.items():
        for batch in batches:
            if batches_seen >= max_batches:
                break
            batch_path = Path(batch.get("path", ""))
            if not batch_path.is_absolute():
                batch_path = (path.parent / batch_path).resolve()
            if not batch_path.exists():
                batch_path = Path(batch.get("path", ""))
            try:
                if batch_path.suffix == ".npy":
                    values = numpy.load(batch_path, mmap_mode="r", allow_pickle=False)
                    if "offset_start" in batch:
                        values = values[int(batch["offset_start"]):int(batch["offset_end"])]
                else:
                    values = numpy.loadtxt(batch_path, delimiter=",", dtype=numpy.float32)
                values = numpy.asarray(values, dtype=numpy.float32)
                if values.ndim == 1:
                    values = values.reshape(1, -1)
                ensure_state_width(state, values.shape[1])
                update_numeric_stats(values, state)
                labels.extend([int(class_label)] * int(values.shape[0]))
                batches_seen += 1
            except Exception:
                continue
    if state["rows"] == 0:
        return None
    summary = finalize_numeric_stats(path, "synthetic_manifest_batches", state)
    summary.data_space = manifest.get("data_space")
    summary.transform_id = manifest.get("transform_id")
    history = manifest.get("transform_history")
    summary.transform_history_count = len(history) if isinstance(history, list) else None
    if labels:
        labels_array = numpy.asarray(labels)
        summary.label_min = float(labels_array.min())
        summary.label_max = float(labels_array.max())
        summary.label_unique_count = int(numpy.unique(labels_array).shape[0])
    if batches_seen >= max_batches:
        summary.error = f"sampled first {max_batches} manifest batches"
    return summary


def summarize_label_selection(path: Path, data: dict[str, Any]) -> RangeSummary | None:
    counts = data.get("selected_counts_by_class") or data.get("observed_counts_by_class")
    if not isinstance(counts, dict) or not counts:
        return None
    try:
        labels = numpy.asarray([int(label) for label in counts.keys()], dtype=numpy.int64)
        total = sum(int(count) for count in counts.values())
    except (TypeError, ValueError):
        return None
    return RangeSummary(
        path=path,
        kind="json_label_selection",
        rows=int(total),
        columns=1,
        label_min=float(labels.min()) if labels.size else None,
        label_max=float(labels.max()) if labels.size else None,
        label_unique_count=int(labels.shape[0]),
        error=f"split={data.get('split_name', 'unknown')}; counts={counts}",
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def collect_files(result_dir: Path, include_related: bool, raw_root: Path) -> dict[str, list[Path]]:
    files = [path for path in result_dir.rglob("*") if path.is_file()]
    if include_related:
        parent = result_dir.parent
        files.extend(parent.glob("*.log"))
        metrics_path = parent / "baseline_real_only" / "metrics.json"
        if metrics_path.is_file():
            files.append(metrics_path)
        if raw_root.is_dir():
            files.extend(sorted(raw_root.glob("*.npy")))
    files = _dedupe_paths(files)
    return {
        "json": [path for path in files if path.suffix == ".json"],
        "logs": [path for path in files if path.suffix == ".log"],
        "joblib": [path for path in files if path.suffix == ".joblib"],
        "arrays": [path for path in files if path.suffix in NUMERIC_SUFFIXES],
    }


def collect_metrics(json_paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in json_paths:
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
            continue
        metrics = {}
        for key, value in data["metrics"].items():
            if is_number(value):
                metrics[key] = float(value)
        if metrics:
            summaries.append({
                "path": str(path),
                "mode": data.get("mode"),
                "classifier": data.get("classifier"),
                "metrics": metrics,
                "scaler": data.get("scaler"),
            })
    return summaries


def collect_manifests(json_paths: list[Path]) -> list[tuple[Path, Any]]:
    manifests = []
    for path in json_paths:
        if (
            path.name in JSON_NAMES
            or path.name.endswith(".space.json")
            or path.name.endswith("_stratified_selection.json")
            or "manifest" in path.name.lower()
        ):
            data = load_json(path)
            if data is not None:
                manifests.append((path, data))
    return manifests


def summarize_scaler(path: Path) -> dict[str, Any]:
    try:
        state = joblib.load(path)
    except Exception as error:
        return {"path": str(path), "error": str(error)}
    if isinstance(state, dict):
        scaler = state.get("scaler")
        return {
            "path": str(path),
            "operation": state.get("operation") or state.get("scaler_name"),
            "stage": state.get("stage"),
            "fit_split": state.get("fit_split"),
            "transform_id": state.get("transform_id"),
            "policy": state.get("policy"),
            "has_data_min": hasattr(scaler, "data_min_"),
            "has_mean": hasattr(scaler, "mean_"),
        }
    return {"path": str(path), "type": type(state).__name__}


def read_log_text(log_paths: list[Path]) -> str:
    chunks = []
    for path in log_paths:
        try:
            chunks.append(f"\n### {path}\n")
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:2_000_000])
        except Exception:
            continue
    return "\n".join(chunks)


def log_evidence(log_text: str) -> dict[str, list[str]]:
    patterns = {
        "fit_transform": r"fit_transform|fit transform",
        "scaler": r"scaler|MinMaxScaler|StandardScaler|minmax|standard",
        "inverse_transform": r"inverse_transform|inverse transform",
        "synthetic": r"synthetic|DataOutput|synthetic_batches",
        "sgd": r"SGDClassifier|StochasticGradientDescent",
        "loader_normalization": r"normalization|normalize|normalized",
        "command": r"Command line|Campaign Command",
    }
    evidence = {}
    lines = log_text.splitlines()
    for key, pattern in patterns.items():
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [line.strip() for line in lines if regex.search(line)]
        evidence[key] = matches[:25]
    return evidence


def manifest_transform_entries(manifests: list[tuple[Path, Any]]) -> list[dict[str, Any]]:
    entries = []
    for path, data in manifests:
        if isinstance(data, dict):
            transformations = data.get("transformations") or data.get("transform_history") or []
            if isinstance(transformations, list):
                for entry in transformations:
                    if isinstance(entry, dict):
                        entries.append({"manifest_path": str(path), **entry})
            splits = data.get("split_usage")
            if isinstance(splits, dict):
                for split_name, split_data in splits.items():
                    if isinstance(split_data, dict):
                        entries.append({
                            "manifest_path": str(path),
                            "split": split_name,
                            "transform_id": split_data.get("transform_id"),
                            "used_scaler_fit_split": split_data.get("used_scaler_fit_split"),
                        })
    return entries


def infer_transform_count(role: str, ranges: list[RangeSummary], transform_entries: list[dict[str, Any]]) -> tuple[str, str]:
    role_ranges = [
        summary for summary in ranges
        if path_role(summary.path) == role and summary.feature_min is not None and summary.feature_max is not None
    ]
    explicit_entries = [
        entry for entry in transform_entries
        if entry.get("split") in {role.replace("_x", ""), "evaluation" if role == "evaluation_x" else role}
    ]
    operations = [
        entry for entry in transform_entries
        if entry.get("operation") not in {None, "preserve"} and (
            entry.get("split") in {role.replace("_x", ""), "evaluation" if role == "evaluation_x" else role}
            or entry.get("output_space") in {"generator", "classifier", "transformed"}
        )
    ]
    if explicit_entries or operations:
        return str(len(operations) or len(explicit_entries)), "manifest transform entries"
    if role_ranges:
        range_descriptions = ", ".join(
            f"{summary.path.name}=[{fmt_value(summary.feature_min)},{fmt_value(summary.feature_max)}]"
            for summary in role_ranges[:4]
        )
        return "unknown", f"ranges available but no transform history: {range_descriptions}"
    return "unknown", "no matching array/manifest evidence"


def collect_range_summaries(
        file_groups: dict[str, list[Path]],
        manifests: list[tuple[Path, Any]],
        max_csv_rows: int,
        max_manifest_batches: int,
        result_dir: Path,
        raw_root: Path) -> list[RangeSummary]:
    def assign_scope(summary: RangeSummary) -> RangeSummary:
        try:
            summary.path.relative_to(result_dir)
            summary.scope = "result_dir"
        except ValueError:
            try:
                summary.path.relative_to(raw_root.resolve())
                summary.scope = "raw_reference"
            except ValueError:
                summary.scope = "related"
        return summary

    summaries = []
    manifest_paths = {path for path, _ in manifests}
    for path, data in manifests:
        if isinstance(data, dict):
            label_summary = summarize_label_selection(path, data)
            if label_summary is not None:
                summaries.append(assign_scope(label_summary))
            manifest_summary = summarize_synthetic_manifest(path, data, max_manifest_batches)
            if manifest_summary is not None:
                summaries.append(assign_scope(manifest_summary))
    for path in file_groups["arrays"]:
        if path in manifest_paths or path.name.endswith(".space.json"):
            continue
        if path.suffix == ".npy":
            summaries.append(assign_scope(summarize_npy(path)))
        elif path.suffix in {".csv", ".txt"}:
            summaries.append(assign_scope(summarize_csv(path, max_csv_rows)))
    return summaries


def build_checklist(
        ranges: list[RangeSummary],
        manifests: list[tuple[Path, Any]],
        scalers: list[dict[str, Any]],
        transform_entries: list[dict[str, Any]],
        evidence: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    checks = []
    for index, (key, label) in enumerate([
        ("train_x", "Quantas vezes train_x foi transformado"),
        ("test_x", "Quantas vezes test_x foi transformado"),
        ("synthetic_x", "Quantas vezes synthetic_x foi transformado"),
    ], start=1):
        count, reason = infer_transform_count(key, ranges, transform_entries)
        checks.append((str(index), count, reason))

    fit_splits = [scaler.get("fit_split") for scaler in scalers if scaler.get("fit_split")]
    manifest_fit_splits = [
        data.get("train_fit", {}).get("fit_split")
        for _, data in manifests
        if isinstance(data, dict) and isinstance(data.get("train_fit"), dict)
    ]
    all_fit_splits = [split for split in [*fit_splits, *manifest_fit_splits] if split]
    if all_fit_splits:
        status = "yes" if set(all_fit_splits) == {"train"} else "no"
        detail = f"fit_split values: {sorted(set(all_fit_splits))}"
    else:
        status = "unknown"
        detail = "no scaler fit metadata found"
    checks.append(("4", status, detail))

    synthetic_fit_evidence = [
        entry for entry in transform_entries
        if str(entry.get("fit_split", "")).lower() == "synthetic"
        or str(entry.get("split", "")).lower() == "synthetic"
    ]
    checks.append((
        "5",
        "yes" if synthetic_fit_evidence else "unknown",
        "synthetic fit metadata found" if synthetic_fit_evidence else "no explicit synthetic scaler fit evidence found",
    ))

    transform_ids = {
        entry.get("transform_id") for entry in transform_entries
        if entry.get("transform_id")
    }
    if transform_ids:
        checks.append((
            "6",
            "yes" if len(transform_ids) == 1 else "no",
            f"transform_id values: {sorted(transform_ids)}",
        ))
    else:
        checks.append(("6", "unknown", "no transform_id evidence found"))

    synthetic_ranges = [
        summary for summary in ranges
        if path_role(summary.path) == "synthetic_x"
        and summary.feature_min is not None
        and summary.feature_max is not None
    ]
    synthetic_01 = [summary for summary in synthetic_ranges if in_range(summary, GENERATOR_MIN, GENERATOR_MAX)]
    checks.append((
        "7",
        "yes" if synthetic_01 else ("no" if synthetic_ranges else "unknown"),
        "synthetic ranges in [0,1]" if synthetic_01 else "no synthetic ranges found" if not synthetic_ranges else "synthetic ranges not fully in [0,1]",
    ))

    real_ranges = [
        summary for summary in ranges
        if path_role(summary.path) in {"train_x", "test_x", "evaluation_x"}
        and summary.feature_min is not None
        and summary.feature_max is not None
    ]
    real_source = [summary for summary in real_ranges if in_range(summary, APPCLASSNET_SOURCE_MIN, APPCLASSNET_SOURCE_MAX)]
    checks.append((
        "8",
        "yes" if real_source else ("no" if real_ranges else "unknown"),
        "real ranges compatible with [-0.5,0.5]" if real_source else "no real ranges found" if not real_ranges else "real ranges not compatible with [-0.5,0.5]",
    ))

    inverse_flags = [
        data.get("inverse_transform_synthetic")
        for _, data in manifests
        if isinstance(data, dict) and "inverse_transform_synthetic" in data
    ]
    inverse_log = bool(evidence.get("inverse_transform"))
    synthetic_source_ranges = [
        summary for summary in ranges
        if path_role(summary.path) == "synthetic_x"
        and summary.feature_min is not None
        and summary.feature_max is not None
        and in_range(summary, APPCLASSNET_SOURCE_MIN, APPCLASSNET_SOURCE_MAX)
    ]
    synthetic_generator_ranges = [
        summary for summary in ranges
        if path_role(summary.path) == "synthetic_x"
        and summary.feature_min is not None
        and summary.feature_max is not None
        and in_range(summary, GENERATOR_MIN, GENERATOR_MAX)
        and not in_range(summary, APPCLASSNET_SOURCE_MIN, APPCLASSNET_SOURCE_MAX)
    ]
    if synthetic_source_ranges and (True in inverse_flags or inverse_log):
        inverse_status = "yes"
    elif synthetic_generator_ranges and False in inverse_flags:
        inverse_status = "no"
    elif True in inverse_flags or inverse_log:
        inverse_status = "configured_only"
    elif False in inverse_flags:
        inverse_status = "no"
    else:
        inverse_status = "unknown"
    checks.append((
        "9",
        inverse_status,
        f"inverse flags={inverse_flags}; log_matches={len(evidence.get('inverse_transform', []))}",
    ))

    generator_spaces = [
        data.get("generator_input_space")
        for _, data in manifests
        if isinstance(data, dict) and data.get("generator_input_space")
    ]
    evaluation_spaces = [
        data.get("evaluation_space")
        for _, data in manifests
        if isinstance(data, dict) and data.get("evaluation_space")
    ]
    synthetic_spaces = [
        data.get("synthetic_output_space") or data.get("data_space")
        for _, data in manifests
        if isinstance(data, dict) and (data.get("synthetic_output_space") or data.get("data_space"))
    ]
    mismatch = bool(generator_spaces and evaluation_spaces and synthetic_spaces and set(synthetic_spaces) != set(evaluation_spaces))
    checks.append((
        "10",
        "yes" if mismatch else "unknown",
        f"generator_input_space={generator_spaces}; synthetic_space={synthetic_spaces}; evaluation_space={evaluation_spaces}",
    ))

    mode_policies = [
        (path, data.get("source_profile"), data.get("feature_transform"), data.get("generator_transform"))
        for path, data in manifests
        if isinstance(data, dict) and ("source_profile" in data or "feature_transform" in data)
    ]
    checks.append((
        "11",
        "unknown",
        "single mode or insufficient mode policy manifests found" if len(mode_policies) < 2 else str(mode_policies),
    ))

    sgd_scaler_evidence = [line for line in evidence.get("sgd", []) if "scaler" in line.lower() or "standard" in line.lower()]
    checks.append((
        "12",
        "unknown" if not sgd_scaler_evidence else "yes",
        "no SGD scaler evidence found" if not sgd_scaler_evidence else "; ".join(sgd_scaler_evidence[:3]),
    ))

    loader_norm = evidence.get("loader_normalization", [])
    checks.append((
        "13",
        "unknown" if not loader_norm else "possible",
        "loader/classifier normalization log evidence: " + "; ".join(loader_norm[:5]) if loader_norm else "no loader normalization evidence found",
    ))

    copied_transformed = [
        summary for summary in ranges
        if path_role(summary.path) in {"train_x", "test_x", "evaluation_x"}
        and summary.feature_min is not None
        and summary.feature_max is not None
        and in_range(summary, GENERATOR_MIN, GENERATOR_MAX)
        and not in_range(summary, APPCLASSNET_SOURCE_MIN, APPCLASSNET_SOURCE_MAX)
    ]
    checks.append((
        "14",
        "possible" if copied_transformed else "unknown",
        "real-looking files in [0,1]: " + ", ".join(str(item.path) for item in copied_transformed[:5])
        if copied_transformed else "no conclusive copied-transformed-as-raw evidence",
    ))

    label_summaries = [summary for summary in ranges if summary.label_unique_count is not None]
    invalid_labels = [
        summary for summary in label_summaries
        if summary.label_min is not None and (summary.label_min < 0 or summary.label_max is None)
    ]
    checks.append((
        "15",
        "yes" if label_summaries and not invalid_labels else "unknown",
        "label summaries found: " + "; ".join(
            f"{summary.path.name} unique={summary.label_unique_count} min={fmt_value(summary.label_min)} max={fmt_value(summary.label_max)}"
            for summary in label_summaries[:8]
        ) if label_summaries else "no label arrays/columns found",
    ))
    return checks


def infer_probable_cause(
        checks: list[tuple[str, str, str]],
        ranges: list[RangeSummary],
        metric_summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    evidence = []
    by_number = {number: (status, detail) for number, status, detail in checks}
    synthetic_in_01 = by_number.get("7", ("unknown", ""))[0] == "yes"
    real_source = by_number.get("8", ("unknown", ""))[0] == "yes"
    inverse_no = by_number.get("9", ("unknown", ""))[0] == "no"
    generator_eval_mismatch = by_number.get("10", ("unknown", ""))[0] == "yes"
    synthetic_ranges = [
        summary for summary in ranges
        if path_role(summary.path) == "synthetic_x"
        and summary.feature_min is not None
        and summary.feature_max is not None
    ]
    near_0005_metrics = []
    observed_metrics = []
    for metric_summary in metric_summaries:
        for metric_name, metric_value in metric_summary.get("metrics", {}).items():
            observed_metrics.append(f"{Path(metric_summary['path']).name}:{metric_name}={metric_value:.6g}")
            if 0.0 <= metric_value <= 0.01:
                near_0005_metrics.append(f"{Path(metric_summary['path']).name}:{metric_name}={metric_value:.6g}")

    if synthetic_in_01 and real_source and (inverse_no or generator_eval_mismatch):
        evidence.extend([
            by_number.get("7", ("", ""))[1],
            by_number.get("8", ("", ""))[1],
            by_number.get("9", ("", ""))[1],
            by_number.get("10", ("", ""))[1],
        ])
        return (
            "Provavel inconsistencia de espaco numerico: sintetico em [0,1] comparado com real em source space.",
            evidence,
        )

    if not synthetic_ranges:
        details = ["Nenhum synthetic_x foi encontrado no diretorio auditado ou artefatos relacionados."]
        if observed_metrics:
            details.append("Metricas observadas: " + "; ".join(observed_metrics[:12]))
        if not near_0005_metrics:
            details.append("Nenhuma metrica proxima de 0.005 foi encontrada nos JSONs disponiveis.")
        return (
            "A causa nao pode ser determinada: faltam sinteticos/manifests da execucao que teria produzido metricas ~0.005.",
            details,
        )

    if not ranges:
        return (
            "A causa nao pode ser determinada: o diretorio auditado nao contem arrays, manifests ou logs suficientes.",
            ["Nenhum range de entrada/sintetico foi calculado."],
        )

    unknown_count = sum(1 for _, status, _ in checks if status == "unknown")
    if unknown_count >= 8:
        return (
            "A causa nao pode ser determinada com os artefatos disponiveis.",
            ["Muitas verificacoes ficaram sem evidencia direta; nao ha base para atribuir a causa ao MinMaxScaler sozinho."],
        )

    return (
        "A causa provavel nao pode ser isolada automaticamente; revisar inconsistencias e evidencias abaixo.",
        ["Nao houve combinacao suficiente de sintetico [0,1], real source e ausencia de inverse_transform."],
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [str(value).replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def build_report(
        result_dir: Path,
        ranges: list[RangeSummary],
        manifests: list[tuple[Path, Any]],
        scalers: list[dict[str, Any]],
        transform_entries: list[dict[str, Any]],
        evidence: dict[str, list[str]],
        checks: list[tuple[str, str, str]],
        metric_summaries: list[dict[str, Any]]) -> str:
    cause, cause_evidence = infer_probable_cause(checks, ranges, metric_summaries)
    lines = [
        "# Preprocessing Audit",
        "",
        f"- Result dir: `{result_dir}`",
        f"- Generated at: `{datetime.datetime.now(datetime.timezone.utc).isoformat()}`",
        f"- Ranges summarized: `{len(ranges)}`",
        f"- Manifests/JSON metadata inspected: `{len(manifests)}`",
        f"- Scaler joblib files inspected: `{len(scalers)}`",
        "",
        "## Causa Provavel",
        "",
        cause,
        "",
        "### Evidencias da causa",
        "",
    ]
    lines.extend(f"- {item}" for item in cause_evidence)

    lines.extend(["", "## Metricas Observadas", ""])
    if metric_summaries:
        metric_rows = []
        for metric_summary in metric_summaries:
            metrics_text = ", ".join(
                f"{name}={value:.6g}" for name, value in sorted(metric_summary.get("metrics", {}).items())
            )
            scaler = metric_summary.get("scaler") if isinstance(metric_summary.get("scaler"), dict) else {}
            metric_rows.append([
                metric_summary.get("path", ""),
                metric_summary.get("mode", ""),
                metric_summary.get("classifier", ""),
                metrics_text,
                scaler.get("legacy_name") or scaler.get("name") or "",
                scaler.get("classifier_transform") or "",
                scaler.get("transform_id") or "",
            ])
        lines.extend(markdown_table(
            ["path", "mode", "classifier", "metrics", "scaler", "classifier_transform", "transform_id"],
            metric_rows,
        ))
    else:
        lines.append("Nenhum JSON de metricas foi encontrado.")

    lines.extend(["", "## Ranges Encontrados", ""])
    range_rows = []
    for summary in ranges:
        range_rows.append([
            str(summary.path),
            summary.scope,
            summary.kind,
            path_role(summary.path),
            fmt_value(summary.rows),
            fmt_value(summary.columns),
            f"[{fmt_value(summary.feature_min)}, {fmt_value(summary.feature_max)}]",
            fmt_value(summary.feature_mean),
            fmt_value(summary.feature_std),
            fmt_value(summary.data_space),
            fmt_value(summary.transform_id),
            fmt_value(summary.transform_history_count),
            fmt_value(summary.label_unique_count),
            summary.error or "",
        ])
    if range_rows:
        lines.extend(markdown_table(
            [
                "path", "scope", "kind", "role", "rows", "cols", "feature range", "mean", "std",
                "data_space", "transform_id", "history_n", "labels_n", "notes",
            ],
            range_rows,
        ))
    else:
        lines.append("Nenhum array ou CSV com features foi encontrado no diretorio auditado.")

    lines.extend(["", "## Transform History / Transform IDs", ""])
    if transform_entries:
        lines.extend(markdown_table(
            ["manifest", "operation", "fit_split", "split", "input_space", "output_space", "transform_id"],
            [
                [
                    entry.get("manifest_path", ""),
                    entry.get("operation", ""),
                    entry.get("fit_split") or entry.get("used_scaler_fit_split", ""),
                    entry.get("split", ""),
                    entry.get("input_space", ""),
                    entry.get("output_space", ""),
                    entry.get("transform_id", ""),
                ]
                for entry in transform_entries
            ],
        ))
    else:
        lines.append("Nenhum `transform_history` ou `transform_id` foi encontrado.")

    lines.extend(["", "## Scalers", ""])
    if scalers:
        lines.extend(markdown_table(
            ["path", "operation", "stage", "fit_split", "transform_id", "has_data_min", "has_mean", "error"],
            [
                [
                    scaler.get("path", ""),
                    scaler.get("operation", ""),
                    scaler.get("stage", ""),
                    scaler.get("fit_split", ""),
                    scaler.get("transform_id", ""),
                    scaler.get("has_data_min", ""),
                    scaler.get("has_mean", ""),
                    scaler.get("error", ""),
                ]
                for scaler in scalers
            ],
        ))
    else:
        lines.append("Nenhum arquivo `.joblib` de scaler foi encontrado.")

    lines.extend(["", "## Verificacoes Solicitadas", ""])
    lines.extend(markdown_table(["#", "status", "evidencia"], [[number, status, detail] for number, status, detail in checks]))

    inconsistencies = []
    for number, status, detail in checks:
        if status in {"no", "possible", "yes"} and number in {"5", "10", "13", "14"}:
            inconsistencies.append(f"Check {number}: {status}. {detail}")
    if not inconsistencies and "nao pode ser determinada" in cause.lower():
        inconsistencies.append("Nao ha evidencia suficiente para comprovar uma inconsistencia especifica.")
    lines.extend(["", "## Inconsistencias", ""])
    lines.extend(f"- {item}" for item in inconsistencies)

    lines.extend(["", "## Logs: Evidencias Relevantes", ""])
    for key, matches in evidence.items():
        lines.append(f"### {key}")
        if not matches:
            lines.append("- nenhum match")
            continue
        lines.extend(f"- `{match[:240]}`" for match in matches[:10])
        lines.append("")

    lines.extend(["", "## Correcoes Recomendadas", ""])
    lines.extend([
        "- Preservar AppClassNet em source space por padrao e aplicar MinMax somente quando o estagio exigir explicitamente.",
        "- Ajustar scalers apenas no treino e persistir `transform_id` para train/valid/test/sintetico.",
        "- Salvar sinteticos no source space por padrao; se forem salvos em generator space, bloquear avaliacao.",
        "- Manter manifests junto dos resultados antigos e novos para tornar a causa auditavel.",
        "- Reexecutar a auditoria no diretorio completo da execucao que produziu metricas ~0.005, incluindo `preprocessing_manifest.json`, `scaler.joblib`, arrays reais e sinteticos.",
    ])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")

    raw_root = Path(args.raw_root)
    if not raw_root.is_absolute():
        raw_root = (Path.cwd() / raw_root).resolve()

    file_groups = collect_files(
        result_dir,
        include_related=not args.no_related_artifacts,
        raw_root=raw_root,
    )
    manifests = collect_manifests(file_groups["json"])
    ranges = collect_range_summaries(
        file_groups,
        manifests,
        args.max_csv_rows,
        args.max_manifest_batches,
        result_dir,
        raw_root,
    )
    scalers = [summarize_scaler(path) for path in file_groups["joblib"]]
    log_text = read_log_text(file_groups["logs"])
    evidence = log_evidence(log_text)
    transform_entries = manifest_transform_entries(manifests)
    metric_summaries = collect_metrics(file_groups["json"])
    checks = build_checklist(ranges, manifests, scalers, transform_entries, evidence)
    report = build_report(
        result_dir,
        ranges,
        manifests,
        scalers,
        transform_entries,
        evidence,
        checks,
        metric_summaries,
    )

    output_path = result_dir / "PREPROCESSING_AUDIT.md"
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
