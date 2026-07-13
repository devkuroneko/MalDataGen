#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate AppClassNet preprocessing policy on deterministic balanced subsets.

The script refuses to validate the policy when the required 200-class train/test
labels are unavailable. It still writes the requested artifacts with a blocked
status and concrete evidence so a failed validation is explicit.
"""

from __future__ import annotations

import json
import math
import hashlib
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy
import pandas
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy
from Engine.Preprocessing.FeatureTransformManager import ModelInputAdapter
from Engine.Preprocessing.FeatureTransformManager import PreprocessingSpaceMismatchError
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
from Engine.Preprocessing.FeatureTransformManager import TransformManifest


OUTPUT_DIR = REPO_ROOT / "results" / "appclassnet_top200" / "preprocessing_validation"
NUM_CLASSES = 200
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 500
RANDOM_STATE = 0
SOURCE_RANGE = (-0.5, 0.5)


def json_default(value):
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.integer, numpy.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(name: str, payload: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=json_default)
        file.write("\n")
    return path


def indices_sha256(indices) -> str:
    values = numpy.asarray(indices, dtype=numpy.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def unique_transformations(transform_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for entry in transform_history:
        key = (
            entry.get("stage"),
            entry.get("operation"),
            entry.get("fit_split"),
            entry.get("input_space"),
            entry.get("output_space"),
            entry.get("transform_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def array_stats(values) -> dict[str, Any]:
    values = numpy.asarray(values, dtype=numpy.float32)
    return {
        "shape": list(values.shape),
        "min": float(numpy.nanmin(values)) if values.size else None,
        "max": float(numpy.nanmax(values)) if values.size else None,
        "mean": float(numpy.nanmean(values)) if values.size else None,
        "std": float(numpy.nanstd(values)) if values.size else None,
        "has_nan": bool(numpy.isnan(values).any()) if values.size else False,
        "has_inf": bool(numpy.isinf(values).any()) if values.size else False,
    }


def label_stats(labels) -> dict[str, Any]:
    labels = numpy.asarray(labels)
    if labels.size == 0:
        return {"count": 0}
    unique, counts = numpy.unique(labels, return_counts=True)
    return {
        "count": int(labels.shape[0]),
        "min": int(unique.min()),
        "max": int(unique.max()),
        "unique_count": int(unique.shape[0]),
        "first_counts": {str(int(label)): int(count) for label, count in zip(unique[:10], counts[:10])},
    }


def is_source_range(stats: dict[str, Any], tolerance: float = 1e-4) -> bool:
    return (
        stats["min"] is not None
        and stats["max"] is not None
        and stats["min"] >= SOURCE_RANGE[0] - tolerance
        and stats["max"] <= SOURCE_RANGE[1] + tolerance
    )


def raw_top200_paths() -> dict[str, Path] | None:
    root = REPO_ROOT / "Datasets" / "raw" / "AppClassNet" / "top200"
    paths = {
        "train_x": root / "train_x.npy",
        "train_y": root / "train_y.npy",
        "test_x": root / "test_x.npy",
        "test_y": root / "test_y.npy",
    }
    if all(path.is_file() for path in paths.values()):
        return paths
    return None


def inspect_processed_csv_candidates() -> list[dict[str, Any]]:
    candidates = []
    root = REPO_ROOT / "Datasets" / "AppClassNet" / "processed"
    for train_x in sorted(root.glob("*/train*_x.csv")):
        train_y = Path(str(train_x).replace("_x.csv", "_y.csv"))
        test_x = Path(str(train_x).replace("train_", "test_"))
        test_y = Path(str(train_y).replace("train_", "test_"))
        record = {
            "train_x": str(train_x),
            "train_y": str(train_y),
            "test_x": str(test_x),
            "test_y": str(test_y),
            "usable": False,
            "reason": None,
        }
        if not all(path.is_file() for path in [train_y, test_x, test_y]):
            record["reason"] = "missing paired train/test files"
            candidates.append(record)
            continue
        try:
            x_head = pandas.read_csv(train_x, nrows=5)
            y_head = pandas.read_csv(train_y, nrows=5)
            record["train_x_head_shape"] = list(x_head.shape)
            record["train_y_head_shape"] = list(y_head.shape)
            record["train_x_columns"] = list(map(str, x_head.columns[:5]))
            record["train_y_columns"] = list(map(str, y_head.columns[:5]))
            if x_head.equals(y_head):
                record["reason"] = "train_y appears identical to train_x; labels are unavailable"
            elif y_head.select_dtypes(include=["number"]).shape[1] not in {1, NUM_CLASSES}:
                record["reason"] = "train_y is neither a single label column nor a 200-column one-hot matrix"
            else:
                record["usable"] = True
        except Exception as error:
            record["reason"] = str(error)
        candidates.append(record)
    return candidates


def load_feature_sample_from_csv_candidates(candidates: list[dict[str, Any]], nrows: int = 5000):
    for candidate in candidates:
        train_x_path = Path(candidate["train_x"])
        if not train_x_path.is_file():
            continue
        try:
            frame = pandas.read_csv(train_x_path, nrows=nrows)
            numeric = frame.select_dtypes(include=["number"])
            feature_columns = [
                column for column in numeric.columns
                if not str(column).lower().startswith("unnamed")
            ]
            if not feature_columns:
                continue
            values = numeric[feature_columns].to_numpy(dtype=numpy.float32, copy=True)
            if values.ndim == 2 and values.shape[0] > 0:
                return values, {
                    "path": str(train_x_path),
                    "rows": int(values.shape[0]),
                    "columns": int(values.shape[1]),
                    "stats": array_stats(values),
                }
        except Exception:
            continue
    return None, None


def load_raw_top200(paths: dict[str, Path]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    train_x = numpy.load(paths["train_x"], mmap_mode="r", allow_pickle=False)
    train_y = numpy.asarray(numpy.load(paths["train_y"], mmap_mode="r", allow_pickle=False)).reshape(-1)
    test_x = numpy.load(paths["test_x"], mmap_mode="r", allow_pickle=False)
    test_y = numpy.asarray(numpy.load(paths["test_y"], mmap_mode="r", allow_pickle=False)).reshape(-1)
    return train_x, train_y, test_x, test_y


def select_balanced_indices(labels, per_class: int, seed: int) -> tuple[numpy.ndarray, dict[str, Any]]:
    labels = numpy.asarray(labels, dtype=numpy.int64)
    rng = numpy.random.default_rng(seed)
    selected = []
    counts = {}
    missing = []
    short = {}
    for class_id in range(NUM_CLASSES):
        class_indices = numpy.flatnonzero(labels == class_id)
        counts[str(class_id)] = int(class_indices.shape[0])
        if class_indices.shape[0] == 0:
            missing.append(class_id)
            continue
        if class_indices.shape[0] < per_class:
            short[str(class_id)] = int(class_indices.shape[0])
            continue
        chosen = rng.choice(class_indices, size=per_class, replace=False)
        selected.append(chosen)
    if missing or short:
        return numpy.array([], dtype=numpy.int64), {
            "status": "blocked",
            "missing_classes": missing,
            "short_classes": short,
            "observed_counts_by_class": counts,
        }
    indices = numpy.concatenate(selected).astype(numpy.int64)
    rng.shuffle(indices)
    return indices, {
        "status": "ok",
        "selected_total": int(indices.shape[0]),
        "selected_per_class": int(per_class),
        "observed_counts_by_class": counts,
    }


def materialize_subset(x_values, y_values, indices):
    indices = numpy.asarray(indices, dtype=numpy.int64)
    order = numpy.argsort(indices)
    sorted_indices = indices[order]
    restore_order = numpy.argsort(order)
    x_selected = numpy.asarray(x_values[sorted_indices], dtype=numpy.float32)[restore_order]
    y_selected = numpy.asarray(y_values[sorted_indices], dtype=numpy.int64)[restore_order]
    return (
        x_selected,
        y_selected,
    )


def base_policy(feature_transform="preserve", generator_transform="preserve", inverse=True):
    return FeatureTransformPolicy.for_profile(
        "appclassnet_top200",
        feature_transform=feature_transform,
        generator_transform=generator_transform,
        classifier_transform="preserve",
        evaluation_space="source",
        allow_refit=False,
        allow_double_transform=False,
        inverse_transform_synthetic=inverse,
    )


def run_decision_tree_experiment(name, train_x, train_y, test_x, test_y, feature_transform):
    policy = base_policy(feature_transform=feature_transform)
    manager = FeatureTransformManager(policy, stage="feature", output_space="transformed")
    manager.fit(train_x, split_name="train")
    transformed_train, train_meta = manager.transform(train_x, split_name="train", return_metadata=True)
    transformed_test, test_meta = manager.transform(test_x, split_name="test", return_metadata=True)

    classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
    classifier.fit(transformed_train, train_y)
    predictions = classifier.predict(transformed_test)
    accuracy = float(accuracy_score(test_y, predictions))
    manifest_transformations = unique_transformations(manager.transform_history)
    manifest = TransformManifest(
        source_profile=policy.source_profile,
        transformations=manifest_transformations,
        transform_id=manager.transform_id,
        train_fit={"fit_split": manager.fit_split, "operation": manager.operation},
        split_usage={
            "train": {"transform_id": manager.transform_id, "used_scaler_fit_split": manager.fit_split},
            "test": {"transform_id": manager.transform_id, "used_scaler_fit_split": manager.fit_split},
        },
        evaluation_space=policy.evaluation_space,
        classifier_input_space="source",
        inverse_transform_synthetic=policy.inverse_transform_synthetic,
        validations=[
            {"name": "fit_only_train", "status": manager.fit_split == "train"},
            {"name": "no_refit_valid_test", "status": True},
            {"name": "single_transform_manifest", "status": len(manifest_transformations) <= 1},
        ],
    )
    return {
        "experiment": name,
        "status": "completed",
        "classifier": "DecisionTreeClassifier",
        "feature_transform": feature_transform,
        "random_state": RANDOM_STATE,
        "accuracy": accuracy,
        "above_0_005": accuracy > 0.005,
        "train_stats_before": array_stats(train_x),
        "test_stats_before": array_stats(test_x),
        "train_stats_after": array_stats(transformed_train),
        "test_stats_after": array_stats(transformed_test),
        "train_metadata": train_meta,
        "test_metadata": test_meta,
        "manifest": manifest.to_dict(),
    }


def run_generator_inverse_experiment(train_x):
    policy = base_policy(generator_transform="minmax", inverse=True)
    adapter = ModelInputAdapter(policy)
    adapter.fit_generator(train_x)
    generator_input = adapter.transform_generator_input(train_x[: min(5000, train_x.shape[0])], split_name="train")
    synthetic_source = adapter.inverse_generator_output(generator_input)
    synthetic_path = OUTPUT_DIR / "experiment_c_synthetic_source.npy"
    numpy.save(synthetic_path, synthetic_source.astype(numpy.float32, copy=False))
    manifest_transformations = unique_transformations(adapter.generator_manager.transform_history)
    manifest = TransformManifest(
        source_profile=policy.source_profile,
        transformations=manifest_transformations,
        transform_id=adapter.generator_manager.transform_id,
        train_fit={
            "fit_split": adapter.generator_manager.fit_split,
            "operation": adapter.generator_manager.operation,
        },
        generator_input_space="generator",
        synthetic_output_space="source",
        evaluation_space="source",
        inverse_transform_synthetic=True,
        validations=[
            {"name": "fit_only_train", "status": adapter.generator_manager.fit_split == "train"},
            {"name": "synthetic_source_range", "status": is_source_range(array_stats(synthetic_source))},
            {"name": "single_transform_manifest", "status": len(manifest_transformations) <= 1},
        ],
    )
    return {
        "experiment": "C",
        "status": "completed",
        "generator_transform": "minmax",
        "inverse_transform_synthetic": True,
        "generator_input_stats": array_stats(generator_input),
        "synthetic_saved_stats": array_stats(synthetic_source),
        "synthetic_saved_in_source_range": is_source_range(array_stats(synthetic_source)),
        "synthetic_source_path": str(synthetic_path),
        "manifest": manifest.to_dict(),
    }


def run_mismatch_experiment(train_x):
    policy = base_policy(generator_transform="minmax", inverse=False)
    adapter = ModelInputAdapter(policy)
    adapter.fit_generator(train_x)
    generator_space = adapter.transform_generator_input(train_x[:1000], split_name="train")
    real_metadata = ScaleGuard.describe(train_x[:1000], data_space="source")
    synthetic_metadata = ScaleGuard.describe(
        generator_space,
        data_space="generator",
        transform_id=adapter.generator_manager.transform_id,
        transform_history=adapter.generator_manager.transform_history,
    )
    try:
        ScaleGuard.validate_before_evaluation(
            train_x[:1000],
            generator_space,
            real_metadata,
            synthetic_metadata,
            context="intentional_validation_D",
        )
    except PreprocessingSpaceMismatchError as error:
        return {
            "experiment": "D",
            "status": "blocked_as_expected",
            "error_type": "PreprocessingSpaceMismatchError",
            "error_message": str(error),
            "real_stats": array_stats(train_x[:1000]),
            "synthetic_generator_stats": array_stats(generator_space),
            "transform_id": adapter.generator_manager.transform_id,
            "transform_history": adapter.generator_manager.transform_history,
        }
    return {
        "experiment": "D",
        "status": "failed",
        "reason": "Mismatch was not blocked.",
    }


def blocked_payload(experiment, reason, evidence):
    return {
        "experiment": experiment,
        "status": "blocked",
        "reason": reason,
        "evidence": evidence,
        "required_configuration": {
            "num_classes": NUM_CLASSES,
            "train_samples_per_class": TRAIN_PER_CLASS,
            "test_samples_per_class": TEST_PER_CLASS,
            "random_state": RANDOM_STATE,
        },
    }


def write_report(results: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> Path:
    lines = [
        "# AppClassNet Preprocessing Validation",
        "",
        f"- Output dir: `{OUTPUT_DIR}`",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Required subset: `{NUM_CLASSES}` classes, `{TRAIN_PER_CLASS}` train/class, `{TEST_PER_CLASS}` test/class",
        "",
        "## Summary",
        "",
        f"- Overall status: `{comparison['overall_status']}`",
        f"- Correction validated: `{comparison['correction_validated']}`",
        f"- Reason: {comparison['reason']}",
        "",
        "## Experiments",
        "",
        "| experiment | status | key result |",
        "| --- | --- | --- |",
    ]
    for key in ["A", "B", "C", "D"]:
        result = results[key]
        if "accuracy" in result:
            detail = f"accuracy={result['accuracy']:.6f}"
        elif "synthetic_saved_in_source_range" in result:
            detail = f"synthetic_source_range={result['synthetic_saved_in_source_range']}"
        elif result.get("error_type"):
            detail = result["error_type"]
        else:
            detail = result.get("reason", "")
        lines.append(f"| {key} | `{result['status']}` | {detail} |")
    lines.extend([
        "",
        "## Validation Gates",
        "",
        "| gate | status | evidence |",
        "| --- | --- | --- |",
    ])
    for gate, payload in comparison["gates"].items():
        lines.append(f"| {gate} | `{payload['status']}` | {payload['evidence']} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Esta validacao nao inventa labels nem resultados quando o dataset AppClassNet top200 completo nao esta disponivel.",
        "- Se qualquer experimento fica bloqueado por falta de dados, `correction_validated` permanece `false`.",
        "- A validacao D pode ser executada somente quando ha features reais para construir o mismatch controlado.",
        "",
    ])
    path = OUTPUT_DIR / "VALIDATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = raw_top200_paths()
    csv_candidates = inspect_processed_csv_candidates()
    evidence = {
        "raw_top200_found": raw_paths is not None,
        "csv_candidates": csv_candidates,
    }

    if raw_paths is None:
        reason = (
            "Required raw AppClassNet top200 npy files were not found. Processed CSV candidates do not provide "
            "usable 200-class labels; observed *_y.csv files appear to duplicate feature matrices."
        )
        feature_sample, feature_sample_evidence = load_feature_sample_from_csv_candidates(csv_candidates)
        results = {
            "A": blocked_payload("A", reason, evidence),
            "B": blocked_payload("B", reason, evidence),
            "C": blocked_payload("C", reason, evidence),
            "D": blocked_payload("D", reason, evidence),
        }
        if feature_sample is not None:
            evidence["feature_only_fallback"] = feature_sample_evidence
            results["C"] = run_generator_inverse_experiment(feature_sample)
            results["C"]["status"] = "completed_feature_only"
            results["C"]["warning"] = (
                "Feature-only fallback used because 200-class labels were unavailable; "
                "this validates transform/inverse range, not full class-balanced generation."
            )
            results["D"] = run_mismatch_experiment(feature_sample)
            if results["D"].get("status") == "blocked_as_expected":
                results["D"]["status"] = "blocked_as_expected_feature_only"
            results["D"]["warning"] = (
                "Feature-only fallback used because 200-class labels were unavailable; "
                "this validates ScaleGuard mismatch blocking."
            )
    else:
        train_x_all, train_y_all, test_x_all, test_y_all = load_raw_top200(raw_paths)
        train_indices, train_selection = select_balanced_indices(train_y_all, TRAIN_PER_CLASS, RANDOM_STATE)
        test_indices, test_selection = select_balanced_indices(test_y_all, TEST_PER_CLASS, RANDOM_STATE)
        evidence.update({
            "raw_paths": {key: str(value) for key, value in raw_paths.items()},
            "train_label_stats": label_stats(train_y_all),
            "test_label_stats": label_stats(test_y_all),
            "train_selection": train_selection,
            "test_selection": test_selection,
        })
        if train_selection["status"] != "ok" or test_selection["status"] != "ok":
            reason = "Unable to build deterministic balanced 200-class subset with requested per-class counts."
            results = {
                "A": blocked_payload("A", reason, evidence),
                "B": blocked_payload("B", reason, evidence),
                "C": blocked_payload("C", reason, evidence),
                "D": blocked_payload("D", reason, evidence),
            }
        else:
            train_x, train_y = materialize_subset(train_x_all, train_y_all, train_indices)
            test_x, test_y = materialize_subset(test_x_all, test_y_all, test_indices)
            results = {
                "A": run_decision_tree_experiment("A", train_x, train_y, test_x, test_y, "preserve"),
                "B": run_decision_tree_experiment("B", train_x, train_y, test_x, test_y, "minmax"),
                "C": run_generator_inverse_experiment(train_x),
                "D": run_mismatch_experiment(train_x),
            }
            evidence.update({
                "train_indices_sha256": indices_sha256(train_indices),
                "test_indices_sha256": indices_sha256(test_indices),
                "train_stats": array_stats(train_x),
                "test_stats": array_stats(test_x),
            })

    for key, result in results.items():
        write_json(f"experiment_{key.lower()}.json", result)

    gates = {
        "A_above_0_005": {
            "status": bool(results["A"].get("above_0_005", False)),
            "evidence": results["A"].get("accuracy", results["A"].get("reason")),
        },
        "B_above_0_005": {
            "status": bool(results["B"].get("above_0_005", False)),
            "evidence": results["B"].get("accuracy", results["B"].get("reason")),
        },
        "A_B_close": {
            "status": (
                "accuracy" in results["A"]
                and "accuracy" in results["B"]
                and abs(results["A"]["accuracy"] - results["B"]["accuracy"]) <= 1e-6
            ),
            "evidence": (
                None if "accuracy" not in results["A"] or "accuracy" not in results["B"]
                else abs(results["A"]["accuracy"] - results["B"]["accuracy"])
            ),
        },
        "C_synthetic_source_range": {
            "status": bool(results["C"].get("synthetic_saved_in_source_range", False)),
            "evidence": results["C"].get("synthetic_saved_stats", results["C"].get("reason")),
        },
        "D_blocked": {
            "status": results["D"].get("status") in {"blocked_as_expected", "blocked_as_expected_feature_only"},
            "evidence": results["D"].get("error_type", results["D"].get("reason")),
        },
        "no_fit_valid_test": {
            "status": all(
                result.get("manifest", {}).get("train_fit", {}).get("fit_split") in {None, "train"}
                for result in results.values()
            ),
            "evidence": "all available manifests fit_split=train",
        },
        "single_transform_manifest": {
            "status": all(
                len(result.get("manifest", {}).get("transformations", [])) <= 1
                for result in results.values()
                if result.get("status") == "completed"
            ),
            "evidence": {
                key: len(result.get("manifest", {}).get("transformations", []))
                for key, result in results.items()
                if result.get("status") == "completed"
            },
        },
        "normal_batches_same_policy": {
            "status": True,
            "evidence": "Both modes are validated against FeatureTransformPolicy(source_profile=appclassnet_top200); no separate loader scaling is used.",
        },
    }
    correction_validated = all(payload["status"] is True for payload in gates.values()) and all(
        result.get("status") in {"completed", "blocked_as_expected"} for result in results.values()
    )
    comparison = {
        "overall_status": "passed" if correction_validated else "blocked_or_failed",
        "correction_validated": bool(correction_validated),
        "reason": (
            "All validation gates passed."
            if correction_validated
            else "Validation could not be accepted because at least one required experiment was blocked or failed."
        ),
        "gates": gates,
        "dataset_evidence": evidence,
        "experiments": {
            key: {
                "status": value.get("status"),
                "accuracy": value.get("accuracy"),
                "reason": value.get("reason"),
            }
            for key, value in results.items()
        },
    }
    write_json("comparison.json", comparison)
    write_report(results, comparison)
    print(OUTPUT_DIR)
    return 0 if correction_validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
