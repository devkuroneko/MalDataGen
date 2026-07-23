#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit Train-Real/Test-Real evaluation pipeline."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.JsonIO import atomic_write_json
from Engine.DataIO.RealClassCountPolicy import build_split_sample_plan
from Engine.Evaluation.TrTrArtifacts import append_summary_csv
from Engine.Evaluation.TrTrArtifacts import build_dataset_manifest
from Engine.Evaluation.TrTrArtifacts import collect_environment
from Engine.Evaluation.TrTrArtifacts import create_tr_tr_run_directory
from Engine.Evaluation.TrTrArtifacts import event
from Engine.Evaluation.TrTrArtifacts import model_disk_size
from Engine.Evaluation.TrTrArtifacts import summary_row
from Engine.Evaluation.TrTrArtifacts import write_class_distribution_csv
from Engine.Evaluation.TrTrArtifacts import write_execution_log
from Engine.Evaluation.TrTrArtifacts import write_status
from Engine.Evaluation.ClassifierFactory import ClassifierConfig
from Engine.Evaluation.ClassifierFactory import build_classifier
from Engine.Evaluation.ClassifierFactory import estimator_statistics
from Engine.Evaluation.ClassifierFactory import sklearn_version
from Engine.Evaluation.MulticlassBatchMetrics import evaluate_classifier_batchwise
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy


TRTR_CLASSIFIERS = {"decision_tree", "random_forest"}
TRTR_SAMPLING_STRATEGIES = {"all", "balanced_per_class", "up_to_available"}


@dataclass(slots=True)
class TrTrConfig:
    classifier: str = "decision_tree"
    train_sampling: str = "all"
    test_sampling: str = "all"
    train_samples_per_class: int | None = None
    test_samples_per_class: int | None = None
    random_state: int = 0
    eval_batch_size: int = 16384
    insufficient_policy: str = "strict"
    classifier_transform: str = "preserve"
    source_profile: str = "appclassnet_top200"
    evaluation_space: str = "source"
    n_estimators: int | None = None
    max_depth: int | None = None
    class_weight: Any | None = None
    decision_tree_max_depth: int | None = None
    decision_tree_class_weight: Any | None = None
    decision_tree_criterion: str = "gini"
    decision_tree_splitter: str = "best"
    decision_tree_min_samples_split: int | float = 2
    decision_tree_min_samples_leaf: int | float = 1
    decision_tree_max_features: int | float | str | None = None
    random_forest_max_depth: int | None = None
    random_forest_class_weight: Any | None = None
    random_forest_criterion: str = "gini"
    random_forest_min_samples_split: int | float = 2
    random_forest_min_samples_leaf: int | float = 1
    random_forest_max_features: int | float | str | None = "sqrt"
    random_forest_bootstrap: bool = True
    random_forest_n_jobs: int = 1
    save_model: bool = False

    def __post_init__(self):
        if self.classifier not in TRTR_CLASSIFIERS:
            raise ValueError(
                "TR-TR classifier must be one of: decision_tree, random_forest. "
                f"Got {self.classifier!r}."
            )
        for field_name in ("train_sampling", "test_sampling"):
            value = getattr(self, field_name)
            if value not in TRTR_SAMPLING_STRATEGIES:
                raise ValueError(
                    f"{field_name} must be one of: all, balanced_per_class, up_to_available. Got {value!r}."
                )
        if int(self.eval_batch_size) <= 0:
            raise ValueError("eval_batch_size must be a positive integer.")
        for field_name in ("train_samples_per_class", "test_samples_per_class"):
            value = getattr(self, field_name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{field_name} must be a positive integer when provided.")
        if self.train_sampling == "balanced_per_class" and self.train_samples_per_class is None:
            raise ValueError("train_sampling=balanced_per_class requires train_samples_per_class.")
        if self.test_sampling == "balanced_per_class" and self.test_samples_per_class is None:
            raise ValueError("test_sampling=balanced_per_class requires test_samples_per_class.")
        if self.train_sampling == "up_to_available" and self.train_samples_per_class is None:
            raise ValueError("train_sampling=up_to_available requires train_samples_per_class.")
        if self.test_sampling == "up_to_available" and self.test_samples_per_class is None:
            raise ValueError("test_sampling=up_to_available requires test_samples_per_class.")
        self.random_state = int(self.random_state)
        self.eval_batch_size = int(self.eval_batch_size)


def run_tr_tr_pipeline(
        bundle: DatasetBundle,
        config: TrTrConfig,
        output_dir,
        *,
        loading_time_seconds: float = 0.0,
        resolved_arguments: dict[str, Any] | None = None,
        repo_root=None,
        summary_path=None) -> dict[str, Any]:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError(f"TR-TR requires DatasetBundle. Got {type(bundle).__name__}.")
    if bundle.test is None:
        raise ValueError("TR-TR requires a provided test split; bundle.test is missing.")
    if bundle.valid is not None:
        valid_usage = "ignored"
    else:
        valid_usage = "missing"
    if getattr(bundle.schema, "split_mode", None) != "provided":
        raise ValueError(
            f"TR-TR requires split_mode=provided for AppClassNet-style data. "
            f"Got split_mode={getattr(bundle.schema, 'split_mode', None)!r}."
        )

    run_id, output_dir = create_tr_tr_run_directory(output_dir, config)
    result_path = output_dir / "tr_tr_result.json"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    run_config_path = output_dir / "run_config.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    resource_usage_path = output_dir / "resource_usage.json"
    environment_path = output_dir / "environment.json"
    execution_log_path = output_dir / "execution.log"
    status_path = output_dir / "status.json"
    train_distribution_path = output_dir / "class_distribution_train.csv"
    test_distribution_path = output_dir / "class_distribution_test.csv"
    model_path = output_dir / "model.joblib"
    summary_path = Path(summary_path) if summary_path is not None else output_dir.parent / "summary.csv"
    events = [event(f"created TR-TR run directory run_id={run_id}")]
    total_start = time.perf_counter()
    stage = "initializing"
    write_status(status_path, run_id=run_id, status="running", stage=stage)
    tracemalloc.start()
    initial_memory, initial_peak_memory = tracemalloc.get_traced_memory()
    train_plan = None
    test_plan = None
    num_classes = None
    resource_usage = {
        "run_id": run_id,
        "protocol": "TR-TR",
        "memory_initial_bytes": int(initial_memory),
        "memory_before_fit_bytes": None,
        "memory_after_fit_bytes": None,
        "memory_current_bytes": None,
        "peak_memory_bytes": int(initial_peak_memory),
        "loading_time_seconds": float(loading_time_seconds),
        "selection_time_seconds": None,
        "training_time_seconds": None,
        "inference_time_seconds": None,
        "total_time_seconds": None,
        "model_size_bytes": 0,
        "model_saved": False,
        "model_save_error": None,
    }
    try:
        stage = "validating_dataset"
        write_status(status_path, run_id=run_id, status="running", stage=stage)
        _validate_split(bundle.train, "train")
        _validate_split(bundle.test, "test")
        if bundle.train.num_features != bundle.test.num_features:
            raise ValueError(
                f"TR-TR shape mismatch: train has {bundle.train.num_features} features, "
                f"test has {bundle.test.num_features} features."
            )

        num_classes = _resolve_num_classes(bundle)
        class_labels = _resolve_class_labels(bundle, num_classes)
        policy = FeatureTransformPolicy.for_profile(
            config.source_profile,
            feature_transform="preserve",
            generator_transform="preserve",
            classifier_transform=config.classifier_transform,
            evaluation_space=config.evaluation_space,
        )
        transformer = FeatureTransformManager(policy, stage="classifier", output_space="classifier")

        stage = "sampling"
        write_status(status_path, run_id=run_id, status="running", stage=stage)
        selection_start = time.perf_counter()
        train_plan = _build_plan(bundle.train.y, num_classes, "train", config.train_sampling,
                                 config.train_samples_per_class, config, seed_offset=0)
        test_plan = _build_plan(bundle.test.y, num_classes, "test", config.test_sampling,
                                config.test_samples_per_class, config, seed_offset=1)

        train_x, train_y = _materialize_selected_rows(bundle.train.X, bundle.train.y, train_plan, "train")
        test_x_source, test_y = _materialize_selected_rows(bundle.test.X, bundle.test.y, test_plan, "test")
        resource_usage["selection_time_seconds"] = float(time.perf_counter() - selection_start)
        if int(train_x.shape[0]) == 0:
            raise ValueError("TR-TR train split is empty after sampling.")
        if int(test_x_source.shape[0]) == 0:
            raise ValueError("TR-TR test split is empty after sampling.")

        write_class_distribution_csv(train_distribution_path, train_plan.selection_table)
        write_class_distribution_csv(test_distribution_path, test_plan.selection_table)
        dataset_manifest = build_dataset_manifest(bundle, train_plan, test_plan, class_labels)
        atomic_write_json(dataset_manifest, dataset_manifest_path, run_id=run_id, protocol="TR-TR",
                          model=config.classifier)
        atomic_write_json(collect_environment(repo_root), environment_path, run_id=run_id, protocol="TR-TR",
                          model=config.classifier)
        events.append(event("dataset validated and sample plans saved"))

        stage = "training"
        write_status(status_path, run_id=run_id, status="running", stage=stage)
        training_start = time.perf_counter()
        transformer.fit(train_x, split_name="train")
        train_x_transformed, train_transform_metadata = transformer.transform(
            train_x,
            split_name="train",
            input_space="source",
            output_space="classifier",
            return_metadata=True,
        )
        classifier = build_tr_tr_classifier(config)
        if hasattr(classifier, "partial_fit"):
            raise ValueError("TR-TR Decision Tree/Random Forest classifiers must not use partial_fit.")
        before_fit_memory, _ = tracemalloc.get_traced_memory()
        resource_usage["memory_before_fit_bytes"] = int(before_fit_memory)
        classifier.fit(train_x_transformed, train_y)
        training_time_seconds = time.perf_counter() - training_start
        after_fit_memory, _ = tracemalloc.get_traced_memory()
        resource_usage["memory_after_fit_bytes"] = int(after_fit_memory)
        resource_usage["training_time_seconds"] = float(training_time_seconds)
        events.append(event(f"classifier fit completed in {training_time_seconds:.6f}s"))

        stage = "inference"
        write_status(status_path, run_id=run_id, status="running", stage=stage)
        accumulator, inference_stats = evaluate_classifier_batchwise(
            classifier,
            test_x_source,
            test_y,
            class_labels=class_labels,
            batch_size=config.eval_batch_size,
            transform_batch=lambda batch_x: transformer.transform(
                batch_x,
                split_name="test",
                input_space="source",
                output_space="classifier",
            ),
            split_name="test",
        )
        inference_time_seconds = inference_stats["inference_time_seconds"]
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        resource_usage["memory_current_bytes"] = int(current_memory)
        resource_usage["peak_memory_bytes"] = int(peak_memory)
        resource_usage["inference_time_seconds"] = float(inference_time_seconds)
        events.append(event(f"batch-wise inference completed in {inference_time_seconds:.6f}s"))
    except Exception as error:
        current_memory, peak_memory = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (0, 0)
        resource_usage["memory_current_bytes"] = int(current_memory)
        resource_usage["peak_memory_bytes"] = int(peak_memory)
        resource_usage["total_time_seconds"] = float(time.perf_counter() - total_start)
        tracemalloc.stop()
        if isinstance(error, MemoryError):
            reason = (
                "TR-TR ran out of memory while fitting or predicting. "
                "Use mmap, reduce train_sampling, reduce random_forest_n_estimators/max_depth, "
                "or lower eval_batch_size."
            )
        else:
            reason = str(error)
        failure = _failure_result(
            config,
            bundle,
            train_plan,
            test_plan,
            num_classes or getattr(bundle.schema, "num_classes", None) or 0,
            type(error).__name__,
            reason,
            current_memory,
            peak_memory,
            result_path,
            metrics_path,
            manifest_path,
        )
        failure["run_id"] = run_id
        failure["stage"] = stage
        failure["run_dir"] = str(output_dir)
        failure["artifacts"].update({
            "run_config": str(run_config_path),
            "dataset_manifest": str(dataset_manifest_path),
            "class_distribution_train": str(train_distribution_path),
            "class_distribution_test": str(test_distribution_path),
            "resource_usage": str(resource_usage_path),
            "environment": str(environment_path),
            "execution_log": str(execution_log_path),
            "status": str(status_path),
        })
        atomic_write_json(failure, result_path, protocol="TR-TR", model=config.classifier)
        atomic_write_json({"protocol": "TR-TR", "status": "failed", "reason": failure["reason"]}, metrics_path,
                          protocol="TR-TR", model=config.classifier)
        atomic_write_json(resource_usage, resource_usage_path, run_id=run_id, protocol="TR-TR",
                          model=config.classifier)
        atomic_write_json(_run_config_payload(
            run_id,
            config,
            None,
            None,
            None,
            resolved_arguments,
            None,
            None,
        ), run_config_path, run_id=run_id, protocol="TR-TR", model=config.classifier)
        write_status(status_path, run_id=run_id, status="failed", stage=stage, exception=error)
        events.append(event(f"failed stage={stage} error={type(error).__name__}: {reason}"))
        write_execution_log(execution_log_path, events)
        return failure
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    stage = "metrics"
    write_status(status_path, run_id=run_id, status="running", stage=stage)
    metrics = accumulator.metrics(
        batch_size=config.eval_batch_size,
        inference_time_seconds=inference_time_seconds,
    )
    classifier_params = _classifier_params(classifier)
    model_statistics = estimator_statistics(classifier, config.classifier)
    metrics_payload = {
        "protocol": "TR-TR",
        "status": "completed",
        "classifier": config.classifier,
        "num_classes": num_classes,
        "class_labels": class_labels,
        **metrics,
    }
    metric_artifacts = accumulator.save_artifacts(output_dir, metrics_payload)
    events.append(event("metrics, confusion matrix, and per-class metrics saved"))
    result = {
        "run_id": run_id,
        "run_dir": str(output_dir),
        "status": "completed",
        "protocol": "TR-TR",
        "source_train": "real",
        "source_test": "real",
        "valid_usage": valid_usage,
        "split_mode": bundle.schema.split_mode,
        "data_format": getattr(bundle.schema, "data_format", getattr(bundle.schema, "source_format", None)),
        "sampling_mode": _sampling_mode(config),
        "classifier_subset": False,
        "classifier_mode": "full_classifier",
        "classifier": config.classifier,
        "classifier_display_name": _classifier_display_name(config.classifier),
        "classifier_params": classifier_params,
        "classifier_config": build_classifier_config(config).to_dict(),
        "classifier_statistics": model_statistics,
        "sklearn_version": sklearn_version(),
        "supports_partial_fit": False,
        "random_state": config.random_state,
        "eval_batch_size": config.eval_batch_size,
        "num_classes": num_classes,
        "class_labels": class_labels,
        "num_features": int(bundle.train.num_features),
        "files": {
            "train_x": bundle.train.x_path,
            "train_y": bundle.train.y_path,
            "valid_x": getattr(bundle.valid, "x_path", None) if bundle.valid is not None else None,
            "valid_y": getattr(bundle.valid, "y_path", None) if bundle.valid is not None else None,
            "test_x": bundle.test.x_path,
            "test_y": bundle.test.y_path,
        },
        "samples": {
            "train_total_available": int(bundle.train.num_rows),
            "test_total_available": int(bundle.test.num_rows),
            "train_total_effective": int(train_y.shape[0]),
            "test_total_effective": int(test_y.shape[0]),
            "train_by_class": train_plan.metadata["selected_counts_by_class"],
            "test_by_class": test_plan.metadata["selected_counts_by_class"],
        },
        "sample_plans": {
            "train": _sample_plan_payload(train_plan),
            "test": _sample_plan_payload(test_plan),
        },
        "transformation": {
            "source_profile": policy.source_profile,
            "classifier_transform": policy.classifier_transform,
            "evaluation_space": policy.evaluation_space,
            "operation": transformer.operation,
            "transform_id": transformer.transform_id,
            "transform_applied": bool(train_transform_metadata.get("transform_applied")),
            "transform_fitted": transformer.scaler is not None,
            "input_range": transformer.input_range,
            "output_range": transformer.output_range,
        },
        "timing": {
            "training_time_seconds": float(training_time_seconds),
            "inference_time_seconds": float(inference_time_seconds),
            "throughput_samples_per_second": metrics["throughput_samples_per_second"],
        },
        "evaluation": {
            "batch_count": metrics["batch_count"],
            "batch_size": metrics["batch_size"],
            "total_predictions": metrics["total_predictions"],
            "zero_division": metrics["zero_division"],
            "classes_without_predictions": metrics["classes_without_predictions"],
            "classes_without_support": metrics["classes_without_support"],
        },
        "memory": {
            "current_bytes": int(current_memory),
            "peak_bytes": int(peak_memory),
        },
        "metrics": metrics,
        "confusion_matrix_shape": [num_classes, num_classes],
    }

    result["artifacts"] = {
        "result": str(result_path),
        "run_config": str(run_config_path),
        "dataset_manifest": str(dataset_manifest_path),
        "class_distribution_train": str(train_distribution_path),
        "class_distribution_test": str(test_distribution_path),
        "metrics": metric_artifacts["metrics"],
        "confusion_matrix": metric_artifacts["confusion_matrix"],
        "per_class_metrics": metric_artifacts["per_class_metrics"],
        "manifest": str(manifest_path),
        "resource_usage": str(resource_usage_path),
        "environment": str(environment_path),
        "execution_log": str(execution_log_path),
        "status": str(status_path),
        "summary": str(summary_path),
        "model": str(model_path) if config.save_model else None,
    }
    atomic_write_json(result, result_path, protocol="TR-TR", model=config.classifier)
    atomic_write_json({**metrics_payload, "artifacts": result["artifacts"]}, metrics_path,
                      protocol="TR-TR", model=config.classifier)
    model_save_error = None
    if config.save_model:
        stage = "saving_model"
        write_status(status_path, run_id=run_id, status="running", stage=stage)
        try:
            import joblib
            joblib.dump(classifier, model_path)
            resource_usage["model_saved"] = True
            resource_usage["model_size_bytes"] = model_disk_size(model_path)
            result["model_saved"] = True
            events.append(event(f"model saved to {model_path}"))
        except Exception as error:  # pragma: no cover - environment dependent
            model_save_error = str(error)
            result["artifacts"]["model"] = None
            result["model_save_error"] = model_save_error
            resource_usage["model_save_error"] = model_save_error
            atomic_write_json(result, result_path, protocol="TR-TR", model=config.classifier)
            events.append(event(f"model save failed: {model_save_error}"))
    else:
        result["model_saved"] = False
        result["model_save_reason"] = "save_model=False"
        events.append(event("model was not saved because save_model=False"))
    resource_usage["total_time_seconds"] = float(time.perf_counter() - total_start)
    resource_usage["memory_current_bytes"] = int(current_memory)
    resource_usage["peak_memory_bytes"] = int(peak_memory)
    resource_usage["training_time_seconds"] = float(training_time_seconds)
    resource_usage["inference_time_seconds"] = float(inference_time_seconds)
    result["resource_usage"] = resource_usage
    run_config_payload = _run_config_payload(
        run_id,
        config,
        build_classifier_config(config).to_dict(),
        classifier_params,
        result["transformation"],
        resolved_arguments,
        result["sample_plans"],
        resource_usage,
    )
    atomic_write_json(run_config_payload, run_config_path, run_id=run_id, protocol="TR-TR",
                      model=config.classifier)
    atomic_write_json(resource_usage, resource_usage_path, run_id=run_id, protocol="TR-TR",
                      model=config.classifier)
    atomic_write_json(result, result_path, protocol="TR-TR", model=config.classifier)
    atomic_write_json({
        "run_id": run_id,
        "protocol": "TR-TR",
        "source_train": "real",
        "source_test": "real",
        "split_mode": bundle.schema.split_mode,
        "classifier": config.classifier,
        "classifier_params": classifier_params,
        "classifier_statistics": model_statistics,
        "sklearn_version": result["sklearn_version"],
        "files": result["files"],
        "sample_plans": result["sample_plans"],
        "transformation": result["transformation"],
        "artifacts": result["artifacts"],
        "model_save_error": model_save_error,
    }, manifest_path, protocol="TR-TR", model=config.classifier)
    append_summary_csv(summary_path, summary_row(result, resource_usage))
    write_status(status_path, run_id=run_id, status="completed", stage="completed")
    events.append(event("run completed"))
    write_execution_log(execution_log_path, events)
    return result


def build_tr_tr_classifier(config: TrTrConfig):
    return build_classifier(build_classifier_config(config))


def _classifier_params(classifier):
    try:
        return classifier.get_params(deep=True)
    except TypeError:
        return classifier.get_params()


def build_classifier_config(config: TrTrConfig) -> ClassifierConfig:
    return ClassifierConfig(
        classifier=config.classifier,
        random_state=config.random_state,
        decision_tree_criterion=config.decision_tree_criterion,
        decision_tree_splitter=config.decision_tree_splitter,
        decision_tree_max_depth=config.decision_tree_max_depth if config.decision_tree_max_depth is not None else config.max_depth,
        decision_tree_min_samples_split=config.decision_tree_min_samples_split,
        decision_tree_min_samples_leaf=config.decision_tree_min_samples_leaf,
        decision_tree_max_features=config.decision_tree_max_features,
        decision_tree_class_weight=(
            config.decision_tree_class_weight if config.decision_tree_class_weight is not None else config.class_weight
        ),
        random_forest_n_estimators=config.n_estimators or 100,
        random_forest_criterion=config.random_forest_criterion,
        random_forest_max_depth=config.random_forest_max_depth if config.random_forest_max_depth is not None else config.max_depth,
        random_forest_min_samples_split=config.random_forest_min_samples_split,
        random_forest_min_samples_leaf=config.random_forest_min_samples_leaf,
        random_forest_max_features=config.random_forest_max_features,
        random_forest_bootstrap=config.random_forest_bootstrap,
        random_forest_class_weight=(
            config.random_forest_class_weight if config.random_forest_class_weight is not None else config.class_weight
        ),
        random_forest_n_jobs=config.random_forest_n_jobs,
    )


def _validate_split(split, expected_name):
    if split.name != expected_name:
        raise ValueError(f"TR-TR expected {expected_name} split; got split.name={split.name!r}.")
    if split.X.ndim != 2:
        raise ValueError(f"TR-TR {expected_name} X must be 2D. Got shape {split.X.shape}.")
    if numpy.asarray(split.y).reshape(-1).ndim != 1:
        raise ValueError(f"TR-TR {expected_name} y must be 1D. Got shape {numpy.asarray(split.y).shape}.")
    if int(split.X.shape[0]) != int(numpy.asarray(split.y).reshape(-1).shape[0]):
        raise ValueError(
            f"TR-TR {expected_name} X/y length mismatch: X rows={int(split.X.shape[0])} "
            f"y rows={int(numpy.asarray(split.y).reshape(-1).shape[0])}."
        )
    if int(split.X.shape[0]) == 0:
        raise ValueError(f"TR-TR {expected_name} split is empty.")


def _resolve_num_classes(bundle):
    if bundle.schema.num_classes is not None:
        return int(bundle.schema.num_classes)
    labels = numpy.concatenate([
        numpy.asarray(bundle.train.y).reshape(-1),
        numpy.asarray(bundle.test.y).reshape(-1),
    ])
    if labels.size == 0:
        raise ValueError("TR-TR cannot infer classes from empty labels.")
    if int(labels.min()) < 0:
        raise ValueError("TR-TR labels must be non-negative.")
    return int(labels.max()) + 1


def _resolve_class_labels(bundle, num_classes):
    classes = getattr(bundle.schema, "classes", None)
    if classes is None:
        return list(range(num_classes))
    class_labels = [int(label) for label in classes]
    if len(class_labels) != int(num_classes):
        raise ValueError(
            f"TR-TR schema classes length mismatch: classes={len(class_labels)} "
            f"num_classes={int(num_classes)}."
        )
    return class_labels


def _build_plan(labels, num_classes, split_name, strategy, samples_per_class, config, seed_offset):
    return build_split_sample_plan(
        labels,
        samples_per_class,
        num_classes,
        config.random_state + int(seed_offset),
        split_name,
        strategy=strategy,
        insufficient_policy=config.insufficient_policy,
        replacement=False,
        require_all_classes=True,
    )


def _materialize_selected_rows(x_values, y_values, plan, split_name):
    y_values = numpy.asanyarray(y_values).reshape(-1)
    if plan.selected_indices is None:
        return x_values, y_values
    indices = numpy.asarray(plan.selected_indices, dtype=numpy.int64)
    if indices.size:
        min_index = int(indices.min())
        max_index = int(indices.max())
        if min_index < 0 or max_index >= int(x_values.shape[0]) or max_index >= int(y_values.shape[0]):
            raise IndexError(
                f"TR-TR {split_name} sample plan produced invalid index range "
                f"min={min_index} max={max_index}; X rows={int(x_values.shape[0])}; y rows={int(y_values.shape[0])}."
            )
    return numpy.asanyarray(x_values[indices]), numpy.asarray(y_values[indices], dtype=numpy.int64)


def _sample_plan_payload(plan):
    return {
        "mode": plan.mode,
        "split_name": plan.split_name,
        "split_size": plan.split_size,
        "total_rows": plan.total_rows,
        "samples_per_class": plan.samples_per_class,
        "number_classes": plan.number_classes,
        "random_state": plan.random_state,
        "insufficient_policy": plan.insufficient_policy,
        "replacement": plan.replacement,
        "class_counts": plan.class_counts,
        "selection_table": plan.selection_table,
        "metadata": plan.metadata,
    }


def _classifier_display_name(classifier):
    if classifier == "decision_tree":
        return "Decision Tree"
    if classifier == "random_forest":
        return "Random Forest"
    return classifier


def _sampling_mode(config):
    if config.train_sampling == "all" and config.test_sampling == "all":
        return "full_provided_split"
    return "balanced_subset"


def tr_tr_config_from_namespace(arguments) -> TrTrConfig:
    classifier = getattr(arguments, "baseline_classifier", "decision_tree")
    if classifier == "random_forest_light":
        classifier = "random_forest"
    class_weight = getattr(arguments, "class_weight", None)
    return TrTrConfig(
        classifier=classifier,
        train_sampling=getattr(arguments, "train_sampling", None) or _legacy_sampling(
            getattr(arguments, "train_samples_per_class", None)
        ),
        test_sampling=getattr(arguments, "test_sampling", None) or _legacy_sampling(
            getattr(arguments, "test_samples_per_class", None)
        ),
        train_samples_per_class=getattr(arguments, "train_samples_per_class", None),
        test_samples_per_class=getattr(arguments, "test_samples_per_class", None),
        random_state=getattr(arguments, "random_state", 0),
        eval_batch_size=getattr(arguments, "eval_batch_size", 16384),
        insufficient_policy=getattr(arguments, "real_class_count_policy", "strict"),
        classifier_transform=getattr(arguments, "classifier_transform", "preserve"),
        source_profile=getattr(arguments, "source_profile", "appclassnet_top200"),
        evaluation_space=getattr(arguments, "evaluation_space", "source"),
        n_estimators=getattr(arguments, "random_forest_n_estimators", getattr(arguments, "n_estimators", None)),
        max_depth=getattr(arguments, "max_depth", None),
        class_weight=class_weight,
        decision_tree_max_depth=getattr(arguments, "decision_tree_max_depth", None),
        decision_tree_class_weight=getattr(arguments, "decision_tree_class_weight", None),
        decision_tree_criterion=getattr(arguments, "decision_tree_criterion", "gini"),
        decision_tree_splitter=getattr(arguments, "decision_tree_splitter", "best"),
        decision_tree_min_samples_split=getattr(arguments, "decision_tree_min_samples_split", 2),
        decision_tree_min_samples_leaf=getattr(arguments, "decision_tree_min_samples_leaf", 1),
        decision_tree_max_features=getattr(arguments, "decision_tree_max_features", None),
        random_forest_max_depth=getattr(arguments, "random_forest_max_depth", None),
        random_forest_class_weight=getattr(arguments, "random_forest_class_weight", None),
        random_forest_criterion=getattr(arguments, "random_forest_criterion", "gini"),
        random_forest_min_samples_split=getattr(arguments, "random_forest_min_samples_split", 2),
        random_forest_min_samples_leaf=getattr(arguments, "random_forest_min_samples_leaf", 1),
        random_forest_max_features=getattr(arguments, "random_forest_max_features", "sqrt"),
        random_forest_bootstrap=getattr(arguments, "random_forest_bootstrap", True),
        random_forest_n_jobs=getattr(arguments, "random_forest_n_jobs", 1),
        save_model=bool(
            getattr(arguments, "save_tr_tr_model", False)
            or getattr(arguments, "save_models", False)
        ),
    )


def _legacy_sampling(samples_per_class):
    return "all" if samples_per_class is None else "balanced_per_class"


def result_to_legacy_baseline_metrics(result):
    return {
        "mode": "baseline_real_only",
        "protocol": "TR-TR",
        "classifier": result["classifier"],
        "classifier_params": result["classifier_params"],
        "classifier_statistics": result.get("classifier_statistics", {}),
        "sklearn_version": result.get("sklearn_version"),
        "num_classes": result["num_classes"],
        "train_shape": [
            result["samples"]["train_total_effective"],
            result["num_features"],
        ],
        "test_shape": [
            result["samples"]["test_total_effective"],
            result["num_features"],
        ],
        "train_samples_per_class_requested": result["sample_plans"]["train"]["samples_per_class"],
        "test_samples_per_class_requested": result["sample_plans"]["test"]["samples_per_class"],
        "real_class_count_policy": result["sample_plans"]["train"]["insufficient_policy"],
        "train_class_counts": result["samples"]["train_by_class"],
        "test_class_counts": result["samples"]["test_by_class"],
        "data_space": "source",
        "feature_range": None,
        "metrics": result["metrics"],
        "paths": {
            **result["files"],
            **result["artifacts"],
        },
        "duration_seconds": (
            float(result["timing"]["training_time_seconds"])
            + float(result["timing"]["inference_time_seconds"])
        ),
        "structured_result": result,
    }


def config_to_dict(config: TrTrConfig):
    return asdict(config)


def _run_config_payload(
        run_id,
        config,
        classifier_config,
        classifier_params,
        transformation,
        resolved_arguments,
        sample_plans,
        resource_usage):
    return {
        "run_id": run_id,
        "protocol": "TR-TR",
        "resolved_arguments": resolved_arguments if resolved_arguments is not None else config_to_dict(config),
        "tr_tr_config": config_to_dict(config),
        "classifier": config.classifier,
        "classifier_config": classifier_config,
        "classifier_params": classifier_params,
        "seed": config.random_state,
        "random_state": config.random_state,
        "transformations": {
            "feature_transform": "preserve",
            "classifier_transform": config.classifier_transform,
            "generator_transform": "preserve",
            "evaluation_space": config.evaluation_space,
            "details": transformation,
        },
        "sampling": {
            "train_sampling": config.train_sampling,
            "test_sampling": config.test_sampling,
            "train_samples_per_class": config.train_samples_per_class,
            "test_samples_per_class": config.test_samples_per_class,
            "insufficient_policy": config.insufficient_policy,
            "sample_plans": sample_plans,
        },
        "resource_usage": resource_usage,
    }


def _empty_sample_plan_payload(split_name):
    return {
        "mode": None,
        "split_name": split_name,
        "split_size": None,
        "total_rows": None,
        "samples_per_class": None,
        "number_classes": None,
        "random_state": None,
        "insufficient_policy": None,
        "replacement": False,
        "class_counts": {},
        "selection_table": [],
        "metadata": {},
    }


def _failure_result(
        config,
        bundle,
        train_plan,
        test_plan,
        num_classes,
        error_type,
        reason,
        current_memory,
        peak_memory,
        result_path,
        metrics_path,
        manifest_path):
    return {
        "status": "failed",
        "protocol": "TR-TR",
        "reason": reason,
        "error_type": error_type,
        "source_train": "real",
        "source_test": "real",
        "split_mode": bundle.schema.split_mode,
        "classifier": config.classifier,
        "classifier_config": build_classifier_config(config).to_dict(),
        "classifier_subset": False,
        "classifier_mode": "full_classifier",
        "sklearn_version": sklearn_version(),
        "num_classes": int(num_classes),
        "files": {
            "train_x": bundle.train.x_path,
            "train_y": bundle.train.y_path,
            "valid_x": getattr(bundle.valid, "x_path", None) if bundle.valid is not None else None,
            "valid_y": getattr(bundle.valid, "y_path", None) if bundle.valid is not None else None,
            "test_x": bundle.test.x_path,
            "test_y": bundle.test.y_path,
        },
        "sample_plans": {
            "train": _sample_plan_payload(train_plan) if train_plan is not None else _empty_sample_plan_payload("train"),
            "test": _sample_plan_payload(test_plan) if test_plan is not None else _empty_sample_plan_payload("test"),
        },
        "memory": {
            "current_bytes": int(current_memory),
            "peak_bytes": int(peak_memory),
        },
        "metrics": {},
        "artifacts": {
            "result": str(result_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
            "model": None,
        },
    }
