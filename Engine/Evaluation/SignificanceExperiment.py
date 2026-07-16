#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reproducible paired significance experiments for AppClassNet synthetic data."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

try:
    import psutil
except ImportError:  # pragma: no cover - dependency exists in project config
    psutil = None

from Engine.Classifiers.BatchClassifiers import make_batch_classifier
from Engine.DataIO.DatasetContracts import validate_xy_alignment
from Engine.Evaluation.EvaluationRunner import dataset_hash


PRIMARY_SCENARIOS = {
    "r200_r500": {
        "label": "A. R200 -> R500",
        "real_per_class": 200,
        "synthetic_per_class": 0,
        "control": "none",
    },
    "s200_r500": {
        "label": "B. S200 -> R500",
        "real_per_class": 0,
        "synthetic_per_class": 200,
        "control": "none",
    },
    "r50_r500": {
        "label": "C. R50 -> R500",
        "real_per_class": 50,
        "synthetic_per_class": 0,
        "control": "none",
    },
    "r50_s150_r500": {
        "label": "D. R50 + S150 -> R500",
        "real_per_class": 50,
        "synthetic_per_class": 150,
        "control": "none",
    },
    "r200_s200_r500": {
        "label": "E. R200 + S200 -> R500",
        "real_per_class": 200,
        "synthetic_per_class": 200,
        "control": "none",
    },
    "real_resample_r500": {
        "label": "F. real_resample -> R500",
        "real_per_class": 200,
        "synthetic_per_class": 0,
        "control": "real_resample",
    },
    "label_permutation_r500": {
        "label": "G. label_permutation -> R500",
        "real_per_class": 200,
        "synthetic_per_class": 0,
        "control": "label_permutation",
    },
}

MAIN_COMPARISONS = [
    ("s200_vs_r200", "s200_r500", "r200_r500", "S200 versus R200: capacidade de substituicao"),
    ("r50_s150_vs_r50", "r50_s150_r500", "r50_r500", "R50+S150 versus R50: ganho por aumento"),
    ("r50_s150_vs_r200", "r50_s150_r500", "r200_r500", "R50+S150 versus R200: recuperacao com menos dados reais"),
    ("real_resample_vs_r200", "real_resample_r500", "r200_r500", "real_resample versus R200: validacao do pipeline"),
]


@dataclass(slots=True)
class SignificanceConfig:
    output_dir: str
    classifiers: list[str]
    seeds: list[int]
    test_samples_per_class: int = 500
    num_classes: int | None = None
    class_labels: list[int] | None = None
    eval_batch_size: int = 4096
    random_state: int = 0
    n_estimators: int | None = None
    max_depth: int | None = None
    max_samples: float | int | None = None
    class_weight: str | dict | None = None
    decision_tree_criterion: str = "gini"
    decision_tree_max_features: Any = None
    decision_tree_max_leaf_nodes: Any = None
    noninferiority_margin: float | None = None
    noninferiority_margin_justification: str | None = None
    command: list[str] | None = None
    resolved_config: dict[str, Any] | None = None
    paths: dict[str, str] | None = None


class SignificanceExperimentRunner:
    def __init__(
            self,
            real_train_x,
            real_train_y,
            real_test_x,
            real_test_y,
            synthetic_x=None,
            synthetic_y=None,
            config: SignificanceConfig | None = None):
        if config is None:
            raise ValueError("SignificanceConfig is required.")
        self.config = config
        self.real_train_x, self.real_train_y = validate_xy_alignment(
            real_train_x,
            real_train_y,
            "significance real train",
        )
        self.real_test_x, self.real_test_y = validate_xy_alignment(
            real_test_x,
            real_test_y,
            "significance real test",
        )
        self.real_train_x = numpy.asarray(self.real_train_x, dtype=numpy.float32)
        self.real_test_x = numpy.asarray(self.real_test_x, dtype=numpy.float32)
        self.real_train_y = numpy.asarray(self.real_train_y, dtype=numpy.int64)
        self.real_test_y = numpy.asarray(self.real_test_y, dtype=numpy.int64)
        self.synthetic_x = None
        self.synthetic_y = None
        if synthetic_x is not None or synthetic_y is not None:
            if synthetic_x is None or synthetic_y is None:
                raise ValueError("synthetic_x and synthetic_y must be provided together.")
            self.synthetic_x, self.synthetic_y = validate_xy_alignment(
                synthetic_x,
                synthetic_y,
                "significance synthetic train",
            )
            self.synthetic_x = numpy.asarray(self.synthetic_x, dtype=numpy.float32)
            self.synthetic_y = numpy.asarray(self.synthetic_y, dtype=numpy.int64)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_labels = self._resolve_class_labels()
        self._validate_config()

    def run(self) -> dict[str, Path]:
        started = time.perf_counter()
        per_seed_rows = []
        matrix = self._experiment_matrix()
        paths = {
            "experiment_matrix": self.output_dir / "experiment_matrix.json",
            "per_seed_results": self.output_dir / "per_seed_results.csv",
            "aggregate_results": self.output_dir / "aggregate_results.csv",
            "paired_comparisons": self.output_dir / "paired_comparisons.csv",
            "statistical_report": self.output_dir / "statistical_report.md",
        }
        try:
            for seed in self.config.seeds:
                test_x, test_y, test_indices = self._sample_real_test(seed)
                for classifier_key in self.config.classifiers:
                    for scenario_id in PRIMARY_SCENARIOS:
                        row = self._run_single(seed, classifier_key, scenario_id, test_x, test_y, test_indices)
                        per_seed_rows.append(row)

            aggregate_rows = aggregate_results(per_seed_rows)
            paired_rows = paired_comparisons(
                per_seed_rows,
                class_labels=self.class_labels,
                noninferiority_margin=self.config.noninferiority_margin,
                noninferiority_margin_justification=self.config.noninferiority_margin_justification,
            )
            matrix["duration_seconds"] = float(time.perf_counter() - started)
            matrix["status"] = "completed"

            paths["experiment_matrix"].write_text(json.dumps(_json_ready(matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_csv(paths["per_seed_results"], per_seed_rows)
            _write_csv(paths["aggregate_results"], aggregate_rows)
            _write_csv(paths["paired_comparisons"], paired_rows)
            paths["statistical_report"].write_text(
                build_statistical_report(matrix, aggregate_rows, paired_rows),
                encoding="utf-8",
            )
        except Exception as error:
            matrix["duration_seconds"] = float(time.perf_counter() - started)
            matrix["status"] = "failed"
            matrix["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            paths["experiment_matrix"].write_text(json.dumps(_json_ready(matrix), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
        return paths

    def _run_single(self, seed, classifier_key, scenario_id, test_x, test_y, test_indices):
        scenario = PRIMARY_SCENARIOS[scenario_id]
        train_x, train_y, train_parts = self._build_training_set(seed, scenario_id)
        args = self._classifier_arguments(seed)
        classifier = make_batch_classifier(classifier_key, args)
        memory_before = _memory_mb()
        train_start = time.perf_counter()
        classifier.fit(train_x, train_y)
        train_seconds = time.perf_counter() - train_start
        inference_start = time.perf_counter()
        predictions = _predict_in_batches(classifier, test_x, self.config.eval_batch_size)
        inference_seconds = time.perf_counter() - inference_start
        memory_after = _memory_mb()
        metrics = classification_metrics(test_y, predictions, self.class_labels)
        return {
            "seed": int(seed),
            "classifier": classifier_key,
            "scenario_id": scenario_id,
            "scenario_label": scenario["label"],
            "control": scenario["control"],
            "Accuracy": metrics["Accuracy"],
            "BalancedAccuracy": metrics["BalancedAccuracy"],
            "MacroF1": metrics["MacroF1"],
            "WeightedF1": metrics["WeightedF1"],
            "recall_by_class": json.dumps(metrics["recall_by_class"], sort_keys=True),
            "precision_by_class": json.dumps(metrics["precision_by_class"], sort_keys=True),
            "f1_by_class": json.dumps(metrics["f1_by_class"], sort_keys=True),
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "train_seconds": float(train_seconds),
            "inference_seconds": float(inference_seconds),
            "memory_mb_before": memory_before,
            "memory_mb_after": memory_after,
            "memory_mb_delta": None if memory_before is None or memory_after is None else float(memory_after - memory_before),
            "effective_train_rows": int(train_y.shape[0]),
            "effective_test_rows": int(test_y.shape[0]),
            "eval_batch_size": int(self.config.eval_batch_size),
            "train_class_counts": json.dumps(_counts_by_class(train_y), sort_keys=True),
            "test_class_counts": json.dumps(_counts_by_class(test_y), sort_keys=True),
            "train_parts": json.dumps(train_parts, sort_keys=True),
            "test_indices_hash": _indices_hash(test_indices),
            "train_hash": dataset_hash(train_x),
            "test_hash": dataset_hash(test_x),
            "status": "completed",
        }

    def _build_training_set(self, seed, scenario_id):
        scenario = PRIMARY_SCENARIOS[scenario_id]
        real_quota = int(scenario["real_per_class"])
        synthetic_quota = int(scenario["synthetic_per_class"])
        parts = []
        x_parts = []
        y_parts = []

        if real_quota > 0:
            real_context = f"{scenario_id}:real_train"
            if scenario["control"] in {"real_resample", "label_permutation"}:
                real_context = "r200_r500:real_train"
            real_x, real_y, real_indices = self._sample_by_class(
                self.real_train_x,
                self.real_train_y,
                real_quota,
                seed,
                real_context,
            )
            x_parts.append(real_x)
            y_parts.append(real_y)
            parts.append({
                "source": "real_resample_control" if scenario["control"] == "real_resample" else "real_train",
                "quota_per_class": real_quota,
                "rows": int(real_y.shape[0]),
                "indices_hash": _indices_hash(real_indices),
                "sampling_context": real_context,
            })

        if synthetic_quota > 0:
            if self.synthetic_x is None:
                raise ValueError(f"Scenario {scenario_id} requires synthetic data.")
            synthetic_x, synthetic_y, synthetic_indices = self._sample_by_class(
                self.synthetic_x,
                self.synthetic_y,
                synthetic_quota,
                seed,
                f"{scenario_id}:synthetic_train",
            )
            x_parts.append(synthetic_x)
            y_parts.append(synthetic_y)
            parts.append({"source": "synthetic_train", "quota_per_class": synthetic_quota, "rows": int(synthetic_y.shape[0]), "indices_hash": _indices_hash(synthetic_indices)})

        if scenario["control"] == "label_permutation":
            if not x_parts:
                raise ValueError("label_permutation requires a real-resample base.")
            rng = numpy.random.default_rng(int(seed) + 1000003)
            y_parts = [rng.permutation(numpy.concatenate(y_parts)).astype(numpy.int64)]
            x_parts = [numpy.vstack(x_parts).astype(numpy.float32)]
            parts.append({"source": "label_permutation", "seed": int(seed) + 1000003})

        if not x_parts:
            raise ValueError(f"Scenario {scenario_id} produced no training rows.")
        train_x = numpy.vstack(x_parts).astype(numpy.float32)
        train_y = numpy.concatenate(y_parts).astype(numpy.int64)
        permutation = numpy.random.default_rng(int(seed) + 2000003).permutation(train_y.shape[0])
        return train_x[permutation], train_y[permutation], parts

    def _sample_real_test(self, seed):
        return self._sample_by_class(
            self.real_test_x,
            self.real_test_y,
            int(self.config.test_samples_per_class),
            seed,
            "real_test",
        )

    def _sample_by_class(self, x_values, y_values, samples_per_class, seed, context):
        rng = numpy.random.default_rng(_stable_seed(self.config.random_state, seed, context))
        selected = []
        for class_id in self.class_labels:
            indices = numpy.flatnonzero(y_values == int(class_id))
            if indices.shape[0] < int(samples_per_class):
                raise ValueError(
                    f"{context} class={class_id} has {indices.shape[0]} rows; "
                    f"requires {samples_per_class}."
                )
            selected.append(rng.choice(indices, size=int(samples_per_class), replace=False))
        selected_indices = numpy.concatenate(selected).astype(numpy.int64)
        permutation = rng.permutation(selected_indices.shape[0])
        selected_indices = selected_indices[permutation]
        return x_values[selected_indices], y_values[selected_indices], selected_indices

    def _resolve_class_labels(self):
        if self.config.class_labels is not None:
            return [int(label) for label in self.config.class_labels]
        if self.config.num_classes is not None:
            return list(range(int(self.config.num_classes)))
        labels = set(numpy.unique(self.real_train_y).astype(int).tolist())
        labels &= set(numpy.unique(self.real_test_y).astype(int).tolist())
        if self.synthetic_y is not None:
            labels &= set(numpy.unique(self.synthetic_y).astype(int).tolist())
        return sorted(labels)

    def _validate_config(self):
        if len(self.config.seeds) < 2:
            raise ValueError("At least two seeds are required for paired statistics; use five or more for the default protocol.")
        if not self.config.classifiers:
            raise ValueError("At least one classifier is required.")
        if self.config.noninferiority_margin is not None and not self.config.noninferiority_margin_justification:
            raise ValueError("noninferiority_margin requires noninferiority_margin_justification.")
        if not self.class_labels:
            raise ValueError("No effective classes selected.")

    def _classifier_arguments(self, seed):
        return SimpleNamespace(
            random_state=int(seed),
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            max_samples=self.config.max_samples,
            class_weight=self.config.class_weight,
            decision_tree_criterion=self.config.decision_tree_criterion,
            decision_tree_max_features=self.config.decision_tree_max_features,
            decision_tree_max_leaf_nodes=self.config.decision_tree_max_leaf_nodes,
        )

    def _experiment_matrix(self):
        return {
            "schema_version": "appclassnet_significance/v1",
            "created_at_unix": int(time.time()),
            "git": git_metadata(),
            "command": self.config.command or sys.argv,
            "canonical_command": shlex.join(map(str, self.config.command or sys.argv)),
            "resolved_config": self.config.resolved_config or asdict(self.config),
            "dataset_hashes": {
                "real_train_x": dataset_hash(self.real_train_x),
                "real_train_y": dataset_hash(self.real_train_y),
                "real_test_x": dataset_hash(self.real_test_x),
                "real_test_y": dataset_hash(self.real_test_y),
                "synthetic_x": None if self.synthetic_x is None else dataset_hash(self.synthetic_x),
                "synthetic_y": None if self.synthetic_y is None else dataset_hash(self.synthetic_y),
            },
            "seeds": [int(seed) for seed in self.config.seeds],
            "classifiers": list(self.config.classifiers),
            "class_labels": [int(label) for label in self.class_labels],
            "scenarios": PRIMARY_SCENARIOS,
            "main_comparisons": [
                {"comparison_id": cid, "candidate": candidate, "baseline": baseline, "question": question}
                for cid, candidate, baseline, question in MAIN_COMPARISONS
            ],
            "superiority_rule": "IC95% of MacroF1 delta completely above zero.",
            "noninferiority": {
                "margin": self.config.noninferiority_margin,
                "justification": self.config.noninferiority_margin_justification,
            },
            "statistics_protocol": {
                "paired": True,
                "normality_condition": "Shapiro-Wilk on paired deltas when n>=3.",
                "test_selection": "paired t-test if Shapiro p>=0.05; otherwise Wilcoxon signed-rank when non-zero deltas exist.",
                "multiple_comparisons": "Holm correction within paired MacroF1 comparisons.",
            },
            "dependencies": dependency_versions(),
            "hardware": hardware_metadata(),
            "paths": self.config.paths or {},
            "status": "running",
        }


def classification_metrics(y_true, y_pred, class_labels):
    labels = [int(label) for label in class_labels]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "WeightedF1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "precision_by_class": {str(label): float(value) for label, value in zip(labels, precision)},
        "recall_by_class": {str(label): float(value) for label, value in zip(labels, recall)},
        "f1_by_class": {str(label): float(value) for label, value in zip(labels, f1)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def _predict_in_batches(classifier, x_values, batch_size):
    predictions = []
    batch_size = int(batch_size)
    for start in range(0, x_values.shape[0], batch_size):
        end = min(start + batch_size, x_values.shape[0])
        predictions.extend(classifier.predict(x_values[start:end]))
    return numpy.asarray(predictions)


def aggregate_results(rows):
    grouped = {}
    metrics = [
        "Accuracy",
        "BalancedAccuracy",
        "MacroF1",
        "WeightedF1",
        "train_seconds",
        "inference_seconds",
        "memory_mb_before",
        "memory_mb_after",
        "memory_mb_delta",
        "effective_train_rows",
        "effective_test_rows",
    ]
    for row in rows:
        key = (row["classifier"], row["scenario_id"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (classifier, scenario_id), group_rows in sorted(grouped.items()):
        base = {
            "classifier": classifier,
            "scenario_id": scenario_id,
            "scenario_label": group_rows[0]["scenario_label"],
            "n_seeds": len(group_rows),
        }
        for metric in metrics:
            summary = summarize([_optional_float(row.get(metric)) for row in group_rows])
            for key, value in summary.items():
                base[f"{metric}_{key}"] = value
        output.append(base)
    return output


def paired_comparisons(rows, class_labels, noninferiority_margin=None, noninferiority_margin_justification=None):
    by_key = {
        (row["classifier"], row["scenario_id"], int(row["seed"])): row
        for row in rows
    }
    output = []
    raw_p_rows = []
    for classifier in sorted({row["classifier"] for row in rows}):
        for comparison_id, candidate_id, baseline_id, question in MAIN_COMPARISONS:
            output.append(_paired_row(
                rows_by_key=by_key,
                classifier=classifier,
                comparison_id=comparison_id,
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                question=question,
                noninferiority_margin=noninferiority_margin,
                noninferiority_margin_justification=noninferiority_margin_justification,
            ))
        output.append(_chance_row(by_key, classifier, class_labels))

    raw_p_rows = [row for row in output if row.get("comparison_family") == "main_macro_f1" and row.get("p_value") not in {"", None}]
    corrected = holm_correction([float(row["p_value"]) for row in raw_p_rows])
    for row, adjusted in zip(raw_p_rows, corrected):
        row["p_value_holm"] = adjusted
    for row in output:
        row.setdefault("p_value_holm", "")
    return output


def _paired_row(
        rows_by_key,
        classifier,
        comparison_id,
        candidate_id,
        baseline_id,
        question,
        noninferiority_margin,
        noninferiority_margin_justification):
    candidate = _scenario_seed_values(rows_by_key, classifier, candidate_id)
    baseline = _scenario_seed_values(rows_by_key, classifier, baseline_id)
    seeds = sorted(set(candidate) & set(baseline))
    deltas = numpy.asarray([candidate[seed]["MacroF1"] - baseline[seed]["MacroF1"] for seed in seeds], dtype=float)
    stats_report = paired_test(deltas)
    ci_low, ci_high = confidence_interval(deltas)
    superiority = bool(ci_low is not None and ci_low > 0.0)
    noninferiority = ""
    if noninferiority_margin is not None and ci_low is not None:
        noninferiority = bool(ci_low > -float(noninferiority_margin))
    return {
        "comparison_id": comparison_id,
        "comparison_family": "main_macro_f1",
        "classifier": classifier,
        "candidate": candidate_id,
        "baseline": baseline_id,
        "question": question,
        "metric": "MacroF1",
        "n_pairs": len(seeds),
        "seeds": json.dumps(seeds),
        "delta_by_seed": json.dumps({str(seed): float(delta) for seed, delta in zip(seeds, deltas)}),
        "mean_delta": _nullable_float(numpy.mean(deltas)) if deltas.size else None,
        "ci95_delta_low": ci_low,
        "ci95_delta_high": ci_high,
        "test_name": stats_report["test_name"],
        "test_conditions": stats_report["conditions"],
        "p_value": stats_report["p_value"],
        "effect_size": stats_report["effect_size"],
        "effect_size_name": stats_report["effect_size_name"],
        "superiority": superiority,
        "superiority_rule": "ci95_delta_low > 0",
        "noninferiority": noninferiority,
        "noninferiority_margin": noninferiority_margin,
        "noninferiority_margin_justification": noninferiority_margin_justification or "",
    }


def _chance_row(rows_by_key, classifier, class_labels):
    chance = 1.0 / float(len(class_labels))
    values = _scenario_seed_values(rows_by_key, classifier, "label_permutation_r500")
    seeds = sorted(values)
    deltas = numpy.asarray([values[seed]["MacroF1"] - chance for seed in seeds], dtype=float)
    stats_report = paired_test(deltas)
    ci_low, ci_high = confidence_interval(deltas)
    return {
        "comparison_id": "label_permutation_vs_chance",
        "comparison_family": "negative_control",
        "classifier": classifier,
        "candidate": "label_permutation_r500",
        "baseline": "chance",
        "question": "label_permutation versus acaso: controle negativo",
        "metric": "MacroF1",
        "n_pairs": len(seeds),
        "seeds": json.dumps(seeds),
        "delta_by_seed": json.dumps({str(seed): float(delta) for seed, delta in zip(seeds, deltas)}),
        "mean_delta": _nullable_float(numpy.mean(deltas)) if deltas.size else None,
        "ci95_delta_low": ci_low,
        "ci95_delta_high": ci_high,
        "test_name": stats_report["test_name"],
        "test_conditions": stats_report["conditions"],
        "p_value": stats_report["p_value"],
        "effect_size": stats_report["effect_size"],
        "effect_size_name": stats_report["effect_size_name"],
        "superiority": False,
        "superiority_rule": "negative control should not exceed chance materially",
        "noninferiority": "",
        "noninferiority_margin": "",
        "noninferiority_margin_justification": "",
    }


def paired_test(deltas):
    deltas = numpy.asarray(deltas, dtype=float)
    deltas = deltas[numpy.isfinite(deltas)]
    conditions = {"n": int(deltas.size), "normality_test": None, "normality_p": None}
    if deltas.size < 2:
        return _test_result("not_applicable", conditions, None, None, "insufficient pairs")
    if numpy.allclose(deltas, 0.0):
        return _test_result("not_applicable", conditions, 1.0, 0.0, "all paired deltas are zero")
    normal = False
    if 3 <= deltas.size <= 5000:
        shapiro = stats.shapiro(deltas)
        conditions["normality_test"] = "shapiro_wilk"
        conditions["normality_p"] = float(shapiro.pvalue)
        normal = bool(shapiro.pvalue >= 0.05)
    conditions["selected_test_rule"] = "paired_ttest_if_normal_else_wilcoxon"
    effect = _cohens_dz(deltas)
    if normal:
        result = stats.ttest_1samp(deltas, popmean=0.0)
        return _test_result("paired_ttest_on_deltas", conditions, float(result.pvalue), effect, "cohens_dz")
    try:
        result = stats.wilcoxon(deltas)
        return _test_result("wilcoxon_signed_rank", conditions, float(result.pvalue), effect, "cohens_dz")
    except ValueError as error:
        return _test_result("not_applicable", conditions, None, effect, str(error))


def _test_result(test_name, conditions, p_value, effect_size, effect_size_name):
    return {
        "test_name": test_name,
        "conditions": json.dumps(conditions, sort_keys=True),
        "p_value": "" if p_value is None else float(p_value),
        "effect_size": "" if effect_size is None else float(effect_size),
        "effect_size_name": effect_size_name,
    }


def summarize(values):
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    ci_low, ci_high = confidence_interval(values)
    return {
        "mean": _nullable_float(numpy.mean(values)) if values.size else None,
        "median": _nullable_float(numpy.median(values)) if values.size else None,
        "std": _nullable_float(numpy.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": _nullable_float(numpy.min(values)) if values.size else None,
        "max": _nullable_float(numpy.max(values)) if values.size else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def confidence_interval(values):
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    if values.size == 0:
        return None, None
    if values.size == 1:
        value = float(values[0])
        return value, value
    mean = float(numpy.mean(values))
    sem = float(stats.sem(values))
    radius = float(stats.t.ppf(0.975, df=values.size - 1) * sem)
    return mean - radius, mean + radius


def holm_correction(p_values):
    if not p_values:
        return []
    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, value)
        adjusted[original_index] = running_max
    return adjusted


def build_statistical_report(matrix, aggregate_rows, paired_rows):
    lines = [
        "# AppClassNet Synthetic Significance Report",
        "",
        f"Status: `{matrix.get('status')}`",
        f"Git commit: `{matrix.get('git', {}).get('commit')}`",
        f"Git dirty: `{matrix.get('git', {}).get('dirty')}`",
        f"Seeds: `{matrix.get('seeds')}`",
        f"Classifiers: `{matrix.get('classifiers')}`",
        "",
        "## Protocol",
        "",
        "- Experiments are paired by seed, classifier, class selection, transform and real TEST selection.",
        "- Superiority requires the 95% CI of MacroF1 delta to be completely above zero.",
        "- Statistical test selection records normality conditions before choosing the paired test.",
        "- Holm correction is applied to main paired MacroF1 comparisons.",
        "",
        "## Aggregate MacroF1",
        "",
        "| Classifier | Scenario | Mean | CI95 | N |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['classifier']} | {row['scenario_id']} | {row['MacroF1_mean']} | "
            f"[{row['MacroF1_ci95_low']}, {row['MacroF1_ci95_high']}] | {row['n_seeds']} |"
        )
    lines.extend([
        "",
        "## Paired Comparisons",
        "",
        "| Comparison | Classifier | Mean Delta | CI95 Delta | Test | p | p Holm | Superiority |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ])
    for row in paired_rows:
        lines.append(
            f"| {row['comparison_id']} | {row['classifier']} | {row['mean_delta']} | "
            f"[{row['ci95_delta_low']}, {row['ci95_delta_high']}] | {row['test_name']} | "
            f"{row['p_value']} | {row.get('p_value_holm', '')} | {row['superiority']} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        "- `experiment_matrix.json`",
        "- `per_seed_results.csv`",
        "- `aggregate_results.csv`",
        "- `paired_comparisons.csv`",
    ])
    return "\n".join(lines) + "\n"


def _scenario_seed_values(rows_by_key, classifier, scenario_id):
    return {
        seed: row
        for (row_classifier, row_scenario, seed), row in rows_by_key.items()
        if row_classifier == classifier and row_scenario == scenario_id
    }


def _counts_by_class(labels):
    unique, counts = numpy.unique(numpy.asarray(labels, dtype=numpy.int64), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def _stable_seed(base_seed, seed, context):
    digest = hashlib.sha256(f"{int(base_seed)}:{int(seed)}:{context}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def _indices_hash(indices):
    return hashlib.sha256(numpy.asarray(indices, dtype=numpy.int64).tobytes()).hexdigest()


def _memory_mb():
    if psutil is None:
        return None
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def _cohens_dz(deltas):
    if deltas.size < 2:
        return None
    std = float(numpy.std(deltas, ddof=1))
    if std <= 0.0:
        return None
    return float(numpy.mean(deltas) / std)


def _nullable_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _optional_float(value):
    if value in {"", None}:
        return numpy.nan
    return float(value)


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _json_ready(value):
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def git_metadata():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {"commit": None, "dirty": None, "status_short": None}
    return {"commit": commit, "dirty": bool(status.strip()), "status_short": status.splitlines()}


def dependency_versions():
    packages = ["numpy", "scipy", "scikit-learn", "tensorflow", "keras", "pandas"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_metadata():
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_total_mb": None if psutil is None else float(psutil.virtual_memory().total / (1024 ** 2)),
    }
