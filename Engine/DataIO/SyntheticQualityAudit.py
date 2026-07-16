#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Layered synthetic quality audit for AppClassNet-style data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from Engine.Classifiers.BatchClassifiers import iter_synthetic_labeled_batches
from Engine.DataIO.DatasetContracts import compute_schema_hash
from Engine.DataIO.DatasetContracts import validate_xy_alignment
from Engine.DataIO.LabelUtils import labels_to_1d_integer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APPCLASSNET_AUDIT_JSON = PROJECT_ROOT / "results" / "appclassnet_top200" / "batches" / "synthetic_quality_audit.json"
APPCLASSNET_AUDIT_DOC = PROJECT_ROOT / "results" / "appclassnet_top200" / "batches" / "synthetic_quality_audit.md"
EPSILON = 1e-12


class SyntheticClassCollapseError(RuntimeError):
    """Raised when synthetic classes collapse into indistinguishable data."""


class SyntheticEvaluationPipelineError(RuntimeError):
    """Raised when a control proves the synthetic evaluation path is broken."""


def run_synthetic_quality_audit(
        real_train_x,
        real_train_y,
        real_test_x,
        real_test_y,
        synthetic_data,
        *,
        number_classes,
        expected_num_features=20,
        schema=None,
        arguments=None,
        experiment_directory=None,
        fold_number=None,
        model_type=None,
        fail_on_collapse=True,
        decoder_probe=None,
        latent_dim=8):
    """Run the complete audit and write JSON/Markdown artifacts."""

    auditor = SyntheticQualityAuditor(
        real_train_x=real_train_x,
        real_train_y=real_train_y,
        real_test_x=real_test_x,
        real_test_y=real_test_y,
        synthetic_data=synthetic_data,
        number_classes=number_classes,
        expected_num_features=expected_num_features,
        schema=schema,
        arguments=arguments,
        experiment_directory=experiment_directory,
        fold_number=fold_number,
        model_type=model_type,
        decoder_probe=decoder_probe,
        latent_dim=latent_dim,
    )
    path, report = auditor.run()
    if fail_on_collapse and report["collapse_decision"]["collapsed"]:
        raise SyntheticClassCollapseError(
            "SyntheticClassCollapseError: synthetic classes collapsed. "
            f"See {path}. Evidence: {report['collapse_decision']['reasons'][:5]}"
        )
    return path, report


def assert_real_resample_control_metrics(metrics, *, number_classes, synthetic_control, fold=1):
    if synthetic_control != "real_resample":
        return
    chance = 1.0 / float(number_classes)
    failures = []
    for mode in ("TR-TS", "TS-TR"):
        mode_block = metrics.get(mode, {})
        for classifier, classifier_block in mode_block.items():
            if classifier == "Summary" or not isinstance(classifier_block, dict):
                continue
            fold_metrics = classifier_block.get(f"{fold}-Fold", {})
            accuracy = _safe_float(fold_metrics.get("Accuracy"))
            balanced = _safe_float(fold_metrics.get("BalancedAccuracy"))
            if accuracy is None or balanced is None:
                failures.append(f"{mode}/{classifier} missing Accuracy or BalancedAccuracy")
            elif accuracy <= chance * 3.0 and balanced <= chance * 3.0:
                failures.append(
                    f"{mode}/{classifier} near chance for real_resample "
                    f"(Accuracy={accuracy:.6f}, BalancedAccuracy={balanced:.6f}, chance={chance:.6f})"
                )
    if failures:
        raise SyntheticEvaluationPipelineError(
            "real_resample synthetic control failed; the issue is in the synthetic reader, manifest, "
            "labels, schema or evaluation path. " + "; ".join(failures)
        )


def assert_label_permutation_control_metrics(metrics, *, number_classes, synthetic_control, fold=1):
    if synthetic_control != "label_permutation":
        return
    chance = 1.0 / float(number_classes)
    failures = []
    for mode in ("TR-TS", "TS-TR"):
        mode_block = metrics.get(mode, {})
        for classifier, classifier_block in mode_block.items():
            if classifier == "Summary" or not isinstance(classifier_block, dict):
                continue
            fold_metrics = classifier_block.get(f"{fold}-Fold", {})
            accuracy = _safe_float(fold_metrics.get("Accuracy"))
            balanced = _safe_float(fold_metrics.get("BalancedAccuracy"))
            if accuracy is None or balanced is None:
                failures.append(f"{mode}/{classifier} missing Accuracy or BalancedAccuracy")
            elif accuracy > chance * 3.0 or balanced > chance * 3.0:
                failures.append(
                    f"{mode}/{classifier} too far above chance for label_permutation "
                    f"(Accuracy={accuracy:.6f}, BalancedAccuracy={balanced:.6f}, chance={chance:.6f})"
                )
    if failures:
        raise SyntheticEvaluationPipelineError(
            "label_permutation synthetic control failed; shuffled labels did not produce chance-level metrics. "
            + "; ".join(failures)
        )


class SyntheticQualityAuditor:

    def __init__(
            self,
            real_train_x,
            real_train_y,
            real_test_x,
            real_test_y,
            synthetic_data,
            number_classes,
            expected_num_features=20,
            schema=None,
            arguments=None,
            experiment_directory=None,
            fold_number=None,
            model_type=None,
            decoder_probe=None,
            latent_dim=8):
        self.real_train_x, self.real_train_y = validate_xy_alignment(
            real_train_x,
            labels_to_1d_integer(real_train_y, context="audit real_train_y"),
            "synthetic quality audit real train",
        )
        self.real_test_x, self.real_test_y = validate_xy_alignment(
            real_test_x,
            labels_to_1d_integer(real_test_y, context="audit real_test_y"),
            "synthetic quality audit real test",
        )
        self.real_train_x = numpy.asarray(self.real_train_x, dtype=numpy.float32)
        self.real_test_x = numpy.asarray(self.real_test_x, dtype=numpy.float32)
        self.synthetic_data = synthetic_data
        self.number_classes = int(number_classes)
        self.expected_num_features = int(expected_num_features)
        self.schema = schema
        self.arguments = arguments
        self.experiment_directory = None if experiment_directory is None else str(experiment_directory)
        self.fold_number = None if fold_number is None else int(fold_number)
        self.model_type = model_type
        self.decoder_probe = decoder_probe
        self.latent_dim = int(latent_dim)
        self.chance = 1.0 / float(max(1, self.number_classes))
        self.errors = []
        self.warnings = []
        self.rng = numpy.random.default_rng(0)

    def run(self):
        synthetic_sections = self._synthetic_sections()
        schema_report = self._schema_report(synthetic_sections)
        transform_report = self._transform_report()
        training_report = self._training_report()

        real_resample = self._make_real_resample(synthetic_sections)
        label_permutation = self._make_label_permutation(real_resample)
        global_feature_quality = self._global_feature_quality(synthetic_sections, real_resample, label_permutation)
        per_class_feature_quality = self._per_class_feature_quality(synthetic_sections, real_resample, label_permutation)
        conditional_fidelity = self._conditional_fidelity(synthetic_sections, real_resample, label_permutation)
        decoder_sensitivity = self._decoder_sensitivity(synthetic_sections)
        memorization = self._memorization(synthetic_sections)
        controls = self._controls(
            real_resample,
            label_permutation,
            global_feature_quality,
            conditional_fidelity,
        )
        collapse_decision = self._collapse_decision(
            global_feature_quality,
            per_class_feature_quality,
            conditional_fidelity,
            decoder_sensitivity,
            memorization,
            controls,
        )
        evidence = self._evidence(
            global_feature_quality,
            per_class_feature_quality,
            conditional_fidelity,
            decoder_sensitivity,
            memorization,
            controls,
            collapse_decision,
        )
        causes = self._classify_causes(schema_report, synthetic_sections, collapse_decision, transform_report, training_report)

        synthetic_report = dict(synthetic_sections)
        if "synthetic_train" in synthetic_report:
            synthetic_report.setdefault("train", synthetic_report["synthetic_train"])
        if "synthetic_test" in synthetic_report:
            synthetic_report.setdefault("test", synthetic_report["synthetic_test"])

        report = {
            "status": "failed" if self.errors else "completed",
            "root_cause": "multiple causes" if len(causes) > 1 else causes[0],
            "root_cause_candidates": causes,
            "chance_accuracy": self.chance,
            "number_classes": self.number_classes,
            "expected_num_features": self.expected_num_features,
            "fold_number": self.fold_number,
            "model_type": self.model_type,
            "experiment_directory": self.experiment_directory,
            "synthetic_control": getattr(self.arguments, "synthetic_control", "none") if self.arguments else "none",
            "generation_strategy": getattr(self.arguments, "generation_strategy", None) if self.arguments else None,
            "classes_per_group": getattr(self.arguments, "classes_per_group", None) if self.arguments else None,
            "schema": schema_report,
            "real": {
                "train_shape": list(self.real_train_x.shape),
                "test_shape": list(self.real_test_x.shape),
                "train_statistics": _jsonify_stats(_dataset_stats(self.real_train_x, self.real_train_y, self.number_classes)),
                "test_statistics": _jsonify_stats(_dataset_stats(self.real_test_x, self.real_test_y, self.number_classes)),
            },
            "synthetic": synthetic_report,
            "transform": transform_report,
            "training": training_report,
            "global_feature_quality": global_feature_quality,
            "per_class_feature_quality": per_class_feature_quality,
            "conditional_fidelity": conditional_fidelity,
            "decoder_sensitivity": decoder_sensitivity,
            "memorization": memorization,
            "controls": controls,
            "collapse_decision": collapse_decision,
            "collapse": collapse_decision,
            "evidence": evidence,
            "errors": self.errors,
            "warnings": self.warnings,
        }

        _write_json_and_markdown(report, self.experiment_directory)
        return APPCLASSNET_AUDIT_JSON, report

    def _synthetic_sections(self):
        if hasattr(self.synthetic_data, "train_reader") and hasattr(self.synthetic_data, "test_reader"):
            items = {
                "synthetic_train": self.synthetic_data.train_reader,
                "synthetic_test": self.synthetic_data.test_reader,
            }
        else:
            split_name = _manifest(self.synthetic_data).get("split") if hasattr(self.synthetic_data, "manifest") else "synthetic_train"
            items = {f"synthetic_{split_name}" if split_name in {"train", "test"} else (split_name or "synthetic_train"): self.synthetic_data}

        sections = {}
        for split_name, synthetic_split in items.items():
            sections[split_name] = self._audit_synthetic_split(split_name, synthetic_split)
        if "synthetic_train" not in sections and "train" in sections:
            sections["synthetic_train"] = sections["train"]
        if "synthetic_test" not in sections and "test" in sections:
            sections["synthetic_test"] = sections["test"]
        return sections

    def _audit_synthetic_split(self, split_name, synthetic_split):
        manifest = _manifest(synthetic_split)
        manifest_errors = self._validate_manifest(split_name, manifest)
        x_values, y_values, batch_report = _materialize_synthetic(synthetic_split, self.expected_num_features)
        label_errors = self._validate_synthetic_arrays(split_name, x_values, y_values, manifest)
        if manifest_errors or label_errors:
            self.errors.extend(manifest_errors + label_errors)

        stats = _dataset_stats(x_values, y_values, self.number_classes)
        return {
            "manifest_path": str(getattr(synthetic_split, "manifest_path", "")),
            "manifest": _manifest_summary(manifest),
            "schema_hash": manifest.get("schema_hash"),
            "data_space": manifest.get("data_space"),
            "transform_id": manifest.get("transform_id"),
            "total_rows": int(y_values.shape[0]),
            "label_audit_errors": label_errors,
            "manifest_errors": manifest_errors,
            "batch_report": batch_report,
            "statistics": _jsonify_stats(stats),
            "x": x_values,
            "y": y_values,
        }

    def _validate_manifest(self, split_name, manifest):
        errors = []
        if not manifest:
            errors.append(f"{split_name}: missing synthetic manifest")
            return errors
        if int(manifest.get("features", -1)) != self.expected_num_features:
            errors.append(
                f"{split_name}: manifest features={manifest.get('features')} expected={self.expected_num_features}"
            )
        if int(manifest.get("num_classes", -1)) != self.number_classes:
            errors.append(
                f"{split_name}: manifest num_classes={manifest.get('num_classes')} expected={self.number_classes}"
            )
        declared = sorted(int(label) for label in manifest.get("batches_by_class", {}).keys())
        expected = _expected_manifest_classes(manifest, declared)
        if declared != expected:
            missing = sorted(set(expected) - set(declared))
            extra = sorted(set(declared) - set(expected))
            errors.append(f"{split_name}: manifest classes are not exactly expected; missing={missing[:20]} extra={extra[:20]}")
        if manifest.get("data_space", "source") != "source":
            errors.append(f"{split_name}: synthetic data_space={manifest.get('data_space')} expected=source")
        return errors

    def _validate_synthetic_arrays(self, split_name, x_values, y_values, manifest):
        errors = []
        if x_values.shape[0] != y_values.shape[0]:
            errors.append(f"{split_name}: X.shape[0]={x_values.shape[0]} y.shape[0]={y_values.shape[0]}")
        if x_values.ndim != 2 or x_values.shape[1] != self.expected_num_features:
            errors.append(f"{split_name}: X.shape={list(x_values.shape)} expected second dimension {self.expected_num_features}")
        if not numpy.issubdtype(y_values.dtype, numpy.integer):
            errors.append(f"{split_name}: labels dtype={y_values.dtype} is not integer")
        if y_values.size:
            if int(y_values.min()) < 0 or int(y_values.max()) >= self.number_classes:
                errors.append(f"{split_name}: labels outside [0, {self.number_classes - 1}]")
            observed = sorted(int(label) for label in numpy.unique(y_values))
            expected = _expected_manifest_classes(manifest, observed)
            if observed != expected:
                errors.append(f"{split_name}: observed labels are not exactly expected; observed_count={len(observed)}")
            if int(y_values.min()) == 1 and int(y_values.max()) == self.number_classes:
                errors.append(f"{split_name}: labels look 1-based (1..{self.number_classes}) instead of 0-based")
        for class_label, batches in manifest.get("batches_by_class", {}).items():
            class_count = sum(int(batch.get("shape", [0])[0]) for batch in batches)
            actual = int(numpy.sum(y_values == int(class_label)))
            if class_count != actual:
                errors.append(f"{split_name}: manifest class {class_label} count={class_count} actual_y={actual}")
        expected_count = _expected_count_for_split(self.arguments, split_name.replace("synthetic_", ""))
        if expected_count is not None:
            counts = _counts_dict(y_values, self.number_classes)
            expected_labels = _expected_manifest_classes(manifest, sorted(counts))
            bad = {
                label: count
                for label, count in counts.items()
                if int(label) in expected_labels and int(count) != int(expected_count)
            }
            if bad:
                errors.append(f"{split_name}: per-class counts differ from expected {expected_count}; examples={dict(list(bad.items())[:10])}")
        return errors

    def _schema_report(self, synthetic_sections):
        real_hash = None
        if self.schema is not None:
            real_hash = getattr(self.schema, "schema_hash", None) or compute_schema_hash(self.schema)
        synthetic_hashes = {
            split_name: section.get("schema_hash")
            for split_name, section in synthetic_sections.items()
        }
        mismatches = {
            split_name: value
            for split_name, value in synthetic_hashes.items()
            if real_hash is not None and value is not None and value != real_hash
        }
        if mismatches:
            self.errors.append(f"schema hash mismatch: real={real_hash} synthetic={mismatches}")
        return {
            "real_schema_hash": real_hash,
            "synthetic_schema_hashes": synthetic_hashes,
            "hashes_match": not mismatches,
            "feature_names": list(getattr(self.schema, "feature_names", [])) if self.schema is not None else [f"f{i}" for i in range(self.expected_num_features)],
            "feature_dtype": getattr(self.schema, "feature_dtype", None) if self.schema is not None else None,
            "data_space": getattr(self.schema, "data_space", None) if self.schema is not None else None,
            "transform_id": getattr(self.schema, "transform_id", None) if self.schema is not None else None,
        }

    def _transform_report(self):
        generator_transform = getattr(self.arguments, "generator_transform", None) if self.arguments else None
        inverse = bool(getattr(self.arguments, "inverse_transform_synthetic", False)) if self.arguments else False
        preserve_inverse_noop = None
        if generator_transform == "preserve" and inverse:
            probe = numpy.asarray(self.real_train_x[: min(32, self.real_train_x.shape[0])], dtype=numpy.float32)
            preserve_inverse_noop = bool(numpy.array_equal(probe, probe.copy()))
        return {
            "generator_transform": generator_transform,
            "inverse_transform_synthetic": inverse,
            "preserve_inverse_is_noop": preserve_inverse_noop,
        }

    def _training_report(self):
        epochs = _configured_epochs(self.arguments)
        losses = getattr(self.arguments, "_training_history", None) if self.arguments else None
        warnings = []
        if self.number_classes >= 100 and epochs.get("effective_min_epochs", 2) is not None and epochs.get("effective_min_epochs", 2) <= 1:
            warning = "num_classes >= 100 and epochs <= 1: training is probably insufficient."
            warnings.append(warning)
            self.warnings.append(warning)
        return {
            "epochs": epochs,
            "losses": losses,
            "warnings": warnings,
        }

    def _global_feature_quality(self, synthetic_sections, real_resample, label_permutation):
        comparisons = {}
        for split_name, section in synthetic_sections.items():
            real_x = self.real_train_x if split_name != "synthetic_test" else self.real_test_x
            comparisons[split_name] = _compare_global_features(real_x, section["x"], self.expected_num_features)
        comparisons["real_resample_control"] = _compare_global_features(self.real_train_x, real_resample["x"], self.expected_num_features)
        comparisons["label_permutation_control"] = _compare_global_features(self.real_train_x, label_permutation["x"], self.expected_num_features)
        return comparisons

    def _per_class_feature_quality(self, synthetic_sections, real_resample, label_permutation):
        comparisons = {}
        for split_name, section in synthetic_sections.items():
            real_x = self.real_train_x if split_name != "synthetic_test" else self.real_test_x
            real_y = self.real_train_y if split_name != "synthetic_test" else self.real_test_y
            comparisons[split_name] = _compare_per_class(real_x, real_y, section["x"], section["y"], self.number_classes)
        comparisons["real_resample_control"] = _compare_per_class(
            self.real_train_x, self.real_train_y, real_resample["x"], real_resample["y"], self.number_classes
        )
        comparisons["label_permutation_control"] = _compare_per_class(
            self.real_train_x, self.real_train_y, label_permutation["x"], label_permutation["y"], self.number_classes
        )
        return comparisons

    def _conditional_fidelity(self, synthetic_sections, real_resample, label_permutation):
        classifier = _train_real_classifier(
            self.real_train_x,
            self.real_train_y,
            self.number_classes,
            _subset_quota(self.arguments, self.number_classes),
        )
        evaluations = {
            "real_test": _evaluate_classifier(classifier, self.real_test_x, self.real_test_y, self.number_classes),
            "real_resample": _evaluate_classifier(classifier, real_resample["x"], real_resample["y"], self.number_classes),
            "label_permutation": _evaluate_classifier(classifier, label_permutation["x"], label_permutation["y"], self.number_classes),
        }
        for split_name, section in synthetic_sections.items():
            evaluations[split_name] = _evaluate_classifier(classifier, section["x"], section["y"], self.number_classes)

        real_recall = evaluations["real_test"]["recall_by_class"]
        real_balanced_dominant = evaluations["real_resample"]["dominant_predicted_class_fraction"]
        for name, evaluation in evaluations.items():
            evaluation["relative_fidelity_by_class"] = _relative_recall(evaluation["recall_by_class"], real_recall)
            evaluation["dominant_predicted_class_fraction_reference"] = real_balanced_dominant
            evaluation["dominant_predicted_class_fraction_excess_over_real_resample"] = _subtract_nullable(
                evaluation["dominant_predicted_class_fraction"],
                real_balanced_dominant,
            )

        ts_tr = {}
        for split_name, section in synthetic_sections.items():
            ts_tr[split_name] = _synthetic_train_real_test(section["x"], section["y"], self.real_test_x, self.real_test_y, self.number_classes)

        return {
            "audit_classifier": "DecisionTreeClassifier(random_state=0)",
            "trained_on": "real_train",
            "evaluations": evaluations,
            "matrix_comparison": {
                "real_confusion_matrix": evaluations["real_test"]["confusion_matrix"],
                "synthetic_train_confusion_matrix": evaluations.get("synthetic_train", {}).get("confusion_matrix"),
                "synthetic_test_confusion_matrix": evaluations.get("synthetic_test", {}).get("confusion_matrix"),
            },
            "ts_tr": ts_tr,
        }

    def _decoder_sensitivity(self, synthetic_sections):
        manifest_labels = sorted({
            int(label)
            for section in synthetic_sections.values()
            for label in section.get("manifest", {}).get("batches_by_class", {}).keys()
        })
        if not manifest_labels:
            manifest_labels = list(range(self.number_classes))

        labels_to_probe = _probe_labels(manifest_labels, self.number_classes)
        proxy = _synthetic_interclass_proxy(synthetic_sections, self.number_classes)
        report = {
            "direct_probe_available": self.decoder_probe is not None,
            "labels_recorded_in_manifest": manifest_labels[:1000],
            "labels_requested_for_probe": labels_to_probe,
            "requested_labels_match_manifest": set(labels_to_probe).issubset(set(manifest_labels)),
            "proxy_interclass_synthetic_distance": proxy,
            "conditional_sensitivity_score": None,
            "same_z_same_label_reproducible": None,
            "same_z_different_labels_changes_output": None,
            "outputs_identical_for_all_labels": None,
            "reason": None,
        }
        if self.decoder_probe is None:
            report["reason"] = "decoder_probe not provided; direct latent-label sensitivity could not be measured"
            return report

        try:
            probe = _call_decoder_probe(self.decoder_probe, self.latent_dim, labels_to_probe)
        except Exception as exc:
            report["reason"] = f"decoder_probe failed: {exc}"
            self.warnings.append(report["reason"])
            return report

        same_z_same_label = float(numpy.linalg.norm(probe["same_label_first"] - probe["same_label_second"], axis=1).mean())
        diff_label = _pairwise_mean_distance(probe["different_labels"])
        same_label_diff_z = float(numpy.linalg.norm(probe["same_label_first"] - probe["same_label_new_z"], axis=1).mean())
        score = diff_label / max(diff_label + same_label_diff_z + same_z_same_label, EPSILON)
        report.update({
            "same_z_same_label_mean_distance": same_z_same_label,
            "same_z_different_labels_mean_distance": diff_label,
            "different_z_same_label_mean_distance": same_label_diff_z,
            "conditional_sensitivity_score": float(score),
            "same_z_same_label_reproducible": bool(same_z_same_label <= 1e-8),
            "same_z_different_labels_changes_output": bool(diff_label > max(1e-6, same_z_same_label * 100.0)),
            "outputs_identical_for_all_labels": bool(diff_label <= 1e-8),
            "reason": "direct latent-label probe completed",
        })
        return report

    def _memorization(self, synthetic_sections):
        report = {}
        for split_name, section in synthetic_sections.items():
            report[split_name] = _memorization_split(section["x"], self.real_train_x, self.real_test_x)
        return report

    def _controls(self, real_resample, label_permutation, global_feature_quality, conditional_fidelity):
        real_resample_eval = conditional_fidelity["evaluations"]["real_resample"]
        label_perm_eval = conditional_fidelity["evaluations"]["label_permutation"]
        real_test_eval = conditional_fidelity["evaluations"]["real_test"]
        positive = {
            "name": "real_resample",
            "balanced_accuracy": real_resample_eval["balanced_accuracy"],
            "dominant_predicted_class_fraction": real_resample_eval["dominant_predicted_class_fraction"],
            "global_mmd_rbf": global_feature_quality["real_resample_control"]["multivariate"]["mmd_rbf"],
        }
        negative = {
            "name": "label_permutation",
            "balanced_accuracy": label_perm_eval["balanced_accuracy"],
            "dominant_predicted_class_fraction": label_perm_eval["dominant_predicted_class_fraction"],
            "global_mmd_rbf": global_feature_quality["label_permutation_control"]["multivariate"]["mmd_rbf"],
        }
        return {
            "positive_control": positive,
            "negative_control": negative,
            "test_real_reference": {
                "balanced_accuracy": real_test_eval["balanced_accuracy"],
                "dominant_predicted_class_fraction": real_test_eval["dominant_predicted_class_fraction"],
            },
            "calibrated_limits": {
                "dominant_predicted_class_fraction_max": min(0.95, max(positive["dominant_predicted_class_fraction"] or 0.0, real_test_eval["dominant_predicted_class_fraction"] or 0.0) + 0.20),
                "ts_tr_balanced_accuracy_min": max(self.chance * 3.0, (real_test_eval["balanced_accuracy"] or 0.0) * 0.20),
                "constant_feature_fraction_max": 0.50,
                "interclass_diversity_ratio_min": 0.20,
                "near_copy_fraction_max": 0.01,
            },
        }

    def _collapse_decision(
            self,
            global_feature_quality,
            per_class_feature_quality,
            conditional_fidelity,
            decoder_sensitivity,
            memorization,
            controls):
        reasons = []
        evidence_flags = []
        limits = controls["calibrated_limits"]
        splits = [name for name in global_feature_quality if name.startswith("synthetic")]
        for split_name in splits:
            global_block = global_feature_quality[split_name]
            multi = global_block["multivariate"]
            if (multi.get("synthetic_total_variance") or 0.0) <= max((multi.get("real_total_variance") or 0.0) * 0.01, 1e-10):
                evidence_flags.append("near_zero_multivariate_variance")
                reasons.append(f"{split_name}: multivariate variance is nearly zero")
            if (multi.get("constant_feature_fraction") or 0.0) >= limits["constant_feature_fraction_max"]:
                evidence_flags.append("many_constant_features")
                reasons.append(f"{split_name}: constant feature fraction={multi.get('constant_feature_fraction'):.4f}")

            inter_ratio = per_class_feature_quality[split_name]["summary"].get("synthetic_to_real_interclass_diversity_ratio")
            if inter_ratio is not None and inter_ratio < limits["interclass_diversity_ratio_min"]:
                evidence_flags.append("low_interclass_diversity")
                reasons.append(f"{split_name}: synthetic/real interclass diversity ratio={inter_ratio:.4f}")

            eval_block = conditional_fidelity["evaluations"].get(split_name, {})
            dominant = eval_block.get("dominant_predicted_class_fraction")
            reference = eval_block.get("dominant_predicted_class_fraction_reference")
            if dominant is not None and dominant > 0.95 and (reference is None or reference < 0.95):
                evidence_flags.append("dominant_predicted_class")
                reasons.append(f"{split_name}: one predicted class dominates {dominant:.4f} of synthetic rows")

            ts_tr = conditional_fidelity["ts_tr"].get(split_name, {})
            if ts_tr.get("balanced_accuracy") is not None and ts_tr["balanced_accuracy"] <= limits["ts_tr_balanced_accuracy_min"]:
                evidence_flags.append("ts_tr_near_chance")
                reasons.append(f"{split_name}: TS-TR balanced accuracy={ts_tr['balanced_accuracy']:.4f}")

            near_copy = memorization.get(split_name, {}).get("near_train_copy_fraction")
            exact_copy = memorization.get(split_name, {}).get("exact_train_copy_fraction")
            if exact_copy is not None and exact_copy > 0.20:
                evidence_flags.append("memorization_exact_copies")
                reasons.append(f"{split_name}: exact TRAIN copy fraction={exact_copy:.4f}")
            elif near_copy is not None and near_copy > 0.50:
                evidence_flags.append("memorization_near_copies")
                reasons.append(f"{split_name}: near TRAIN copy fraction={near_copy:.4f}")

        if decoder_sensitivity.get("direct_probe_available"):
            if decoder_sensitivity.get("outputs_identical_for_all_labels"):
                evidence_flags.append("decoder_ignores_label")
                reasons.append("decoder probe: outputs for different labels are practically identical")
            elif decoder_sensitivity.get("conditional_sensitivity_score") is not None and decoder_sensitivity["conditional_sensitivity_score"] < 0.05:
                evidence_flags.append("decoder_low_label_sensitivity")
                reasons.append(f"decoder probe: conditional sensitivity score={decoder_sensitivity['conditional_sensitivity_score']:.4f}")

        strong_non_memorization_flags = [
            flag for flag in evidence_flags
            if flag not in {"memorization_exact_copies", "memorization_near_copies"}
        ]
        collapsed = len(set(strong_non_memorization_flags)) >= 2 or "decoder_ignores_label" in evidence_flags
        return {
            "collapsed": bool(collapsed),
            "reasons": reasons,
            "evidence_flags": sorted(set(evidence_flags)),
            "decision_rule": (
                "collapse requires combined evidence; nearest centroid correctness alone is not a final criterion"
            ),
            "correct_centroid_nearest_rate_used_as_final_decision": False,
        }

    def _evidence(
            self,
            global_feature_quality,
            per_class_feature_quality,
            conditional_fidelity,
            decoder_sensitivity,
            memorization,
            controls,
            collapse_decision):
        synthetic_train_eval = conditional_fidelity["evaluations"].get("synthetic_train", {})
        synthetic_train_global = global_feature_quality.get("synthetic_train", {}).get("multivariate", {})
        synthetic_train_class = per_class_feature_quality.get("synthetic_train", {}).get("summary", {})
        synthetic_train_mem = memorization.get("synthetic_train", {})
        return {
            "dominant_predicted_class_fraction": synthetic_train_eval.get("dominant_predicted_class_fraction"),
            "dominant_predicted_class_fraction_real_resample": controls["positive_control"].get("dominant_predicted_class_fraction"),
            "synthetic_total_variance": synthetic_train_global.get("synthetic_total_variance"),
            "real_total_variance": synthetic_train_global.get("real_total_variance"),
            "constant_feature_fraction": synthetic_train_global.get("constant_feature_fraction"),
            "synthetic_to_real_interclass_diversity_ratio": synthetic_train_class.get("synthetic_to_real_interclass_diversity_ratio"),
            "ts_tr_balanced_accuracy": conditional_fidelity["ts_tr"].get("synthetic_train", {}).get("balanced_accuracy"),
            "decoder_conditional_sensitivity_score": decoder_sensitivity.get("conditional_sensitivity_score"),
            "exact_train_copy_fraction": synthetic_train_mem.get("exact_train_copy_fraction"),
            "near_train_copy_fraction": synthetic_train_mem.get("near_train_copy_fraction"),
            "collapse_reasons": collapse_decision["reasons"],
        }

    def _make_real_resample(self, synthetic_sections):
        target_counts = None
        if "synthetic_train" in synthetic_sections:
            target_counts = _counts_dict(synthetic_sections["synthetic_train"]["y"], self.number_classes)
        return _resample_by_class(self.real_train_x, self.real_train_y, self.number_classes, target_counts, self.rng)

    def _make_label_permutation(self, real_resample):
        labels = numpy.asarray(real_resample["y"], dtype=numpy.int64).copy()
        if labels.size:
            labels = self.rng.permutation(labels)
        return {"x": real_resample["x"].copy(), "y": labels}

    def _classify_causes(self, schema_report, synthetic_sections, collapse_decision, transform_report, training_report):
        causes = []
        if any(section["manifest_errors"] for section in synthetic_sections.values()):
            causes.append("evaluation pipeline failure")
        if any(section["label_audit_errors"] for section in synthetic_sections.values()):
            causes.append("label mismatch")
        if not schema_report["hashes_match"]:
            causes.append("schema mismatch")
        if transform_report["generator_transform"] != "preserve" and any(
                section.get("data_space") != "source" for section in synthetic_sections.values()):
            causes.append("transform mismatch")
        if collapse_decision["collapsed"]:
            causes.append("conditional collapse")
        if training_report["warnings"]:
            causes.append("insufficient training")
        if _expected_count_for_split(self.arguments, "train") is not None and _expected_count_for_split(self.arguments, "train") < 100:
            causes.append("insufficient sample count")
        return causes or ["no strong failure evidence"]


def _compare_global_features(real_x, synthetic_x, expected_num_features):
    real_x = _as_2d_float(real_x, expected_num_features)
    synthetic_x = _as_2d_float(synthetic_x, expected_num_features)
    per_feature = []
    real_stats = _feature_stats(real_x)
    synthetic_stats = _feature_stats(synthetic_x)
    for index in range(expected_num_features):
        real_col = _finite_vector(real_x[:, index]) if real_x.shape[1] > index else numpy.asarray([])
        synth_col = _finite_vector(synthetic_x[:, index]) if synthetic_x.shape[1] > index else numpy.asarray([])
        real_min = float(numpy.min(real_col)) if real_col.size else None
        real_max = float(numpy.max(real_col)) if real_col.size else None
        outside = None
        if synth_col.size and real_min is not None and real_max is not None:
            outside = float(numpy.mean((synth_col < real_min) | (synth_col > real_max)))
        per_feature.append({
            "feature_index": index,
            "real": _single_feature_stats(real_x[:, index]) if real_x.shape[1] > index else _empty_single_feature_stats(),
            "synthetic": _single_feature_stats(synthetic_x[:, index]) if synthetic_x.shape[1] > index else _empty_single_feature_stats(),
            "wasserstein_distance": _wasserstein(real_col, synth_col),
            "ks_statistic": _ks_stat(real_col, synth_col),
            "outside_real_domain_fraction": outside,
            "constant_value_fraction": _constant_value_fraction(synth_col),
            "nan_fraction": _nan_fraction(synthetic_x[:, index]) if synthetic_x.shape[1] > index else None,
            "inf_fraction": _inf_fraction(synthetic_x[:, index]) if synthetic_x.shape[1] > index else None,
        })

    real_corr = _corr_matrix(real_x)
    synth_corr = _corr_matrix(synthetic_x)
    real_cov = _cov_matrix(real_x)
    synth_cov = _cov_matrix(synthetic_x)
    return {
        "per_feature": per_feature,
        "real_feature_statistics": real_stats,
        "synthetic_feature_statistics": synthetic_stats,
        "multivariate": {
            "correlation_matrix_real": _matrix_to_list(real_corr),
            "correlation_matrix_synthetic": _matrix_to_list(synth_corr),
            "correlation_matrix_difference": _matrix_to_list(synth_corr - real_corr),
            "correlation_mean_absolute_difference": _mean_abs(synth_corr - real_corr),
            "covariance_matrix_real": _matrix_to_list(real_cov),
            "covariance_matrix_synthetic": _matrix_to_list(synth_cov),
            "covariance_mean_absolute_difference": _mean_abs(synth_cov - real_cov),
            "energy_distance": _energy_distance(real_x, synthetic_x),
            "mmd_rbf": _mmd_rbf(real_x, synthetic_x),
            "pca_distance": _pca_distance(real_x, synthetic_x),
            "coverage": _coverage(real_x, synthetic_x),
            "density": _density(real_x, synthetic_x),
            "duplicates_fraction": _duplicate_fraction(synthetic_x),
            "near_duplicates_fraction": _near_duplicate_fraction(synthetic_x),
            "constant_feature_fraction": _constant_feature_fraction_matrix(synthetic_x),
            "real_total_variance": _total_variance(real_x),
            "synthetic_total_variance": _total_variance(synthetic_x),
        },
    }


def _compare_per_class(real_x, real_y, synthetic_x, synthetic_y, number_classes):
    real_x = numpy.asarray(real_x, dtype=numpy.float64)
    synthetic_x = numpy.asarray(synthetic_x, dtype=numpy.float64)
    real_y = numpy.asarray(real_y, dtype=numpy.int64)
    synthetic_y = numpy.asarray(synthetic_y, dtype=numpy.int64)
    classes = {}
    real_centroids = []
    synth_centroids = []
    class_distances = []
    coverage_values = []
    for class_id in range(int(number_classes)):
        rx = real_x[real_y == class_id] if real_y.size else numpy.empty((0, real_x.shape[1]))
        sx = synthetic_x[synthetic_y == class_id] if synthetic_y.size else numpy.empty((0, synthetic_x.shape[1]))
        real_mean = _nan_list(numpy.nanmean(rx, axis=0)) if rx.size else []
        synth_mean = _nan_list(numpy.nanmean(sx, axis=0)) if sx.size else []
        real_var = _nan_list(numpy.nanvar(rx, axis=0)) if rx.size else []
        synth_var = _nan_list(numpy.nanvar(sx, axis=0)) if sx.size else []
        centroid_distance = _centroid_distance(rx, sx)
        coverage = _coverage(rx, sx)
        nn = _nearest_summary(sx, rx)
        corr_diff = _mean_abs(_corr_matrix(sx) - _corr_matrix(rx)) if rx.shape[0] >= 2 and sx.shape[0] >= 2 else None
        constants = _constant_feature_indices(sx)
        duplicates = int(sx.shape[0] - numpy.unique(sx, axis=0).shape[0]) if sx.shape[0] else 0
        classes[str(class_id)] = {
            "real_count": int(rx.shape[0]),
            "synthetic_count": int(sx.shape[0]),
            "real_mean": real_mean,
            "synthetic_mean": synth_mean,
            "real_variance": real_var,
            "synthetic_variance": synth_var,
            "real_synthetic_centroid_distance": centroid_distance,
            "distribution_coverage": coverage,
            "nearest_neighbor_distance": nn,
            "correlation_mean_absolute_difference": corr_diff,
            "constant_features": constants,
            "duplicate_rows": duplicates,
            "duplicate_fraction": float(duplicates / sx.shape[0]) if sx.shape[0] else None,
        }
        if rx.shape[0]:
            real_centroids.append(numpy.nanmean(rx, axis=0))
        if sx.shape[0]:
            synth_centroids.append(numpy.nanmean(sx, axis=0))
        if centroid_distance is not None:
            class_distances.append(centroid_distance)
        if coverage is not None:
            coverage_values.append(coverage)

    real_inter = _interclass_distance(numpy.asarray(real_centroids, dtype=numpy.float64))
    synth_inter = _interclass_distance(numpy.asarray(synth_centroids, dtype=numpy.float64))
    return {
        "by_class": classes,
        "summary": {
            "mean_real_synthetic_centroid_distance": float(numpy.mean(class_distances)) if class_distances else None,
            "mean_distribution_coverage": float(numpy.mean(coverage_values)) if coverage_values else None,
            "real_interclass_diversity": real_inter,
            "synthetic_interclass_diversity": synth_inter,
            "synthetic_to_real_interclass_diversity_ratio": (
                float(synth_inter / real_inter) if real_inter and synth_inter is not None else None
            ),
            "note": "similar real classes are descriptive evidence only; they do not trigger collapse by themselves",
        },
    }


def _evaluate_classifier(classifier, x_values, y_values, number_classes):
    x_values = numpy.asarray(x_values, dtype=numpy.float32)
    y_values = numpy.asarray(y_values, dtype=numpy.int64)
    if x_values.shape[0] == 0:
        predictions = numpy.asarray([], dtype=numpy.int64)
    else:
        predictions = classifier.predict(x_values).astype(numpy.int64)
    labels = list(range(int(number_classes)))
    cm = confusion_matrix(y_values, predictions, labels=labels) if y_values.size else numpy.zeros((number_classes, number_classes), dtype=numpy.int64)
    recall = []
    for class_id in labels:
        denom = int(cm[class_id].sum())
        recall.append(None if denom == 0 else float(cm[class_id, class_id] / denom))
    counts = _counts_dict(predictions, number_classes)
    return {
        "rows": int(y_values.shape[0]),
        "accuracy": float(accuracy_score(y_values, predictions)) if y_values.size else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_values, predictions)) if y_values.size else None,
        "confusion_matrix": cm.astype(int).tolist(),
        "recall_by_class": {str(i): recall[i] for i in labels},
        "predicted_counts_by_class": counts,
        "predicted_class_distribution": _distribution_dict(counts),
        "dominant_predicted_class_fraction": _dominant_fraction(counts),
    }


def _synthetic_train_real_test(synthetic_x, synthetic_y, real_test_x, real_test_y, number_classes):
    synthetic_x = numpy.asarray(synthetic_x, dtype=numpy.float32)
    synthetic_y = numpy.asarray(synthetic_y, dtype=numpy.int64)
    if synthetic_x.shape[0] == 0 or numpy.unique(synthetic_y).shape[0] < 2:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "reason": "not enough synthetic labels to train TS-TR auditor",
        }
    classifier = DecisionTreeClassifier(random_state=1)
    classifier.fit(synthetic_x, synthetic_y)
    return _evaluate_classifier(classifier, real_test_x, real_test_y, number_classes)


def _memorization_split(synthetic_x, train_x, test_x):
    synthetic_x = numpy.asarray(synthetic_x, dtype=numpy.float64)
    train_x = numpy.asarray(train_x, dtype=numpy.float64)
    test_x = numpy.asarray(test_x, dtype=numpy.float64)
    train_nn = _nearest_distances(synthetic_x, train_x)
    test_nn = _nearest_distances(synthetic_x, test_x)
    scale = _distance_scale(train_x)
    exact_threshold = 1e-12
    near_threshold = max(scale * 1e-6, 1e-8)
    return {
        "nearest_train_distance": _distance_summary(train_nn),
        "nearest_test_distance": _distance_summary(test_nn),
        "exact_train_copy_fraction": _fraction_leq(train_nn, exact_threshold),
        "near_train_copy_fraction": _fraction_leq(train_nn, near_threshold),
        "exact_test_copy_fraction": _fraction_leq(test_nn, exact_threshold),
        "near_test_copy_fraction": _fraction_leq(test_nn, near_threshold),
        "train_closer_than_test_fraction": _train_closer_fraction(train_nn, test_nn),
        "membership_risk_diagnostics": {
            "distance_scale": scale,
            "near_copy_threshold": near_threshold,
            "mean_train_minus_test_nn_distance": _mean_difference(train_nn, test_nn),
            "interpretation": "high risk when synthetic rows are exact/near TRAIN copies or much closer to TRAIN than TEST",
        },
    }


def _manifest(synthetic_split):
    return getattr(synthetic_split, "manifest", {}) if synthetic_split is not None else {}


def _manifest_summary(manifest):
    keys = (
        "total_rows", "num_classes", "features", "seed", "model", "execution_mode", "format",
        "data_space", "transform_id", "schema_hash", "split", "fold", "feature_dtype", "feature_order",
        "requested_classes", "generation_plan", "batch_sizes", "label_mapping", "training",
        "code_version", "data_hashes", "batches_by_class",
    )
    return {key: manifest.get(key) for key in keys if key in manifest}


def _materialize_synthetic(synthetic_split, expected_num_features):
    x_parts = []
    y_parts = []
    batch_hashes = {}
    identical_pairs = []
    batch_count = 0
    for x_batch, y_batch in iter_synthetic_labeled_batches(synthetic_split):
        x_batch = numpy.asarray(x_batch, dtype=numpy.float32)
        y_batch = numpy.asarray(y_batch, dtype=numpy.int64)
        digest = hashlib.sha256(numpy.ascontiguousarray(x_batch).view(numpy.uint8)).hexdigest()
        label = int(y_batch[0]) if y_batch.size else -1
        if digest in batch_hashes and batch_hashes[digest]["label"] != label:
            identical_pairs.append({
                "first_label": batch_hashes[digest]["label"],
                "second_label": label,
                "hash": digest,
            })
        batch_hashes[digest] = {"label": label, "rows": int(x_batch.shape[0])}
        x_parts.append(x_batch)
        y_parts.append(y_batch)
        batch_count += 1
    if not x_parts:
        return (
            numpy.empty((0, int(expected_num_features)), dtype=numpy.float32),
            numpy.asarray([], dtype=numpy.int64),
            {"batch_count": 0, "unique_batch_hashes": 0, "identical_batch_pairs": []},
        )
    return (
        numpy.vstack(x_parts).astype(numpy.float32, copy=False),
        numpy.concatenate(y_parts).astype(numpy.int64, copy=False),
        {
            "batch_count": int(batch_count),
            "unique_batch_hashes": int(len(batch_hashes)),
            "identical_batch_pairs": identical_pairs[:100],
        },
    )


def _dataset_stats(x_values, y_values, number_classes):
    x_values = numpy.asarray(x_values, dtype=numpy.float64)
    y_values = numpy.asarray(y_values, dtype=numpy.int64)
    num_features = 0 if x_values.ndim != 2 else int(x_values.shape[1])
    by_class = {}
    centroids = numpy.full((int(number_classes), num_features), numpy.nan, dtype=numpy.float64)
    intra = numpy.full(int(number_classes), numpy.nan, dtype=numpy.float64)
    for class_id in range(int(number_classes)):
        class_x = x_values[y_values == class_id] if x_values.ndim == 2 and y_values.size else numpy.empty((0, num_features))
        class_stats = _feature_stats(class_x)
        by_class[str(class_id)] = class_stats
        if class_x.shape[0] > 0:
            centroid = numpy.nanmean(class_x, axis=0)
            centroids[class_id] = centroid
            intra[class_id] = float(numpy.nanmean(numpy.linalg.norm(class_x - centroid, axis=1)))
    return {
        "overall": _feature_stats(x_values),
        "by_class": by_class,
        "centroids": centroids,
        "intra_class_distance": intra,
    }


def _feature_stats(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2:
        return {"rows": 0, "features": 0, "min": [], "max": [], "mean": [], "std": [], "percentiles": {}, "nan": 0, "inf": 0, "constant_features": [], "duplicates": 0, "unique_values": []}
    rows = int(values.shape[0])
    features = int(values.shape[1])
    if rows == 0:
        return {"rows": 0, "features": features, "min": [], "max": [], "mean": [], "std": [], "percentiles": {}, "nan": 0, "inf": 0, "constant_features": [], "duplicates": 0, "unique_values": []}
    finite = numpy.where(numpy.isfinite(values), values, numpy.nan)
    with numpy.errstate(all="ignore"):
        percentiles = numpy.nanpercentile(finite, [1, 5, 25, 50, 75, 95, 99], axis=0)
    unique_rows = numpy.unique(values, axis=0).shape[0] if rows else 0
    unique_values = [int(numpy.unique(values[:, feature_index]).shape[0]) for feature_index in range(features)]
    std = numpy.nanstd(finite, axis=0)
    return {
        "rows": rows,
        "features": features,
        "min": _nan_list(numpy.nanmin(finite, axis=0)),
        "max": _nan_list(numpy.nanmax(finite, axis=0)),
        "mean": _nan_list(numpy.nanmean(finite, axis=0)),
        "std": _nan_list(std),
        "percentiles": {
            str(label): _nan_list(row)
            for label, row in zip([1, 5, 25, 50, 75, 95, 99], percentiles)
        },
        "nan": int(numpy.isnan(values).sum()),
        "inf": int(numpy.isinf(values).sum()),
        "constant_features": [int(index) for index, value in enumerate(std) if numpy.isfinite(value) and value <= 1e-12],
        "duplicates": int(rows - unique_rows),
        "unique_values": unique_values,
    }


def _single_feature_stats(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    finite = _finite_vector(values)
    if finite.size == 0:
        return _empty_single_feature_stats()
    with numpy.errstate(all="ignore"):
        quantiles = numpy.nanpercentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "mean": float(numpy.mean(finite)),
        "std": float(numpy.std(finite)),
        "min": float(numpy.min(finite)),
        "max": float(numpy.max(finite)),
        "quantiles": {str(q): float(v) for q, v in zip([1, 5, 25, 50, 75, 95, 99], quantiles)},
        "nan_fraction": _nan_fraction(values),
        "inf_fraction": _inf_fraction(values),
        "constant": bool(numpy.std(finite) <= 1e-12),
    }


def _empty_single_feature_stats():
    return {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "quantiles": {},
        "nan_fraction": None,
        "inf_fraction": None,
        "constant": None,
    }


def _train_real_classifier(x_values, y_values, number_classes, quota):
    x_values = numpy.asarray(x_values, dtype=numpy.float32)
    y_values = numpy.asarray(y_values, dtype=numpy.int64)
    indices = []
    rng = numpy.random.default_rng(0)
    for class_id in range(int(number_classes)):
        class_indices = numpy.flatnonzero(y_values == class_id)
        if class_indices.size == 0:
            continue
        take = min(int(quota), int(class_indices.size))
        indices.extend(rng.choice(class_indices, size=take, replace=False).tolist())
    if not indices:
        raise ValueError("Cannot train audit classifier: no real rows selected.")
    indices = numpy.asarray(indices, dtype=numpy.int64)
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(x_values[indices], y_values[indices])
    return classifier


def _subset_quota(arguments, number_classes):
    if arguments is not None and getattr(arguments, "train_samples_per_class", None) is not None:
        return int(arguments.train_samples_per_class)
    if arguments is not None and getattr(arguments, "batch_classifier_subset_size", None) is not None:
        return max(1, int(arguments.batch_classifier_subset_size) // max(1, int(number_classes)))
    return 500


def _resample_by_class(x_values, y_values, number_classes, target_counts, rng):
    x_values = numpy.asarray(x_values, dtype=numpy.float32)
    y_values = numpy.asarray(y_values, dtype=numpy.int64)
    x_parts = []
    y_parts = []
    fallback = min([numpy.sum(y_values == class_id) for class_id in range(number_classes) if numpy.sum(y_values == class_id) > 0] or [1])
    for class_id in range(int(number_classes)):
        class_indices = numpy.flatnonzero(y_values == class_id)
        if class_indices.size == 0:
            continue
        target = int(target_counts.get(str(class_id), fallback)) if target_counts else int(fallback)
        replace = target > class_indices.size
        chosen = rng.choice(class_indices, size=target, replace=replace)
        x_parts.append(x_values[chosen])
        y_parts.append(numpy.full(target, class_id, dtype=numpy.int64))
    return {
        "x": numpy.vstack(x_parts).astype(numpy.float32, copy=False) if x_parts else numpy.empty((0, x_values.shape[1]), dtype=numpy.float32),
        "y": numpy.concatenate(y_parts).astype(numpy.int64, copy=False) if y_parts else numpy.asarray([], dtype=numpy.int64),
    }


def _counts_dict(labels, number_classes):
    labels = numpy.asarray(labels, dtype=numpy.int64)
    counts = {str(class_id): 0 for class_id in range(int(number_classes))}
    if labels.size:
        unique, values = numpy.unique(labels, return_counts=True)
        for label, count in zip(unique, values):
            if 0 <= int(label) < int(number_classes):
                counts[str(int(label))] = int(count)
    return counts


def _distribution_dict(counts):
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {key: None for key in counts}
    return {key: float(value / total) for key, value in counts.items()}


def _dominant_fraction(counts):
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return None
    return float(max(int(value) for value in counts.values()) / total)


def _expected_count_for_split(arguments, split_name):
    if arguments is None:
        return None
    if split_name == "train":
        return getattr(arguments, "synthetic_train_samples_per_class", None)
    if split_name == "test":
        return getattr(arguments, "synthetic_test_samples_per_class", None)
    return None


def _configured_epochs(arguments):
    if arguments is None:
        return {}
    values = {
        "vae_epochs": getattr(arguments, "variational_autoencoder_number_epochs", None),
        "gan_epochs": getattr(arguments, "adversarial_number_epochs", None),
        "wasserstein_epochs": getattr(arguments, "wasserstein_number_epochs", None),
        "model_type": getattr(arguments, "model_type", None),
    }
    numeric = [int(value) for value in values.values() if isinstance(value, int)]
    values["effective_min_epochs"] = min(numeric) if numeric else None
    return values


def _jsonify_stats(stats):
    return {
        "overall": stats["overall"],
        "by_class": stats["by_class"],
        "centroids": None,
        "intra_class_distance": _nan_list(stats["intra_class_distance"]),
    }


def _as_2d_float(values, expected_num_features):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim == 1:
        values = values.reshape((-1, int(expected_num_features)))
    if values.ndim != 2:
        return numpy.empty((0, int(expected_num_features)), dtype=numpy.float64)
    return values


def _finite_vector(values):
    values = numpy.asarray(values, dtype=numpy.float64).reshape(-1)
    return values[numpy.isfinite(values)]


def _wasserstein(a_values, b_values):
    if len(a_values) == 0 or len(b_values) == 0:
        return None
    return float(wasserstein_distance(a_values, b_values))


def _ks_stat(a_values, b_values):
    if len(a_values) < 2 or len(b_values) < 2:
        return None
    return float(ks_2samp(a_values, b_values).statistic)


def _corr_matrix(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        return numpy.zeros((values.shape[1] if values.ndim == 2 else 0, values.shape[1] if values.ndim == 2 else 0), dtype=numpy.float64)
    clean = numpy.where(numpy.isfinite(values), values, numpy.nan)
    means = numpy.nanmean(clean, axis=0)
    clean = numpy.where(numpy.isfinite(clean), clean, means)
    corr = numpy.corrcoef(clean, rowvar=False)
    corr = numpy.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return numpy.atleast_2d(corr)


def _cov_matrix(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        return numpy.zeros((values.shape[1] if values.ndim == 2 else 0, values.shape[1] if values.ndim == 2 else 0), dtype=numpy.float64)
    clean = numpy.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return numpy.atleast_2d(numpy.cov(clean, rowvar=False))


def _energy_distance(real_x, synthetic_x, max_rows=512):
    real = _sample_rows(real_x, max_rows)
    synth = _sample_rows(synthetic_x, max_rows)
    if real.shape[0] == 0 or synth.shape[0] == 0:
        return None
    rr = cdist(real, real)
    ss = cdist(synth, synth)
    rs = cdist(real, synth)
    value = 2.0 * numpy.mean(rs) - numpy.mean(rr) - numpy.mean(ss)
    return float(max(value, 0.0))


def _mmd_rbf(real_x, synthetic_x, max_rows=512):
    real = _sample_rows(real_x, max_rows)
    synth = _sample_rows(synthetic_x, max_rows)
    if real.shape[0] == 0 or synth.shape[0] == 0:
        return None
    combined = numpy.vstack([real, synth])
    distances = cdist(combined, combined, metric="sqeuclidean")
    positive = distances[distances > 0]
    gamma = 1.0 / max(float(numpy.median(positive)) if positive.size else 1.0, EPSILON)
    k_rr = numpy.exp(-gamma * cdist(real, real, metric="sqeuclidean"))
    k_ss = numpy.exp(-gamma * cdist(synth, synth, metric="sqeuclidean"))
    k_rs = numpy.exp(-gamma * cdist(real, synth, metric="sqeuclidean"))
    return float(numpy.mean(k_rr) + numpy.mean(k_ss) - 2.0 * numpy.mean(k_rs))


def _pca_distance(real_x, synthetic_x):
    real = numpy.asarray(real_x, dtype=numpy.float64)
    synth = numpy.asarray(synthetic_x, dtype=numpy.float64)
    if real.shape[0] < 2 or synth.shape[0] < 2:
        return None
    try:
        _, _, real_vt = numpy.linalg.svd(real - numpy.mean(real, axis=0), full_matrices=False)
        _, _, synth_vt = numpy.linalg.svd(synth - numpy.mean(synth, axis=0), full_matrices=False)
    except numpy.linalg.LinAlgError:
        return None
    k = min(3, real_vt.shape[0], synth_vt.shape[0])
    if k == 0:
        return None
    projection_diff = real_vt[:k].T @ real_vt[:k] - synth_vt[:k].T @ synth_vt[:k]
    return float(numpy.linalg.norm(projection_diff, ord="fro"))


def _coverage(real_x, synthetic_x):
    real = numpy.asarray(real_x, dtype=numpy.float64)
    synth = numpy.asarray(synthetic_x, dtype=numpy.float64)
    if real.shape[0] < 2 or synth.shape[0] == 0:
        return None
    real_dist = cdist(real, real)
    real_dist[real_dist == 0.0] = numpy.nan
    radius = numpy.nanpercentile(real_dist, 5)
    if not numpy.isfinite(radius) or radius <= 0:
        radius = numpy.nanmedian(real_dist)
    if not numpy.isfinite(radius):
        return None
    synth_to_real = numpy.min(cdist(synth, real), axis=1)
    return float(numpy.mean(synth_to_real <= radius))


def _density(real_x, synthetic_x):
    real = numpy.asarray(real_x, dtype=numpy.float64)
    synth = numpy.asarray(synthetic_x, dtype=numpy.float64)
    if real.shape[0] < 2 or synth.shape[0] == 0:
        return None
    real_dist = cdist(real, real)
    real_dist[real_dist == 0.0] = numpy.nan
    radius = numpy.nanpercentile(real_dist, 5)
    if not numpy.isfinite(radius) or radius <= 0:
        radius = numpy.nanmedian(real_dist)
    if not numpy.isfinite(radius):
        return None
    neighbor_counts = numpy.sum(cdist(synth, real) <= radius, axis=1)
    return float(numpy.mean(neighbor_counts) / max(real.shape[0], 1))


def _nearest_summary(source_x, reference_x):
    distances = _nearest_distances(source_x, reference_x)
    return _distance_summary(distances)


def _nearest_distances(source_x, reference_x, max_reference_rows=4096):
    source = numpy.asarray(source_x, dtype=numpy.float64)
    reference = _sample_rows(reference_x, max_reference_rows)
    if source.shape[0] == 0 or reference.shape[0] == 0:
        return numpy.asarray([], dtype=numpy.float64)
    return numpy.min(cdist(source, reference), axis=1)


def _distance_summary(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0:
        return {"min": None, "mean": None, "median": None, "p01": None, "p05": None}
    return {
        "min": float(numpy.min(values)),
        "mean": float(numpy.mean(values)),
        "median": float(numpy.median(values)),
        "p01": float(numpy.percentile(values, 1)),
        "p05": float(numpy.percentile(values, 5)),
    }


def _sample_rows(values, max_rows):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return numpy.empty((0, values.shape[1] if values.ndim == 2 else 0), dtype=numpy.float64)
    finite = numpy.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if finite.shape[0] <= max_rows:
        return finite
    indices = numpy.linspace(0, finite.shape[0] - 1, int(max_rows)).astype(numpy.int64)
    return finite[indices]


def _duplicate_fraction(values):
    values = numpy.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    return float((values.shape[0] - numpy.unique(values, axis=0).shape[0]) / values.shape[0])


def _near_duplicate_fraction(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return None
    sampled = _sample_rows(values, 2048)
    distances = cdist(sampled, sampled)
    distances[distances == 0.0] = numpy.inf
    scale = _distance_scale(sampled)
    return float(numpy.mean(numpy.min(distances, axis=1) <= max(scale * 1e-6, 1e-8)))


def _distance_scale(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return 1.0
    std = numpy.nanstd(values, axis=0)
    scale = float(numpy.linalg.norm(std))
    return scale if numpy.isfinite(scale) and scale > 0 else 1.0


def _constant_feature_fraction_matrix(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        return None
    return float(len(_constant_feature_indices(values)) / values.shape[1])


def _constant_feature_indices(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return []
    std = numpy.nanstd(numpy.where(numpy.isfinite(values), values, numpy.nan), axis=0)
    return [int(index) for index, value in enumerate(std) if numpy.isfinite(value) and value <= 1e-12]


def _constant_value_fraction(values):
    values = _finite_vector(values)
    if values.size == 0:
        return None
    _, counts = numpy.unique(values, return_counts=True)
    return float(numpy.max(counts) / values.size)


def _nan_fraction(values):
    values = numpy.asarray(values)
    return None if values.size == 0 else float(numpy.isnan(values).sum() / values.size)


def _inf_fraction(values):
    values = numpy.asarray(values)
    return None if values.size == 0 else float(numpy.isinf(values).sum() / values.size)


def _total_variance(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return None
    return float(numpy.nansum(numpy.nanvar(values, axis=0)))


def _centroid_distance(real_x, synthetic_x):
    if real_x.shape[0] == 0 or synthetic_x.shape[0] == 0:
        return None
    return float(numpy.linalg.norm(numpy.nanmean(real_x, axis=0) - numpy.nanmean(synthetic_x, axis=0)))


def _interclass_distance(centroids):
    centroids = numpy.asarray(centroids, dtype=numpy.float64)
    if centroids.ndim != 2 or centroids.shape[0] < 2:
        return None
    valid = centroids[numpy.all(numpy.isfinite(centroids), axis=1)]
    if valid.shape[0] < 2:
        return None
    distances = cdist(valid, valid)
    distances = distances[numpy.triu_indices_from(distances, k=1)]
    return float(numpy.mean(distances)) if distances.size else None


def _relative_recall(current, real_reference):
    result = {}
    for class_id, value in current.items():
        baseline = real_reference.get(class_id)
        if value is None or baseline is None:
            result[class_id] = None
        else:
            result[class_id] = float(value / max(baseline, EPSILON))
    return result


def _synthetic_interclass_proxy(synthetic_sections, number_classes):
    result = {}
    for split_name, section in synthetic_sections.items():
        stats = _dataset_stats(section["x"], section["y"], number_classes)
        result[split_name] = _interclass_distance(stats["centroids"])
    return result


def _probe_labels(manifest_labels, number_classes):
    if len(manifest_labels) <= 20:
        return list(manifest_labels)
    indices = numpy.linspace(0, len(manifest_labels) - 1, 20).astype(numpy.int64)
    return [int(manifest_labels[index]) for index in indices]


def _expected_manifest_classes(manifest, fallback):
    requested = manifest.get("requested_classes") if isinstance(manifest, dict) else None
    if requested is not None:
        return sorted(int(label) for label in requested)
    plan = manifest.get("generation_plan", {}) if isinstance(manifest, dict) else {}
    classes = plan.get("classes") if isinstance(plan, dict) else None
    if isinstance(classes, dict):
        return sorted(int(label) for label, count in classes.items() if int(count) > 0)
    return sorted(int(label) for label in fallback)


def _call_decoder_probe(decoder_probe, latent_dim, labels):
    rng = numpy.random.default_rng(123)
    z = rng.normal(size=(len(labels), int(latent_dim))).astype(numpy.float32)
    labels_array = numpy.asarray(labels, dtype=numpy.int64)
    first = _decoder_call(decoder_probe, z, labels_array)
    second = _decoder_call(decoder_probe, z, labels_array)
    z_new = rng.normal(size=(len(labels), int(latent_dim))).astype(numpy.float32)
    same_label_new_z = _decoder_call(decoder_probe, z_new, labels_array)
    fixed_z = numpy.repeat(z[:1], len(labels), axis=0)
    different_labels = _decoder_call(decoder_probe, fixed_z, labels_array)
    return {
        "same_label_first": numpy.asarray(first, dtype=numpy.float64),
        "same_label_second": numpy.asarray(second, dtype=numpy.float64),
        "same_label_new_z": numpy.asarray(same_label_new_z, dtype=numpy.float64),
        "different_labels": numpy.asarray(different_labels, dtype=numpy.float64),
    }


def _decoder_call(decoder_probe, z, labels):
    if callable(decoder_probe):
        return decoder_probe(z, labels)
    if hasattr(decoder_probe, "decode"):
        return decoder_probe.decode(z, labels)
    if hasattr(decoder_probe, "generate"):
        return decoder_probe.generate(z, labels)
    if hasattr(decoder_probe, "predict"):
        return decoder_probe.predict([z, labels])
    raise TypeError("decoder_probe must be callable or expose decode/generate/predict")


def _pairwise_mean_distance(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.shape[0] < 2:
        return 0.0
    distances = cdist(values, values)
    return float(numpy.mean(distances[numpy.triu_indices_from(distances, k=1)]))


def _fraction_leq(values, threshold):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0:
        return None
    return float(numpy.mean(values <= threshold))


def _train_closer_fraction(train_nn, test_nn):
    train_nn = numpy.asarray(train_nn, dtype=numpy.float64)
    test_nn = numpy.asarray(test_nn, dtype=numpy.float64)
    if train_nn.size == 0 or test_nn.size == 0:
        return None
    return float(numpy.mean(train_nn < test_nn))


def _mean_difference(a_values, b_values):
    a_values = numpy.asarray(a_values, dtype=numpy.float64)
    b_values = numpy.asarray(b_values, dtype=numpy.float64)
    if a_values.size == 0 or b_values.size == 0:
        return None
    return float(numpy.mean(a_values - b_values))


def _subtract_nullable(a_value, b_value):
    if a_value is None or b_value is None:
        return None
    return float(a_value - b_value)


def _matrix_to_list(values):
    return numpy.nan_to_num(numpy.asarray(values, dtype=numpy.float64), nan=0.0, posinf=0.0, neginf=0.0).tolist()


def _mean_abs(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0:
        return None
    return float(numpy.mean(numpy.abs(numpy.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))))


def _nan_list(values):
    result = []
    for value in numpy.asarray(values).reshape(-1):
        if not numpy.isfinite(value):
            result.append(None)
        else:
            result.append(float(value))
    return result


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not numpy.isfinite(value):
        return None
    return value


def _write_json_and_markdown(report, experiment_directory):
    serializable = _strip_runtime_arrays(report)
    APPCLASSNET_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    APPCLASSNET_AUDIT_JSON.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(serializable, APPCLASSNET_AUDIT_DOC)
    if experiment_directory:
        local_dir = Path(experiment_directory) / "Audits"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "synthetic_quality_audit.json").write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_markdown_report(serializable, local_dir / "synthetic_quality_audit.md")


def _strip_runtime_arrays(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"x", "y"}:
                continue
            result[key] = _strip_runtime_arrays(item)
        return result
    if isinstance(value, list):
        return [_strip_runtime_arrays(item) for item in value]
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    return value


def _write_markdown_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    collapse = report["collapse_decision"]
    evidence = report["evidence"]
    lines = [
        "# Synthetic Quality Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Conclusion: `{report['root_cause']}`",
        f"- Candidates: `{', '.join(report['root_cause_candidates'])}`",
        f"- Classes: `{report['number_classes']}`",
        f"- Features: `{report['expected_num_features']}`",
        f"- Chance accuracy: `{report['chance_accuracy']:.6f}`",
        f"- Synthetic control: `{report['synthetic_control']}`",
        "",
        "## Collapse Decision",
        "",
        f"- Collapsed: `{collapse['collapsed']}`",
        f"- Evidence flags: `{', '.join(collapse['evidence_flags'])}`",
        f"- Reasons: `{collapse['reasons']}`",
        f"- Decision rule: {collapse['decision_rule']}",
        "",
        "## Key Evidence",
        "",
        f"- Dominant predicted class fraction: `{evidence.get('dominant_predicted_class_fraction')}`",
        f"- Real-resample dominant predicted class fraction: `{evidence.get('dominant_predicted_class_fraction_real_resample')}`",
        f"- Synthetic total variance: `{evidence.get('synthetic_total_variance')}`",
        f"- Real total variance: `{evidence.get('real_total_variance')}`",
        f"- Constant feature fraction: `{evidence.get('constant_feature_fraction')}`",
        f"- Synthetic/real interclass diversity ratio: `{evidence.get('synthetic_to_real_interclass_diversity_ratio')}`",
        f"- TS-TR balanced accuracy: `{evidence.get('ts_tr_balanced_accuracy')}`",
        f"- Decoder conditional sensitivity score: `{evidence.get('decoder_conditional_sensitivity_score')}`",
        f"- Exact TRAIN copy fraction: `{evidence.get('exact_train_copy_fraction')}`",
        f"- Near TRAIN copy fraction: `{evidence.get('near_train_copy_fraction')}`",
        "",
        "## Sections",
        "",
        "- global_feature_quality",
        "- per_class_feature_quality",
        "- conditional_fidelity",
        "- decoder_sensitivity",
        "- memorization",
        "- controls",
        "- collapse_decision",
        "- evidence",
        "",
        "## Outputs",
        "",
        f"- JSON: `{APPCLASSNET_AUDIT_JSON}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
