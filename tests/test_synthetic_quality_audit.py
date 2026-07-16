import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

import Engine.DataIO.SyntheticQualityAudit as audit_module
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.SyntheticBatchIO import SyntheticBatchWriter
from Engine.DataIO.SyntheticBatchIO import SyntheticSplitBatchReaders
from Engine.DataIO.SyntheticQualityAudit import SyntheticClassCollapseError
from Engine.DataIO.SyntheticQualityAudit import assert_label_permutation_control_metrics
from Engine.DataIO.SyntheticQualityAudit import assert_real_resample_control_metrics
from Engine.DataIO.SyntheticQualityAudit import run_synthetic_quality_audit


class SyntheticQualityAuditTest(unittest.TestCase):

    def _schema(self, num_classes=2, num_features=2):
        return DatasetSchema(
            feature_names=[f"f{index}" for index in range(num_features)],
            feature_type="continuous",
            target_type="multiclass",
            num_classes=num_classes,
            source_format="npy_xy",
            source_profile="appclassnet_top200",
            feature_dtype="float32",
            data_space="source",
        )

    def _reader(self, directory, split, class0, class1, schema):
        writer = SyntheticBatchWriter(
            root_dir=directory,
            num_classes=2,
            num_features=2,
            seed=0,
            model_name="control",
            execution_mode="batches",
            output_format="npy_batches",
            data_space="source",
            feature_names=schema.feature_names,
            feature_dtype=schema.feature_dtype,
            schema_hash=schema.schema_hash,
            split_name=split,
        )
        writer.write_batch(0, 0, numpy.asarray(class0, dtype=numpy.float32))
        writer.write_batch(1, 0, numpy.asarray(class1, dtype=numpy.float32))
        return writer.close()

    def _reader_from_arrays(self, directory, split, x_values, y_values, schema):
        writer = SyntheticBatchWriter(
            root_dir=directory,
            num_classes=schema.num_classes,
            num_features=len(schema.feature_names),
            seed=0,
            model_name="control",
            execution_mode="batches",
            output_format="npy_batches",
            data_space="source",
            feature_names=schema.feature_names,
            feature_dtype=schema.feature_dtype,
            schema_hash=schema.schema_hash,
            split_name=split,
        )
        x_values = numpy.asarray(x_values, dtype=numpy.float32)
        y_values = numpy.asarray(y_values, dtype=numpy.int64)
        for class_id in range(schema.num_classes):
            writer.write_batch(class_id, 0, x_values[y_values == class_id])
        return writer.close()

    def _args(self, samples_per_class, control="none"):
        return SimpleNamespace(
            synthetic_control=control,
            synthetic_train_samples_per_class=samples_per_class,
            synthetic_test_samples_per_class=samples_per_class,
            train_samples_per_class=samples_per_class,
            batch_classifier_subset_size=max(10, samples_per_class * 200),
            generator_transform="preserve",
            inverse_transform_synthetic=True,
            generation_strategy="single_conditional",
            classes_per_group=10,
            variational_autoencoder_number_epochs=2,
            adversarial_number_epochs=2,
            wasserstein_number_epochs=2,
            model_type="control",
        )

    def _separable_data(self, num_classes, samples_per_class, num_features=20, offset=0.0, noise=0.02):
        rng = numpy.random.default_rng(42 + int(offset * 1000) + num_classes + samples_per_class)
        x_parts = []
        y_parts = []
        for class_id in range(num_classes):
            center = numpy.zeros(num_features, dtype=numpy.float32)
            center[0] = float(class_id) * 4.0
            center[1] = float(class_id % 7)
            center[2] = float(class_id // 7)
            samples = center + offset + rng.normal(scale=noise, size=(samples_per_class, num_features)).astype(numpy.float32)
            x_parts.append(samples)
            y_parts.append(numpy.full(samples_per_class, class_id, dtype=numpy.int64))
        return numpy.vstack(x_parts), numpy.concatenate(y_parts)

    def _overlapping_data(self, num_classes, samples_per_class, num_features=20, offset=0.0):
        rng = numpy.random.default_rng(123 + int(offset * 1000))
        total = num_classes * samples_per_class
        x_values = rng.normal(loc=offset, scale=1.0, size=(total, num_features)).astype(numpy.float32)
        y_values = numpy.repeat(numpy.arange(num_classes, dtype=numpy.int64), samples_per_class)
        return x_values, y_values

    def _run_array_audit(
            self,
            directory,
            real_train_x,
            real_train_y,
            real_test_x,
            real_test_y,
            synthetic_train_x,
            synthetic_train_y,
            synthetic_test_x=None,
            synthetic_test_y=None,
            *,
            num_classes,
            num_features=20,
            fail_on_collapse=False,
            decoder_probe=None):
        schema = self._schema(num_classes=num_classes, num_features=num_features)
        synthetic_test_x = synthetic_train_x if synthetic_test_x is None else synthetic_test_x
        synthetic_test_y = synthetic_train_y if synthetic_test_y is None else synthetic_test_y
        train = self._reader_from_arrays(directory, "train", synthetic_train_x, synthetic_train_y, schema)
        test = self._reader_from_arrays(directory, "test", synthetic_test_x, synthetic_test_y, schema)
        with self._patched_audit_outputs(directory):
            return run_synthetic_quality_audit(
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                SyntheticSplitBatchReaders(train, test),
                number_classes=num_classes,
                expected_num_features=num_features,
                schema=schema,
                arguments=self._args(samples_per_class=int(numpy.sum(synthetic_train_y == 0))),
                experiment_directory=Path(directory),
                fail_on_collapse=fail_on_collapse,
                decoder_probe=decoder_probe,
            )

    def test_audit_writes_report_for_real_like_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            schema = self._schema()
            real_train_x = numpy.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=numpy.float32)
            real_train_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            real_test_x = numpy.array([[0, 0.2], [0, 0.8], [10, 10.2], [10, 10.8]], dtype=numpy.float32)
            real_test_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            train = self._reader(directory, "train", real_train_x[:2], real_train_x[2:], schema)
            test = self._reader(directory, "test", real_test_x[:2], real_test_x[2:], schema)

            with self._patched_audit_outputs(directory):
                _, report = run_synthetic_quality_audit(
                    real_train_x,
                    real_train_y,
                    real_test_x,
                    real_test_y,
                    SyntheticSplitBatchReaders(train, test),
                    number_classes=2,
                    expected_num_features=2,
                    schema=schema,
                    arguments=SimpleNamespace(
                        synthetic_control="real_resample",
                        synthetic_train_samples_per_class=2,
                        synthetic_test_samples_per_class=2,
                        train_samples_per_class=2,
                        batch_classifier_subset_size=10,
                        generator_transform="preserve",
                        inverse_transform_synthetic=True,
                        generation_strategy="single_conditional",
                        classes_per_group=10,
                        variational_autoencoder_number_epochs=2,
                        adversarial_number_epochs=2,
                        wasserstein_number_epochs=2,
                        model_type="control",
                    ),
                    experiment_directory=Path(directory),
                    fail_on_collapse=False,
                )

            self.assertEqual(report["schema"]["real_schema_hash"], schema.schema_hash)
            self.assertTrue(report["schema"]["hashes_match"])
            self.assertEqual(report["synthetic"]["train"]["manifest_errors"], [])
            self.assertEqual(report["synthetic"]["test"]["label_audit_errors"], [])
            self.assertEqual(report["transform"]["preserve_inverse_is_noop"], True)

    def test_identical_classes_raise_collapse_error(self):
        with tempfile.TemporaryDirectory() as directory:
            schema = self._schema()
            real_x = numpy.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=numpy.float32)
            real_y = numpy.array([0, 0, 1, 1], dtype=numpy.int64)
            collapsed = numpy.array([[0, 0], [0, 1]], dtype=numpy.float32)
            train = self._reader(directory, "train", collapsed, collapsed, schema)
            test = self._reader(directory, "test", collapsed, collapsed, schema)

            with self._patched_audit_outputs(directory), self.assertRaises(SyntheticClassCollapseError):
                run_synthetic_quality_audit(
                    real_x,
                    real_y,
                    real_x,
                    real_y,
                    SyntheticSplitBatchReaders(train, test),
                    number_classes=2,
                    expected_num_features=2,
                    schema=schema,
                    arguments=SimpleNamespace(
                        synthetic_control="none",
                        synthetic_train_samples_per_class=2,
                        synthetic_test_samples_per_class=2,
                        train_samples_per_class=2,
                        batch_classifier_subset_size=10,
                        generator_transform="preserve",
                        inverse_transform_synthetic=True,
                        variational_autoencoder_number_epochs=2,
                        adversarial_number_epochs=2,
                        wasserstein_number_epochs=2,
                    ),
                    experiment_directory=Path(directory),
                )

    def test_real_resample_control_near_chance_fails(self):
        metrics = {
            "TR-TS": {"DecisionTreeSubset": {"1-Fold": {"Accuracy": 0.01, "BalancedAccuracy": 0.01}}},
            "TS-TR": {"DecisionTreeSubset": {"1-Fold": {"Accuracy": 0.01, "BalancedAccuracy": 0.01}}},
        }

        with self.assertRaisesRegex(RuntimeError, "real_resample synthetic control failed"):
            assert_real_resample_control_metrics(
                metrics,
                number_classes=200,
                synthetic_control="real_resample",
                fold=1,
            )

    def test_label_permutation_control_above_chance_fails(self):
        metrics = {
            "TR-TS": {"DecisionTreeSubset": {"1-Fold": {"Accuracy": 0.50, "BalancedAccuracy": 0.50}}},
            "TS-TR": {"DecisionTreeSubset": {"1-Fold": {"Accuracy": 0.50, "BalancedAccuracy": 0.50}}},
        }

        with self.assertRaisesRegex(RuntimeError, "label_permutation synthetic control failed"):
            assert_label_permutation_control_metrics(
                metrics,
                number_classes=200,
                synthetic_control="label_permutation",
                fold=1,
            )

    def _patched_audit_outputs(self, directory):
        return mock.patch.multiple(
            audit_module,
            APPCLASSNET_AUDIT_JSON=Path(directory) / "synthetic_quality_audit.json",
            APPCLASSNET_AUDIT_DOC=Path(directory) / "SYNTHETIC_QUALITY_ROOT_CAUSE.md",
        )

    def test_perfect_twenty_feature_synthetic_has_layered_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(4, 8, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 8, 20, offset=0.1)
            synthetic_x, synthetic_y = self._separable_data(4, 8, 20, offset=0.02)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                synthetic_y,
                num_classes=4,
                num_features=20,
            )

            self.assertFalse(report["collapse_decision"]["collapsed"])
            self.assertEqual(len(report["global_feature_quality"]["synthetic_train"]["per_feature"]), 20)
            self.assertIn("conditional_fidelity", report)
            self.assertIn("memorization", report)
            self.assertTrue((Path(directory) / "Audits" / "synthetic_quality_audit.md").exists())

    def test_label_permuted_synthetic_is_negative_control_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(10, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(10, 6, 20, offset=0.1)
            synthetic_y = numpy.roll(real_train_y, 7)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                real_train_x,
                synthetic_y,
                num_classes=10,
                num_features=20,
            )

            synthetic_balanced = report["conditional_fidelity"]["evaluations"]["synthetic_train"]["balanced_accuracy"]
            real_resample_balanced = report["controls"]["positive_control"]["balanced_accuracy"]
            self.assertLess(synthetic_balanced, real_resample_balanced)
            self.assertIn("label_permutation", report["conditional_fidelity"]["evaluations"])

    def test_constant_features_are_reported_as_collapse_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(4, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 6, 20, offset=0.1)
            synthetic_x = numpy.zeros_like(real_train_x)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                real_train_y,
                num_classes=4,
                num_features=20,
            )

            self.assertTrue(report["collapse_decision"]["collapsed"])
            self.assertIn("many_constant_features", report["collapse_decision"]["evidence_flags"])

    def test_decoder_ignoring_label_raises_collapse(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(4, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 6, 20, offset=0.1)
            synthetic_x, synthetic_y = self._separable_data(4, 6, 20, offset=0.02)

            def decoder_ignores_label(z_values, labels):
                del labels
                return numpy.repeat(z_values[:, :1], 20, axis=1)

            with self.assertRaises(SyntheticClassCollapseError):
                self._run_array_audit(
                    directory,
                    real_train_x,
                    real_train_y,
                    real_test_x,
                    real_test_y,
                    synthetic_x,
                    synthetic_y,
                    num_classes=4,
                    num_features=20,
                    fail_on_collapse=True,
                    decoder_probe=decoder_ignores_label,
                )

    def test_decoder_sensitive_to_label_passes_fixed_z_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(4, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 6, 20, offset=0.1)
            synthetic_x, synthetic_y = self._separable_data(4, 6, 20, offset=0.02)

            def decoder_uses_label(z_values, labels):
                values = numpy.zeros((len(labels), 20), dtype=numpy.float32)
                values[:, 0] = numpy.asarray(labels, dtype=numpy.float32)
                values[:, 1:1 + z_values.shape[1]] = z_values
                return values

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                synthetic_y,
                num_classes=4,
                num_features=20,
                decoder_probe=decoder_uses_label,
            )

            sensitivity = report["decoder_sensitivity"]
            self.assertTrue(sensitivity["same_z_same_label_reproducible"])
            self.assertTrue(sensitivity["same_z_different_labels_changes_output"])
            self.assertGreater(sensitivity["conditional_sensitivity_score"], 0.05)

    def test_manifest_requested_classes_allow_sparse_effective_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            schema = self._schema(num_classes=4, num_features=20)
            real_train_x, real_train_y = self._separable_data(4, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 6, 20, offset=0.1)
            writer = SyntheticBatchWriter(
                root_dir=directory,
                num_classes=4,
                num_features=20,
                seed=0,
                model_name="variational",
                execution_mode="batches",
                split_name="train",
                requested_classes=[0, 2],
                generation_plan={"classes": {"0": 3, "2": 3}, "number_classes": 4},
                feature_names=schema.feature_names,
                feature_dtype=schema.feature_dtype,
                schema_hash=schema.schema_hash,
            )
            writer.write_batch(0, 0, real_train_x[real_train_y == 0][:3])
            writer.write_batch(2, 0, real_train_x[real_train_y == 2][:3])
            reader = writer.close()

            with self._patched_audit_outputs(directory):
                _, report = run_synthetic_quality_audit(
                    real_train_x,
                    real_train_y,
                    real_test_x,
                    real_test_y,
                    reader,
                    number_classes=4,
                    expected_num_features=20,
                    schema=schema,
                    arguments=self._args(samples_per_class=3),
                    experiment_directory=Path(directory),
                    fail_on_collapse=False,
                )

            self.assertEqual(report["synthetic"]["synthetic_train"]["manifest_errors"], [])
            self.assertEqual(report["synthetic"]["synthetic_train"]["label_audit_errors"], [])

    def test_memorized_data_is_detected_without_being_labeled_class_collapse_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(4, 6, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(4, 6, 20, offset=0.2)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                real_train_x,
                real_train_y,
                num_classes=4,
                num_features=20,
            )

            self.assertFalse(report["collapse_decision"]["collapsed"])
            self.assertGreater(report["memorization"]["synthetic_train"]["exact_train_copy_fraction"], 0.99)

    def test_overlapping_real_classes_do_not_cause_false_collapse_by_themselves(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._overlapping_data(3, 12, 20, offset=0.0)
            real_test_x, real_test_y = self._overlapping_data(3, 12, 20, offset=0.1)
            synthetic_x, synthetic_y = self._overlapping_data(3, 12, 20, offset=0.02)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                synthetic_y,
                num_classes=3,
                num_features=20,
            )

            self.assertFalse(report["collapse_decision"]["collapsed"])
            self.assertIn("similar real classes", report["per_class_feature_quality"]["synthetic_train"]["summary"]["note"])

    def test_subset_of_ten_classes_is_audited_per_class(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(10, 4, 20, offset=0.0)
            real_test_x, real_test_y = self._separable_data(10, 4, 20, offset=0.1)
            synthetic_x, synthetic_y = self._separable_data(10, 4, 20, offset=0.02)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                synthetic_y,
                num_classes=10,
                num_features=20,
            )

            self.assertEqual(report["number_classes"], 10)
            self.assertEqual(len(report["per_class_feature_quality"]["synthetic_train"]["by_class"]), 10)

    def test_two_hundred_classes_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            real_train_x, real_train_y = self._separable_data(200, 2, 20, offset=0.0, noise=0.001)
            real_test_x, real_test_y = self._separable_data(200, 2, 20, offset=0.1, noise=0.001)
            synthetic_x, synthetic_y = self._separable_data(200, 2, 20, offset=0.02, noise=0.001)

            _, report = self._run_array_audit(
                directory,
                real_train_x,
                real_train_y,
                real_test_x,
                real_test_y,
                synthetic_x,
                synthetic_y,
                num_classes=200,
                num_features=20,
            )

            self.assertEqual(report["number_classes"], 200)
            self.assertEqual(len(report["per_class_feature_quality"]["synthetic_train"]["by_class"]), 200)
            self.assertFalse(report["collapse_decision"]["collapsed"])


if __name__ == "__main__":
    unittest.main()
