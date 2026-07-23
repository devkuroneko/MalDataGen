import json
import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.Arguments.ArgumentsDataLoader import DEFAULT_FEATURE_TRANSFORM
from Engine.Arguments.ArgumentsDataLoader import DEFAULT_SCALER
from Engine.Arguments.ArgumentsDataLoader import DEFAULT_SOURCE_PROFILE
from Engine.Preprocessing.FeatureTransformManager import DoubleTransformError
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy
from Engine.Preprocessing.FeatureTransformManager import ModelInputAdapter
from Engine.Preprocessing.FeatureTransformManager import PreprocessingSpaceMismatchError
from Engine.Preprocessing.FeatureTransformManager import ScaleGuard
from Engine.Preprocessing.FeatureTransformManager import TransformManifest


class FeatureTransformManagerTest(unittest.TestCase):

    def _app_policy(self, **overrides):
        return FeatureTransformPolicy.for_profile("appclassnet_top200", **overrides)

    def test_appclassnet_preserve_keeps_values_and_records_no_transform(self):
        values = numpy.array([[-0.5, 0.0, 0.5], [-0.25, 0.25, 0.4]], dtype=numpy.float64)
        manager = FeatureTransformManager(self._app_policy(), stage="feature")

        manager.fit(values, split_name="train")
        transformed, metadata = manager.transform(values, return_metadata=True)

        self.assertIs(transformed, values)
        numpy.testing.assert_array_equal(transformed, values)
        self.assertEqual(transformed.dtype, values.dtype)
        self.assertFalse(manager.transform_applied)
        self.assertIsNone(manager.scaler)
        self.assertIsNone(manager.transform_id)
        self.assertFalse(metadata["transform_applied"])
        self.assertEqual(metadata["transform_history"], [])
        self.assertEqual(metadata["data_space"], "source")

    def test_appclassnet_preserve_inverse_transform_keeps_values_and_dtype(self):
        values = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float64)
        manager = FeatureTransformManager(self._app_policy(), stage="feature")

        manager.fit(values, split_name="train")
        inverse = manager.inverse_transform(values)

        self.assertIs(inverse, values)
        numpy.testing.assert_array_equal(inverse, values)
        self.assertEqual(inverse.dtype, values.dtype)

    def test_appclassnet_generator_minmax_inverse_returns_source_scale(self):
        source = numpy.array([[-0.5, 0.0], [0.5, 0.25], [0.0, 0.5]], dtype=numpy.float32)
        policy = self._app_policy(generator_transform="minmax", inverse_transform_synthetic=True)
        adapter = ModelInputAdapter(policy)

        adapter.fit_generator(source)
        generator_values = adapter.transform_generator_input(source)
        synthetic_source = adapter.inverse_generator_output(generator_values)

        self.assertGreaterEqual(float(generator_values.min()), 0.0)
        self.assertLessEqual(float(generator_values.max()), 1.0)
        numpy.testing.assert_allclose(synthetic_source, source, atol=1e-6)
        self.assertEqual(adapter.synthetic_space_after_generation(), "source")

    def test_appclassnet_preserve_inverse_synthetic_batch_is_noop(self):
        source = numpy.array([[-0.5, 0.0], [0.5, 0.25], [0.0, 0.5]], dtype=numpy.float32)
        policy = self._app_policy(generator_transform="preserve", inverse_transform_synthetic=True)
        adapter = ModelInputAdapter(policy)

        adapter.fit_generator(source)
        before = adapter.transform_generator_input(source)
        after = adapter.inverse_synthetic_batch(before)

        numpy.testing.assert_array_equal(before, source)
        numpy.testing.assert_array_equal(after, before)
        self.assertEqual(adapter.synthetic_space_after_generation(), "source")

    def test_valid_and_test_use_train_fitted_scaler_without_fit(self):
        train = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        valid = numpy.array([[0.0, 0.1]], dtype=numpy.float32)
        test = numpy.array([[0.25, 0.2]], dtype=numpy.float32)
        manager = FeatureTransformManager(self._app_policy(feature_transform="minmax"), stage="feature")

        manager.fit(train, split_name="train")
        original_fit = manager.scaler.fit

        def fail_fit(*args, **kwargs):
            raise AssertionError("fit must not be called for valid/test")

        manager.scaler.fit = fail_fit
        manager.transform(valid, split_name="valid")
        manager.transform(test, split_name="test")
        manager.scaler.fit = original_fit

    def test_duplicate_transform_is_blocked_by_history(self):
        values = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        manager = FeatureTransformManager(self._app_policy(feature_transform="minmax"), stage="feature")
        manager.fit(values, split_name="train")
        _, metadata = manager.transform(values, split_name="train", return_metadata=True)

        with self.assertRaises(DoubleTransformError):
            manager.transform(
                values,
                split_name="train",
                transform_history=metadata["transform_history"],
            )

    def test_preserve_rejects_loaded_scaler_parameters(self):
        values = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        manager = FeatureTransformManager(self._app_policy(), stage="feature")
        manager.scaler = object()

        with self.assertRaisesRegex(ValueError, "preserve.*scaler"):
            manager.transform(values)

    def test_real_synthetic_space_mismatch_blocks_evaluation(self):
        real = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        synthetic = numpy.array([[0.0, 1.0], [0.5, 0.75]], dtype=numpy.float32)
        real_metadata = ScaleGuard.describe(real, data_space="source")
        synthetic_metadata = ScaleGuard.describe(synthetic, data_space="generator", transform_id="abc")

        with self.assertRaises(PreprocessingSpaceMismatchError):
            ScaleGuard.validate_before_evaluation(
                real,
                synthetic,
                real_metadata,
                synthetic_metadata,
                context="TS-TR",
            )

    def test_evaluation_space_source_rejects_transformed_history(self):
        metadata = {
            "data_space": "classifier",
            "transform_id": "abc",
            "transform_history": [
                {"operation": "minmax", "output_space": "classifier", "transform_id": "abc"},
            ],
        }

        with self.assertRaises(PreprocessingSpaceMismatchError):
            ScaleGuard.validate_evaluation_space_contract(
                metadata,
                "source",
                context="TR-TR",
            )

    def test_tree_classifier_policy_preserves_appclassnet_scale(self):
        values = numpy.array([[-0.5, 0.5], [0.0, 0.25]], dtype=numpy.float32)
        policy = self._app_policy(classifier_transform="preserve")
        manager = FeatureTransformManager(policy, stage="classifier", output_space="classifier")

        manager.fit(values, split_name="train")
        transformed = manager.transform(values, split_name="train")

        numpy.testing.assert_array_equal(transformed, values)
        self.assertIsNone(manager.transform_id)

    def test_sgd_explicit_standard_transform_is_persisted_and_reused(self):
        train = numpy.array([[-0.5, 0.0], [0.5, 0.25], [0.0, 0.5]], dtype=numpy.float32)
        test = numpy.array([[0.25, 0.2]], dtype=numpy.float32)
        policy = self._app_policy(classifier_transform="standard")
        manager = FeatureTransformManager(policy, stage="classifier", output_space="classifier")
        manager.fit(train, split_name="train")
        transformed_test = manager.transform(test, split_name="test")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier_scaler.joblib"
            manager.save(path)
            loaded = FeatureTransformManager.load(path)

        self.assertEqual(loaded.transform_id, manager.transform_id)
        numpy.testing.assert_allclose(
            loaded.transform(test, split_name="test"),
            transformed_test,
            atol=1e-6,
        )

    def test_full_and_batch_fit_produce_same_minmax_contract(self):
        values = numpy.linspace(-0.5, 0.5, 40, dtype=numpy.float32).reshape(10, 4)
        policy = self._app_policy(feature_transform="minmax")
        full = FeatureTransformManager(policy, stage="feature")
        batches = FeatureTransformManager(policy, stage="feature")

        full.fit(values, split_name="train")
        batches.partial_fit_batches((values[start:start + 2] for start in range(0, values.shape[0], 2)))

        self.assertEqual(full.transform_id, batches.transform_id)
        numpy.testing.assert_allclose(
            full.transform(values, split_name="train"),
            batches.transform(values, split_name="train"),
            atol=1e-6,
        )

    def test_legacy_csv_defaults_are_preserved(self):
        self.assertEqual(DEFAULT_SOURCE_PROFILE, "legacy_csv")
        self.assertEqual(DEFAULT_SCALER, "none")
        self.assertEqual(DEFAULT_FEATURE_TRANSFORM, "preserve")

    def test_manifest_records_transform_history(self):
        values = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        manager = FeatureTransformManager(self._app_policy(feature_transform="minmax"), stage="feature")
        manager.fit(values, split_name="train")
        manager.transform(values, split_name="train")

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "preprocessing_manifest.json"
            TransformManifest(
                source_profile="appclassnet_top200",
                transformations=manager.transform_history,
                transform_id=manager.transform_id,
                train_fit={"fit_split": "train", "operation": manager.operation},
                synthetic_output_space="source",
                evaluation_space="source",
                inverse_transform_synthetic=True,
            ).save(manifest_path)
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["source_profile"], "appclassnet_top200")
        self.assertEqual(manifest["transform_id"], manager.transform_id)
        self.assertEqual(manifest["transformations"][0]["operation"], "minmax")
        self.assertEqual(manifest["transformations"][0]["fit_split"], "train")

    def test_manifest_records_explicit_preserve_policy(self):
        values = numpy.array([[-0.5, 0.0], [0.5, 0.25]], dtype=numpy.float32)
        manager = FeatureTransformManager(self._app_policy(), stage="feature")
        manager.fit(values, split_name="train")

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "preprocessing_manifest.json"
            TransformManifest(
                source_profile="appclassnet_top200",
                feature_transform="preserve",
                classifier_transform="preserve",
                generator_transform="preserve",
                evaluation_space="source",
                source_min=-0.5,
                source_max=0.5,
                transformed_min=-0.5,
                transformed_max=0.5,
                transform_fitted=False,
                transform_parameters={},
                transformations=manager.transform_history,
            ).save(manifest_path)
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["feature_transform"], "preserve")
        self.assertEqual(manifest["classifier_transform"], "preserve")
        self.assertEqual(manifest["generator_transform"], "preserve")
        self.assertEqual(manifest["evaluation_space"], "source")
        self.assertFalse(manifest["transform_fitted"])
        self.assertEqual(manifest["transform_parameters"], {})
        self.assertEqual(manifest["source_min"], -0.5)
        self.assertEqual(manifest["transformed_max"], 0.5)


if __name__ == "__main__":
    unittest.main()
