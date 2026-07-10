import unittest

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SamplePlan
from Engine.DataIO.DatasetContracts import SplitData


class DatasetContractsTest(unittest.TestCase):

    def test_dataset_bundle_accepts_train_valid_test(self):
        schema = DatasetSchema(
            feature_names=["f0", "f1", "f2"],
            feature_type="continuous",
            target_name="label",
            target_type="multiclass",
            num_classes=3,
            class_labels=[0, 1, 2],
            source_format="npy_xy",
        )

        bundle = DatasetBundle(
            train=SplitData(
                X=numpy.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]),
                y=numpy.array([0, 1, 2]),
                name="train",
            ),
            valid=SplitData(
                X=numpy.array([[3.1, 3.2, 3.3], [4.1, 4.2, 4.3]]),
                y=numpy.array([[1], [2]]),
                name="valid",
            ),
            test=SplitData(
                X=numpy.array([[5.1, 5.2, 5.3]]),
                y=numpy.array([0]),
                name="test",
            ),
            schema=schema,
            metadata={"dataset": "appclassnet_top200"},
        )

        self.assertEqual(bundle.train.num_rows, 3)
        self.assertEqual(bundle.valid.y.shape, (2,))
        self.assertEqual(bundle.test.num_features, 3)
        self.assertEqual(set(bundle.splits.keys()), {"train", "valid", "test"})
        self.assertEqual(bundle.metadata["dataset"], "appclassnet_top200")

    def test_split_data_rejects_non_2d_features(self):
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            SplitData(X=numpy.array([1.0, 2.0, 3.0]), y=numpy.array([0, 1, 0]))

    def test_split_data_rejects_row_mismatch(self):
        with self.assertRaisesRegex(ValueError, "row mismatch"):
            SplitData(X=numpy.zeros((3, 2)), y=numpy.array([0, 1]), name="train")

    def test_split_data_rejects_y_that_cannot_be_squeezed_to_1d(self):
        with self.assertRaisesRegex(ValueError, "safely convertible"):
            SplitData(X=numpy.zeros((2, 2)), y=numpy.zeros((2, 2)), name="train")

    def test_schema_rejects_invalid_multiclass_num_classes(self):
        with self.assertRaisesRegex(ValueError, "num_classes"):
            DatasetSchema(
                feature_names=["f0"],
                target_type="multiclass",
                num_classes=1,
            )

    def test_schema_rejects_class_label_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "class_labels length"):
            DatasetSchema(
                feature_names=["f0"],
                target_type="multiclass",
                num_classes=3,
                class_labels=[0, 1],
            )

    def test_bundle_rejects_feature_count_mismatch(self):
        schema = DatasetSchema(
            feature_names=["f0", "f1"],
            target_type="binary",
            num_classes=2,
        )

        with self.assertRaisesRegex(ValueError, "schema defines 2"):
            DatasetBundle(
                train=SplitData(X=numpy.zeros((2, 3)), y=numpy.array([0, 1])),
                schema=schema,
            )

    def test_bundle_rejects_multiclass_label_outside_num_classes(self):
        schema = DatasetSchema(
            feature_names=["f0", "f1"],
            target_type="multiclass",
            num_classes=3,
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            DatasetBundle(
                train=SplitData(X=numpy.zeros((3, 2)), y=numpy.array([0, 1, 3])),
                schema=schema,
            )

    def test_bundle_accepts_declared_string_class_labels(self):
        schema = DatasetSchema(
            feature_names=["f0", "f1"],
            target_type="multiclass",
            num_classes=2,
            class_labels=["benign", "malware"],
        )

        bundle = DatasetBundle(
            train=SplitData(
                X=numpy.zeros((2, 2)),
                y=numpy.array(["benign", "malware"]),
            ),
            schema=schema,
        )

        self.assertEqual(bundle.schema.class_labels, ("benign", "malware"))

    def test_bundle_accepts_target_type_none_without_y(self):
        schema = DatasetSchema(
            feature_names=["f0", "f1"],
            target_type="none",
            target_name=None,
        )

        bundle = DatasetBundle(
            train=SplitData(X=numpy.zeros((2, 2)), name="train"),
            schema=schema,
        )

        self.assertIsNone(bundle.train.y)
        self.assertIsNone(bundle.schema.target_name)

    def test_sample_plan_infers_total_from_class_counts(self):
        plan = SamplePlan(class_counts={0: 2, 1: 3})

        self.assertEqual(plan.total_rows, 5)
        self.assertEqual(plan.class_counts, {0: 2, 1: 3})


if __name__ == "__main__":
    unittest.main()
