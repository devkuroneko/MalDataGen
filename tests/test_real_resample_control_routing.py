import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from main import AggregateDataUsedAsRawSamplesError
from main import SynDataGen
from main import SyntheticControlSplitMismatchError


NUM_CLASSES = 200
NUM_FEATURES = 20


class RealResampleControlRoutingTest(unittest.TestCase):

    def test_synthesize_data_routes_train_and_test_to_dataset_bundle_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, train_rows_per_class=60, test_rows_per_class=60)
            bundle = owner._dataset_bundle

            result = owner.synthesize_data(
                numpy.asarray(bundle.valid.X),
                numpy.asarray(bundle.valid.y),
                numpy.asarray(bundle.valid.X),
                numpy.asarray(bundle.valid.y),
                real_train_source_data=bundle.train,
                real_test_source_data=bundle.test,
            )

            self.assertEqual(result.train_reader.manifest["source_split"], "train")
            self.assertEqual(result.train_reader.manifest["source_x_path"], "/raw/train_x.npy")
            self.assertEqual(result.test_reader.manifest["source_split"], "test")
            self.assertEqual(result.test_reader.manifest["source_x_path"], "/raw/test_x.npy")

    def test_real_resample_never_accepts_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory)

            with self.assertRaises(SyntheticControlSplitMismatchError):
                owner._synthesize_control_batches(
                    source_data=owner._dataset_bundle.valid,
                    split_name="valid",
                    samples_per_class=50,
                    random_state=0,
                )

    def test_centroid_matrix_with_one_row_per_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory)
            centroid_split = _split("train", 1, "/raw/centroids_x.npy", "/raw/centroids_y.npy")

            with self.assertRaises(AggregateDataUsedAsRawSamplesError):
                owner._synthesize_control_batches(
                    source_data=centroid_split,
                    split_name="train",
                    samples_per_class=1,
                    random_state=0,
                )

    def test_wrong_split_raises_split_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory)

            with self.assertRaises(SyntheticControlSplitMismatchError):
                owner._synthesize_control_batches(
                    source_data=owner._dataset_bundle.test,
                    split_name="train",
                    samples_per_class=50,
                    random_state=0,
                )

    def test_provided_split_without_test_does_not_fallback_to_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory)
            owner._dataset_bundle.test = None
            dictionary_data = {
                "dataset_bundle": owner._dataset_bundle,
                "x_training_real": owner._dataset_bundle.train.X,
                "y_training_real": owner._dataset_bundle.train.y,
                "x_evaluation_real": owner._dataset_bundle.valid.X,
                "y_evaluation_real": owner._dataset_bundle.valid.y,
            }

            with self.assertRaisesRegex(SyntheticControlSplitMismatchError, "valid split is not allowed"):
                owner._control_source_splits(dictionary_data)

    def test_fifty_per_class_generates_exactly_ten_thousand_rows_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, train_rows_per_class=60)

            reader = owner._synthesize_control_batches(
                source_data=owner._dataset_bundle.train,
                split_name="train",
                samples_per_class=50,
                random_state=123,
            )

            self.assertEqual(reader.total_rows, 10000)
            self.assertEqual(reader.manifest["total_samples"], 10000)
            for class_id in range(NUM_CLASSES):
                indices = reader.manifest["source_indices"][str(class_id)]
                self.assertEqual(len(indices), 50)
                self.assertEqual(len(set(indices)), 50)
                self.assertTrue(all(0 <= index < owner._dataset_bundle.train.X.shape[0] for index in indices))

    def test_selected_x_rows_remain_aligned_with_class_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, train_rows_per_class=60)

            reader = owner._synthesize_control_batches(
                source_data=owner._dataset_bundle.train,
                split_name="train",
                samples_per_class=50,
                random_state=5,
            )

            for class_id, batch in reader.iter_batches():
                self.assertTrue(numpy.all(batch[:, 0] == class_id))

    def test_manifest_records_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, train_rows_per_class=60)

            reader = owner._synthesize_control_batches(
                source_data=owner._dataset_bundle.train,
                split_name="train",
                samples_per_class=50,
                random_state=9,
            )

            manifest = reader.manifest
            self.assertEqual(manifest["control_type"], "real_resample")
            self.assertEqual(manifest["source_split"], "train")
            self.assertEqual(manifest["source_y_path"], "/raw/train_y.npy")
            self.assertEqual(manifest["samples_per_class"], 50)
            self.assertEqual(manifest["num_classes"], NUM_CLASSES)
            self.assertEqual(manifest["feature_count"], NUM_FEATURES)
            self.assertEqual(manifest["data_space"], "source")
            self.assertEqual(manifest["transform_id"], None)
            self.assertEqual(manifest["random_state"], 9)
            self.assertIsNotNone(manifest["schema_hash"])

    def test_test_minimum_1539_allows_selecting_50(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, test_rows_per_class=1539)

            reader = owner._synthesize_control_batches(
                source_data=owner._dataset_bundle.test,
                split_name="test",
                samples_per_class=50,
                random_state=0,
            )

            self.assertEqual(reader.total_rows, 10000)
            self.assertEqual(reader.manifest["source_class_counts"]["0"], 1539)

    def test_train_minimum_1388_allows_selecting_50(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = _owner(directory, train_rows_per_class=1388)

            reader = owner._synthesize_control_batches(
                source_data=owner._dataset_bundle.train,
                split_name="train",
                samples_per_class=50,
                random_state=0,
            )

            self.assertEqual(reader.total_rows, 10000)
            self.assertEqual(reader.manifest["source_class_counts"]["0"], 1388)

    def test_batch_command_for_provided_split_uses_one_effective_fold(self):
        import run_appclassnet_top200 as runner

        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            for split in ("train", "valid", "test"):
                numpy.save(raw_root / f"{split}_x.npy", numpy.zeros((NUM_CLASSES, NUM_FEATURES), dtype=numpy.float32))
                numpy.save(raw_root / f"{split}_y.npy", numpy.arange(NUM_CLASSES, dtype=numpy.int64))

            args = SimpleNamespace(
                dataset_split="train",
                batch_size=64,
                eval_batch_size=64,
                generation_batch_size=64,
                eval_classifier="decision_tree_subset",
                batch_classifier_subset_size=100,
                min_samples_per_class_required=1,
                generation_strategy="single_conditional",
                classes_per_group=10,
                source_profile="appclassnet_top200",
                generator_transform="preserve",
                classifier_transform="preserve",
                evaluation_space="source",
                evaluation_mode="both",
                random_state=0,
                synthetic_control="real_resample",
                class_subset=None,
                num_classes_subset=None,
                vae_epochs=None,
                gan_epochs=None,
                use_mmap=True,
                max_train_samples=None,
                max_samples_per_class=None,
                train_samples_per_class=None,
                test_samples_per_class=None,
                synthetic_train_samples_per_class=50,
                synthetic_test_samples_per_class=50,
                generated_samples_per_class=100,
                real_class_count_policy=None,
                samples_per_class_scope=None,
                n_estimators=None,
                max_depth=None,
                max_samples=None,
                class_weight=None,
                full=False,
                run_mode_effective="demo",
                dry_run_memory=False,
                run_tr_tr_effective=False,
                save_synthetic_format=None,
                materialize_synthetic=False,
                allow_double_transform=False,
                allow_scaler_refit=False,
                inverse_transform_synthetic=True,
            )

            command = runner.build_batch_main_command(
                "python3",
                raw_root,
                raw_root / "out",
                {"model_type": "copy", "number_k_folds": 2},
                20,
                "multiclass",
                args,
            )

        self.assertIn("--number_k_folds", command)
        self.assertEqual(command[command.index("--number_k_folds") + 1], "1")


def _owner(directory, train_rows_per_class=60, test_rows_per_class=60):
    owner = SynDataGen.__new__(SynDataGen)
    owner.arguments = SimpleNamespace(
        synthetic_control="real_resample",
        save_synthetic_format="npy_batches",
        execution_mode="batches",
        batch_size=64,
        num_classes=NUM_CLASSES,
        num_features=NUM_FEATURES,
        random_state=0,
        split_mode="provided",
        data_type="continuous",
        target_type="multiclass",
        sample_plan="class_counts",
        number_samples_per_class={
            "classes": {class_id: 100 for class_id in range(NUM_CLASSES)},
            "number_classes": NUM_CLASSES,
        },
        _legacy_number_samples_per_class_explicit=True,
        synthetic_train_samples_per_class=50,
        synthetic_test_samples_per_class=50,
        train_samples_per_class=None,
        test_samples_per_class=None,
    )
    owner.fold_number = 0
    owner.directory_output_data = Path(directory)
    owner._dictionary_metrics = {}
    owner._data_loaded = numpy.zeros((1, NUM_FEATURES), dtype=numpy.float32)
    owner._data_type = "continuous"
    owner._dataset_bundle = _bundle(train_rows_per_class, test_rows_per_class)
    owner._model_input_adapter = None
    owner._current_synthetic_metadata = None
    return owner


def _bundle(train_rows_per_class, test_rows_per_class):
    schema = DatasetSchema(
        feature_names=[f"f{index}" for index in range(NUM_FEATURES)],
        target_type="multiclass",
        feature_type="continuous",
        num_classes=NUM_CLASSES,
        class_labels=tuple(range(NUM_CLASSES)),
        source_format="npy_xy",
        source_profile="appclassnet_top200",
        data_space="source",
    )
    return DatasetBundle(
        train=_split("train", train_rows_per_class, "/raw/train_x.npy", "/raw/train_y.npy"),
        valid=_split("valid", 2, "/raw/valid_x.npy", "/raw/valid_y.npy"),
        test=_split("test", test_rows_per_class, "/raw/test_x.npy", "/raw/test_y.npy"),
        schema=schema,
    )


def _split(name, rows_per_class, x_path, y_path):
    y = numpy.repeat(numpy.arange(NUM_CLASSES, dtype=numpy.int64), rows_per_class)
    X = numpy.zeros((y.shape[0], NUM_FEATURES), dtype=numpy.float32)
    X[:, 0] = y.astype(numpy.float32)
    X[:, 1] = numpy.arange(y.shape[0], dtype=numpy.float32)
    return SplitData(
        X=X,
        y=y,
        name=name,
        x_path=x_path,
        y_path=y_path,
        dataset_id="appclassnet_top200",
    )


if __name__ == "__main__":
    unittest.main()
