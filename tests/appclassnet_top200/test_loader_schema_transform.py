import numpy
import pytest

from Engine.DataIO.NpyXYLoader import NpyXYLoader
from Engine.Evaluation.CrossValidation import load_dataset_from_args
from Engine.Preprocessing.FeatureTransformManager import DoubleTransformError
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformManager
from Engine.Preprocessing.FeatureTransformManager import FeatureTransformPolicy


def test_loader_accepts_six_npy_xy_files_and_preserves_splits(small_npy_dir):
    bundle = NpyXYLoader(
        small_npy_dir / "train_x.npy",
        small_npy_dir / "train_y.npy",
        small_npy_dir / "valid_x.npy",
        small_npy_dir / "valid_y.npy",
        small_npy_dir / "test_x.npy",
        small_npy_dir / "test_y.npy",
        mmap_mode=None,
        expected_num_features=5,
    ).load()

    assert bundle.train.name == "train", "train split must remain train"
    assert bundle.valid.name == "valid", "valid split must remain valid"
    assert bundle.test.name == "test", "test split must remain test"
    assert bundle.train.X.shape == (12, 5), "loader changed train shape"
    assert bundle.schema.split_mode == "provided"


def test_loader_mmap_preserves_memmap_and_values(small_npy_dir):
    expected = numpy.load(small_npy_dir / "train_x.npy", allow_pickle=False)
    bundle = NpyXYLoader(
        small_npy_dir / "train_x.npy",
        small_npy_dir / "train_y.npy",
        mmap_mode="r",
    ).load()

    assert isinstance(bundle.train.X, numpy.memmap), "X must remain memmap when mmap is enabled"
    assert isinstance(bundle.train.y, numpy.memmap), "y must remain memmap when mmap is enabled"
    numpy.testing.assert_array_equal(bundle.train.X, expected)


def test_loader_rejects_shape_mismatch(tmp_path):
    numpy.save(tmp_path / "train_x.npy", numpy.zeros((3, 5), dtype=numpy.float32))
    numpy.save(tmp_path / "train_y.npy", numpy.asarray([0, 1], dtype=numpy.int64))

    with pytest.raises(ValueError, match="row mismatch"):
        NpyXYLoader(tmp_path / "train_x.npy", tmp_path / "train_y.npy").load()


def test_loader_rejects_feature_width_mismatch(small_npy_dir):
    numpy.save(small_npy_dir / "test_x.npy", numpy.zeros((12, 6), dtype=numpy.float32))

    with pytest.raises(ValueError, match="expected 5"):
        NpyXYLoader(
            small_npy_dir / "train_x.npy",
            small_npy_dir / "train_y.npy",
            test_x_path=small_npy_dir / "test_x.npy",
            test_y_path=small_npy_dir / "test_y.npy",
        ).load()


def test_loader_rejects_negative_labels(tmp_path):
    numpy.save(tmp_path / "train_x.npy", numpy.zeros((2, 5), dtype=numpy.float32))
    numpy.save(tmp_path / "train_y.npy", numpy.asarray([0, -1], dtype=numpy.int64))

    with pytest.raises(ValueError, match="negative labels"):
        NpyXYLoader(tmp_path / "train_x.npy", tmp_path / "train_y.npy").load()


def test_loader_accepts_labels_zero_to_199(top200_npy_dir):
    bundle = NpyXYLoader(
        top200_npy_dir / "train_x.npy",
        top200_npy_dir / "train_y.npy",
        top200_npy_dir / "valid_x.npy",
        top200_npy_dir / "valid_y.npy",
        top200_npy_dir / "test_x.npy",
        top200_npy_dir / "test_y.npy",
        source_profile="appclassnet_top200",
        expected_num_features=20,
    ).load()

    assert bundle.schema.num_classes == 200, "Top200 schema must keep 200 classes"
    assert bundle.schema.classes[0] == 0
    assert bundle.schema.classes[-1] == 199


def test_loader_preserves_feature_dtypes(small_npy_dir):
    bundle = NpyXYLoader(small_npy_dir / "train_x.npy", small_npy_dir / "train_y.npy").load()

    assert bundle.schema.feature_dtype == "float32"
    assert bundle.schema.target_dtype == "int64"


def test_csv_compatibility_still_uses_legacy_loader(csv_args):
    class Owner:
        def __init__(self):
            self.load_csv_called = False

        def load_csv(self):
            self.load_csv_called = True

    owner = Owner()
    result = load_dataset_from_args(csv_args, owner=owner)

    assert result is None, "CSV path must remain legacy and not return DatasetBundle"
    assert owner.load_csv_called, "legacy CSV owner.load_csv was not called"


def test_schema_dynamic_classes_for_custom_profile(small_npy_dir):
    bundle = NpyXYLoader(small_npy_dir / "train_x.npy", small_npy_dir / "train_y.npy").load()

    assert bundle.schema.target_type == "multiclass"
    assert bundle.schema.num_classes == 4
    assert bundle.schema.classes == (0, 1, 2, 3)


def test_schema_rejects_wrong_top200_feature_count(top200_npy_dir):
    with pytest.raises(ValueError, match="expected 21"):
        NpyXYLoader(
            top200_npy_dir / "train_x.npy",
            top200_npy_dir / "train_y.npy",
            source_profile="appclassnet_top200",
            expected_num_features=21,
        ).load()


def test_schema_records_train_interval_only(small_npy_dir):
    numpy.save(small_npy_dir / "test_x.npy", numpy.full((12, 5), 99.0, dtype=numpy.float32))
    bundle = NpyXYLoader(
        small_npy_dir / "train_x.npy",
        small_npy_dir / "train_y.npy",
        test_x_path=small_npy_dir / "test_x.npy",
        test_y_path=small_npy_dir / "test_y.npy",
        expected_num_features=5,
    ).load()

    assert bundle.schema.train_feature_max < 1.0, "test values must not influence train feature max"


def test_preserve_transform_keeps_minus_half_space_identity():
    x = numpy.asarray([[-0.5, 0.0], [0.25, 0.5]], dtype=numpy.float32)
    manager = FeatureTransformManager(
        FeatureTransformPolicy.for_profile("appclassnet_top200"),
        stage="classifier",
        output_space="classifier",
    )

    manager.fit(x, split_name="train")
    transformed, metadata = manager.transform(x, split_name="train", return_metadata=True)

    numpy.testing.assert_array_equal(transformed, x)
    assert metadata["transform_applied"] is False
    assert manager.scaler is None, "preserve mode must not fit a scaler"


def test_preserve_does_not_renormalize_to_zero_one():
    x = numpy.asarray([[-0.5, 0.0], [0.25, 0.5]], dtype=numpy.float32)
    manager = FeatureTransformManager(FeatureTransformPolicy.for_profile("appclassnet_top200"), stage="classifier")
    manager.fit(x, split_name="train")

    transformed = manager.transform(x, split_name="test")

    assert float(numpy.min(transformed)) == -0.5, "preserve must not shift [-0.5, 0.5] to [0, 1]"
    assert float(numpy.max(transformed)) == 0.5


def test_transform_rejects_double_scaling_history():
    policy = FeatureTransformPolicy.for_profile("custom", classifier_transform="minmax")
    manager = FeatureTransformManager(policy, stage="classifier")
    manager.fit(numpy.asarray([[0.0], [1.0]], dtype=numpy.float32), split_name="train")

    with pytest.raises(DoubleTransformError, match="duplicate"):
        manager.transform(
            numpy.asarray([[0.5]], dtype=numpy.float32),
            transform_history=[{"operation": "minmax", "stage": "classifier"}],
        )


def test_transform_fit_uses_train_not_test():
    train = numpy.asarray([[-0.5], [0.5]], dtype=numpy.float32)
    test = numpy.asarray([[99.0]], dtype=numpy.float32)
    manager = FeatureTransformManager(
        FeatureTransformPolicy.for_profile("custom", classifier_transform="minmax"),
        stage="classifier",
    )

    manager.fit(train, split_name="train")
    manager.transform(test, split_name="test")

    assert manager.input_range == [-0.5, 0.5], "test values must not change fitted scaler range"
