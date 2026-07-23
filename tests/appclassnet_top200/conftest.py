from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData


def make_xy(num_classes=4, rows_per_class=3, num_features=5, *, offset=0.0, dtype=numpy.float32):
    x_rows = []
    y_rows = []
    for class_id in range(num_classes):
        center = (class_id / max(1, num_classes - 1)) - 0.5 + float(offset)
        for row in range(rows_per_class):
            x_rows.append([
                center + (feature * 0.001) + (row * 0.0001)
                for feature in range(num_features)
            ])
            y_rows.append(class_id)
    return numpy.asarray(x_rows, dtype=dtype), numpy.asarray(y_rows, dtype=numpy.int64)


def save_npy_splits(root: Path, *, num_classes=4, rows_per_class=3, num_features=5):
    paths = {}
    for split, offset in (("train", 0.0), ("valid", 0.05), ("test", 0.1)):
        x_values, y_values = make_xy(
            num_classes=num_classes,
            rows_per_class=rows_per_class,
            num_features=num_features,
            offset=offset,
        )
        x_path = root / f"{split}_x.npy"
        y_path = root / f"{split}_y.npy"
        numpy.save(x_path, x_values)
        numpy.save(y_path, y_values)
        paths[f"{split}_x"] = x_path
        paths[f"{split}_y"] = y_path
    return paths


@pytest.fixture
def small_npy_dir(tmp_path):
    save_npy_splits(tmp_path, num_classes=4, rows_per_class=3, num_features=5)
    return tmp_path


@pytest.fixture
def top200_npy_dir(tmp_path):
    save_npy_splits(tmp_path, num_classes=200, rows_per_class=1, num_features=20)
    return tmp_path


@pytest.fixture
def small_bundle():
    return make_bundle(num_classes=4, rows_per_class=3, num_features=5)


@pytest.fixture
def top200_bundle():
    return make_bundle(num_classes=200, rows_per_class=1, num_features=20)


@pytest.fixture
def imbalanced_labels():
    return numpy.asarray([0] * 600 + [1] * 499 + [2] * 501, dtype=numpy.int64)


def make_bundle(num_classes=4, rows_per_class=3, num_features=5):
    train_x, train_y = make_xy(num_classes, rows_per_class, num_features, offset=0.0)
    valid_x, valid_y = make_xy(num_classes, 1, num_features, offset=0.05)
    test_x, test_y = make_xy(num_classes, rows_per_class, num_features, offset=0.1)
    schema = DatasetSchema(
        feature_names=[f"f{index}" for index in range(num_features)],
        num_features=num_features,
        feature_type="continuous",
        target_type="multiclass",
        num_classes=num_classes,
        classes=tuple(range(num_classes)),
        source_format="npy_xy",
        data_format="npy_xy",
        split_mode="provided",
        source_profile="appclassnet_top200" if num_classes == 200 else "custom",
        train_feature_min=float(numpy.min(train_x)),
        train_feature_max=float(numpy.max(train_x)),
        feature_dtype=str(train_x.dtype),
        target_dtype=str(train_y.dtype),
        data_space="source",
    )
    return DatasetBundle(
        train=SplitData(train_x, train_y, name="train", x_path="/fake/train_x.npy", y_path="/fake/train_y.npy"),
        valid=SplitData(valid_x, valid_y, name="valid", x_path="/fake/valid_x.npy", y_path="/fake/valid_y.npy"),
        test=SplitData(test_x, test_y, name="test", x_path="/fake/test_x.npy", y_path="/fake/test_y.npy"),
        schema=schema,
    )


def tr_tr_cli_args(*extra):
    return [
        "--pipeline", "tr_tr",
        "--source_profile", "appclassnet_top200",
        "--data_format", "npy_xy",
        "--split_mode", "provided",
        "--execution_mode", "batches",
        "--use_mmap",
        "--mmap_npy",
        "--feature_transform", "preserve",
        "--classifier_transform", "preserve",
        "--generator_transform", "preserve",
        "--evaluation_space", "source",
        "--train_sampling", "all",
        "--test_sampling", "all",
        "--eval_batch_size", "16",
        "--random_state", "123",
        *extra,
    ]


@pytest.fixture
def csv_args():
    return SimpleNamespace(
        data_format="csv",
        data_load_label_column=-1,
        data_load_max_samples=-1,
        data_load_max_columns=-1,
        data_load_start_column=0,
        data_load_end_column=-1,
        data_load_path_file_input="dataset.csv",
        data_load_path_file_output="OutputDir",
        data_load_exclude_columns=-1,
        data_type="binary",
        number_samples_per_class={"classes": {0: 1}, "number_classes": 1},
    )
