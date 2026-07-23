import inspect
import unittest

import numpy

from Engine.DataIO.DatasetContracts import DatasetBundle
from Engine.DataIO.DatasetContracts import DatasetSchema
from Engine.DataIO.DatasetContracts import SplitData
from Engine.Pipelines.Interfaces import SyntheticGenerationConfig
from Engine.Pipelines.Interfaces import build_synthetic_generation_request
from Engine.Pipelines.Interfaces import partition_generation_classes
import Engine.Evaluation.TrTrPipeline as tr_tr_pipeline


def make_bundle():
    schema = DatasetSchema(
        feature_names=["f0", "f1"],
        num_features=2,
        feature_type="continuous",
        target_type="multiclass",
        num_classes=4,
        classes=(0, 1, 2, 3),
        source_format="npy_xy",
        data_format="npy_xy",
        split_mode="provided",
        source_profile="appclassnet_top200",
        feature_dtype="float32",
        target_dtype="int64",
        data_space="source",
    )
    return DatasetBundle(
        train=SplitData(
            X=numpy.ones((8, 2), dtype=numpy.float32),
            y=numpy.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=numpy.int64),
            name="train",
            x_path="/data/train_x.npy",
            y_path="/data/train_y.npy",
        ),
        valid=SplitData(
            X=numpy.ones((4, 2), dtype=numpy.float32),
            y=numpy.asarray([0, 1, 2, 3], dtype=numpy.int64),
            name="valid",
            x_path="/data/valid_x.npy",
            y_path="/data/valid_y.npy",
        ),
        test=SplitData(
            X=numpy.full((4, 2), 9, dtype=numpy.float32),
            y=numpy.asarray([0, 1, 2, 3], dtype=numpy.int64),
            name="test",
            x_path="/data/test_x.npy",
            y_path="/data/test_y.npy",
        ),
        schema=schema,
    )


class PipelineArchitectureTest(unittest.TestCase):

    def test_tr_tr_pipeline_does_not_import_generator(self):
        source = inspect.getsource(tr_tr_pipeline)

        self.assertNotIn("GenerativeModels", source)
        self.assertNotIn("import_models", source)

    def test_synthetic_generation_request_excludes_test(self):
        bundle = make_bundle()
        request = build_synthetic_generation_request(
            bundle,
            SyntheticGenerationConfig(
                generator="adversarial",
                generation_strategy="single_conditional",
                generator_transform="preserve",
                synthetic_samples_per_class=5,
            ),
        )

        self.assertIs(request.train_split, bundle.train)
        self.assertIs(request.valid_split, bundle.valid)
        self.assertFalse(request.uses_test_split)
        self.assertFalse(hasattr(request, "test_split"))
        self.assertEqual(request.config.generator_transform, "preserve")

    def test_grouped_generation_keeps_global_labels(self):
        units = partition_generation_classes(
            {0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 199: 1},
            "grouped_classes",
            2,
        )

        self.assertEqual(units, [
            {"unit_type": "group", "classes": [0, 1]},
            {"unit_type": "group", "classes": [2, 3]},
            {"unit_type": "group", "classes": [199]},
        ])

    def test_single_conditional_keeps_all_active_global_labels(self):
        units = partition_generation_classes({0: 2, 3: 1, 199: 4}, "single_conditional", 10)

        self.assertEqual(units, [{"unit_type": "global", "classes": [0, 3, 199]}])


if __name__ == "__main__":
    unittest.main()
