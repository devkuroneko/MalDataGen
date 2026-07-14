import unittest
from dataclasses import dataclass

import numpy


SPEC_SKIP_REASON = (
    "Integration specification skeleton only. Enable after the evaluation "
    "pipeline contract is implemented without changing legacy behavior."
)


@dataclass(frozen=True)
class ControlledSplitDataset:
    train_x: numpy.ndarray
    train_y: numpy.ndarray
    valid_x: numpy.ndarray
    valid_y: numpy.ndarray
    test_x: numpy.ndarray
    test_y: numpy.ndarray
    num_classes: int
    num_features: int = 20
    data_space: str = "source"


def make_controlled_dataset(
        num_classes,
        samples_per_class_train=12,
        samples_per_class_valid=6,
        samples_per_class_test=6,
        num_features=20,
        seed=1234):
    """Create independent, separable splits in source scale [-0.5, 0.5]."""
    rng = numpy.random.default_rng(seed)
    centroid_values = numpy.array([-0.35, 0.0, 0.35], dtype=numpy.float32)

    def centroid_for_class(class_id):
        code = numpy.zeros(num_features, dtype=numpy.int64)
        value = class_id
        for feature_index in range(num_features):
            code[feature_index] = value % 3
            value //= 3
        return centroid_values[code]

    def build_split(samples_per_class, split_offset):
        xs = []
        ys = []
        for class_id in range(num_classes):
            centroid = centroid_for_class(class_id)
            noise = rng.normal(
                loc=split_offset,
                scale=0.01,
                size=(samples_per_class, num_features),
            ).astype(numpy.float32)
            xs.append(numpy.clip(centroid + noise, -0.5, 0.5))
            ys.append(numpy.full(samples_per_class, class_id, dtype=numpy.int64))
        return numpy.vstack(xs), numpy.concatenate(ys)

    train_x, train_y = build_split(samples_per_class_train, 0.000)
    valid_x, valid_y = build_split(samples_per_class_valid, 0.003)
    test_x, test_y = build_split(samples_per_class_test, -0.003)
    return ControlledSplitDataset(
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
        num_classes=num_classes,
        num_features=num_features,
    )


def make_compatible_synthetic(dataset, samples_per_class=8, seed=5678):
    """Synthetic fixture preserving class centroids without reusing exact rows."""
    rng = numpy.random.default_rng(seed)
    xs = []
    ys = []
    for class_id in range(dataset.num_classes):
        real_class_x = dataset.train_x[dataset.train_y == class_id]
        centroid = real_class_x.mean(axis=0)
        synthetic = centroid + rng.normal(
            loc=0.0,
            scale=0.012,
            size=(samples_per_class, dataset.num_features),
        ).astype(numpy.float32)
        xs.append(numpy.clip(synthetic, -0.5, 0.5))
        ys.append(numpy.full(samples_per_class, class_id, dtype=numpy.int64))
    return numpy.vstack(xs), numpy.concatenate(ys)


@unittest.skip(SPEC_SKIP_REASON)
class EvaluationIntegrationSpecTest(unittest.TestCase):

    def test_tr_tr_learns_separable_classes(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # classifier = DecisionTreeClassifier(random_state=seed)
        # classifier.fit(dataset.train_x, dataset.train_y)
        # pred = classifier.predict(dataset.test_x)
        # assert accuracy and macro_f1 are high, and train/test row IDs differ.
        self.assertEqual(dataset.train_x.shape[1], 20)

    def test_tr_tr_fails_with_shuffled_labels(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # shuffled_train_y = deterministic_permutation(dataset.train_y)
        # train TR-TR with shuffled labels, test on true test_y.
        # assert accuracy is close to 1 / 3 and diagnostics record label shuffle.
        self.assertEqual(dataset.num_classes, 3)

    def test_tr_ts_works_with_statistically_compatible_synthetic(self):
        dataset = make_controlled_dataset(num_classes=3)
        synthetic_x, synthetic_y = make_compatible_synthetic(dataset)
        # Expected execution:
        # train classifier on real train split.
        # test classifier on synthetic_x/synthetic_y in source space.
        # assert high accuracy and matching data_space metadata.
        self.assertEqual(synthetic_x.shape[1], 20)
        self.assertEqual(set(synthetic_y.tolist()), {0, 1, 2})

    def test_ts_tr_works_when_synthetic_preserves_classes(self):
        dataset = make_controlled_dataset(num_classes=3)
        synthetic_x, synthetic_y = make_compatible_synthetic(dataset)
        # Expected execution:
        # train classifier on synthetic_x/synthetic_y.
        # test classifier on dataset.test_x/dataset.test_y.
        # assert high accuracy and independent real test split.
        self.assertEqual(synthetic_x.shape[0], synthetic_y.shape[0])

    def test_tr_ts_and_ts_tr_drop_to_chance_when_synthetic_loses_label_relation(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # build label_broken synthetic fixture by permuting synthetic_y or sampling
        # X independently of labels.
        # assert TR-TS and TS-TR accuracy are near 1 / num_classes.
        self.assertEqual(dataset.num_classes, 3)

    def test_mismatched_real_and_synthetic_spaces_block_evaluation(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # real metadata: data_space=source.
        # synthetic metadata: data_space=generator, transform_id non-null.
        # assert PreprocessingSpaceMismatchError or documented warning/error.
        self.assertEqual(dataset.data_space, "source")

    def test_missing_classes_raise_error_or_documented_warning(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # remove one class from synthetic_x/synthetic_y.
        # assert label audit reports missing class before metrics are trusted.
        self.assertEqual(set(dataset.train_y.tolist()), {0, 1, 2})

    def test_misaligned_labels_are_detected(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # keep X order and rotate y by one position.
        # assert sanity check or integration guard rejects/flags X-y misalignment.
        self.assertEqual(dataset.train_x.shape[0], dataset.train_y.shape[0])

    def test_normal_and_batches_modes_produce_close_results(self):
        dataset = make_controlled_dataset(num_classes=10)
        # Expected execution:
        # run same controlled synthetic fixture through normal and batches modes.
        # assert metrics are within a small tolerance.
        self.assertEqual(dataset.num_classes, 10)

    def test_batches_do_not_drop_classes(self):
        dataset = make_controlled_dataset(num_classes=10)
        # Expected execution:
        # force batch boundaries that split class groups.
        # assert all labels 0..9 are observed in generated/read batches.
        self.assertEqual(set(dataset.train_y.tolist()), set(range(10)))

    def test_batches_do_not_use_only_first_block(self):
        dataset = make_controlled_dataset(num_classes=10)
        # Expected execution:
        # tag samples by block or use distinguishable later-block values.
        # assert later-block markers affect output counts and metrics.
        self.assertGreater(dataset.train_x.shape[0], 10)

    def test_scaler_is_fit_only_on_training_data(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # install spy scaler/transform manager.
        # assert fit is called only with train_x, never valid_x/test_x/synthetic_x.
        self.assertEqual(dataset.train_x.shape[1], 20)

    def test_no_transformation_is_applied_twice(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # run source-preserving AppClassNet profile.
        # assert transform_history has no duplicate transform_id/application.
        self.assertEqual(dataset.data_space, "source")

    def test_not_executed_metrics_are_not_applicable(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # run evaluation_mode=tr_ts and inspect TS-TR result.
        # assert skipped metrics are status=not_applicable, not numeric zero.
        self.assertEqual(dataset.num_classes, 3)

    def test_two_hundred_classes_use_zero_based_labels(self):
        dataset = make_controlled_dataset(
            num_classes=200,
            samples_per_class_train=1,
            samples_per_class_valid=1,
            samples_per_class_test=1,
        )
        # Expected execution:
        # load NPY XY, build metadata and one-hot.
        # assert labels are exactly 0..199 and one-hot width is 200.
        self.assertEqual(dataset.num_classes, 200)
        self.assertEqual(set(dataset.train_y.tolist()), set(range(200)))

    def test_synthetic_train_is_not_reused_as_synthetic_test(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # materialize synthetic train and synthetic test with origin IDs.
        # assert no row IDs/hashes are shared unless explicitly allowed.
        self.assertEqual(dataset.num_classes, 3)

    def test_seeds_make_pipeline_reproducible(self):
        dataset = make_controlled_dataset(num_classes=3, seed=42)
        # Expected execution:
        # run same command twice with same seed.
        # assert metrics, selected indices and synthetic hashes match.
        self.assertEqual(dataset.train_x.shape[1], 20)

    def test_legacy_csv_command_still_works(self):
        dataset = make_controlled_dataset(num_classes=3)
        # Expected execution:
        # write legacy CSV with label column and run old command shape.
        # assert command completes and legacy outputs remain compatible.
        self.assertEqual(dataset.train_x.shape[0], dataset.train_y.shape[0])


if __name__ == "__main__":
    unittest.main()
