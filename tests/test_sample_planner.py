import unittest
from types import SimpleNamespace

import numpy

from Engine.DataIO.RealClassCountPolicy import build_split_sample_plan
from Engine.DataIO.SamplePlanner import build_sample_plan_from_args
from Engine.DataIO.SamplePlanner import sample_plan_to_legacy_metadata
import run_appclassnet_top200 as runner


def args(**overrides):
    base = {
        "data_format": "csv",
        "sample_plan": "legacy",
        "number_samples_per_class": {"classes": {0: 5, 1: 7}, "number_classes": 2},
        "samples_per_class": None,
        "total_synthetic_rows": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class SamplePlannerTest(unittest.TestCase):

    def test_legacy_number_samples_per_class_is_preserved(self):
        plan = build_sample_plan_from_args(
            args(),
            y_train=[0, 0, 1],
            number_classes=2,
            data_type="binary",
        )
        metadata = sample_plan_to_legacy_metadata(plan, data_type="binary")

        self.assertEqual(plan.mode, "legacy")
        self.assertEqual(metadata["classes"], {0: 5, 1: 7})
        self.assertEqual(metadata["number_classes"], 2)

    def test_balanced_per_class_with_200_classes(self):
        plan = build_sample_plan_from_args(
            args(
                data_format="npy_xy",
                sample_plan="balanced_per_class",
                number_samples_per_class="1:256,2:256",
                samples_per_class=1000,
            ),
            y_train=[0, 3, 199],
            number_classes=200,
            data_type="continuous",
        )

        self.assertEqual(plan.mode, "balanced_per_class")
        self.assertEqual(len(plan.class_counts), 200)
        self.assertEqual(plan.class_counts[0], 1000)
        self.assertEqual(plan.class_counts[199], 1000)
        self.assertEqual(plan.total_rows, 200000)

    def test_match_train_distribution_preserves_proportion(self):
        plan = build_sample_plan_from_args(
            args(
                data_format="npy_xy",
                sample_plan="match_train_distribution",
                number_samples_per_class="1:256,2:256",
                total_synthetic_rows=100,
            ),
            y_train=[0, 0, 0, 1],
            number_classes=2,
            data_type="continuous",
        )

        self.assertEqual(plan.class_counts, {0: 75, 1: 25})
        self.assertEqual(plan.total_rows, 100)

    def test_npy_legacy_without_explicit_counts_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "explicit sampling plan"):
            build_sample_plan_from_args(
                args(data_format="npy_xy", number_samples_per_class="1:256,2:256"),
                y_train=[0, 1],
                number_classes=2,
            )

    def test_split_plan_never_produces_negative_or_out_of_bounds_indices(self):
        labels = numpy.repeat(numpy.arange(3), [5, 6, 7])

        plan = build_split_sample_plan(
            labels,
            4,
            3,
            123,
            "train",
            strategy="balanced_per_class",
            insufficient_policy="strict",
        )

        self.assertIsNotNone(plan.selected_indices)
        self.assertGreaterEqual(int(plan.selected_indices.min()), 0)
        self.assertLess(int(plan.selected_indices.max()), labels.shape[0])

    def test_balanced_per_class_strict_fails_on_short_class(self):
        labels = numpy.repeat(numpy.arange(2), [3, 5])

        with self.assertRaisesRegex(ValueError, "train split has fewer than requested 4 samples"):
            build_split_sample_plan(
                labels,
                4,
                2,
                0,
                "train",
                strategy="balanced_per_class",
                insufficient_policy="strict",
            )

    def test_balanced_per_class_can_cap_to_available(self):
        labels = numpy.repeat(numpy.arange(2), [3, 5])

        plan = build_split_sample_plan(
            labels,
            4,
            2,
            0,
            "train",
            strategy="balanced_per_class",
            insufficient_policy="cap_to_available",
        )

        self.assertEqual(plan.metadata["selected_counts_by_class"], {"0": 3, "1": 4})
        self.assertEqual(plan.selection_table[0]["available"], 3)
        self.assertEqual(plan.selection_table[0]["requested"], 4)
        self.assertEqual(plan.selection_table[0]["selected"], 3)

    def test_up_to_available_records_requested_and_effective_by_class(self):
        labels = numpy.repeat(numpy.arange(3), [2, 4, 6])

        plan = build_split_sample_plan(
            labels,
            5,
            3,
            0,
            "test",
            strategy="up_to_available",
            insufficient_policy="strict",
        )

        self.assertEqual(plan.metadata["selected_counts_by_class"], {"0": 2, "1": 4, "2": 5})
        self.assertEqual(
            [(row["class_id"], row["available"], row["requested"], row["selected"], row["split"])
             for row in plan.selection_table],
            [(0, 2, 5, 2, "test"), (1, 4, 5, 4, "test"), (2, 6, 5, 5, "test")],
        )

    def test_same_random_state_reproduces_indices(self):
        labels = numpy.repeat(numpy.arange(4), 20)

        first = build_split_sample_plan(labels, 8, 4, 42, "train")
        second = build_split_sample_plan(labels, 8, 4, 42, "train")

        numpy.testing.assert_array_equal(first.selected_indices, second.selected_indices)

    def test_different_random_state_can_change_indices(self):
        labels = numpy.repeat(numpy.arange(4), 20)

        first = build_split_sample_plan(labels, 8, 4, 42, "train")
        second = build_split_sample_plan(labels, 8, 4, 43, "train")

        self.assertFalse(numpy.array_equal(first.selected_indices, second.selected_indices))

    def test_changing_samples_per_class_recalculates_plan(self):
        labels = numpy.repeat(numpy.arange(2), 10)

        first = build_split_sample_plan(labels, 3, 2, 42, "train")
        second = build_split_sample_plan(labels, 4, 2, 42, "train")

        self.assertNotEqual(first.total_rows, second.total_rows)
        self.assertEqual(first.metadata["cache_key"]["samples_per_class"], 3)
        self.assertEqual(second.metadata["cache_key"]["samples_per_class"], 4)

    def test_train_and_test_plans_do_not_share_indices(self):
        train_labels = numpy.repeat(numpy.arange(2), 10)
        test_labels = numpy.repeat(numpy.arange(2), 8)

        train_plan = build_split_sample_plan(train_labels, 4, 2, 42, "train")
        test_plan = build_split_sample_plan(test_labels, 4, 2, 42, "test")

        self.assertIsNot(train_plan.selected_indices, test_plan.selected_indices)
        self.assertEqual(train_plan.metadata["cache_key"]["split"], "train")
        self.assertEqual(test_plan.metadata["cache_key"]["split"], "test")
        self.assertEqual(train_plan.metadata["cache_key"]["split_size"], 20)
        self.assertEqual(test_plan.metadata["cache_key"]["split_size"], 16)

    def test_out_of_bounds_indices_fail_before_indexing(self):
        x_values = numpy.zeros((81832, 2), dtype=numpy.float32)
        y_values = numpy.zeros((81832,), dtype=numpy.int64)
        bad_indices = numpy.array([0, 98070], dtype=numpy.int64)

        with self.assertRaisesRegex(IndexError, "98070.*out of bounds"):
            runner.load_selected_rows(numpy, x_values, y_values, bad_indices, seed=0)

    def test_all_strategy_does_not_materialize_selection_indices(self):
        labels = numpy.repeat(numpy.arange(2), 5)
        x_values = numpy.zeros((labels.shape[0], 2), dtype=numpy.float32)

        plan = build_split_sample_plan(labels, None, 2, 0, "train", strategy="all")
        selected_x, selected_y = runner.load_selected_rows(numpy, x_values, labels, plan.selected_indices, seed=0)

        self.assertIsNone(plan.selected_indices)
        self.assertIs(selected_x, x_values)
        self.assertIs(selected_y, labels)
        self.assertEqual(plan.total_rows, labels.shape[0])
        self.assertEqual(plan.metadata["selected_counts_by_class"], {"0": 5, "1": 5})


if __name__ == "__main__":
    unittest.main()
