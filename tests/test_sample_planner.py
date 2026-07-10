import unittest
from types import SimpleNamespace

from Engine.DataIO.SamplePlanner import build_sample_plan_from_args
from Engine.DataIO.SamplePlanner import sample_plan_to_legacy_metadata


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


if __name__ == "__main__":
    unittest.main()
