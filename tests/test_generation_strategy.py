import unittest
from argparse import Namespace

from main import SynDataGen


def _strategy_instance(strategy="single_conditional", classes_per_group=10, model_type="adversarial"):
    instance = object.__new__(SynDataGen)
    instance.arguments = Namespace(
        execution_mode="batches",
        generation_strategy=strategy,
        classes_per_group=classes_per_group,
        model_type=model_type,
    )
    instance._dictionary_metrics = {}
    return instance


class GenerationStrategyTest(unittest.TestCase):
    def test_partitioned_generation_only_for_batch_per_class_or_grouped(self):
        instance = _strategy_instance(strategy="single_conditional")
        self.assertFalse(instance._uses_partitioned_generation())

        instance.arguments.generation_strategy = "per_class"
        self.assertTrue(instance._uses_partitioned_generation())

        instance.arguments.execution_mode = "normal"
        self.assertFalse(instance._uses_partitioned_generation())

    def test_per_class_units_include_only_classes_with_requested_samples(self):
        instance = _strategy_instance(strategy="per_class")

        units = instance._partition_generation_units({0: 5, 1: 0, 2: 7})

        self.assertEqual(
            units,
            [
                {"unit_type": "class", "classes": [0]},
                {"unit_type": "class", "classes": [2]},
            ],
        )

    def test_grouped_units_follow_classes_per_group(self):
        instance = _strategy_instance(strategy="grouped_classes", classes_per_group=2)

        units = instance._partition_generation_units({0: 5, 1: 3, 2: 7, 3: 1, 4: 0})

        self.assertEqual(
            units,
            [
                {"unit_type": "group", "classes": [0, 1]},
                {"unit_type": "group", "classes": [2, 3]},
            ],
        )

    def test_split_generation_plan_preserves_effective_classes(self):
        instance = _strategy_instance(strategy="single_conditional")

        split_plan = instance._generation_metadata_for_split(
            {
                "classes": {0: 4, 2: 4},
                "number_classes": 3,
                "generation_batch_size": 16,
            },
            samples_per_class=2,
            split_name="train",
        )

        self.assertEqual(split_plan["classes"], {0: 2, 2: 2})
        self.assertNotIn(1, split_plan["classes"])

    def test_invalid_classes_per_group_fails_fast(self):
        instance = _strategy_instance(strategy="grouped_classes", classes_per_group=0)

        with self.assertRaisesRegex(ValueError, "classes_per_group"):
            instance._partition_generation_units({0: 5})

    def test_not_supported_records_metrics_before_raising(self):
        instance = _strategy_instance(strategy="per_class", model_type="ctgan")
        instance.get_evaluation_results_path = lambda: "/tmp/eval"
        instance.save_dictionary_to_json = lambda path: setattr(instance, "saved_metrics_path", path)

        with self.assertRaisesRegex(NotImplementedError, "not supported"):
            instance._raise_generation_strategy_not_supported(0, "not supported")

        self.assertEqual(instance.saved_metrics_path, "/tmp/eval/Results.json")
        self.assertEqual(
            instance._dictionary_metrics["GenerationStrategy"]["1-Fold"]["status"],
            "not_supported",
        )
        self.assertEqual(
            instance._dictionary_metrics["GenerationStrategy"]["1-Fold"]["model_type"],
            "ctgan",
        )


if __name__ == "__main__":
    unittest.main()
