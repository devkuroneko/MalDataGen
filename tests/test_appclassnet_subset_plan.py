import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy

import run_appclassnet_top200 as runner


def _args(**overrides):
    base = {
        "class_subset": None,
        "num_classes_subset": 10,
        "dataset_split": "all",
        "use_mmap": True,
        "batch_size": 32,
        "eval_batch_size": 64,
        "generation_batch_size": 16,
        "eval_classifier": "decision_tree_subset",
        "batch_classifier_subset_size": 100,
        "min_samples_per_class_required": 1,
        "generation_strategy": "single_conditional",
        "classes_per_group": 10,
        "synthetic_control": "none",
        "max_train_samples": None,
        "max_samples_per_class": None,
        "train_samples_per_class": 2,
        "test_samples_per_class": 2,
        "synthetic_train_samples_per_class": 1,
        "synthetic_test_samples_per_class": 1,
        "generated_samples_per_class": 2,
        "real_class_count_policy": runner.DEFAULT_REAL_CLASS_COUNT_POLICY,
        "samples_per_class_scope": runner.DEFAULT_SAMPLES_PER_CLASS_SCOPE,
        "n_estimators": None,
        "max_depth": None,
        "max_samples": None,
        "class_weight": None,
        "full": False,
        "run_mode_effective": "demo",
        "dry_run_memory": False,
        "run_tr_tr_effective": False,
        "save_synthetic_format": "npy_batches",
        "materialize_synthetic": False,
        "allow_double_transform": False,
        "allow_scaler_refit": False,
        "inverse_transform_synthetic": True,
        "source_profile": "appclassnet_top200",
        "feature_transform": "preserve",
        "generator_transform": "preserve",
        "classifier_transform": "preserve",
        "evaluation_space": "source",
        "evaluation_mode": "all",
        "random_state": 0,
        "pipeline_effective": "all",
        "run_mode_origin": "test",
        "pipeline_origin": "test",
        "_execution_profile_parameters": {},
        "_effective_parameters": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class AppClassNetSubsetPlanTest(unittest.TestCase):

    def test_number_samples_plan_uses_effective_domain(self):
        plan = runner.build_number_samples_per_class_plan(7, num_classes=10)

        self.assertEqual(plan.split(",")[0], "0:7")
        self.assertEqual(plan.split(",")[-1], "9:7")
        self.assertNotIn("10:7", plan)

    def test_batch_command_materializes_subset_and_passes_ten_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory) / "raw"
            raw_root.mkdir()
            labels = numpy.tile(numpy.arange(12, dtype=numpy.int64), 2)
            for split_name in ("train", "valid", "test"):
                numpy.save(raw_root / f"{split_name}_x.npy", numpy.zeros((labels.shape[0], 20), dtype=numpy.float32))
                numpy.save(raw_root / f"{split_name}_y.npy", labels)

            combination = {
                "model_type": "variational",
                "effective_number_k_folds": 1,
                "requested_number_k_folds": 2,
                "number_k_folds_origin": "test",
                "origin": "test",
                "split_mode": "provided",
                "variational_autoencoder_number_classes": 200,
            }
            original_results_root = runner.RESULTS_ROOT
            try:
                runner.RESULTS_ROOT = Path(directory) / "results"
                command = runner.build_batch_main_command(
                    "python",
                    raw_root,
                    Path(directory) / "out",
                    combination,
                    20,
                    "continuous",
                    _args(),
                )
            finally:
                runner.RESULTS_ROOT = original_results_root

            num_classes_index = command.index("--num_classes") + 1
            vae_classes_index = command.index("--variational_autoencoder_number_classes") + 1
            number_samples_index = command.index("--number_samples_per_class") + 1
            train_x_index = command.index("--train_x_path") + 1

            self.assertEqual(command[num_classes_index], "10")
            self.assertEqual(command[vae_classes_index], "10")
            self.assertEqual(command[number_samples_index].split(",")[-1], "9:2")
            self.assertNotIn("--num_classes_subset", command)
            self.assertIn(str(Path(directory) / "results" / "batches" / "subsets"), command[train_x_index])


if __name__ == "__main__":
    unittest.main()
