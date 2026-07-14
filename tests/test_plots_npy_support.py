import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy

import plots


class PlotsNpySupportTest(unittest.TestCase):

    def test_load_dataset_processor_uses_numpy_for_npy(self):
        with tempfile.TemporaryDirectory() as directory:
            npy_path = Path(directory) / "train_x.npy"
            numpy.save(npy_path, numpy.zeros((4, 3), dtype=numpy.float32))

            with mock.patch.object(plots.CSVDataProcessor, "load_csv", side_effect=AssertionError("CSV loader used")):
                processor = plots.load_dataset_processor(str(npy_path))

        self.assertEqual(processor.get_number_columns(), 3)
        self.assertEqual(processor.get_features_by_label(0).shape, (4, 3))

    def test_main_does_not_look_for_legacy_synthetic_txt_when_batches_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "train_x.npy"
            numpy.save(dataset_path, numpy.zeros((4, 3), dtype=numpy.float32))
            results_path = root / "EvaluationResults" / "Results.json"
            results_path.parent.mkdir(parents=True)
            results_path.write_text("{}", encoding="utf-8")
            manifest_path = root / "DataGenerated" / "synthetic_batches" / "train" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps({
                    "total_rows": 0,
                    "num_classes": 3,
                    "features": 3,
                    "batches_by_class": {},
                    "format": "npy_batches",
                }),
                encoding="utf-8",
            )
            argv = [
                "plots.py",
                "--results",
                str(results_path),
                "--dataset",
                str(dataset_path),
                "--output_dir",
                str(root),
                "--model",
                "copy",
                "--folds",
                "1",
            ]

            with mock.patch("sys.argv", argv), \
                    mock.patch.object(plots, "PlotClassificationMetrics"), \
                    mock.patch.object(plots, "PlotDistanceMetrics"), \
                    mock.patch.object(plots, "PlotConfusionMatrix"), \
                    mock.patch.object(plots, "PlotTrainingCurve"), \
                    mock.patch.object(plots, "plot_heatmaps_from_dataset") as heatmap, \
                    mock.patch.object(plots, "plot_heatmaps_from_dataset_comparison") as comparison:
                plots.main()

        heatmap.assert_called_once()
        self.assertNotIn("DataOutput_K_fold", str(heatmap.call_args))
        comparison.assert_not_called()


if __name__ == "__main__":
    unittest.main()
