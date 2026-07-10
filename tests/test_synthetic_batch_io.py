import json
import tempfile
import unittest
from pathlib import Path

import numpy

from Engine.DataIO.SyntheticBatchIO import SyntheticBatchReader
from Engine.DataIO.SyntheticBatchIO import SyntheticBatchWriter


class SyntheticBatchIOTest(unittest.TestCase):

    def test_writes_npy_batches_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SyntheticBatchWriter(
                root_dir=directory,
                num_classes=200,
                num_features=3,
                seed=42,
                model_name="variational",
                execution_mode="batches",
                output_format="npy_batches",
            )
            writer.write_batch(0, 0, numpy.ones((2, 3), dtype=numpy.float32))
            writer.write_batch(0, 1, numpy.full((1, 3), 2, dtype=numpy.float32))
            writer.write_batch(199, 0, numpy.full((2, 3), 3, dtype=numpy.float32))
            reader = writer.close()

            manifest_path = Path(directory) / "synthetic_batches" / "manifest.json"
            self.assertEqual(reader.manifest_path, manifest_path)
            with manifest_path.open() as manifest_file:
                manifest = json.load(manifest_file)

            self.assertEqual(manifest["total_rows"], 5)
            self.assertEqual(manifest["num_classes"], 200)
            self.assertEqual(manifest["features"], 3)
            self.assertEqual(manifest["model"], "variational")
            self.assertEqual(manifest["execution_mode"], "batches")
            self.assertEqual(len(manifest["batches_by_class"]["0"]), 2)
            self.assertEqual(manifest["batches_by_class"]["199"][0]["shape"], [2, 3])

            batches = list(reader.iter_batches())
            self.assertEqual([label for label, _ in batches], [0, 0, 199])
            self.assertEqual(sum(batch.shape[0] for _, batch in batches), 5)

    def test_single_npy_reader_uses_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SyntheticBatchWriter(
                root_dir=directory,
                num_classes=2,
                num_features=2,
                seed=7,
                model_name="adversarial",
                execution_mode="batches",
                output_format="single_npy",
            )
            writer.initialize_single_npy(total_rows=3)
            writer.write_batch(0, 0, numpy.array([[1, 2], [3, 4]], dtype=numpy.float32))
            writer.write_batch(1, 0, numpy.array([[5, 6]], dtype=numpy.float32))
            reader = writer.close()

            batches = list(SyntheticBatchReader(reader.manifest_path).iter_batches())

            self.assertEqual([batch.shape for _, batch in batches], [(2, 2), (1, 2)])
            numpy.testing.assert_array_equal(batches[0][1], [[1, 2], [3, 4]])
            numpy.testing.assert_array_equal(batches[1][1], [[5, 6]])


if __name__ == "__main__":
    unittest.main()

