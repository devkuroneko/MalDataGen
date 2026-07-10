import unittest

import numpy

from Engine.DataIO.LabelUtils import to_one_hot_batch


class OneHotBatchTest(unittest.TestCase):

    def test_three_classes(self):
        encoded = to_one_hot_batch(numpy.array([0, 2, 1]), num_classes=3)

        expected = numpy.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=numpy.float32,
        )
        numpy.testing.assert_array_equal(encoded, expected)

    def test_two_hundred_classes(self):
        encoded = to_one_hot_batch(numpy.array([0, 50, 199]), num_classes=200)

        self.assertEqual(encoded.shape, (3, 200))
        self.assertEqual(encoded.dtype, numpy.float32)
        self.assertEqual(float(encoded[0, 0]), 1.0)
        self.assertEqual(float(encoded[1, 50]), 1.0)
        self.assertEqual(float(encoded[2, 199]), 1.0)
        self.assertEqual(float(encoded.sum()), 3.0)

    def test_out_of_range_labels_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "num_classes=3"):
            to_one_hot_batch(numpy.array([0, 3]), num_classes=3)

    def test_dtype_float32_by_default(self):
        encoded = to_one_hot_batch([1], num_classes=2)

        self.assertEqual(encoded.dtype, numpy.float32)


if __name__ == "__main__":
    unittest.main()

