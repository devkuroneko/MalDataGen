import unittest

import numpy

from Engine.Algorithms.Adversarial.AdversarialAlgorithm import AdversarialAlgorithm


class ConditionalGanLabelTest(unittest.TestCase):

    def test_one_hot_labels_remain_rank_two(self):
        algorithm = object.__new__(AdversarialAlgorithm)
        algorithm._number_classes = 3

        labels = algorithm._prepare_condition_labels(
            numpy.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=numpy.float32,
            )
        )

        self.assertEqual(tuple(labels.shape), (2, 3))

    def test_rank_one_labels_are_encoded_with_valid_class_width(self):
        algorithm = object.__new__(AdversarialAlgorithm)
        algorithm._number_classes = 3

        labels = algorithm._prepare_condition_labels(numpy.array([0, 2], dtype=numpy.int64))

        self.assertEqual(tuple(labels.shape), (2, 3))
        numpy.testing.assert_array_equal(numpy.argmax(labels.numpy(), axis=1), [0, 2])


if __name__ == "__main__":
    unittest.main()
