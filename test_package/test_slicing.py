import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.slices import overlapping_slices, fixed_shape_slices


class TestSlicing(unittest.TestCase):
    def test_overlapping_slices(self):
        t = np.ndarray([4, 4, 4])
        l = list(overlapping_slices(t, 8, 1))
        self.assertEqual(len(l), 1)
        self.assertTrue(np.all(l[0] == t))

        t = np.ndarray([4, 512, 512, 89])
        l = list(overlapping_slices(t, 8, 3))
        self.assertEqual(len(l), 12)
        self.assertTrue(np.all(l[0] == t[:, :, :, 0:8]))

    def test_fixed_shape_slices(self):
        t = np.ndarray([15, 4, 8, 10, 18])
        shape = (10, 10, 10)
        dims = (2, 3, 4)
        l = list(fixed_shape_slices(t, shape, dims))
        self.assertEqual(len(l), 2)
        self.assertTrue(all([s.shape == (15, 4, 8, 10, 10) for s in l]))
