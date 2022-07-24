import pickle
import tempfile
import unittest
from pathlib import Path
from random import randint

import nibabel
import numpy as np

from dataset import ndarray


class TestNdarrayIO(unittest.TestCase):
    print("Here we are")
    def test_ndarray(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            #Mock some data
            data = (2000*np.random.rand(512, 512, 64) - 1000).astype(np.int16)
            diag = np.random.rand(4)
            matrix = np.diagflat(diag / diag[3])
            matrix[0:3, 3] = 1 + np.random.rand(3)
            image = nibabel.Nifti1Image(data, affine=matrix)

            # Test saving and loading originals
            ndarray.save_original(
                image=image,
                path=tmp_path,
                phase='v'
            )

            _data, _matrix = ndarray.load_original(
                path=tmp_path,
                phase='v'
            )
            self.assertIsInstance(_data, np.ndarray)
            self.assertIsInstance(_matrix, np.ndarray)
            self.assertEqual(_data.shape, (512, 512, 64))
            self.assertEqual(_matrix.shape, (4, 4))
            self.assertTrue(np.all(_data == data))  # Integer are preserved exactly
            self.assertTrue(np.allclose(_matrix, matrix))  # Floating point goes from 32bit to 16bit and back

            # Mock some data
            bottom, top, height = randint(0, 10), randint(54, 64), 64
            regs = {
                "v": data[..., bottom:top]
            }
            for phase in ["b", "a", "t"]:
                regs[phase] = data[..., bottom:top] + (10*np.random.rand(*data[..., bottom:top].shape)).astype(np.int16)

            ndarray.save_registereds(regs, path=tmp_path, affine=matrix, bottom=bottom, top=top, height=height)
            ndarray.save_scan(regs, path=tmp_path, affine=matrix, bottom=bottom, top=top, height=height)

            _data, _matrix = ndarray.load_registered(tmp_path, phase='a')
            self.assertIsInstance(_data, np.ndarray)
            self.assertIsInstance(_matrix, np.ndarray)
            self.assertEqual(_data.shape, (512, 512, 64))
            self.assertEqual(_matrix.shape, (4, 4))
            self.assertTrue(np.all(_data[..., bottom:top] == regs['a']))  # Integer are preserved exactly
            self.assertTrue(np.allclose(_matrix, matrix))  # Floating point goes from 32bit to 16bit and back

            _data = ndarray.load_scan(tmp_path)
            self.assertIsInstance(_data, np.ndarray)
            self.assertEqual(_data.shape, (4, 512, 512, top-bottom))
            self.assertTrue(np.all(_data[0] == regs['b']))
            self.assertTrue(np.all(_data[1] == regs['a']))
            self.assertTrue(np.all(_data[2] == regs['v']))
            self.assertTrue(np.all(_data[3] == regs['t']))

            _data = ndarray.load_scan_from_regs(tmp_path)
            self.assertIsInstance(_data, np.ndarray)
            self.assertEqual(_data.shape, (4, 512, 512, top-bottom))
            self.assertTrue(np.all(_data[0] == regs['b']))
            self.assertTrue(np.all(_data[1] == regs['a']))
            self.assertTrue(np.all(_data[2] == regs['v']))
            self.assertTrue(np.all(_data[3] == regs['t']))
