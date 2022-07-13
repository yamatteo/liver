from __future__ import  annotations

import torch
from torch import Tensor

from.base import BaseWrappedTensor
from .decorators import short, floating, bidimensional, tridimensional, channels, batch

@short
@tridimensional
@channels(["b", "a", "v", "t"])
class Scan(BaseWrappedTensor):
    def get_plane(self, phase: str, z: int) -> Slice:
        return Slice(self.select_channel(phase).select_section(z))

    def boundaries(self) -> tuple[int, int]:
        phases = self.fixed_channels
        z_range = range(self.size("Z"))
        a = 0
        for z in z_range:
            a = z
            if not any(self.get_plane(ph, z).is_empty() for ph in phases):
                break
        b = self.size("Z")
        for z in reversed(z_range):
            if not any(self.get_plane(ph, z).is_empty() for ph in phases):
                break
            b = z
        return a, b

    @classmethod
    def from_float(cls, scan: FloatScan) -> Scan:
        return cls(scan.to(dtype=torch.int16))

    @classmethod
    def from_niigz(cls, path: Path) -> Scan:
        return Scan(torch.stack([
            torch.tensor(np.array(nibabel.load(
                path / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16))
            for phase in ["b", "a", "v", "t"]
        ]))

@short
@tridimensional
class Segm(BaseWrappedTensor):

    def __init__(self, data, **kwargs):
        super(Segm, self).__init__(data, **kwargs)
        if torch.min(self) < 0 or torch.max(self) > 2:
            raise ValueError("Segmentation indices can only be 0, 1 or 2.")

    def get_mask(self, channel: str, z: int) -> Mask:
        channel = ["backg", "liver", "tumor"].index(channel)
        return Mask((self.select_section(z) == channel).to(dtype=torch.int16))

    @classmethod
    def from_float(cls, segm: FloatSegm) -> Segm:
        dim = segm.dim("C")
        return cls(torch.argmax(segm, dim=dim).to(dtype=torch.int16))

    @classmethod
    def from_niigz(cls, path: Path) -> Segm:
        return Segm(torch.tensor(np.array(nibabel.load(
            path / f"segmentation.nii.gz"
        ).dataobj, dtype=np.int16)))


@short
@tridimensional
@channels(["b", "a", "v", "t", "segm"])
class Bundle(BaseWrappedTensor):
    def separate(self) -> tuple[Scan, Segm]:
        return Scan(self[0:4]), Segm(self[4])

    @classmethod
    def from_join(cls, scan: Scan, segm: Segm) -> Bundle:
        if scan.size("Z") != segm.size("Z"):
            raise ValueError("Scan and segm have different lengths along z axis.")
        a, b = scan.boundaries()
        return cls(torch.cat([
            scan[..., a:b].torch(),
            segm[..., a:b].torch().unsqueeze(0)
        ], dim=0))

@channels(["b", "a", "v", "t"])
@tridimensional
@floating
class FloatScan(BaseWrappedTensor):
    def get_plane(self, phase: str, z: int) -> Slice:
        return Scan.from_float(self).get_plane(phase, z)

    @classmethod
    def from_int(cls, scan: Scan) -> FloatScan:
        return cls(scan.to(dtype=torch.float32))


@channels(["backg", "liver", "tumor"])
@tridimensional
@floating
class FloatSegm(BaseWrappedTensor):
    def get_mask(self, channel: str, z: int) -> Mask:
        return Segm.from_float(self).get_mask(channel, z)

    @classmethod
    def from_int(cls, segm: Segm) -> FloatSegm:
        return cls(functional.one_hot(segm.to(dtype=torch.int64), 3).permute(3, 0, 1, 2).float())


@channels(["b", "a", "v", "t", "backg", "liver", "tumor"])
@tridimensional
@floating
class FloatBundle(BaseWrappedTensor):
    # def separate(self) -> tuple[FloatScan, FloatSegm]:
    #     return FloatScan(self[0:4]), FloatSegm(self[4:7])

    @classmethod
    def from_int(cls, bundle: Bundle) -> FloatBundle:
        scan, segm = bundle.separate()
        return cls(torch.cat([
            FloatScan.from_int(scan).torch(),
            FloatSegm.from_int(segm).torch()
        ], dim=0))

### Tests
import numpy as np
import unittest

class TestTensors(unittest.TestCase):
    def test_fixed_dtype(self):
        @short
        class MockTensor(BaseWrappedTensor):
            pass

        # Correct type, no error
        t = MockTensor(np.ones((2, 3, 4), dtype=np.int16))

        # Wrong type, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((2, 3, 4)))

    def test_fixed_shape(self):
        @tridimensional
        class MockTensor(BaseWrappedTensor):
            pass

        # Correct shape, no error
        x = MockTensor(np.zeros((4, 2, 7)))
        x = MockTensor(np.ones((4, 2, 13)))

        # Tensor.dim override
        self.assertEqual(x.dim(), 3)
        self.assertEqual(x.dim("X"), 0)
        self.assertEqual(x.dim("Y"), 1)
        self.assertEqual(x.dim("Z"), 2)
        self.assertRaises(ValueError, x.dim, "?")

        # Tensor.size override
        self.assertEqual(x.size(), torch.Size((4, 2, 13)))
        self.assertEqual(x.size(0), 4)
        self.assertEqual(x.size("X"), 4)
        self.assertEqual(x.size(2), 13)
        self.assertEqual(x.size("Z"), 13)
        self.assertRaises(ValueError, x.size, "?")

        # Wrong shape, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 3, 5, 5)))
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 2)))

    def test_inheritance(self):
        @floating
        @tridimensional
        @channels(["a", "b", "c"])
        class Mock(BaseWrappedTensor):
            pass

        @batch
        class MockBatch(Mock):
            pass

        x = Mock(torch.zeros((3, 8, 8, 5)))
        self.assertEqual(x.fixed_shape, {"C": ["a", "b", "c"], "X": ..., "Y": ..., "Z": ...})

        m = torch.nn.AvgPool3d((2, 2, 1))
        x = Mock(m(x.t))
        self.assertEqual(x.fixed_shape, {"C": ["a", "b", "c"], "X": ..., "Y": ..., "Z": ...})
        self.assertIsInstance(x, Mock)

        x = MockBatch(x.t.unsqueeze(0))
        self.assertIsInstance(x, MockBatch)
        self.assertEqual(x.fixed_shape, {"N": ..., "C": ["a", "b", "c"], "X": ..., "Y": ..., "Z": ...})

    def test_get_plane(self):
        t = Scan(np.ndarray([4, 8, 8, 5], dtype=np.int16))
        p = t.get_plane("v", 3)

        self.assertIsInstance(p, Slice)
        self.assertTrue(np.all(p.numpy() == t[2, :, :, 3].numpy()))

    def test_scan_boundaries(self):
        random_scan = np.random.uniform(-400, 1000, (4, 8, 8, 11)).astype(np.int16)
        # Phase b is zeroed on planes 0 and 1
        random_scan[0, :, :, 0:2] = 0
        # Phase a is zeroed on planes 7 to 10
        random_scan[1, :, :, 7:11] = 0
        t = Scan(random_scan)
        # The scan is interesting only in planes [2, 7[
        self.assertEqual((2, 7), t.boundaries())

    def test_fbb_from_list(self):
        l = [
            FloatBundle(np.random.uniform(-10, +10, [7, 16, 16, 13]).astype(dtype=np.float32))
            for _ in range(20)
        ]
        fbb = FloatBundleBatch.from_list(l)
        n = random.randint(0, 19)
        self.assertTrue(torch.all(fbb[n] == l[n]).item())

        fscb, fseb = fbb.separate()

    def test_fb_from_int(self):
        b = np.random.uniform(-1000, +1000, [5, 16, 16, 13]).astype(dtype=np.int16)
        b[4] = np.clip(b[4], 0, 2)
        b = Bundle(b)
        fb = FloatBundle.from_int(b)
