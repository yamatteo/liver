from __future__ import annotations

import random
from pathlib import Path

import nibabel
import torch
from torch import Tensor
from torch.nn import functional

from .decorators import integer, floating, bidimensional, tridimensional, channels, batch
from .strict_tensor import StrictTensor


@integer
@bidimensional
class Slice(StrictTensor):
    def is_empty(self) -> bool:
        return bool(torch.all(self == 0))


@integer
@bidimensional
class Mask(StrictTensor):
    def __init__(self, data, **kwargs):
        super(Mask, self).__init__(data, **kwargs)
        if torch.min(self) < 0 or torch.max(self) > 1:
            raise ValueError("Mask indices can only be 0 or 1.")


@channels(["b", "a", "v", "t"])
@tridimensional
@integer
class Scan(StrictTensor):
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

@tridimensional
@integer
class Segm(StrictTensor):

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


@channels(["b", "a", "v", "t", "segm"])
@tridimensional
@integer
class Bundle(StrictTensor):
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


@channels(["b", "a", "v", "t", "aid", "segm"])
@tridimensional
@integer
class ExtraBundle(StrictTensor):
    def separate(self) -> tuple[Scan, Segm, Segm]:
        return Scan(self[0:4]), Segm(self[4]), Segm(self[5])

    # @classmethod
    # def from_join(cls, scan: Scan, segm: Segm) -> Bundle:
    #     if scan.size("Z") != segm.size("Z"):
    #         raise ValueError("Scan and segm have different lengths along z axis.")
    #     a, b = scan.boundaries()
    #     return cls(torch.cat([
    #         scan[..., a:b].torch(),
    #         segm[..., a:b].torch().unsqueeze(0)
    #     ], dim=0))


@channels(["b", "a", "v", "t"])
@tridimensional
@floating
class FloatScan(StrictTensor):
    def get_plane(self, phase: str, z: int) -> Slice:
        return Scan.from_float(self).get_plane(phase, z)

    @classmethod
    def from_int(cls, scan: Scan) -> FloatScan:
        return cls(scan.to(dtype=torch.float32))


@channels(["backg", "liver", "tumor"])
@tridimensional
@floating
class FloatSegm(StrictTensor):
    def get_mask(self, channel: str, z: int) -> Mask:
        return Segm.from_float(self).get_mask(channel, z)

    @classmethod
    def from_int(cls, segm: Segm) -> FloatSegm:
        return cls(functional.one_hot(segm.to(dtype=torch.int64), 3).permute(3, 0, 1, 2).float())


@channels(["b", "a", "v", "t", "backg", "liver", "tumor"])
@tridimensional
@floating
class FloatBundle(StrictTensor):
    # def separate(self) -> tuple[FloatScan, FloatSegm]:
    #     return FloatScan(self[0:4]), FloatSegm(self[4:7])

    @classmethod
    def from_int(cls, bundle: Bundle) -> FloatBundle:
        scan, segm = bundle.separate()
        return cls(torch.cat([
            FloatScan.from_int(scan).torch(),
            FloatSegm.from_int(segm).torch()
        ], dim=0))


@batch
class FloatScanBatch(FloatScan):
    def get_plane(self, n: int, phase: str, z: int) -> Slice:
        return FloatScan(self.select_item(n)).get_plane(phase, z)


@batch
class FloatSegmBatch(FloatSegm):
    def get_mask(self, n: int, channel: str, z: int) -> Mask:
        return FloatSegm(self.select_item(n)).get_mask(channel, z)

    def get_slice(self, n: int, z: int) -> Slice:
        fs = FloatSegm(self.select_item(n))
        s = Segm.from_float(fs)
        return Slice(s.select_section(z))

    @classmethod
    def from_int(cls, segm: Segm) -> FloatSegmBatch:
        x = cls(functional.one_hot(segm.to(dtype=torch.int64), 3).permute(0, 4, 1, 2, 3).float())
        return x

@batch
class FloatBundleBatch(FloatBundle):
    def separate(self) -> tuple[FloatScanBatch, FloatSegmBatch]:
        return FloatScanBatch(self[:, 0:4]), FloatSegmBatch(self[:, 4:7])

    @classmethod
    def from_list(cls, bundles: list[FloatBundle]) -> FloatBundleBatch:
        return cls(torch.stack(bundles, dim=0))



### Tests
import numpy as np
import unittest

class TestTensors(unittest.TestCase):
    def test_fixed_dtype(self):
        @integer
        class MockTensor(StrictTensor):
            pass

        # Correct type, no error
        t = MockTensor(np.ones((2, 3, 4), dtype=np.int16))

        # Wrong type, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((2, 3, 4)))

        self.assertIsInstance(t * 5, MockTensor)
        self.assertNotIsInstance(t * 0.5, MockTensor)  # If values are float it's not a MockTensor
        self.assertIsInstance(t * 0.5, Tensor)

    def test_fixed_shape(self):
        @tridimensional
        class MockTensor(StrictTensor):
            pass

        # Correct shape, no error
        t = MockTensor(np.zeros((4, 2, 7)))
        t = MockTensor(np.ones((4, 2, 13)))

        # Tensor.dim override
        self.assertEqual(t.dim(), 3)
        self.assertEqual(t.dim("X"), 0)
        self.assertEqual(t.dim("Y"), 1)
        self.assertEqual(t.dim("Z"), 2)
        self.assertRaises(ValueError, t.dim, "?")

        # Tensor.size override
        self.assertEqual(t.size(), torch.Size((4, 2, 13)))
        self.assertEqual(t.size(0), 4)
        self.assertEqual(t.size("X"), 4)
        self.assertEqual(t.size(2), 13)
        self.assertEqual(t.size("Z"), 13)
        self.assertRaises(ValueError, t.size, "?")

        # Wrong shape, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 3, 5, 5)))
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 2)))

        # Changes of shape loose strictness
        self.assertIsInstance(torch.mean(t, dim=0, keepdim=True), MockTensor)
        self.assertIsInstance(torch.mean(t, dim=0, keepdim=False), Tensor)

    def test_inheritance(self):
        @channels(["a", "b", "c"])
        @tridimensional
        @floating
        class Mock(StrictTensor):
            pass

        @batch
        class MockBatch(Mock):
            pass

        t = Mock(torch.zeros((3, 8, 8, 5)))
        # self.assertEqual({"C": 3, "X": None, "Y": None, "Z": None}, t.fixed_shape)
        m = torch.nn.AvgPool3d((2, 2, 1))
        t = m(t)
        # t.__init__()
        self.assertEqual(t.fixed_shape, {"C": 3, "X": None, "Y": None, "Z": None})
        self.assertIsInstance(t, Mock)

        t = t.unsqueeze(0)
        self.assertIsInstance(t, Tensor)
        self.assertNotIsInstance(t, Mock)

        t = MockBatch(t)
        self.assertIsInstance(t, MockBatch)

    def test_get_plane(self):
        t = Scan(np.empty([4, 8, 8, 5], dtype=np.int16))
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
