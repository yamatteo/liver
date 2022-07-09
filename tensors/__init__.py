from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.nn import functional

from .batch import Batch
from .dimensional import ShapedTensor, Bidimensional, Tridimensional
from .multichannel import MultiChannel, Phases, ColdBundle, HotSegm, HotBundle
from .typed import TypedTensor, Integer, Floating


#
#
# class Plane(ShapedTensor):
#     fixed_shape = {"H": None, "W": None}
#
#     def is_empty(self) -> bool:
#         return bool(torch.all(self == 0))
#
#
# class Volume(ShapedTensor):
#     fixed_shape = {"H": None, "W": None, "D": None}
#
#     def plane(self, z: int) -> Plane:
#         return Plane(self[:, :, z])
#
#     def with_neighbours(self, minimum: int = 1, kernel_size: tuple[int, int, int] = (3, 3, 3)) -> Volume:
#         kx, ky, kz = kernel_size
#         assert all(k % 2 == 1 for k in kernel_size)
#         kernel = torch.nn.Conv3d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=kernel_size,
#             padding=(kx // 2, ky // 2, kz // 2),
#             device=self.device,
#             dtype=torch.float32,
#         )
#         kernel.bias = Parameter(torch.tensor([0.1 - minimum]), requires_grad=False)
#         kernel.weight = Parameter(torch.ones((1, 1, *kernel_size)), requires_grad=False)
#         return Volume(
#             torch.clamp(kernel(self.unsqueeze(0).to(dtype=torch.float32)).squeeze(0), 0, 1).to(dtype=self.dtype))
#
#     def set_difference(self, other: Volume) -> Volume:
#         return Volume(torch.clamp((self - other), 0, 1).to(dtype=torch.int16))
#
#     @classmethod
#     def masks(cls, segm: Segm) -> tuple[Volume, Volume, Volume]:
#         orig_liver = segm.liver_mask()
#         tumor = segm.tumor_mask()
#         ext_tumor = tumor.with_neighbours(2, (9, 9, 1))
#         liver = orig_liver.set_difference(ext_tumor)
#         perit = orig_liver.set_difference(liver)
#         return liver, perit, tumor
#
#
class Slice(Integer, Bidimensional):
    def is_empty(self) -> bool:
        return bool(torch.all(self == 0))


class Mask(Integer, Bidimensional):

    def __init__(self, *args, **kwargs):
        super(Mask, self).__init__()
        if torch.min(self) < 0 or torch.max(self) > 1:
            raise ValueError("Mask indices can only be 0 or 1.")


class Scan(Integer, Tridimensional, Phases):
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


class Segm(Integer, Tridimensional):

    def __init__(self, *args, **kwargs):
        super(Segm, self).__init__()
        if torch.min(self) < 0 or torch.max(self) > 2:
            raise ValueError("Segmentation indices can only be 0, 1 or 2.")

    def get_mask(self, channel: str, z: int) -> Mask:
        channel = ["backg", "liver", "tumor"].index(channel)
        return Mask((self.select_section(z) == channel).to(dtype=torch.int16))

    @classmethod
    def from_float(cls, segm: FloatSegm) -> Segm:
        dim = segm.dim("C")
        return cls(torch.argmax(segm, dim=dim).to(dtype=torch.int16))


class Bundle(Integer, Tridimensional, ColdBundle):
    def separate(self) -> tuple[Scan, Segm]:
        return Scan(self[0:4]), Segm(self[4])

    @classmethod
    def from_join(cls, scan: Scan, segm: Segm) -> Bundle:
        if scan.size("Z") != segm.size("Z"):
            raise ValueError("Scan and segm have different lengths along z axis.")
        a, b = scan.boundaries()
        return cls(torch.cat([
            scan[..., a:b],
            segm[..., a:b]
        ], dim=0))


class FloatScan(Floating, Tridimensional, Phases):
    def get_plane(self, phase: str, z: int) -> Slice:
        return Scan.from_float(self).get_plane(phase, z)

    @classmethod
    def from_int(cls, scan: Scan) -> FloatScan:
        return cls(scan.to(dtype=torch.float32))


class FloatSegm(Floating, Tridimensional, HotSegm):
    def get_mask(self, channel: str, z: int) -> Mask:
        return Segm.from_float(self).get_mask(channel, z)

    @classmethod
    def from_int(cls, segm: Segm) -> FloatSegm:
        return cls(functional.one_hot(segm, 3).permute(3, 0, 1, 2).float())


class FloatBundle(Floating, Tridimensional, HotBundle):
    def separate(self) -> tuple[FloatScan, FloatSegm]:
        return FloatScan(self[0:4]), FloatSegm(self[4:7])

    @classmethod
    def from_int(cls, bundle: Bundle) -> FloatBundle:
        scan, segm = bundle.separate()
        return cls(torch.cat([
            FloatScan.from_int(scan),
            FloatSegm.from_int(segm)
        ], dim=0))


class FloatScanBatch(Floating, Tridimensional, Phases, Batch):
    def get_plane(self, n: int, phase: str, z: int) -> Slice:
        return self.select_item(n).get_plane(phase, z)


class FloatSegmBatch(Floating, Tridimensional, HotSegm, Batch):
    def get_mask(self, n: int, channel: str, z: int) -> Mask:
        return self.select_item(n).get_mask(channel, z)


class FloatBatchBundle(FloatBundle, Batch):
    def separate(self) -> tuple[FloatScanBatch, FloatSegmBatch]:
        return FloatScanBatch(self[:, 0:4]), FloatSegmBatch(self[:, 4:7])

    # def dimensional_slices(self, thickness: int, dim: int) -> Iterator["FloatBatchBundle"]:
    #     """Iterate over slices of self along dimension `dim`.
    #
    #      Slices may overlap if `thickness` does not divide `self.size(dim)` evenly.
    #      If `self.size(dim)` is less than `thickness`, yields only `self`."""
    #     length = self.size(dim)
    #     if length <= thickness:
    #         yield self
    #         return
    #     num_slices = math.ceil(length / thickness)
    #     for j in range(num_slices):
    #         i = int(j * (length - thickness) / (num_slices - 1))
    #         yield torch.narrow(self, dim, i, thickness)
    #
    # def slices(self, shape: tuple[int, int, int]) -> Iterator["FloatBatchBundle"]:
    #     """Iterate over FloatBatchBundle slices of given shape. Possibly overlapping."""
    #     for h_slice in self.dimensional_slices(shape[0], 2):
    #         for w_slice in h_slice.dimensional_slices(shape[1], 3):
    #             yield from w_slice.dimensional_slices(shape[2], 4)


# def distance_from(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
#     items = {}
#     asyml1, asyml1_items = self.asyml1_df(other)
#     cross, cross_items = self.cross_entropy_df(other)
#     items.update(asyml1_items)
#     items.update(cross_items)
#     return cross, items
#
# def asyml1_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
#     channel_distances = torch.mean(
#         functional.relu(functional.softmax(other, dim=1) - functional.softmax(self, dim=1)),
#         dim=(2, 3, 4)
#     )
#     channel_weights = torch.tensor([[1, 5, 20]]).to(device=self.device)
#     items = {
#         "back": torch.mean(channel_distances[:, 0]).item(),
#         "livr": torch.mean(channel_distances[:, 1]).item(),
#         "tumr": torch.mean(channel_distances[:, 2]).item(),
#     }
#     return BatchDistance(torch.sum(channel_weights * channel_distances, dim=1)), items
#
# def cross_entropy_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
#     batch_distances = BatchDistance(torch.mean(
#         functional.cross_entropy(self, target=other, reduction='none'),
#         dim=(1, 2, 3)
#     ))
#     items = {
#         "cros": torch.mean(batch_distances).item(),
#     }
#     return batch_distances, items

# def as_int(self) -> IntSegmBatch:
#     return IntSegmBatch(torch.argmax(self, dim=1))


######################################################################################################
#


### Tests
import numpy as np
import unittest


class TestInheritance(unittest.TestCase):
    def test_inheritance(self):
        class ScanBatch(Tridimensional, MultiChannel, Batch, Integer):
            fixed_channels = ("b", "a", "v", "t")

        t = ScanBatch(torch.zeros((12, 4, 8, 8, 5), dtype=torch.int16))
        t = 0


class TestTensors(unittest.TestCase):
    def test_fixed_dtype(self):
        class MockTensor(Integer):
            pass

        # Correct type, no error
        t = MockTensor(np.ones((2, 3, 4), dtype=np.int16))

        # Wrong type, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((2, 3, 4)))

    def test_fixed_shape(self):
        class MockTensor(Tridimensional):
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


class TestScan(unittest.TestCase):
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
