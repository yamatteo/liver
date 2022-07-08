from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import functional, Parameter


class ShapedTensor(Tensor):
    """Class of tensor that checks its shape and data type on creation."""
    fixed_shape = ...
    fixed_dtype = ...

    def __new__(cls, data, *args, **kwargs):
        # From pytorch documentation:
        #
        #   torch.as_tensor(data)
        #
        # Converts data into a tensor, sharing data and preserving autograd history if possible.
        # If data is a NumPy array then a tensor is constructed using torch.from_numpy().
        _t = torch.as_tensor(data)
        _t.__class__ = cls
        return _t

    def __init__(self, *args, **kwargs):
        super(ShapedTensor, self).__init__()
        if self.fixed_shape is not ...:
            shape = self.shape
            error_message = (
                    f"{type(self).__name__} should be a ("
                    + ', '.join([
                key if value is None else f"{key}={value}"
                for key, value in self.fixed_shape.items()
            ])
                    + f") shaped vector, got {tuple(shape)}."
            )
            if len(shape) != len(self.fixed_shape):
                raise ValueError(error_message)
            for value, fixed in zip(shape, self.fixed_shape.values()):
                if fixed is not None and fixed != value:
                    raise ValueError(error_message)
        if self.fixed_dtype is not ... and self.fixed_dtype != self.dtype:
            raise ValueError(f"{type(self).__name__} expect values of type {self.fixed_dtype}, got {self.dtype}")

    def size(self, dim: None | int | str = None) -> int | torch.Size:
        if dim is None:
            return super().size()
        if isinstance(dim, str):
            dim = list(self.fixed_shape).index(dim)
        return super().size(dim)


class Plane(ShapedTensor):
    fixed_shape = {"H": None, "W": None}

    def is_empty(self) -> bool:
        return bool(torch.all(self == 0))


class Volume(ShapedTensor):
    fixed_shape = {"H": None, "W": None, "D": None}

    def plane(self, z: int) -> Plane:
        return Plane(self[:, :, z])

    def with_neighbours(self, minimum: int = 1, kernel_size: tuple[int, int, int] = (3, 3, 3)) -> Volume:
        kx, ky, kz = kernel_size
        assert all(k % 2 == 1 for k in kernel_size)
        kernel = torch.nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kx // 2, ky // 2, kz // 2),
            device=self.device,
            dtype=torch.float32,
        )
        kernel.bias = Parameter(torch.tensor([0.1 - minimum]), requires_grad=False)
        kernel.weight = Parameter(torch.ones((1, 1, *kernel_size)), requires_grad=False)
        return Volume(
            torch.clamp(kernel(self.unsqueeze(0).to(dtype=torch.float32)).squeeze(0), 0, 1).to(dtype=self.dtype))

    def set_difference(self, other: Volume) -> Volume:
        return Volume(torch.clamp((self - other), 0, 1).to(dtype=torch.int16))

    @classmethod
    def masks(cls, segm: Segm) -> tuple[Volume, Volume, Volume]:
        orig_liver = segm.liver_mask()
        tumor = segm.tumor_mask()
        ext_tumor = tumor.with_neighbours(1, (7, 7, 1))
        liver = orig_liver.set_difference(ext_tumor)
        perit = orig_liver.set_difference(liver)
        return liver, perit, tumor


class Scan(ShapedTensor):
    fixed_shape = {"C": 4, "H": None, "W": None, "D": None}
    fixed_dtype = torch.int16

    def phase(self, phase: int | str) -> Volume:
        if isinstance(phase, str):
            phase = {
                "b": 0,
                "a": 1,
                "v": 2,
                "t": 3
            }[phase]
        return Volume(self[phase, :, :, :])

    def boundaries(self) -> tuple[int, int]:
        ph_range = range(self.size("C"))
        z_range = range(self.size("D"))
        a = 0
        for z in z_range:
            a = z
            if not any(self.phase(ph).plane(z).is_empty() for ph in ph_range):
                break
        b = self.size("D")
        for z in reversed(z_range):
            if not any(self.phase(ph).plane(z).is_empty() for ph in ph_range):
                break
            b = z
        return a, b

    def as_float(self) -> FloatScan:
        return FloatScan(self.to(dtype=torch.float32))


class Segm(Volume, ShapedTensor):
    fixed_shape = {"H": None, "W": None, "D": None}
    fixed_dtype = torch.int16

    def __init__(self, *args, **kwargs):
        super(Segm, self).__init__()
        if torch.min(self) < 0 or torch.max(self) > 2:
            raise ValueError("Segmentation indices can only be 0, 1 or 2.")

    def as_float(self) -> FloatSegm:
        return FloatSegm(functional.one_hot(self, 3).permute(3, 0, 1, 2).float())

    def liver_mask(self) -> Volume:
        return Volume((self == 1).to(dtype=torch.int16))

    def tumor_mask(self) -> Volume:
        return Volume((self == 2).to(dtype=torch.int16))


class Bundle(ShapedTensor):
    fixed_shape = {"C": 5, "H": None, "W": None, "D": None}
    fixed_dtype = torch.int16

    @classmethod
    def __from(cls, scan: Scan, segm: Segm) -> Bundle:
        if scan.size("D") != segm.size("D"):
            raise ValueError("Scan and segm have different lengths along z axis.")
        a, b = scan.boundaries()
        return Bundle(torch.cat([
            scan[..., a:b],
            segm[..., a:b]
        ], dim=0))

    def as_float(self) -> FloatBundle:
        return FloatBundle(torch.cat([
            self[0:4].float(),
            Segm(self[4]).as_float()
        ], dim=0))


class FloatScan(Scan, ShapedTensor):
    fixed_shape = {"C": 4, "H": None, "W": None, "D": None}
    fixed_dtype = torch.float32

    def as_int(self) -> Scan:
        return Scan(self.to(dtype=torch.int16))


class FloatSegm(ShapedTensor):
    fixed_shape = {"C": 3, "H": None, "W": None, "D": None}
    fixed_dtype = torch.float32

    def as_int(self) -> Segm:
        return Segm(torch.argmax(self, dim=0).to(dtype=torch.int16))


class FloatBundle(ShapedTensor):
    fixed_shape = {"C": 7, "H": None, "W": None, "D": None}
    fixed_dtype = torch.float32

    def separate(self) -> tuple[Scan, FloatSegm]:
        return Scan(self[0:4]), FloatSegm(self[4:7])

    def dimensional_slices(self, thickness: int, dim: int) -> Iterator[FloatBundle]:
        """Iterate over slices of self along dimension `dim`.

         Slices may overlap if `thickness` does not divide `self.size(dim)` evenly.
         If `self.size(dim)` is less than `thickness`, yields only `self`."""
        length = self.size(dim)
        if length <= thickness:
            yield self
            return
        num_slices = math.ceil(length / thickness)
        for j in range(num_slices):
            i = int(j * (length - thickness) / (num_slices - 1))
            yield torch.narrow(self, dim, i, thickness)

    def slices(self, shape: tuple[int, int, int]) -> Iterator[FloatBundle]:
        """Iterate over FloatBundle slices of given shape. Possibly overlapping."""
        H, W, D = 1, 2, 3
        for h_slice in self.dimensional_slices(shape[0], H):
            for w_slice in h_slice.dimensional_slices(shape[1], W):
                yield from w_slice.dimensional_slices(shape[2], D)


class FloatScanBatch(ShapedTensor):
    fixed_shape = {"N": None, "C": 4, "H": None, "W": None, "D": None}
    fixed_dtype = torch.float32


class FloatSegmBatch(ShapedTensor):
    fixed_shape = {"N": None, "C": 3, "H": None, "W": None, "D": None}
    fixed_dtype = torch.float32

    def distance_from(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
        items = {}
        asyml1, asyml1_items = self.asyml1_df(other)
        cross, cross_items = self.cross_entropy_df(other)
        items.update(asyml1_items)
        items.update(cross_items)
        return cross, items

    def asyml1_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
        channel_distances = torch.mean(
            functional.relu(functional.softmax(other, dim=1) - functional.softmax(self, dim=1)),
            dim=(2, 3, 4)
        )
        channel_weights = torch.tensor([[1, 5, 20]]).to(device=self.device)
        items = {
            "back": torch.mean(channel_distances[:, 0]).item(),
            "livr": torch.mean(channel_distances[:, 1]).item(),
            "tumr": torch.mean(channel_distances[:, 2]).item(),
        }
        return BatchDistance(torch.sum(channel_weights * channel_distances, dim=1)), items

    def cross_entropy_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
        batch_distances = BatchDistance(torch.mean(
            functional.cross_entropy(self, target=other, reduction='none'),
            dim=(1, 2, 3)
        ))
        items = {
            "cros": torch.mean(batch_distances).item(),
        }
        return batch_distances, items

    def as_int(self) -> IntSegmBatch:
        return IntSegmBatch(torch.argmax(self, dim=1))


######################################################################################################
#
# class FloatBatchBundle(ShapedTensor):
#     fixed_shape = {"N": None, "C": 7, "H": None, "W": None, "D": None}
#     fixed_dtype = torch.float32
#
#     def separate(self) -> tuple[FloatScanBatch, FloatSegmBatch]:
#         return FloatScanBatch(self[:, 0:4]), FloatSegmBatch(self[:, 4:7])
#
#     # def dimensional_slices(self, thickness: int, dim: int) -> Iterator["FloatBatchBundle"]:
#     #     """Iterate over slices of self along dimension `dim`.
#     #
#     #      Slices may overlap if `thickness` does not divide `self.size(dim)` evenly.
#     #      If `self.size(dim)` is less than `thickness`, yields only `self`."""
#     #     length = self.size(dim)
#     #     if length <= thickness:
#     #         yield self
#     #         return
#     #     num_slices = math.ceil(length / thickness)
#     #     for j in range(num_slices):
#     #         i = int(j * (length - thickness) / (num_slices - 1))
#     #         yield torch.narrow(self, dim, i, thickness)
#     #
#     # def slices(self, shape: tuple[int, int, int]) -> Iterator["FloatBatchBundle"]:
#     #     """Iterate over FloatBatchBundle slices of given shape. Possibly overlapping."""
#     #     for h_slice in self.dimensional_slices(shape[0], 2):
#     #         for w_slice in h_slice.dimensional_slices(shape[1], 3):
#     #             yield from w_slice.dimensional_slices(shape[2], 4)
#

### Tests
import numpy as np
import unittest


class TestSuit(unittest.TestCase):
    def test_fixed_dtype(self):
        class MockTensor(ShapedTensor):
            fixed_dtype = torch.int64

        # Correct type, no error
        t = MockTensor(np.ones((2, 3, 4), dtype=int))

        # Wrong type, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((2, 3, 4)))

    def test_fixed_shape(self):
        class MockTensor(ShapedTensor):
            fixed_shape = {"A": 4, "B": 2, "C": None, "D": None}

        # Correct shape, no error
        t = MockTensor(np.zeros((4, 2, 7, 11)))
        t = MockTensor(np.ones((4, 2, 13, 17)))
        self.assertEqual(t.size(), torch.Size((4, 2, 13, 17)))
        self.assertEqual(t.size(0), 4)
        self.assertEqual(t.size("A"), 4)
        self.assertEqual(t.size(2), 13)
        self.assertEqual(t.size("C"), 13)

        # Wrong shape, raises ValueError
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 3, 5, 5)))
        self.assertRaises(ValueError, MockTensor, np.zeros((4, 2, 5)))

    def test_scan_boundaries(self):
        random_scan = np.random.uniform(-400, 1000, (4, 8, 8, 11)).astype(np.int16)
        # Phase b is zeroed on planes 0 and 1
        random_scan[0, :, :, 0:2] = 0
        # Phase a is zeroed on planes 7 to 10
        random_scan[1, :, :, 7:11] = 0
        t = Scan(random_scan)
        # The scan is interesting only in planes [2, 7[
        self.assertEqual((2, 7), t.boundaries())
