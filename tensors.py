from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import functional


class IntBundle(Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.int16)

    def __init__(self, *args, **kwargs):
        super().__init__()
        assert len(self.shape) == 4, \
            f"IntBundle should be a (C, H, W, D) shaped vector, got {self.shape}."
        assert self.size(0) == 5, \
            f"Channels should be 5: four phases and a ground-truth segmentation, got {self.shape}."
        assert self.size(1) == self.size(2) == 512, \
            f"Scans expected to be 512x512xD, got {self.shape}."

    def to_float_batch_bundle(self) -> "FloatBatchBundle":
        return FloatBatchBundle(torch.cat([
            self[0:4].float().unsqueeze(0),
            functional.one_hot(
                self[4].long(),
                3
            ).permute(3, 0, 1, 2).unsqueeze(0).float()
        ], dim=1))


class FloatBatchBundle(Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)

    def __init__(self, *args, **kwargs):
        super(FloatBatchBundle, self).__init__()
        assert len(self.shape) == 5, \
            f"FloatBatchBundle should be a (N, C, H, W, D) shaped vector, got {self.shape}."
        assert self.size(1) == 7, \
            f"Channels should be 7: four phases and three for one-hot ground-truth segmentation, got {self.shape}."
        assert self.size(2) == self.size(3) == 512, \
            f"Scans expected to be 512x512xD, got {self.shape}."

    def separate(self) -> tuple["ScanBatch", "FloatSegmBatch"]:
        return ScanBatch(self[:, 0:4]), FloatSegmBatch(self[:, 4:7])

    def dimensional_slices(self, thickness: int, dim: int) -> Iterator["FloatBatchBundle"]:
        """Iterate over slices of self along dimension `dim`.

         Slices may overlap if `thicknes` does not divide `self.size(dim)` evenly.
         If `self.size(dim)` is less than `thickness`, yields only `self`."""
        length = self.size(dim)
        if length <= thickness:
            yield self
            return
        num_slices = math.ceil(length / thickness)
        for j in range(num_slices):
            i = int(j * (length - thickness) / (num_slices - 1))
            yield torch.narrow(self, dim, i, thickness)

    def slices(self, shape: tuple[int, int, int]) -> Iterator["FloatBatchBundle"]:
        """Iterate over FloatBatchBundle slices of given shape. Possibly overlapping."""
        for h_slice in self.dimensional_slices(shape[0], 2):
            for w_slice in h_slice.dimensional_slices(shape[1], 3):
                yield from w_slice.dimensional_slices(shape[2], 4)


class ScanBatch(Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)

    def __init__(self, *args, **kwargs):
        super(ScanBatch, self).__init__()
        assert len(self.shape) == 5, \
            f"ScanBatch should be a (N, C, H, W, D) shaped vector, got {self.shape}."
        assert self.size(1) == 4, \
            f"Channels should be four phases, got {self.shape}."


class FloatSegmBatch(Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)

    def __init__(self, *args, **kwargs):
        super(FloatSegmBatch, self).__init__()
        assert len(self.shape) == 5, \
            f"FloatSegmBatch should be a (N, C, H, W, D) shaped vector, got {self.shape}."
        assert self.size(1) == 3, \
            f"Channels should be three for one-hot segmentation, got {self.shape}."


class IntSegmTensor(Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.long)

    def __init__(self, *args, **kwargs):
        super(IntSegmTensor, self).__init__()
        assert len(self.shape) == 5, \
            f"ScanTensor should be a (N, C, H, W, D) shaped vector, got {self.shape}."
        assert self.size(1) == 3, \
            f"Channels should be three for one-hot segmentation, got {self.shape}."
