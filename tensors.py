from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor as TorchTensor
from torch.nn import functional


class Tensor(TorchTensor):
    fixed_shape = ...

    def __new__(cls, x, *args, **kwargs):
        _t = torch.as_tensor(x)
        cls.check_shape(_t.shape)
        _t.__class__ = cls
        return _t

    @classmethod
    def check_shape(cls, shape):
        correct_shape_len = len(shape) == len(cls.fixed_shape)
        correct_sizes = all(
            value is None or shape[i] == value
            for i, value in enumerate(cls.fixed_shape.values())
        )
        if not correct_shape_len and correct_sizes:
            shape_error = (
                f"{cls.__name__} should be a ("
                + ', '.join([
                    key if value is None else key + '=' + value
                    for key, value in cls.fixed_shape.items()
                ]) + f") shaped vector, got {tuple(shape)}."
            )
            raise AssertionError(shape_error)


class IntBundle(Tensor):
    fixed_shape = {"C": 5, "H": None, "W": None, "D": None}
    def to_float_batch_bundle(self) -> FloatBatchBundle:
        return FloatBatchBundle(torch.cat([
            self[0:4].float().unsqueeze(0),
            functional.one_hot(
                self[4].long(),
                3
            ).permute(3, 0, 1, 2).unsqueeze(0).float()
        ], dim=1))


class FloatBatchBundle(Tensor):
    fixed_shape = {"N": None, "C": 7, "H": None, "W": None, "D": None}
    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     return torch.tensor(x, dtype=torch.float32)
    #     # return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)
    #
    # def __init__(self, *args, **kwargs):
    #     super(FloatBatchBundle, self).__init__()
    #     assert len(self.shape) == 5, \
    #         f"FloatBatchBundle should be a (N, C, H, W, D) shaped vector, got {self.shape}."
    #     assert self.size(1) == 7, \
    #         f"Channels should be 7: four phases and three for one-hot ground-truth segmentation, got {self.shape}."

    @classmethod
    def cat(cls, inputs: list[FloatBatchBundle]) -> FloatBatchBundle:
        return cls(torch.cat(inputs, dim=0))

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
    fixed_shape = {"N": None, "C": 4, "H": None, "W": None, "D": None}
    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)

    # def __init__(self, *args, **kwargs):
    #     super(ScanBatch, self).__init__()
    #     assert len(self.shape) == 5, \
    #         f"ScanBatch should be a (N, C, H, W, D) shaped vector, got {self.shape}."
    #     assert self.size(1) == 4, \
    #         f"Channels should be four phases, got {self.shape}."


class FloatSegmBatch(Tensor):
    fixed_shape = {"N": None, "C": 3, "H": None, "W": None, "D": None}

    # @staticmethod
    # def __new__(cls, x, *args, **kwargs):
    #     return super().__new__(cls, x, *args, **kwargs).to(dtype=torch.float32)

    # def __init__(self, *args, **kwargs):
    #     super(FloatSegmBatch, self).__init__()
    #     assert len(self.shape) == 5, \
    #         f"FloatSegmBatch should be a (N, C, H, W, D) shaped vector, got {self.shape}."
    #     assert self.size(1) == 3, \
    #         f"Channels should be three for one-hot segmentation, got {self.shape}."

    def distance_from(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
        asyml1, asyml1_items = self.asyml1_df(other)
        return asyml1, asyml1_items

    def asyml1_df(self, other: FloatSegmBatch) -> tuple[BatchDistance, dict]:
        channel_distances = torch.mean(
            functional.relu(other - self),
            dim=(2, 3, 4)
        )
        channel_weights = torch.tensor([[1, 5, 20]]).to(device=self.device)
        items = {
            "back": torch.mean(channel_distances[:, 0]).item(),
            "livr": torch.mean(channel_distances[:, 1]).item(),
            "tumr": torch.mean(channel_distances[:, 2]).item(),
        }
        return BatchDistance(torch.sum(channel_weights * channel_distances, dim=1)), items


# class IntSegmTensor(Tensor):
#     fixed_shape = {"N": None, "C": 7, "H": None, "W": None, "D": None}
#
#     def __init__(self, *args, **kwargs):
#         super(IntSegmTensor, self).__init__()
#         assert len(self.shape) == 5, \
#             f"ScanTensor should be a (N, C, H, W, D) shaped vector, got {self.shape}."
#         assert self.size(1) == 3, \
#             f"Channels should be three for one-hot segmentation, got {self.shape}."


class BatchDistance(Tensor):
    fixed_shape = {"N": None}

    def __init__(self, *args, **kwargs):
        super(BatchDistance, self).__init__()
        assert len(self.shape) == 1, \
            f"BatchDistance should be a (N, ) shaped vector, got {self.shape}."
