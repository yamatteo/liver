from __future__ import annotations

import torch

from .as_tensor import AsTensor


class TypedTensor(AsTensor):
    """Class of tensor that checks dtype of its input data."""
    fixed_dtype = ...

    def __init__(self, *args, **kwargs):
        if self.fixed_dtype != self.dtype:
            raise ValueError(f"{type(self).__name__} expect values of type {self.fixed_dtype}, got {self.dtype}")
        super(TypedTensor, self).__init__()


class Integer(TypedTensor):
    fixed_dtype = torch.int16


class Floating(TypedTensor):
    fixed_dtype = torch.float32
