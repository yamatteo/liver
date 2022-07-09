from __future__ import annotations

import torch
from torch import Tensor

from .as_tensor import AsTensor


class TypedTensor(AsTensor):
    """Class of tensor that checks dtype of its input data."""

    def __init__(self, data, *, fixed_dtype: torch.dtype, **kwargs):
        super(TypedTensor, self).__init__(data, **kwargs)
        self._metadata["fixed_dtype"] = fixed_dtype

    @property
    def dtype(self) -> torch.dtype:
        return self._t.dtype

    @property
    def fixed_dtype(self) -> torch.dtype:
        return self._metadata["fixed_dtype"]

    def dtype_check(self):
        if self.fixed_dtype != self.dtype:
            raise ValueError(f"{type(self).__name__} expect values of type {self.fixed_dtype}, got {self.dtype}")



class Integer(TypedTensor):
    def __init__(self, data, **kwargs):
        super(Integer, self).__init__(data, fixed_dtype=torch.int16, **kwargs)
        self.dtype_check()


class Floating(TypedTensor):
    def __init__(self, data, **kwargs):
        super(Floating, self).__init__(data, fixed_dtype=torch.float32, **kwargs)
        self.dtype_check()
