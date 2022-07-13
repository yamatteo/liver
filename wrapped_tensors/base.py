from __future__ import annotations

import torch
from torch import Tensor


class BaseWrappedTensor:
    def __init__(self, data, **kwargs):
        self.t = torch.as_tensor(data)
        self.fixed_dtype = kwargs.get("fix_dtype", None)
        fixed_shape = None
        if batch := kwargs.get("fix_batch", None):
            fixed_shape = dict(fixed_shape or {}, N=batch)
        if channels := kwargs.get("fix_channels", None):
            fixed_shape = dict(fixed_shape or {}, C=channels)
        if (shape := kwargs.get("fix_shape", None)) is not None:
            fixed_shape = dict(fixed_shape or {}, **shape)
        self.fixed_shape = fixed_shape

        self.check()

    def check(self):
        if msg := self.incorrect_dtype():
            raise ValueError(msg)
        if msg := self.incorrect_shape():
            raise ValueError(msg)

    def dim(self, name: None | str = None) -> int:
        if name is None:
            return self.t.dim()
        return list(self.fixed_shape).index(name)

    def incorrect_dtype(self):
        if self.fixed_dtype is None:
            return False
        if self.t.dtype != self.fixed_dtype:
            return f"{type(self).__name__} expect values of type {self.fixed_dtype}, got {self.t.dtype}"

    def incorrect_shape(self):
        if self.fixed_shape is None:
            return False
        shape = self.t.shape
        fixed_shape = self.fixed_shape
        msg = (
            f"{type(self).__name__} should be a ("
            + ', '.join(map(_fmt, fixed_shape.items()))
            + f") shaped vector, got {tuple(shape)}. "
        )
        if len(shape) != len(fixed_shape):
            return msg
        for value, fixed in zip(shape, fixed_shape.values()):
            if isinstance(fixed, int) and fixed != value:
                return msg
            if isinstance(fixed, (list, tuple)) and len(fixed) != value:
                return msg

    def size(self, dim: None | int | str = None) -> int | torch.Size:
        if dim is None:
            return self.t.size()
        if isinstance(dim, str):
            dim = list(self.fixed_shape).index(dim)
        return self.t.size(dim)



def _fmt(item):
    key, value = item
    if isinstance(value, int):
        return f"{key}={value}"
    if isinstance(value, (list, tuple)):
        return f"{key}={len(value)}"
    return key