from __future__ import annotations

import torch
from torch import Tensor


class StrictTensor(Tensor):
    """Class of tensor that checks its shape on creation."""
    fixed_shape = ...
    fixed_dtype = ...
    fixed_channels = ...

    def __new__(cls, data, **kwargs):
        self = torch.as_tensor(data)
        self.__class__ = cls
        cls.__init__(self, data, **kwargs)
        return self

    def __init__(self, data, **kwargs):
        super(StrictTensor, self).__init__()
        type(self).check(self)

    @classmethod
    def check(cls, data):
        fixed_shape = cls.fixed_shape
        fixed_dtype = cls.fixed_dtype
        if fixed_shape is not ...:
            cls.check_shape(data)
        if fixed_dtype is not ...:
            cls.check_dtype(data)

    @classmethod
    def check_shape(cls, data):
        shape = data.shape
        fixed_shape = cls.fixed_shape
        if len(shape) != len(fixed_shape):
            raise ValueError(
                f"{cls.__name__} should be a ("
                + ', '.join([
                    key if value is None else f"{key}={value}"
                    for key, value in fixed_shape.items()
                ])
                + f") shaped vector, got {tuple(shape)}. "
            )
        for value, fixed in zip(data.shape, fixed_shape.values()):
            if fixed is not None and fixed != value:
                raise ValueError(
                    f"{cls.__name__} should be a ("
                    + ', '.join([
                        key if value is None else f"{key}={value}"
                        for key, value in fixed_shape.items()
                    ])
                    + f") shaped vector, got {tuple(shape)}. "
                )

    @classmethod
    def check_dtype(cls, data):
        dtype = data.dtype
        fixed_dtype = cls.fixed_dtype
        if fixed_dtype != dtype:
            raise ValueError(f"{cls.__name__} expect values of type {fixed_dtype}, got {dtype}")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, StrictTensor):
            try:
                return cls(out)
            except ValueError:
                return out.torch()
        elif isinstance(out, Tensor):
            try:
                return cls(out)
            except ValueError:
                return out
        else:
            return out

    def dim(self, name: None | str = None) -> int:
        if name is None:
            return Tensor.dim(self)
        return list(self.fixed_shape).index(name)

    def select_channel(self, name: str) -> StrictTensor:
        dim = self.dim("C")
        index = list(self.fixed_channels).index(name)
        t = torch.select(self, dim, index)
        t = StrictTensor(t)
        t.fixed_shape = {name: value for name, value in self.fixed_shape.items() if name != "C"}
        t.fixed_dtype = self.fixed_dtype
        t.fixed_channels = ...
        return t

    def select_item(self, n: int) -> StrictTensor:
        dim = self.dim("N")
        t = torch.select(self, dim, n)
        t = StrictTensor(t)
        t.fixed_shape = {name: value for name, value in self.fixed_shape.items() if name != "N"}
        t.fixed_dtype = self.fixed_dtype
        t.fixed_channels = ...
        return t

    def select_section(self, z: int) -> StrictTensor:
        dim = self.dim("Z")
        t = torch.select(self, dim, z)
        t = StrictTensor(t)
        t.fixed_shape = {name: value for name, value in self.fixed_shape.items() if name != "Z"}
        t.fixed_dtype = self.fixed_dtype
        t.fixed_channels = self.fixed_channels
        return t

    def size(self, dim: None | int | str = None) -> int | torch.Size:
        if dim is None:
            return Tensor.size(self)
        if isinstance(dim, str):
            dim = list(self.fixed_shape).index(dim)
        return Tensor.size(self, dim)

    def torch(self) -> Tensor:
        t = torch.as_tensor(self)
        t.__class__ = Tensor
        return t

# Tests
import unittest

class TestCase(unittest.TestCase):
    def test_shaped_tensors(self):
        class Mock(StrictTensor):
            fixed_shape = {"A": 5, "B": None}
            fixed_dtype = torch.float32

        class MockBatch(StrictTensor):
            fixed_shape = {"N": None, "A": 5, "B": None}
            fixed_dtype = torch.float32

        t = Mock(torch.empty([5, 8]))
        s = torch.mean(t, dim=1)
        r = MockBatch(t.unsqueeze(0))
        pass

