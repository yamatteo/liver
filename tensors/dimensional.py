from __future__ import annotations

import torch

from .as_tensor import AsTensor


class ShapedTensor(AsTensor):
    """Class of tensor that checks its shape on creation."""
    fixed_shape = ...

    def __init__(self, *args, **kwargs):
        super(ShapedTensor, self).__init__()
        shape = self.shape
        if len(shape) != len(self.fixed_shape):
            self.shape_error()
        for value, fixed in zip(shape, self.fixed_shape.values()):
            if fixed is not None and fixed != value:
                self.shape_error()

    def shape_error(self):
        error_message = (
            f"{type(self).__name__} should be a ("
            + ', '.join([
                key if value is None else f"{key}={value}"
                for key, value in self.fixed_shape.items()
            ])
            + f") shaped vector, got {tuple(self.shape)}."
        )
        raise ValueError(error_message)

    def dim(self, name: None | str = None) -> int:
        if name is None:
            return super().dim()
        return list(self.fixed_shape).index(name)

    def size(self, dim: None | int | str = None) -> int | torch.Size:
        if dim is None:
            return super().size()
        if isinstance(dim, str):
            dim = list(self.fixed_shape).index(dim)
        return super().size(dim)


class Bidimensional(ShapedTensor):
    fixed_shape = {"X": None, "Y": None}


class Tridimensional(ShapedTensor):
    fixed_shape = {"X": None, "Y": None, "Z": None}

    def select_section(self, z: int):
        return torch.select(self, 2, z)
