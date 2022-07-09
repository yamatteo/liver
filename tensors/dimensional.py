from __future__ import annotations

import torch

from .as_tensor import AsTensor


class ShapedTensor(AsTensor):
    """Class of tensor that checks its shape on creation."""
    def __init__(self, data, *, fixed_shape: dict, **kwargs):
        super(ShapedTensor, self).__init__(data)
        self._metadata["fixed_shape"] = fixed_shape

    @property
    def fixed_shape(self) -> dict:
        return self._metadata["fixed_shape"]

    @property
    def shape(self):
        return self._t.shape

    def shape_check(self):
        if len(self.shape) != len(self.fixed_shape):
            self.shape_error()
        for value, fixed in zip(self.shape, self.fixed_shape.values()):
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
            return self._t.dim()
        return list(self.fixed_shape).index(name)

    def size(self, dim: None | int | str = None) -> int | torch.Size:
        if dim is None:
            return self._t.size()
        if isinstance(dim, str):
            dim = list(self.fixed_shape).index(dim)
        return self._t.size(dim)


class Bidimensional(ShapedTensor):
    def __init__(self, data, **kwargs):
        super(Bidimensional, self).__init__(data, fixed_shape={"X": None, "Y": None}, **kwargs)
        self.shape_check()


class Tridimensional(ShapedTensor):
    def __init__(self, data, **kwargs):
        super(Tridimensional, self).__init__(data, fixed_shape={"X": None, "Y": None, "Z": None}, **kwargs)
        self.shape_check()

    def select_section(self, z: int) -> AsTensor:
        return torch.select(self, 2, z)
