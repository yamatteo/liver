from __future__ import annotations

import torch

from .dimensional import ShapedTensor


class MultiChannel(ShapedTensor):
    """Class of tensor that checks its shape on creation."""
    fixed_channels = ...

    def __init__(self, *args, **kwargs):
        self.fixed_shape = {"C": len(self.fixed_channels), **self.fixed_shape}
        super(MultiChannel, self).__init__()

    def select_channel(self, name: str):
        dim = self.dim("C")
        index = list(self.fixed_channels).index(name)
        return torch.select(self, dim, index)


class Phases(MultiChannel):
    fixed_channels = ("b", "a", "v", "t")


class HotSegm(MultiChannel):
    fixed_channels = ("backg", "liver", "tumor")


class ColdBundle(MultiChannel):
    fixed_channels = ("b", "a", "v", "t", "segm")


class HotBundle(MultiChannel):
    fixed_channels = ("b", "a", "v", "t", "backg", "liver", "tumor")
