from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .as_tensor import AsTensor
from .dimensional import ShapedTensor


class MultiChannel(ShapedTensor):
    """Class of tensor that checks its shape on creation."""

    def __init__(self, data, *, fixed_channels: Sequence[str], **kwargs):
        super(MultiChannel, self).__init__(data, **kwargs)
        self._metadata["fixed_shape"] = {"C": len(fixed_channels), **self.fixed_shape}
        self._metadata["fixed_channels"] = fixed_channels

    @property
    def fixed_channels(self) -> Sequence[str]:
        return self._metadata["fixed_channels"]

    def select_channel(self, name: str) -> AsTensor:
        dim = self.dim("C")
        index = list(self.fixed_channels).index(name)
        return torch.select(self, dim, index)


class Phases(MultiChannel):
    def __init__(self, data, **kwargs):
        super(Phases, self).__init__(data, fixed_channels=("b", "a", "v", "t"), **kwargs)


class HotSegm(MultiChannel):
    def __init__(self, data, **kwargs):
        super(HotSegm, self).__init__(data, fixed_channels=("backg", "liver", "tumor"), **kwargs)


class ColdBundle(MultiChannel):
    def __init__(self, data, **kwargs):
        super(ColdBundle, self).__init__(data, fixed_channels=("b", "a", "v", "t", "segm"), **kwargs)


class HotBundle(MultiChannel):
    def __init__(self, data, **kwargs):
        super(HotBundle, self).__init__(data, fixed_channels=("b", "a", "v", "t", "backg", "liver", "tumor"))
