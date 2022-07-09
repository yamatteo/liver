from __future__ import annotations

import torch

from .dimensional import ShapedTensor


class Batch(ShapedTensor):
    """Class of tensor that checks its shape on creation."""

    def __init__(self, *args, **kwargs):
        if hasattr(self, "fixed_shape"):
            self.fixed_shape = {"N": None, **self.fixed_shape}
        super(Batch, self).__init__()

    def select_item(self, n: int):
        dim = self.dim("N")
        return torch.select(self, dim, n)
