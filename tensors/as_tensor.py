from __future__ import annotations

import torch
from torch import Tensor


class AsTensor(Tensor):
    def __new__(cls, data, *args, **kwargs):
        # From pytorch documentation:
        #
        #   torch.as_tensor(data)
        #
        # Converts data into a tensor, sharing data and preserving autograd history if possible.
        # If data is a NumPy array then a tensor is constructed using torch.from_numpy().
        _t = torch.as_tensor(data)
        _t.__class__ = cls
        return _t
