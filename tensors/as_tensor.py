from __future__ import annotations

import torch
from torch import Tensor


class AsTensor:
    def __init__(self, data, **kwargs):
        self._t = torch.as_tensor(data)
        self._metadata = {}

    def __repr__(self):
        return f"{type(self).__name__}{list(self._t.shape)}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [a._t if hasattr(a, '_t') else a for a in args]
        assert len(metadatas) > 0, "No metadata"
        ret = func(*args, **kwargs)
        try:
            return cls(ret)
        except ValueError:
            return AsTensor(ret, **(metadatas[0] if metadatas else {}))
