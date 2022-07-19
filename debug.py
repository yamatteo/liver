import torch
from torch import Tensor


def check_floatscan(t):
    assert isinstance(t, Tensor)
    assert t.dtype == torch.float32
    (C, X, Y, Z) = t.size()
    assert C == 4


def check_floatsegm(t):
    assert isinstance(t, Tensor)
    assert t.dtype == torch.float32
    (C, X, Y, Z) = t.size()
    assert C == 3