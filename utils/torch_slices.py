from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from utils.debug import dbg

int3d = "tuple[int, int, int]"
str3d = "tuple[str, str, str]"


def covering_indices(length: int, size: int) -> Iterator[int]:
    """Covering slices, maybe overlapping.

    ##################
    #######
         #######
               #######
    """
    num_slices = int(np.ceil(size / length))
    for j in range(num_slices):
        yield int(j * (size - length) / (num_slices - 1))


def jumping_indices(length: int, size: int) -> Iterator[int]:
    """Jumping slices, maybe last is shorter.

    ##################
    #######
           #######
                  ####
    """
    yield from range(0, size, length)


def stepping_indices(length: int, size: int) -> Iterator[int]:
    """Stepping slices, much overlapping.

    ###########
    #######
     #######
      #######
       #######
        #######
    """
    yield from range(0, 1 + size - length)


def _dim_slices(t: Tensor, *, dim: int, length: int, mode: str) -> Iterator[Tensor]:
    size = t.size(dim)
    if size <= length:
        yield t, 0
        return
    if mode == "jump":
        indices = jumping_indices(length, size)
    elif mode == "step":
        indices = stepping_indices(length, size)
    elif mode == "cover":
        indices = covering_indices(length, size)
    else:
        raise ValueError(f"Unexpected mode {mode}.")
    for start in indices:
        yield torch.narrow(t, dim, start, length), start


def slices3d(t: Tensor, shape: int3d, modes: str3d) -> Iterator[Tensor]:
    dims = (t.ndim -3, t.ndim-2, t.ndim - 1)
    for x, _ in _dim_slices(t, dim=dims[0], length=shape[0], mode=modes[0]):
        for y, _ in _dim_slices(x, dim=dims[1], length=shape[1], mode=modes[1]):
            yield from _dim_slices(y, dim=dims[2], length=shape[2], mode=modes[2])


def pad_dim(t: Tensor, *, dim: int, length: int, seed: list) -> Tensor:
    if length <= 0:
        return t
    fill = torch.tile(
        torch.tensor(seed).reshape((len(seed), 1, 1, 1)),
        dims=(
            1,
            length if dim == 1 else t.size(1),
            length if dim == 2 else t.size(2),
            length if dim == 3 else t.size(3),
        )
    )
    return torch.cat([t, fill], dim=dim)


def pad3d(t: Tensor, shape: int3d, seed: list) -> Tensor:
    dims = (t.ndim -3, t.ndim-2, t.ndim - 1)
    shape = (*t.shape[:-3], *shape)
    pad = (
        *[0]*max(0, t.ndim-3),
        shape[1] - t.size(1),
        shape[2] - t.size(2),
        shape[3] - t.size(3),
    )
    for dim in dims:
        t = pad_dim(t, dim=dim, length=pad[dim], seed=seed)
    assert t.shape == shape, f"Shape mismatch: {t.shape} and {shape}."
    return t


def slices(t: Tensor, *, shape: int3d, modes: str3d, pad: str, zstart: bool = False) -> Iterator[Tensor]:
    """Iterator over slices of t with fixed shape.

    Parameter modes is a 3-tuple of "jump", "step" or "cover". If one of the modes is "jump" and , the last slice can be
        smaller than shape along that dimension; in that c
    Parameter pad is one of "scan", "bundle" or "none".
    """
    for t, z in slices3d(t, shape, modes):
        if pad == "scan":
            yield (
                (pad3d(t, shape, [-1024, -1024, -1024, -1024]), z)
                if zstart
                else pad3d(t, shape, [-1024, -1024, -1024, -1024])
            )
        elif pad == "bundle":
            yield (
                (pad3d(t, shape, [-1024, -1024, -1024, -1024, 0]), z)
                if zstart
                else pad3d(t, shape, [-1024, -1024, -1024, -1024, 0])
            )
        elif pad == "none":
            yield (t, z) if zstart else t
        else:
            raise ValueError(f"Undefined pad seed: {pad}")
