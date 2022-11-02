from __future__ import annotations

import types
import typing
from typing import Iterable

import numpy as np

T = typing.TypeVar('T')
throuple = types.GenericAlias(tuple, (T,) * 3)
int3d = throuple[int | None]


def slices(*args, shape: int3d, stride: int3d = (None, None, None), pad_up_to: float = 0.5):
    X, Y, Z = -3, -2, -1
    input_shape = args[0].shape[X:]
    shape = [shape[d] or input_shape[d] for d in (X, Y, Z)]
    nopad = [int((1 - pad_up_to) * shape[d]) for d in (X, Y, Z)]
    stride = [stride[d] or shape[d] for d in (X, Y, Z)]
    for x in range(0, input_shape[X] - nopad[X], stride[X]):
        for y in range(0, input_shape[Y] - nopad[Y], stride[Y]):
            for z in range(0, input_shape[Z] - nopad[Z], stride[Z]):
                yield [
                    pad_xyz(
                        narrow_xyz(arg, (x, y, z), shape),
                        shape
                    )
                    for arg in args
                ]


def narrow_xyz(x: np.ndarray, starts: int3d, lengths: int3d, strict: bool = False):
    if strict:
        raise NotImplemented
    return np.array(x[
                    ...,
                    starts[-3]:starts[-3] + lengths[-3],
                    starts[-2]:starts[-2] + lengths[-2],
                    starts[-1]:starts[-1] + lengths[-1]
                    ])


def pad_xyz(x: np.ndarray, shape: int3d):
    for original, requested in zip(x.shape[-3:], shape):
        assert original <= requested
    dims = x.ndim - 3, x.ndim - 2, x.ndim - 1
    pad_width = [(0, shape[dims.index(n)] - original) if n in dims else (0, 0) for n, original in enumerate(x.shape)]
    return np.pad(x, pad_width, mode="edge")


# def narrow_x(x: np.ndarray, *, start: int, length: int, strict: bool = False):
#     assert not strict or start + length <= x.shape[-3], \
#         f"Requsted slice x[..., {start}:{start + length}, :, :] is not strict when x.shape={x.shape}."
#     return np.array(x[..., start:start + length, :, :])
#
#
# def narrow_y(x: np.ndarray, *, start: int, length: int, strict: bool = False):
#     assert not strict or start + length <= x.shape[-2], \
#         f"Requsted slice x[..., :, {start}:{start + length}, :] is not strict when x.shape={x.shape}."
#     return np.array(x[..., :, start:start + length, :])
#
#
# def narrow_z(x: np.ndarray, *, start: int, length: int, strict: bool = False):
#     assert not strict or start + length <= x.shape[-1], \
#         f"Requsted slice x[..., :, :, {start}:{start + length}] is not strict when x.shape={x.shape}."
#     return np.array(x[..., :, :, start:start + length])
#
#
# def pad(x: np.ndarray, dims: list[int], lengths: list[int]):
#     for dim, length in zip(dims, lengths):
#         assert x.shape[dim] <= length
#     pad_width = [(0, lengths[dims.index(n)] - shape) if n in dims else (0, 0) for n, shape in enumerate(x.shape)]
#     return np.pad(x, pad_width, mode="edge")
#
#
# def halfstep_z_slices(x: np.ndarray, y: np.ndarray | None = None, *, length: int):
#     if y is not None:
#         assert x.shape[-1] == y.shape[-1]
#     for start in range(0, x.shape[-1] - length // 2, max(1, length // 2)):
#         if y is None:
#             yield pad(narrow(x, dim=-1, start=start, length=length), dims=[x.ndim - 1], lengths=[length])
#         else:
#             yield (
#                 pad(narrow(x, dim=-1, start=start, length=length), dims=[x.ndim - 1], lengths=[length]),
#                 pad(narrow(y, dim=-1, start=start, length=length), dims=[y.ndim - 1], lengths=[length])
#             )
#
#
# def dimensional_slices(x: np.ndarray | Iterable[np.ndarray], *, dim: int, starts: Iterable[int], length: int,
#                        strict: bool = False):
#     if isinstance(x, np.ndarray):
#         return [narrow(x, dim=dim, start=s, length=length, strict=strict) for s in starts]
#     else:
#         return [
#             t
#             for xx in x
#             for t in dimensional_slices(xx, dim=dim, starts=starts, length=length, strict=strict)
#         ]
