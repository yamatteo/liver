from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy import ndarray


def overlapping_slices(t: ndarray, thickness: int, dim: int) -> Iterator[ndarray]:
    """Iterate over slices of t along dimension dim.

     Slices may overlap if thickness does not divide t's size along dim evenly.
     If t.shape[dim] is less than thickness, yields only t."""
    length = t.shape[dim]
    if length <= thickness:
        yield t
        return
    num_slices = int(np.ceil(length / thickness))
    for j in range(num_slices):
        i = int(j * (length - thickness) / (num_slices - 1))
        indices = (*([slice(None, None)] * dim), slice(i, i + thickness))
        yield t[indices]

def nonoverlapping_slices(t: ndarray, thickness: int, dim: int) -> Iterator[ndarray]:
    length = t.shape[dim]
    if length <= thickness:
        yield t
        return
    for i in range(0, length, thickness):
        indices = (*([slice(None, None)] * dim), slice(i, i + thickness))
        yield t[indices]

def bundle_pad_z(t: ndarray, shape: tuple[int, int, int]) -> ndarray:
    shape = (5, *shape)
    assert t.shape[:-1] == shape[:-1]
    z_pad = t.shape[3] - shape[3]
    if z_pad > 0:
        fill = np.array([-1024, -1024, -1024, -1024, 0]).reshape((5, 1, 1, 1))
        fill = np.tile(fill, (1, shape[1], shape[2], z_pad))
        return np.concatenate([t, fill], axis=3)
    else:
        return t

def scan_pad_z(t: ndarray, shape: tuple[int, int, int]) -> ndarray:
    shape = (4, *shape)
    assert t.shape[:-1] == shape[:-1]
    z_pad = t.shape[3] - shape[3]
    if z_pad > 0:
        fill = np.array([-1024, -1024, -1024, -1024]).reshape((4, 1, 1, 1))
        fill = np.tile(fill, (1, shape[1], shape[2], z_pad))
        return np.concatenate([t, fill], axis=3)
    else:
        return t


def fixed_shape_slices(t: ndarray, shape: tuple[int, int, int], dims: tuple[int, int, int]) -> Iterator[ndarray]:
    if shape is None:
        yield t
        return
    for x in overlapping_slices(t, shape[0], dims[0]):
        for y in overlapping_slices(x, shape[1], dims[1]):
            for z in overlapping_slices(y, shape[2], dims[2]):
                yield z


def padded_nonoverlapping_bundle_slices(t: ndarray, shape: tuple[int, int, int]) -> Iterator[ndarray]:
    assert t.ndim == 4  # [C=5, X, Y, Z]
    if shape is None:
        yield t
        return
    for x in nonoverlapping_slices(t, shape[0], 1):
        for y in nonoverlapping_slices(x, shape[1], 2):
            for z in nonoverlapping_slices(y, shape[2], 3):
                yield bundle_pad_z(z, shape)


def padded_nonoverlapping_scan_slices(t: ndarray, shape: tuple[int, int, int]) -> Iterator[ndarray]:
    assert t.ndim == 4  # [C=4, X, Y, Z]
    if shape is None:
        yield t
        return
    for x in nonoverlapping_slices(t, shape[0], 1):
        for y in nonoverlapping_slices(x, shape[1], 2):
            for z in nonoverlapping_slices(y, shape[2], 3):
                yield scan_pad_z(z, shape)


def padded_overlapping_bundle_slices(t: ndarray, shape: tuple[int, int, int]) -> Iterator[ndarray]:
    assert t.ndim == 4  # [C=5, X, Y, Z]
    if shape is None:
        yield t
        return
    for x in overlapping_slices(t, shape[0], 1):
        for y in overlapping_slices(x, shape[1], 2):
            for z in overlapping_slices(y, shape[2], 3):
                yield bundle_pad_z(z, shape)
