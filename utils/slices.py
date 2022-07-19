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


def fixed_shape_slices(t: ndarray, shape: tuple[int, int, int], dims: tuple[int, int, int]) -> Iterator[ndarray]:
    for x in overlapping_slices(t, shape[0], dims[0]):
        for y in overlapping_slices(x, shape[1], dims[1]):
            for z in overlapping_slices(y, shape[2], dims[2]):
                yield z
