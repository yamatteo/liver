from __future__ import annotations

from typing import Iterable

import numpy as np
from rich import print


def narrow(x: np.ndarray, *, dim: int, start: int, length: int, strict: bool = False):
    if dim < 0:
        dim = x.ndim + dim
    assert not strict or start + length <= x.shape[dim], \
        f"Requsted slice x[..., {start}:{start + length}] is not strict when x.shape={x.shape}."
    return np.array(x[(*[slice(None)] * dim, slice(start, start + length))])


def pad(x: np.ndarray, dims: list[int], lengths: list[int]):
    for dim, length in zip(dims, lengths):
        assert x.shape[dim] <= length
    pad_width = [(0, lengths[dims.index(n)] - shape) if n in dims else (0, 0) for n, shape in enumerate(x.shape)]
    return np.pad(x, pad_width, mode="edge")


def halfstep_z_slices(x: np.ndarray, y: np.ndarray | None = None, *, length: int):
    if y is not None:
        assert x.shape[-1] == y.shape[-1]
    for start in range(0, x.shape[-1] - length // 2, max(1, length // 2)):
        if y is None:
            yield pad(narrow(x, dim=-1, start=start, length=length), dims=[x.ndim - 1], lengths=[length])
        else:
            yield (
                pad(narrow(x, dim=-1, start=start, length=length), dims=[x.ndim - 1], lengths=[length]),
                pad(narrow(y, dim=-1, start=start, length=length), dims=[y.ndim - 1], lengths=[length])
            )


def dimensional_slices(x: np.ndarray | Iterable[np.ndarray], *, dim: int, starts: Iterable[int], length: int,
                       strict: bool = False):
    if isinstance(x, np.ndarray):
        return [narrow(x, dim=dim, start=s, length=length, strict=strict) for s in starts]
    else:
        return [
            t
            for xx in x
            for t in dimensional_slices(xx, dim=dim, starts=starts, length=length, strict=strict)
        ]


def covering_indexes(x: np.ndarray, *, dim: int, length: int):
    return list(range(0, x.shape[dim], length))


# def covering_slices(x: np.ndarray | Iterable[np.ndarray], *, dim: int, length: int, strict: bool = False):
#     if isinstance(x, np.ndarray):
#         return [narrow(x, dim=dim, start=s, length=length, strict=strict) for s in starts]
#     else:
#         return [
#             t
#             for xx in x
#             for t in covering_slices(xx, dim=dim, length=length, strict=strict)
# ]

############################################################################

import unittest


class TestNarrow(unittest.TestCase):

    def test_one(self):
        x = np.zeros([4, 5, 6])
        self.assertEqual(narrow(x, dim=0, start=0, length=2).shape, (2, 5, 6))
        self.assertEqual(narrow(x, dim=1, start=1, length=3, strict=True).shape, (4, 3, 6))
        self.assertEqual(narrow(x, dim=2, start=4, length=7).shape, (4, 5, 2))
        with self.assertRaises(AssertionError):
            self.assertEqual(narrow(x, dim=2, start=4, length=7, strict=True).shape, (4, 5, 2))

    def test_two(self):
        self.assertEqual(4, len(list(halfstep_z_slices(np.random.randn(3, 4, 5), length=2))))
        self.assertEqual(3, len(list(halfstep_z_slices(np.random.randn(3, 4, 8), length=4))))
        self.assertEqual(4, len(list(halfstep_z_slices(np.random.randn(3, 4, 9), length=4))))
        self.assertEqual(4, len(list(halfstep_z_slices(np.random.randn(3, 4, 10), length=4))))

if __name__ == "__main__":
    unittest.main()
