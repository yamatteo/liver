from __future__ import annotations

import unittest
from math import ceil
from pathlib import Path
from typing import Iterator, Callable

import nibabel
import numpy as np
import torch
from torch import Tensor

from tensors import IntBundle, FloatBatchBundle, ScanBatch, FloatSegmBatch
from utils.path_explorer import criterion


def cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    """Iterate over 'base_path' folders (recursively) that satisfy 'accepted_dir'"""
    base_path = Path(base_path).expanduser().resolve()
    yield from _cases(base_path, accepted_dir=accepted_dir)


def train_bundles(base_path: Path | str) -> Iterator[FloatBatchBundle]:
    """Iterate over NCHWD training tensors. N=1. C=7 (bavt-blt)."""
    for case in cases(base_path, criterion(bundle=True)):
        obj = IntBundle(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16))
        yield obj.to_float_batch_bundle()


# def tensor_slices(t: Tensor, thickness: int, dim: int) -> Iterator[Tensor]:
#     """Iterate over slices of `t` along dimension `dim`.
#
#      Slices may overlap if `thicknes` does not divide `t.size(dim)` evenly.
#      If `t.size(dim)` is less than `thickness`, yields only `t`."""
#     length = t.size(dim)
#     if length <= thickness:
#         yield t
#         return
#     num_slices = ceil(length / thickness)
#     for j in range(num_slices):
#         i = int(j * (length - thickness) / (num_slices - 1))
#         yield torch.narrow(t, dim, i, thickness)
#
#
# def compose_tensor_slices(t_gen: Iterator[Tensor], thickness: int, dim: int) -> Iterator[Tensor]:
#     """Compose generators to return sliced tensors."""
#     for t in t_gen:
#         yield from tensor_slices(t, thickness, dim)


def train_slices(base_path: Path | str, shape: tuple[int, int, int], split=False) \
        -> Iterator[FloatBatchBundle | tuple[ScanBatch, FloatSegmBatch]]:
    """Iterate over NCHWD training tensors of given shape.

    Yields FloatBatchBundle or (ScanBatch, FloatSegmBatch) tuple."""
    if split:
        for fbb in train_bundles(base_path):
            for t in fbb.slices(shape):
                yield t.separate()
    else:
        for fbb in train_bundles(base_path):
            for t in fbb.slices(shape):
                yield t


def _cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    if base_path.is_dir():
        if accepted_dir(base_path):
            yield base_path
        else:
            for sub_path in base_path.iterdir():
                yield from _cases(base_path / sub_path, accepted_dir=accepted_dir)


class TestSlicing(unittest.TestCase):
    @staticmethod
    def mock_train_bundles(ns):
        for n in ns:
            base = torch.arange(0, n, dtype=torch.float32)
            yield base.reshape((1, 1, 1, 1, n)).repeat((1, 7, 512, 512, 1))

    def test_slices(self):
        shape = [64, 64, 16]
        t_gen = self.mock_train_bundles([7])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        for t in t_gen:
            self.assertEqual(t.shape, (1, 7, 64, 64, 7))

        t_gen = self.mock_train_bundles([17, 31, 77])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        for t in t_gen:
            self.assertEqual(t.shape, (1, 7, 64, 64, 16))

        shape = [512, 512, 8]
        t_gen = self.mock_train_bundles([7])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        self.assertEqual(len(list(t_gen)), 1)

        t_gen = self.mock_train_bundles([9])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        self.assertEqual(len(list(t_gen)), 2)

        t_gen = self.mock_train_bundles([16])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        self.assertEqual(len(list(t_gen)), 2)

        t_gen = self.mock_train_bundles([17])
        t_gen = compose_tensor_slices(t_gen, shape[0], 2)
        t_gen = compose_tensor_slices(t_gen, shape[1], 3)
        t_gen = compose_tensor_slices(t_gen, shape[2], 4)
        self.assertEqual(len(list(t_gen)), 3)
