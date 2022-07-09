from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Iterator, Callable, Iterable, TypeVar, Generic, NewType

import nibabel
import numpy as np
import torch
from torch import Tensor

from tensors import Scan, Segm, ColdBundle, FloatBatchBundle, FloatSegmBatch, Bundle, ShapedTensor, Tridimensional
from utils.path_explorer import criterion

T = TypeVar("T")

def dimensional_slices(t: T, thickness: int, dim: int | str) -> Iterator[T]:
    """Iterate over slices of self along dimension `dim`.

     Slices may overlap if `thickness` does not divide `self.size(dim)` evenly.
     If `self.size(dim)` is less than `thickness`, yields only `self`."""
    if isinstance(dim, str):
        dim = t.dim(dim)
    length = t.size(dim)
    if length <= thickness:
        yield t
        return
    num_slices = math.ceil(length / thickness)
    for j in range(num_slices):
        i = int(j * (length - thickness) / (num_slices - 1))
        yield torch.narrow(t, dim, i, thickness)


def slices(t: T, shape: tuple[int, int, int]) -> Iterator[T]:
    """Iterate over t slices of given shape. Possibly overlapping."""
    for x_slice in dimensional_slices(t, shape[0], "X"):
        for y_slice in dimensional_slices(x_slice, shape[1], "Y"):
            yield from dimensional_slices(y_slice, shape[2], "Z")


def cold_bundles(path: Path) -> Iterator[Bundle]:
    for case_path in cases(path, criterion(registered=True, segmented=True)):
        scan = Scan(torch.stack([
            torch.tensor(np.array(nibabel.load(
                case_path / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16))
            for phase in ["b", "a", "v", "t"]
        ]))
        segm = Segm(torch.tensor(np.array(nibabel.load(
            case_path / f"segmentation.nii.gz"
        ).dataobj, dtype=np.int16)))
        yield Bundle.from_join(scan, segm)


def cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    """Iterate over 'base_path' folders (recursively) that satisfy 'accepted_dir'"""
    base_path = Path(base_path).expanduser().resolve()
    yield from _cases(base_path, accepted_dir=accepted_dir)


# def cycle_enum_slices(cases_list: Iterable, shape: tuple[int, int, int]) -> Iterator[tuple[int, FloatBatchBundle]]:
#     while True:
#         k = 0
#         for case in cases_list:
#             fbb = ColdBundle(np.array(nibabel.load(
#                 case / f"train_bundle.nii.gz"
#             ).dataobj, dtype=np.int16)).to_float_batch_bundle()
#             for t in fbb.slices(shape):
#                 yield k, t
#                 k += 1


# def train_bundles(base_path: Path | str) -> Iterator[FloatBatchBundle]:
#     """Iterate over NCHWD training tensors. N=1. C=7 (bavt-blt)."""
#     for case in cases(base_path, criterion(bundle=True)):
#         obj = ColdBundle(np.array(nibabel.load(
#             case / f"train_bundle.nii.gz"
#         ).dataobj, dtype=np.int16))
#         yield obj.to_float_batch_bundle()
#
#
# def train_slices(base_path: Path | str, shape: tuple[int, int, int], split=False) \
#         -> Iterator[FloatBatchBundle | tuple[ScanBatch, FloatSegmBatch]]:
#     """Iterate over NCHWD training tensors of given shape.
#
#     Yields FloatBatchBundle or (ScanBatch, FloatSegmBatch) tuple."""
#     if split:
#         for fbb in train_bundles(base_path):
#             for t in fbb.slices(shape):
#                 yield t.separate()
#     else:
#         for fbb in train_bundles(base_path):
#             for t in fbb.slices(shape):
#                 yield t


def _cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    if base_path.is_dir():
        if accepted_dir(base_path):
            yield base_path
        else:
            for sub_path in base_path.iterdir():
                yield from _cases(base_path / sub_path, accepted_dir=accepted_dir)

#
# class TestSlicing(unittest.TestCase):
#     @staticmethod
#     def mock_train_bundles(ns):
#         for n in ns:
#             base = torch.arange(0, n, dtype=torch.float32)
#             yield FloatBatchBundle(base.reshape((1, 1, 1, 1, n)).repeat((1, 7, 512, 512, 1)))
#
#     def test_slices(self):
#         shape = (64, 64, 16)
#         t_gen = self.mock_train_bundles([7])
#         for fbb in t_gen:
#             for t in fbb.slices(shape):
#                 self.assertEqual(t.shape, (1, 7, 64, 64, 7))
#
#         t_gen = self.mock_train_bundles([17, 31, 77])
#         for fbb in t_gen:
#             for t in fbb.slices(shape):
#                 self.assertEqual(t.shape, (1, 7, 64, 64, 16))
#
#         shape = (512, 512, 8)
#         t_gen = self.mock_train_bundles([7])
#         t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
#         self.assertEqual(len(list(t_gen)), 1)
#
#         t_gen = self.mock_train_bundles([9])
#         t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
#         self.assertEqual(len(list(t_gen)), 2)
#
#         t_gen = self.mock_train_bundles([16])
#         t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
#         self.assertEqual(len(list(t_gen)), 2)
#
#         t_gen = self.mock_train_bundles([17])
#         t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
#         self.assertEqual(len(list(t_gen)), 3)
