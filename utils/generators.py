from __future__ import annotations

import unittest
from pathlib import Path
from typing import Iterator, Callable, Iterable

import nibabel
import numpy as np
import torch

from tensors import IntBundle, FloatBatchBundle, ScanBatch, FloatSegmBatch
from utils.path_explorer import criterion


def cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    """Iterate over 'base_path' folders (recursively) that satisfy 'accepted_dir'"""
    base_path = Path(base_path).expanduser().resolve()
    yield from _cases(base_path, accepted_dir=accepted_dir)


def cycle_enum_slices(cases_list: Iterable, shape: tuple[int, int, int]) -> Iterator[tuple[int, FloatBatchBundle]]:
    while True:
        k = 0
        for case in cases_list:
            fbb = IntBundle(np.array(nibabel.load(
                case / f"train_bundle.nii.gz"
            ).dataobj, dtype=np.int16)).to_float_batch_bundle()
            for t in fbb.slices(shape):
                yield k, t
                k += 1


def train_bundles(base_path: Path | str) -> Iterator[FloatBatchBundle]:
    """Iterate over NCHWD training tensors. N=1. C=7 (bavt-blt)."""
    for case in cases(base_path, criterion(bundle=True)):
        obj = IntBundle(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16))
        yield obj.to_float_batch_bundle()


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
            yield FloatBatchBundle(base.reshape((1, 1, 1, 1, n)).repeat((1, 7, 512, 512, 1)))

    def test_slices(self):
        shape = (64, 64, 16)
        t_gen = self.mock_train_bundles([7])
        for fbb in t_gen:
            for t in fbb.slices(shape):
                self.assertEqual(t.shape, (1, 7, 64, 64, 7))

        t_gen = self.mock_train_bundles([17, 31, 77])
        for fbb in t_gen:
            for t in fbb.slices(shape):
                self.assertEqual(t.shape, (1, 7, 64, 64, 16))

        shape = (512, 512, 8)
        t_gen = self.mock_train_bundles([7])
        t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
        self.assertEqual(len(list(t_gen)), 1)

        t_gen = self.mock_train_bundles([9])
        t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
        self.assertEqual(len(list(t_gen)), 2)

        t_gen = self.mock_train_bundles([16])
        t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
        self.assertEqual(len(list(t_gen)), 2)

        t_gen = self.mock_train_bundles([17])
        t_gen = (t for fbb in t_gen for t in fbb.slices(shape))
        self.assertEqual(len(list(t_gen)), 3)
