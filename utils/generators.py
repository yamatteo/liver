from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterator, Callable

import nibabel
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from utils.path_explorer import criterion


def cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    """Iterate over 'base_path' folders (recursively) that satisfy 'accepted_dir'"""
    base_path = Path(base_path).expanduser().resolve()
    yield from _cases(base_path, accepted_dir=accepted_dir)


def train_bundles(base_path: Path | str) -> Iterator[Tensor]:
    """Iterate over NCHWD training tensors. N=1. C=7 (bavt-blt)."""
    for case in cases(base_path, criterion(bundle=True)):
        obj = torch.tensor(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16))
        scan = obj[0:4].float().unsqueeze(0)
        segm = one_hot(
            obj[4].long(),
            3
        ).permute(3, 0, 1, 2).unsqueeze(0).float()
        yield torch.cat([scan, segm], dim=1)


def tensor_slices(t: Tensor, thickness: int, dim: int) -> Iterator[Tensor]:
    """Iterate over slices of `t` along dimension `dim`.

     Slices may overlap if `thicknes` does not divide `t.size(dim)` evenly.
     If `t.size(dim)` is less than `thickness`, yields only `t`."""
    length = t.size(dim)
    if length <= thickness:
        yield t
        return
    num_slices = ceil(length / thickness)
    for j in range(num_slices):
        i = int(j * (length - thickness) / (num_slices - 1))
        yield torch.narrow(t, dim, i, thickness)


def compose_tensor_slices(t_gen: Iterator[Tensor], thickness: int, dim: int) -> Iterator[Tensor]:
    """Compose generators to return sliced tensors."""
    for t in t_gen:
        yield from tensor_slices(t, thickness, dim)


def train_slices(base_path: Path | str, shape: tuple[int, int, int], split=False) -> Iterator[tuple[Tensor, Tensor]]:
    """Iterate over NCHWD training tensors. Yields (scan, segm) tuple.

    scan: N=1. C=4 (bavt). HWD=shape.
    segm: N=1. C=3 (blt).  HWD=shape."""
    t_gen = train_bundles(base_path)
    t_gen = compose_tensor_slices(t_gen, shape[0], 2)
    t_gen = compose_tensor_slices(t_gen, shape[1], 3)
    t_gen = compose_tensor_slices(t_gen, shape[2], 4)
    if split:
        for t in t_gen:
            yield t[:, 0:4], t[:, 4:7]
    else:
        yield from t_gen


def _cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    if base_path.is_dir():
        if accepted_dir(base_path):
            yield base_path
        else:
            for sub_path in base_path.iterdir():
                yield from _cases(base_path / sub_path, accepted_dir=accepted_dir)


import unittest

class TestSlicing(unittest.TestCase):

    def mock_train_bundles(self, ns):
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
