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


def train_bundles(base_path: Path | str) -> Iterator[tuple[Tensor, Tensor]]:
    for case in cases(base_path, criterion(bundle=True)):
        obj = torch.tensor(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16))
        scan = obj[0:4].float().unsqueeze(0)
        segm = one_hot(
            obj[4].long(),
            3
        ).permute(3, 0, 1, 2).unsqueeze(0).float()
        yield scan, segm


def gen_slices(t: Tensor, thick: int, dim: int) -> Iterator[Tensor]:
    length = t.size(dim)
    num_slices = ceil(length / thick)
    for j in range(num_slices):
        i = int(j * (length - thick) / max(1, num_slices - 1))
        yield torch.narrow(t, dim, i, thick)


def train_slices(base_path: Path | str, side: int) -> Iterator[Tensor]:
    for scan, segm in train_bundles(base_path):
        t = torch.cat([scan, segm], dim=1)
        for tx in gen_slices(t, side, 2):
            for txy in gen_slices(tx, side, 3):
                for txyz in gen_slices(txy, side, 4):
                    yield txyz[:, 0:4], txyz[:, 4:7]


def _cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    if base_path.is_dir():
        if accepted_dir(base_path):
            yield base_path
        else:
            for sub_path in base_path.iterdir():
                yield from _cases(base_path / sub_path, accepted_dir=accepted_dir)
