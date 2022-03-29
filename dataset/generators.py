from __future__ import annotations

import itertools
from math import ceil
from pathlib import Path
from typing import Iterator, Tuple

import nibabel
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool3d, avg_pool3d, one_hot, interpolate
from torch.utils.data import Dataset

from path_explorer import discover, get_criterion


def cases(base_path: Path | str, segmented: bool) -> Iterator[Path]:
    yield from (
        base_path / case
        for case in
        discover(base_path, get_criterion(registered=True, segmented=segmented))
    )


def scan_segm_tuples(base_path: Path | str) -> Iterator[Tuple[Tensor, Tensor]]:
    for case in cases(base_path, segmented=True):
        scan = torch.stack([
            torch.tensor(np.array(nibabel.load(
                case / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16)).float()
            for phase in ["b", "a", "v", "t"]
        ]).unsqueeze(0)
        segm = one_hot(
            torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long(),
            3
        ).permute(3, 0, 1, 2).unsqueeze(0).float()
        yield scan, segm


def get_scans(base_path: Path | str) -> Iterator[Tensor]:
    for case in cases(base_path, segmented=False):
        scan = torch.stack([
            torch.tensor(np.array(nibabel.load(
                case / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16)).float()
            for phase in ["b", "a", "v", "t"]
        ]).unsqueeze(0)
        yield scan


def tensor_slices(x: Tensor, thick: int, step: int, dim: int = -1) -> Iterator[Tensor]:
    height = x.size(dim)
    n = ceil((height - thick) / step) + 1
    for j in range(n):
        i = int(j * (height - thick) / max(1, n-1))
        yield torch.narrow(x, dim, i, thick)


def slices882(base_path: Path, thick: int, step: int) -> Iterator[Tensor]:
    for scan, segm in scan_segm_tuples(base_path):
        for x in tensor_slices(torch.cat([scan, segm], dim=1), thick, step):
            yield avg_pool3d(
                x,
                kernel_size=(8, 8, 2)
            )
