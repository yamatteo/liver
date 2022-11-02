from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data
from torch import nn

import utils.ndarray
import utils.path_explorer
import utils.slices



def split_dataset(
        source_path: Path,
        target_path: Path,
        store_slice_shape: tuple[int, int, int],
        store_pool_kernel: tuple[int, int, int],
        **kwargs,
):
    i, k = 0, 0
    print("Storing dataset:")
    print("  source_path =", source_path)
    print("  target_path =", target_path)
    print("  slice_shape =", store_slice_shape)
    (target_path / "train").mkdir(exist_ok=True)
    (target_path / "valid").mkdir(exist_ok=True)
    for case in utils.path_explorer.iter_trainable(source_path):
        print("  ...working on:", case)
        case_path = source_path / case
        registered_scans = [
            utils.ndarray.load_registered(case_path, phase)
            for phase in ["b", "a", "v", "t"]
        ]
        segm = utils.ndarray.load_segm(case_path)
        affine, bottom, top, height = utils.ndarray.load_registration_data(case_path)
        bundle = np.stack([*registered_scans, segm])[..., bottom:top]
        bundle = np.clip(bundle, a_min=-1024, a_max=1024)
        for slice in utils.slices.slices(bundle, store_slice_shape, overlap=True, pad_seed="bundle"):
            assert tuple(slice.shape[-3:]) == store_slice_shape, f"Can't fix slice shape ({slice.shape} vs {store_slice_shape})!"
            scan = torch.tensor(slice[0:4], dtype=torch.float32)
            segm = torch.tensor(slice[4:5], dtype=torch.float32)
            scan = nn.AvgPool3d(store_pool_kernel)(scan)
            segm = nn.MaxPool3d(store_pool_kernel)(segm).squeeze(0).to(dtype=torch.int64)
            assert 0 <= torch.min(segm) <= torch.max(segm) <= 2

            if k % 10 == 0:
                file_path = (target_path / "valid" / f"{i:06}.pt")
            else:
                file_path = (target_path / "train" / f"{i:06}.pt")
            torch.save({"scan": scan, "segm": segm}, file_path)
            i += 1
        k += 1

def gen_split_dataset(
        source_path: Path,
        target_path: Path,
        slice_shape: tuple[int, int, int],
        gen: Callable,
        **kwargs,
):
    i, k = 0, 0
    print("Storing dataset:")
    print("  source =", source_path)
    print("  target =", target_path)
    print("  original sliceshape =", slice_shape)
    (target_path / "train").mkdir(exist_ok=True)
    (target_path / "valid").mkdir(exist_ok=True)
    for case in utils.path_explorer.iter_trainable(source_path):
        print("  ...working on:", case)
        case_path = source_path / case
        registered_scans = [
            utils.ndarray.load_registered(case_path, phase)
            for phase in ["b", "a", "v", "t"]
        ]
        segm = utils.ndarray.load_segm(case_path)
        affine, bottom, top, height = utils.ndarray.load_registration_data(case_path)
        bundle = np.stack([*registered_scans, segm])[..., bottom:top]
        bundle = np.clip(bundle, a_min=-1024, a_max=1024)
        for slice in utils.slices.slices(bundle, slice_shape, overlap=True, pad_seed="bundle"):
            assert tuple(slice.shape[-3:]) == slice_shape, f"Can't fix slice shape ({slice.shape} vs {slice_shape})!"
            scan = torch.tensor(slice[0:4], dtype=torch.float32)
            segm = torch.tensor(slice[4:5], dtype=torch.float32)
            assert 0 <= torch.min(segm) <= torch.max(segm) <= 2

            if k % 10 == 0:
                file_path = (target_path / "valid" / f"{i:06}.pt")
            else:
                file_path = (target_path / "train" / f"{i:06}.pt")
            torch.save(gen(scan, segm), file_path)
            i += 1
        k += 1
