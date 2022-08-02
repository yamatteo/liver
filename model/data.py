from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from rich.console import Console

import dataset.ndarray as nd
import dataset.path_explorer as px
from dataset.slices import fixed_shape_slices

console = Console()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, format=torch.tensor):
        super(Dataset, self).__init__()
        self.files = list(path.iterdir())
        self.format = format

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        data = nd.load_niftidata(path)
        return self.format(data)


def store_dataset(source_path: Path, target_path: Path, pooler=None, slice_shape: tuple[int, int, int] = None,
                  min_slice_z: int = None):
    i, k = 0, 0

    def iter_slice(bundle):
        if slice_shape:
            yield from fixed_shape_slices(bundle, slice_shape, dims=(1, 2, 3))
        else:
            yield bundle

    def pad_slice(slice):
        if min_slice_z and slice_shape[2] > slice.shape[3]:
            pad = slice_shape[2] - slice.shape[3]
            return np.concatenate([
                np.pad(slice[0:4], ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=-1024),
                np.pad(slice[4:5], ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=0)
            ])
        else:
            return slice

    console.print("Storing dataset:")
    console.print("  source_path =", source_path)
    console.print("  target_path =", target_path)
    console.print("  pooler =", pooler)
    console.print(f"  slice_shape = {slice_shape}")
    (target_path / "train").mkdir(exist_ok=True)
    (target_path / "valid").mkdir(exist_ok=True)
    for case in px.iter_trainable(source_path):
        console.print("  ...working on:", case)
        case_path = source_path / case
        registered_scans = [
            nd.load_registered(case_path, phase)
            for phase in ["b", "a", "v", "t"]
        ]
        segm = nd.load_segm(case_path)
        affine, bottom, top, height = nd.load_registration_data(case_path)
        bundle = np.stack([*registered_scans, segm])[..., bottom:top]
        bundle = np.clip(bundle, a_min=-1024, a_max=1024)
        if pooler:
            bundle = torch.tensor(bundle)
            bundle = pooler(bundle)
            bundle = bundle.numpy()
            for slice in iter_slice(bundle):
                slice = pad_slice(slice)
                assert tuple(
                    slice.shape[-3:]) == slice_shape, f"Can't fix slice shape ({slice.shape} vs {slice_shape})!"
                if k % 10 == 0:
                    nd.save_niftiimage(target_path / "valid" / f"{i:06}.nii.gz", slice)
                else:
                    nd.save_niftiimage(target_path / "train" / f"{i:06}.nii.gz", slice)
                i += 1
        k += 1
