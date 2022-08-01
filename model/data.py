from pathlib import Path

import numpy as np
import torch
import torch.utils.data

import dataset.ndarray as nd
import dataset.path_explorer as px
from dataset.slices import fixed_shape_slices


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


def store_dataset(source_path: Path, target_path: Path, pooler=None, slice_shape=None):
    i, k = 0, 0
    for case in px.iter_trainable(source_path):
        case_path = source_path / case
        registered_scans = [
            nd.load_registered(case_path, phase)
            for phase in ["b", "a", "v", "t"]
        ]
        segm = nd.load_segm(case_path)
        affine, bottom, top, height = nd.load_registration_data(case_path)
        bundle = np.stack([*registered_scans, segm])[..., bottom:top]
        if pooler:
            bundle = torch.tensor(bundle)
            bundle = pooler(bundle)
            bundle = bundle.numpy()
        if slice_shape:
            for slice in fixed_shape_slices(bundle, slice_shape, dims=(1, 2, 3)):
                if k % 10 == 0:
                    nd.save_niftiimage(target_path / "valid" / f"{i:06}.nii.gz", slice)
                else:
                    nd.save_niftiimage(target_path / "train" / f"{i:06}.nii.gz", slice)
                i += 1
        else:
            if k % 10 == 0:
                nd.save_niftiimage(target_path / "valid" / f"{i:06}.nii.gz", bundle)
            else:
                nd.save_niftiimage(target_path / "train" / f"{i:06}.nii.gz", bundle)
            i += 1
        k += 1
