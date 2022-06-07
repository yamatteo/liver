from pathlib import Path

import nibabel
import numpy as np
import torch
from torch.nn.functional import one_hot


def build_tetravoxel(case: Path, segmentation=False):
    affine = nibabel.load(case / f"registered_phase_v.nii.gz").affine
    scan = torch.stack([
        torch.tensor(np.array(nibabel.load(
            case / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16)).float()
        for phase in ["b", "a", "v", "t"]
    ])
    good_z = [not bool(torch.any(torch.all(torch.all(scan[:, :, :, z] == 0, dim=1), dim=1))) for z in
              range(scan.size(3))]
    a = 0
    for z in range(scan.size(3)):
        if any(good_z[:z]):
            break
        else:
            a = z
    b = scan.size(3)
    for z in reversed(range(scan.size(3))):
        if any(good_z[z:]):
            break
        else:
            b = z
    if segmentation:
        segm = one_hot(
            torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long(),
            3
        ).permute(3, 0, 1, 2).float()[:, :, :, a:b]
    else:
        segm = None

    return (a, scan.size(3)), affine, scan[:, :, :, a:b], segm
