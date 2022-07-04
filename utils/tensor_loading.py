from pathlib import Path

import nibabel
import numpy as np
import torch
from torch.nn.functional import one_hot
from tensors import *


def load_scan(case_path: Path) -> tuple[tuple[int, int, int], np.ndarray, FloatScan]:
    affine = nibabel.load(case_path / f"registered_phase_v.nii.gz").affine
    scan = Scan(torch.stack([
        Volume(np.array(nibabel.load(
            case_path / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16))
        for phase in ["b", "a", "v", "t"]
    ]))
    a, b = scan.boundaries()

    return (a, b, scan.size("D")), affine, scan[..., a:b].as_float()
