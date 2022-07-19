from __future__ import annotations

from pathlib import Path

import nibabel

from subclass_tensors import *


def load_scan(case_path: Path) -> tuple[tuple[int, int, int], np.ndarray, FloatScan]:
    affine = nibabel.load(case_path / f"registered_phase_v.nii.gz").affine
    scan = Scan(torch.stack([
        torch.as_tensor(np.array(nibabel.load(
            case_path / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16))
        for phase in ["b", "a", "v", "t"]
    ]))
    a, b = scan.boundaries()

    return (a, b, scan.size(3)), affine, scan[..., a:b].as_float()
