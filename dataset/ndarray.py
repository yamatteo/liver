from __future__ import annotations

from pathlib import Path

import nibabel
import numpy as np


def load_affine(case_path: Path) -> np.ndarray:
    return nibabel.load(case_path / f"registered_phase_v.nii.gz").affine


def load_scan(case_path: Path) -> np.ndarray:
    return np.stack([
        np.array(nibabel.load(
            case_path / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16)
        for phase in ["b", "a", "v", "t"]
    ])


def load_segm(case_path: Path) -> np.ndarray:
    return np.array(nibabel.load(
        case_path / f"segmentation.nii.gz"
    ).dataobj, dtype=np.int16)
