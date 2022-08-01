from __future__ import annotations

import pickle
from pathlib import Path

import nibabel
import numpy as np


def load_niftiimage(path: Path) -> nibabel.Nifti1Image:
    return nibabel.load(path)

def load_niftidata(path: Path) -> np.ndarray:
    image = nibabel.load(path)
    return np.array(image.dataobj, dtype=np.int16)

def save_niftiimage(path: Path, data, affine=np.eye(4)):
    image = nibabel.Nifti1Image(data, affine)
    nibabel.save(image, path)


def save_original(image: nibabel.Nifti1Image, path: Path, phase: str):
    return nibabel.save(image, path / f"original_phase_{phase}.nii.gz")


def load_original(path: Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
    image = nibabel.load(path / f"original_phase_{phase}.nii.gz")
    data = np.array(image.dataobj, dtype=np.int16)
    matrix = image.affine
    return data, matrix


def save_registereds(regs: dict[str, np.ndarray], path: Path, affine: np.ndarray, bottom: int, top: int, height: int):
    # regs is a dictionary {phase: ndarray} of length 4
    # each ndarray is [512, 512, z] and is already cropped (i.e. z = top-bottom)
    for phase, data in regs.items():
        _data = np.full([data.shape[0], data.shape[1], height], fill_value=-1024, dtype=np.int16)
        _data[..., bottom:top] = data
        image = nibabel.Nifti1Image(_data, affine)
        nibabel.save(image, path / f"registered_phase_{phase}.nii.gz")
    with open(path / "registration_data.pickle", "wb") as f:
        pickle.dump({"affine": affine, "bottom": bottom, "top": top, "height": height}, f)


def load_registered_with_matrix(path: Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
    image = nibabel.load(path / f"registered_phase_{phase}.nii.gz")
    data = np.array(image.dataobj, dtype=np.int16)
    matrix = image.affine
    return data, matrix


def load_registered(path: Path, phase: str) -> np.ndarray:
    image = nibabel.load(path / f"registered_phase_{phase}.nii.gz")
    return np.array(image.dataobj, dtype=np.int16)


def save_scan(regs: dict[str, np.ndarray], path: Path, affine: np.ndarray, bottom: int, top: int, height: int):
    # regs is a dictionary {phase: ndarray} of length 4
    # each ndarray is [512, 512, z] and is already cropped (i.e. z = top-bottom)
    scan = np.stack([regs["b"], regs["a"], regs["v"], regs["t"]], axis=0)
    image = nibabel.Nifti1Image(scan, np.eye(4))
    nibabel.save(image, path / f"scan.nii.gz")
    with open(path / "registration_data.pickle", "wb") as f:
        pickle.dump({"affine": affine, "bottom": bottom, "top": top, "height": height}, f)


def load_scan(path: Path) -> np.ndarray:
    image = nibabel.load(path / f"scan.nii.gz")
    return np.array(image.dataobj, dtype=np.int16)


def load_segm(path: Path) -> np.ndarray:
    image = nibabel.load(path / f"segmentation.nii.gz")
    data = np.array(image.dataobj, dtype=np.int16)
    assert np.all(data < 3), "segmentation has indices above 2"
    return data


def load_registration_data(path: Path) -> tuple[np.ndarray, int, int, int]:
    with open(path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def load_scan_from_regs(path: Path) -> np.ndarray:
    data = np.stack([
        np.array(nibabel.load(
            path / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16)
        for phase in ["b", "a", "v", "t"]
    ])
    _, bottom, top, _ = load_registration_data(path)
    return data[..., bottom:top]
