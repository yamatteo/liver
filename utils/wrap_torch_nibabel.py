from __future__ import annotations

import pickle
from pathlib import Path

import nibabel
import numpy as np
import torch


def _load_array(file_path: Path) -> np.ndarray:
    image = nibabel.load(file_path)
    return np.array(image.dataobj, dtype=np.int16)


def load_registration_data(dir_path: Path) -> tuple[np.ndarray, int, int, int]:
    with open(dir_path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def save_prediction(dir_path: Path, t: torch.Tensor):
    dir_path = dir_path.parent
    try:
        affine, bottom, top, height = load_registration_data(dir_path)
    except FileNotFoundError:
        affine = np.eye(4)
        bottom = 0
        top = height = t.size(-1)
    data = np.zeros((512, 512, height))
    data[..., bottom:top] = t.to(dtype=torch.int16, device=torch.device("cpu")).numpy()
    image = nibabel.Nifti1Image(data, affine)
    nibabel.save(image, dir_path / "prediction.nii.gz")


def load_scan(dir_path: Path) -> torch.Tensor:
    _, bottom, top, _ = load_registration_data(dir_path)
    data = np.stack([
        _load_array(dir_path / f"registered_phase_{phase}.nii.gz")
        for phase in ["b", "a", "v", "t"]
    ])
    return torch.tensor(data[..., bottom:top], dtype=torch.float32)


def load_segm(dir_path: Path) -> torch.Tensor:
    _, bottom, top, _ = load_registration_data(dir_path)
    data = _load_array(dir_path / f"segmentation.nii.gz")
    assert np.all(data < 3), "segmentation has indices above 2"
    return torch.tensor(data[..., bottom:top], dtype=torch.int64)


# def save_original(image: nibabel.Nifti1Image, path: Path, phase: str):
#     return nibabel.save(image, path / f"original_phase_{phase}.nii.gz")
#
#
# def load_original(path: Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
#     image = nibabel.load(path / f"original_phase_{phase}.nii.gz")
#     data = np.array(image.dataobj, dtype=np.int16)
#     matrix = image.affine
#     return data, matrix
#
#
# def save_registereds(regs: dict[str, np.ndarray], path: Path, affine: np.ndarray, bottom: int, top: int, height: int):
#     # regs is a dictionary {phase: ndarray} of length 4
#     # each ndarray is [512, 512, z] and is already cropped (i.e. z = top-bottom)
#     for phase, data in regs.items():
#         _data = np.full([data.shape[0], data.shape[1], height], fill_value=-1024, dtype=np.int16)
#         _data[..., bottom:top] = data
#         image = nibabel.Nifti1Image(_data, affine)
#         nibabel.save(image, path / f"registered_phase_{phase}.nii.gz")
#     with open(path / "registration_data.pickle", "wb") as f:
#         pickle.dump({"affine": affine, "bottom": bottom, "top": top, "height": height}, f)
#
#
# def load_registered_with_matrix(path: Path, phase: str) -> tuple[np.ndarray, np.ndarray]:
#     image = nibabel.load(path / f"registered_phase_{phase}.nii.gz")
#     data = np.array(image.dataobj, dtype=np.int16)
#     matrix = image.affine
#     return data, matrix
#
#
# def load_registered(path: Path, phase: str) -> np.ndarray:
#     image = nibabel.load(path / f"registered_phase_{phase}.nii.gz")
#     return np.array(image.dataobj, dtype=np.int16)
#
#
# def save_scan(regs: dict[str, np.ndarray], path: Path, affine: np.ndarray, bottom: int, top: int, height: int):
#     # regs is a dictionary {phase: ndarray} of length 4
#     # each ndarray is [512, 512, z] and is already cropped (i.e. z = top-bottom)
#     scan = np.stack([regs["b"], regs["a"], regs["v"], regs["t"]], axis=0)
#     image = nibabel.Nifti1Image(scan, np.eye(4))
#     nibabel.save(image, path / f"scan.nii.gz")
#     with open(path / "registration_data.pickle", "wb") as f:
#         pickle.dump({"affine": affine, "bottom": bottom, "top": top, "height": height}, f)
#
#
# def load_scan(path: Path) -> np.ndarray:
#     image = nibabel.load(path / f"scan.nii.gz")
#     return np.array(image.dataobj, dtype=np.int16)
#
#
# def load_segm(path: Path, what: str = "segmentation") -> np.ndarray:
#     image = nibabel.load(path / f"{what}.nii.gz")
#     data = np.array(image.dataobj, dtype=np.int16)
#     assert np.all(data < 3), "segmentation has indices above 2"
#     return data
