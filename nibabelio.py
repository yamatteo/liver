from __future__ import annotations

import pickle
from pathlib import Path

import nibabel
import numpy as np
from rich import print


### Nibabel Input/Output

def load_ndarray(file_path: Path) -> np.ndarray:
    """
    Load a NIfTI file using nibabel and return its data as a NumPy array.

    Parameters:
        file_path (Path): The path to the NIfTI file.

    Returns:
        np.ndarray: The NIfTI image data as a NumPy array (int16).
    """
    image = nibabel.load(file_path)
    return np.array(image.dataobj, dtype=np.int16)


def load_registration_data(case_path: Path) -> tuple[np.ndarray, int, int, int]:
    """
    Load registration data from a pickle file associated with a case.

    Parameters:
        case_path (Path): The path to the case directory.

    Returns:
        tuple[np.ndarray, int, int, int]: A tuple containing affine matrix,
            bottom, top, and height values from the registration data.
    """
    with open(case_path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def load(case_path: Path, scan: bool = True, segm: bool = False, clip: tuple[int, int] = None) -> dict:
    """
    Load data from a case directory, including registered scan and segmentation.

    Parameters:
        case_path (Path): The path to the case directory.
        scan (bool, optional): Whether to load the registered scan data. Default is True.
        segm (bool, optional): Whether to load the segmentation data. Default is False.
        clip (tuple[int, int], optional): Tuple specifying the lower and upper clip values for the scan data.

    Returns:
        dict: A dictionary containing the loaded data, including 'scan', 'segm', and 'name'.
            'scan' and 'segm' may be None if not requested.
    """
    print(f"Loading {case_path}...")
    name = str(case_path.name)
    _, bottom, top, _ = load_registration_data(case_path)
    if scan:
        scan = np.stack([
            load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
            for phase in ["b", "a", "v", "t"]
        ])
        scan = scan[..., bottom:top]
        if clip:
            np.clip(scan, *clip, out=scan)
        scan = scan.astype(np.float32)
    else:
        scan = None

    if segm:
        segm = load_ndarray(case_path / f"segmentation.nii.gz")
        assert np.all(segm < 3), "segmentation has indices above 2"
        segm = segm[..., bottom:top]
        segm = segm.astype(np.int64)
    else:
        segm = None

    return dict(scan=scan, segm=segm, name=name)
