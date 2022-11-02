from __future__ import annotations

import pickle
import re
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import dicom2nifti
import nibabel
import numpy as np
import pyelastix
from scipy import ndimage as scipy_ndimage


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


def load_segm(path: Path, what: str = "segmentation") -> np.ndarray:
    image = nibabel.load(path / f"{what}.nii.gz")
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


def get_phase(phase: str) -> Optional[str]:
    """Classify the matched string about phase."""
    if phase == "basale":
        return "b"
    elif phase == "arteriosa":
        return "a"
    elif phase == "venosa" or phase == "portale":
        return "v"
    elif phase == "tardiva":
        return "t"
    else:
        return None


def get_index(index: str) -> Optional[int]:
    """Convert the matched string about index to int, if possible."""
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def get_info(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract index and phase information from filename."""
    r = r"^(?:(?P<index>\d+)_)?(?P<phase>[a-z]+)?(?:_.+)?\.nii(?:\.gz)?"
    m = re.match(r, filename)
    if m is not None:
        return get_index(m.group("index")), get_phase(m.group("phase"))
    else:
        return None, None


def process_dicomdir(source_path: Path, target_path: Path):
    """Convert scans in `source_path` from dicom to nifti."""
    case_name = source_path.name

    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)

        # Convert the whole directory
        log = StringIO()
        with redirect_stderr(log):
            dicom2nifti.convert_directory(
                dicom_directory=str(source_path),
                output_folder=str(temp_path),
                compression=True
            )
        log = log.getvalue()
        if len(log) > 1:
            with open(target_path / "preprocessing.log", "w") as logfile:
                logfile.write(log)

        # List what generated nifti is a 512x512xD image for the appropriate phase
        items = []
        for path in temp_path.iterdir():
            index, phase = get_info(str(path.name))
            if index is not None or phase is not None:
                image = load_niftiimage(temp_path / path.name)
                if image.shape[:2] == (512, 512):
                    items.append((index, phase, image))
                else:
                    # TODO what for other resolutions
                    pass
            else:
                continue

        # Following "b", "a", "v", "t" order, choose an image in this way:
        #   - the image with correct phase and least index (but bigger than the last used)
        #   - any image with correct phase and None index
        #   - any image with None phase and least index (but bigger than the last used)
        #   - any image with None phase and None index
        phases = {}
        min_index = 0

        def ordering(_tuple):
            index, phase, image = _tuple
            if index is None:
                index = 8000
            if phase is None:
                phase = 16000
            else:
                phase = 0
            return index + phase

        for try_phase in ["b", "a", "v", "t"]:
            candidates = sorted(
                [
                    (index, phase, image)
                    for index, phase, image in items
                    if (index is None or index >= min_index) and (phase is None or phase == try_phase)
                ],
                key=ordering
            )
            if len(candidates) > 0:
                phases[try_phase] = candidates[0][2]
                if candidates[0][0] is not None:
                    min_index = candidates[0][0]

        # Check if there is one image per phase
        for phase in ["b", "a", "v", "t"]:
            if phase not in phases.keys():
                print(
                    f"{' ' * len(case_name)}  "
                    f"No image for phase [italic cyan]{phase}[/italic cyan]."
                )
                raise ValueError

        # If there is exactly one image per phase, save them as compressed nifti
        for phase in ["b", "a", "v", "t"]:
            phases[phase].header.set_sform(phases[phase].affine)
            phases[phase].header.set_qform(phases[phase].affine)
            save_original(phases[phase], target_path, phase)
        print(
            f"{' ' * len(case_name)}  "
            f"Original images saved in {target_path.absolute()}."
        )


def move_image(input: np.ndarray, input_matrix: np.ndarray, target: np.ndarray, target_matrix: np.ndarray) \
        -> tuple[np.ndarray, int, int]:
    """Perform affine transformations so that input uses the same homogeneous coordinates as target.

    Usually x and y coordinates don't change that much, and the body is well in the middle of the 512x512
    voxels space available. On the other hand, z varies a lot and significant portions of the body present
    in a phase are missing in another; not the part with the liver, though.
    moved_input has the shape of target and is initialized with value -1024 (value or air in Hounsfield units)
    bottom and top track the z index where input landed on moved_input after the affine transformation,
    so that moved_input[..., bottom:top] = change_in_coordinates(input)
    """
    matrix = np.linalg.inv(input_matrix) @ target_matrix
    moved_input = scipy_ndimage.affine_transform(input, matrix, output_shape=target.shape, cval=-1024)
    max_z = target.shape[2]
    bottom = 0
    for z in range(max_z):
        bottom = z
        if not np.all(moved_input[..., z] <= -1024):
            break
    top = max_z
    for z in reversed(range(max_z)):
        if not np.all(moved_input[..., z] <= -1024):
            break
        top = z
    return moved_input, bottom, top


def regs_dict_from(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, int, int, int]:
    """Phases b, a and t are aligned (with a non-rigid transformation) to phase v on the space they share.

    Returns (regs, matrix, bottom, top, height) where:
        - regs is a dict {phase: data} where, for each phase b, a, v and t, data is aligned with the v-phase
        - matrix is the affine matrix for homogeneous coordinates of original phase v
        - bottom, top and height: if the original v phase has shape [512, 512, height],
          then all registered phases are cropped to the same space equivalent to [512, 512, bottom:top]
    """
    originals = {phase: load_original(path, phase) for phase in ["b", "a", "v", "t"]}
    orig_v, matrix = originals["v"]
    moveds = {phase: move_image(*originals[phase], orig_v, matrix) for phase in ["b", "a", "t"]}
    bottom = max(_bottom for (_, _bottom, _) in moveds.values())
    top = min(_top for (_, _, _top) in moveds.values())
    height = orig_v.shape[2]
    assert 0 <= bottom < top <= height, f"bottom={bottom}, top={top}, height={height}"

    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 32

    regs = dict()
    regs["v"] = orig_v[..., bottom:top]
    for phase in ["b", "a", "t"]:
        data, _ = pyelastix.register(
            np.ascontiguousarray(moveds[phase][0][..., bottom:top]),
            np.ascontiguousarray(regs["v"]),
            params=params,
            verbose=0,
        )
        regs[phase] = data

    return regs, matrix, bottom, top, height


def register_case(path):
    regs, matrix, bottom, top, height = regs_dict_from(path)
    save_registereds(regs, path=path, affine=matrix, bottom=bottom, top=top, height=height)


if __name__ == "__main__":
    import path_explorer as px

    # # Nifti conversion
    # sources = Path("/gpfswork/rech/otc/uiu95bi/HUMANITAS/CT SCAN")
    # targets = Path("/gpfswork/rech/otc/uiu95bi/sources")
    # for folder_name in px.iter_dicom(sources):
    #     match str(folder_name).split():
    #         case [case_name, "ok"]:
    #             pass
    #         case [case_name]:
    #             pass
    #     case_path = targets / case_name
    #     if px.is_original(case_path) or px.is_registered(case_path):
    #         pass
    #     else:
    #         (targets / case_name).mkdir()
    #         # print("Want to process", sources / folder_name, ">>", targets / case_name)
    #         process_dicomdir(sources / folder_name, targets / case_name)

    # Registration
    sources = Path("/gpfswork/rech/otc/uiu95bi/sources")
    #sources = Path("/home/yamatteo/jzfs/sources")
    for case_name in px.iter_original(sources):
        case_path = sources / case_name
        if px.is_registered(case_path):
            print(case_name, "is already registered. Should clear?")
        else:
            try:
                print("Registering", case_name)
                register_case(case_path)
            except AssertionError as err:
                print(err)
