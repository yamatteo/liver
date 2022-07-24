from __future__ import annotations

from pathlib import Path

import numpy as np
import pyelastix
import scipy.ndimage

import dataset.ndarray


def move_image(input: np.ndarray, input_matrix: np.ndarray, target: np.ndarray, target_matrix: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Perform affine transformations so that input uses the same homogeneous coordinates as target.

    Usually x and y coordinates don't change that much, and the body is well in the middle of the 512x512
    voxels space available. On the other hand, z varies a lot and significant portions of the body present
    in a phase are missing in another; not the part with the liver, though.
    moved_input has the shape of target and is initialized with value -1024 (value or air in Hounsfield units)
    bottom and top track the z index where input landed on moved_input after the affine transformation,
    so that moved_input[..., bottom:top] = change_in_coordinates(input)
    """
    matrix = np.linalg.inv(input_matrix) @ target_matrix
    moved_input = scipy.ndimage.affine_transform(input, matrix, output_shape=target.shape, cval=-1024)
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
    originals = {phase: dataset.ndarray.load_original(path, phase) for phase in ["b", "a", "v", "t"]}
    orig_v, matrix = originals["v"]
    moveds = {phase: move_image(*originals[phase], orig_v, matrix) for phase in ["b", "a", "t"]}
    bottom = max(_bottom for (_, _bottom, _) in moveds.values())
    top = min(_top for (_, _, _top) in moveds.values())
    height = orig_v.shape[2]
    assert 0 <= bottom < top <= height

    params = pyelastix.get_default_params()
    params.MaximumNumberOfIterations = 200
    params.FinalGridSpacingInVoxels = 32

    regs = {}
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


def registration_callback(event, path, overwrite):
    from dataset.path_explorer import iter_original
    if path is None:
        for case in iter_original(base_path):
            registration_callback(event, path=base_path / case, overwrite=False)
    else:
        source_path = target_path = path
        target_path_is_complete = all(
            (target_path / f"registered_phase_{phase}.nii.gz").exists()
            for phase in ["b", "a", "v", "t"]
        )
        if target_path_is_complete and not overwrite:
            console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
        else:
            console.print(f"[bold black]{case_path.name}.[/bold black] registering images...")
            target_path.mkdir(parents=True, exist_ok=True)
            register_case(path)