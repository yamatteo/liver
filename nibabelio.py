from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

import elasticdeform
import nibabel
import numpy as np
import torch
import torch.utils.data
from report import print
from torch.nn import functional


class Bundle:
    def __init__(self, scan, segm=None, name=None):
        if isinstance(scan, torch.Tensor):
            scan = scan.clone().detach().to(dtype=torch.float32)
        else:
            scan = torch.tensor(scan, dtype=torch.float32)
        assert (scan.ndim == 4 or scan.ndim == 5) and scan.size(-3) == scan.size(-2) == 512 and scan.size(-4) == 4, \
            f"Scan is expected to be of shape [N, 4, 512, 512, Z] or [4, 512, 512, Z], got {list(scan.shape)}."
        if isinstance(segm, torch.Tensor):
            segm = segm.clone().detach().to(dtype=torch.int64)
        elif segm is not None:
            segm = torch.tensor(segm, dtype=torch.int64)
        if segm is not None:
            assert scan.ndim == segm.ndim + 1, \
                f"Segm is expected to be categorical, got scan {list(scan.shape)} and segm {list(segm.shape)}."
            assert segm.shape[-3:] == scan.shape[-3:], \
                f"Tensors must have the same spacial dimensions, got scan {list(scan.shape)} and segm {list(segm.shape)}."
            assert scan.ndim == 4 or len(scan) == len(segm), \
                f"Batches should have same length, got scan {list(scan.shape)} and segm {list(segm.shape)}."
        self.scan: torch.Tensor = scan
        self.segm: torch.Tensor = segm
        self.name = name

    @property
    def is_batch(self) -> bool:
        return self.scan.ndim == 5

    @property
    def onehot_segm(self) -> torch.Tensor:
        oh_segm = functional.one_hot(self.segm, num_classes=3).to(dtype=torch.float32)
        if self.is_batch:
            return torch.permute(oh_segm, (0, 4, 1, 2, 3))
        else:
            return torch.permute(oh_segm, (3, 0, 1, 2))

    @property
    def shape(self) -> torch.Size:
        return self.scan.shape

    def deformed(self) -> Bundle:
        scan = self.scan.cpu().numpy()
        segm = self.onehot_segm.cpu().numpy()
        axis = (2, 3, 4) if self.is_batch else (1, 2, 3)
        scan, segm = elasticdeform.deform_random_grid(
            [scan, segm],
            sigma=np.broadcast_to(np.array([4, 4, 1]).reshape([3, 1, 1, 1]), [3, 5, 5, 5]),
            points=[5, 5, 5],
            axis=[axis, axis],
        )
        segm = np.argmax(segm, axis=1) if self.is_batch else np.argmax(segm, axis=0)
        return Bundle(scan, segm)

    def narrow(self, dim: int, start: int, length: int, pad: bool = True) -> Bundle:
        assert dim in [-3, -2, -1], "dim should be negative, indicating spatial dimension"
        available_length = min(length, self.scan.size(dim)-start)
        scan = torch.narrow(self.scan, dim=dim, start=start, length=available_length)
        if pad and available_length < length:
            pad_width = tuple(length - available_length if d == dim else 0 for d in (0, -1, 0, -2, 0, -3))
            pad = torch.nn.ReplicationPad3d(pad_width)
        else:
            pad = lambda x: x
        scan = pad(scan)

        if self.segm is None:
            segm = None
        else:
            segm = torch.narrow(self.onehot_segm, dim, start, min(length, self.scan.size(dim)-start))
            segm = pad(segm)
            segm = torch.argmax(segm, dim=1) if self.is_batch else torch.argmax(segm, dim=0)
        return Bundle(scan, segm)

    def slices(self, length: int = 1, step: int = None) -> Iterator[Bundle]:
        if step is None:
            step = max(1, length // 2)
        for start in range(0, self.shape[-1] - length // 2, step):
            yield self.narrow(-1, start, length)


### Nibabel Input/Output

def _load_ndarray(file_path: Path) -> np.ndarray:
    image = nibabel.load(file_path)
    return np.array(image.dataobj, dtype=np.int16)


def load_registration_data(case_path: Path) -> tuple[np.ndarray, int, int, int]:
    with open(case_path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def load(case_path: Path, train: bool = False, clip: tuple[int, int] = None):
    print(f"Loading {case_path}...")
    name = str(case_path.name)
    _, bottom, top, _ = load_registration_data(case_path)
    scan = np.stack([
        _load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
        for phase in ["b", "a", "v", "t"]
    ])
    scan = scan[..., bottom:top]
    if clip:
        np.clip(scan, *clip, out=scan)
    scan = scan.astype(np.float32)

    if train:
        segm = _load_ndarray(case_path / f"segmentation.nii.gz")
        assert np.all(segm < 3), "segmentation has indices above 2"
        segm = segm[..., bottom:top]
        segm = segm.astype(np.int64)
    else:
        segm = None

    return Bundle(scan, segm, name=name)


def load_train_dict(case_path: Path):
    _, bottom, top, _ = load_registration_data(case_path)
    scan = np.stack([
        _load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
        for phase in ["b", "a", "v", "t"]
    ])
    scan = scan[..., bottom:top]
    np.clip(scan, -300, 400, out=scan)
    scan = scan.astype(np.float32)

    segm = _load_ndarray(case_path / f"segmentation.nii.gz")
    assert np.all(segm < 3), "segmentation has indices above 2"
    segm = segm[..., bottom:top]
    segm = segm.astype(np.int64)

    return dict(name=case_path.name, scan=scan, segm=segm)
