# -*- coding: utf-8 -*-
from __future__ import annotations

import heapq
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterator, Iterable

import elasticdeform
import nibabel
import numpy as np
import torch
import torch.utils.data
from rich import print
from rich.progress import Progress

from numpy_slicing import halfstep_z_slices as hzs

### Criteria

def is_anything(path: Path) -> bool:
    """True if path contains something related to this project."""
    return is_dicom(path) or is_original(path) or is_trainable(path)


def is_dicom(path: Path) -> bool:
    """True if path contains DICOMDIR."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return "DICOMDIR" in files


def is_original(path: Path) -> bool:
    """True if path contains original nifti scans."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return all(
        f"original_phase_{phase}.nii.gz" in files
        for phase in ["b", "a", "v", "t"]
    )


def is_registered(path: Path) -> bool:
    """True if path contains registered nifti scans."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return all(
        f"registered_phase_{phase}.nii.gz" in files
        for phase in ["b", "a", "v", "t"]
    )


def is_predicted(path: Path) -> bool:
    """True if path contains prediction."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return "prediction.nii.gz" in files


def is_trainable(path: Path) -> bool:
    """True if path contains segmentation and registered nifti scans."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return "segmentation.nii.gz" in files and all(
        f"registered_phase_{phase}.nii.gz" in files
        for phase in ["b", "a", "v", "t"]
    )


### Discover utility

def discover(path: Path | str, select_dir: Callable = is_anything) -> list[Path]:
    """Recursively list dirs in `path` that respect `select_dir` criterion."""
    path = Path(path).resolve()
    unexplored_paths = [path]
    selected_paths = []
    while len(unexplored_paths) > 0:
        new_path = unexplored_paths.pop(0)
        if select_dir(new_path):
            selected_paths.append(new_path.resolve().relative_to(path))
        elif new_path.is_dir():
            unexplored_paths.extend(new_path.iterdir())
    selected_paths.sort()
    return selected_paths


def recurse(base_path: Path, select_dir: Callable = is_anything, **kwargs):
    opening = kwargs.get("opening", None)
    case_in = kwargs.get("case_in", None)
    case_out = kwargs.get("case_out", None)

    def _recurse(func):
        if opening:
            print(opening)
        returns = {}
        for case in discover(base_path, select_dir):
            if case_in:
                print(case_in.format(case=case))
            ret = func(case_path=base_path / case)
            if ret:
                returns[case] = ret
            if case_out:
                print(case_out.format(case=case))
        return returns

    return _recurse


### Iterators

def iter_dicom(path: Path) -> Iterator[Path]:
    """Iterates over DICOMDIR subfolders."""
    yield from discover(path, is_dicom)


def iter_original(path: Path) -> Iterator[Path]:
    """Iterates over subfolders containing original nifti scans."""
    yield from discover(path, is_original)


def iter_registered(path: Path) -> Iterator[Path]:
    """Iterates over subfolders containing registered nifti scans."""
    yield from discover(path, is_registered)


def iter_trainable(path: Path) -> Iterator[Path]:
    """Iterates over subfolders containing registered nifti scans and segmentations."""
    yield from discover(path, is_trainable)


def split_iter_trainables(path: Path) -> tuple[Iterator[Path], Iterator[Path]]:
    return (
        (path / case for k, case in enumerate(iter_trainable(path)) if k % 10 != 0),
        (path / case for k, case in enumerate(iter_trainable(path)) if k % 10 == 0),
    )


# def train_cases(dir_path: Path):
#     for k, case in enumerate(iter_trainable(dir_path)):
#         if k%10 != 0:
#             yield dir_path / case
#
#
# def cyclic_train_cases(dir_path: Path):
#     while True:
#         for k, case in enumerate(iter_trainable(dir_path)):
#             if k%10 != 0:
#                 yield dir_path / case
#
#
# def valid_cases(dir_path: Path):
#     for k, case in enumerate(iter_trainable(dir_path)):
#         if k%10 == 0:
#             yield dir_path / case


### Nibabel Input/Output

def _load_ndarray(file_path: Path) -> np.ndarray:
    image = nibabel.load(file_path)
    return np.array(image.dataobj, dtype=np.int16)


def load_registration_data(case_path: Path) -> tuple[np.ndarray, int, int, int]:
    with open(case_path / "registration_data.pickle", "rb") as f:
        d = pickle.load(f)
    return d["affine"], d["bottom"], d["top"], d["height"]


def load(case_path: Path, train: bool = False, clip: tuple[int, int] = None):
    global scan
    global segm
    print(f"Loading {case_path}...")
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


def argmax(x):
    x = np.asarray(x)
    assert x.shape[0] == 3
    return np.argmax(x, axis=0)


def onehot(x):
    x = np.asarray(x)
    permutation = (len(x.shape), *range(len(x.shape)))
    x = np.eye(3, dtype=np.float32)[x]
    return np.transpose(x, permutation)


def deformation():
    global scan
    global segm
    print("Applying random elastic deformation...")
    segm = onehot(segm)
    assert scan.ndim == segm.ndim == 4, \
        f"Scan and segm are expected to be of shape [C, X, Y, Z], got {scan.shape}, {segm.shape}."
    scan, segm = elasticdeform.deform_random_grid(
        [scan, segm],
        sigma=np.broadcast_to(np.array([4, 4, 1]).reshape([3, 1, 1, 1]), [3, 5, 5, 5]),
        points=[5, 5, 5],
        axis=[(1, 2, 3), (1, 2, 3)],
    )
    segm = argmax(segm)


# def indices(shape: tuple[int, int, int], step: tuple[int, int, int]) -> Iterator[tuple[int, int, int]]:
#     for i in range(0, shape[0]-step[0], step[0]):
#         for j in range(0, shape[1]-step[1], step[1]):
#             for k in range(0, shape[2], step[2]):
#                 yield (i, j, k)
#
#
# def pad(t: np.ndarray, shape: tuple[int, int, int]):
#     pad_width = [(0, 0) for _ in t.shape]
#     for n in (-3, -2, -1):
#         assert t.shape[n] <= shape[n]
#         pad_width[n] = (0, shape[n] - t.shape[n])
#     return np.pad(t, pad_width, mode="edge")


def slices(cases: Iterator, *, clip: tuple[int, int] = (-300, 400), deform: bool = False, length: int = 1,
           train: bool = False):
    global scan
    global segm
    for case_path in cases:
        load(case_path, train=train, clip=clip)
        if deform:
            deformation()
        for (scan_slice, segm_slice) in hzs(scan, segm, length=length):
            yield dict(
                scan=torch.tensor(scan_slice),
                segm=torch.tensor(segm_slice),
            )
        # for (i, j, k) in indices(scan.shape[-3:], step):
        #     print(f"scan is {scan.shape}, shape is {shape}, indices are {(i, j, k)}")
        #     yield dict(
        #         scan=torch.tensor(pad(scan[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
        #         segm=torch.tensor(pad(segm[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
        #     )


def repeat(generator: Iterable, stop_after: int = None):
    i = 0
    while stop_after is None or i < stop_after:
        for x in iter(generator):
            yield x
        i += 1


def load_generated(path: Path):
    import psutil
    for item in path.iterdir():
        print("DEBUG", "Loading", item)
        print("DEBUG", psutil.virtual_memory().percent, "percent of memory")
        try:
            yield torch.load(item)
        except Exception as err:
            print(err)


### Datasets

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator: Iterator[dict], *, buffer_size: int = None, staging_size: int = None):
        super(GeneratorDataset, self).__init__()
        self.generator = generator
        if buffer_size is None:
            self.buffer = list(generator)
            self.buffer_size = len(self.buffer)
            self.staging_size = None
        else:
            self.buffer = []
            self.buffer_size = buffer_size
            self.staging_size = staging_size
            self.fill()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, i: int):
        return {"keys": i, **self.buffer[i]}

    def fill(self, size=None):
        size = size or self.buffer_size
        for _ in range(size):
            if len(self.buffer) >= size:
                break
            self.buffer.append(next(self.generator))

    def drop(self, scores: dict[int, float]):
        smallest = heapq.nsmallest(len(self.buffer) - self.buffer_size + self.staging_size, list(scores.keys()),
                                   lambda i: scores[i])
        smallest = reversed(sorted(smallest))
        for k in smallest:
            del self.buffer[k]
        self.fill()

    def warmup(self, size, evaluator):
        item_score = namedtuple('item_score', ['item', 'score'])
        buffer = []
        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Populating dataset buffer, {size} item to process.".ljust(50, ' '),
                total=size
            )
            for i in range(size):
                item = next(self.generator)
                score = evaluator(item)
                buffer.append(item_score(item, score))
                if len(buffer) > self.buffer_size:
                    buffer = sorted(buffer, key=lambda t: -t.score)[:self.buffer_size]
                progress.update(task, advance=1)
        self.buffer = [t.item for t in buffer]


# def load_scan_ndarray(dir_path: Path):
#     global loaded_case
#     _, bottom, top, _ = load_registration_data(dir_path)
#     loaded_case.scan = np.stack([
#         _load_ndarray(dir_path / f"registered_phase_{phase}.nii.gz")
#         for phase in ["b", "a", "v", "t"]
#     ])
#     loaded_case.scan = loaded_case.scan[..., bottom:top]
#     loaded_case.scan = loaded_case.scan.astype(np.float32)


# def load_segm_ndarray(dir_path: Path):
#     global loaded_case
#     _, bottom, top, _ = load_registration_data(dir_path)
#     loaded_case.segm = _load_ndarray(dir_path / f"segmentation.nii.gz")
#     assert np.all(loaded_case.segm < 3), "segmentation has indices above 2"
#     loaded_case.segm = loaded_case.segm[..., bottom:top]
#     loaded_case.segm = loaded_case.segm.astype(np.int64)


# def load_train_case_ndarray(dir_path: Path):
#     global loaded_case
#     _, bottom, top, _ = load_registration_data(dir_path)
#     loaded_case.scan = np.stack([
#         _load_ndarray(dir_path / f"registered_phase_{phase}.nii.gz")
#         for phase in ["b", "a", "v", "t"]
#     ])
#     loaded_case.scan = loaded_case.scan[..., bottom:top]
#     loaded_case.scan = loaded_case.scan.astype(np.float32)

#     loaded_case.segm = _load_ndarray(dir_path / f"segmentation.nii.gz")
#     assert np.all(loaded_case.segm < 3), "segmentation has indices above 2"
#     loaded_case.segm = loaded_case.segm[..., bottom:top]
#     loaded_case.segm = loaded_case.segm.astype(np.int64)


# def load_apply_case_ndarray(dir_path: Path):
#     global loaded_case
#     _, bottom, top, _ = load_registration_data(dir_path)
#     loaded_case.scan = np.stack([
#         _load_ndarray(dir_path / f"registered_phase_{phase}.nii.gz")
#         for phase in ["b", "a", "v", "t"]
#     ])
#     loaded_case.scan = loaded_case.scan[..., bottom:top]
#     loaded_case.scan = loaded_case.scan.astype(np.float32)


# def save_ndarray_pred(dir_path: Path, pred: np.ndarray):
#     """pred is supposed to be 512x512xZ int16"""
#     assert False, f"DEBUG: Is the correct path {dir_path} or is it {dir_path.parent}?"
#     dir_path = dir_path.parent
#     try:
#         affine, bottom, top, height = load_registration_data(dir_path)
#     except FileNotFoundError:
#         affine = np.eye(4)
#         bottom = 0
#         top = height = t.size(-1)
#     data = np.zeros((512, 512, height))
#     data[..., bottom:top] = pred
#     # data[..., bottom:top] = t.to(dtype=torch.int16, device=torch.device("cpu")).numpy()
#     image = nibabel.Nifti1Image(data, affine)
#     nibabel.save(image, dir_path / "prediction.nii.gz")


# def load_scan(dir_path: Path) -> torch.Tensor:
#     _, bottom, top, _ = load_registration_data(dir_path)
#     data = np.stack([
#         _load_ndarray(dir_path / f"registered_phase_{phase}.nii.gz")
#         for phase in ["b", "a", "v", "t"]
#     ])

#     return torch.tensor(data[..., bottom:top], dtype=torch.float32)


# def load_segm(dir_path: Path) -> torch.Tensor:
#     _, bottom, top, _ = load_registration_data(dir_path)
#     data = _load_ndarray(dir_path / f"segmentation.nii.gz")
#     assert np.all(data < 3), "segmentation has indices above 2"
#     return torch.tensor(data[..., bottom:top], dtype=torch.int64)


# def narrow(t: np.ndarray, axis: int, start: int, length: int):
#     return t[(*([slice(None, None)] * axis), slice(start, start + length))]


# def pad(t: np.ndarray, axis: int, width: int = 1, mode: str = "edge"):
#     return np.pad(t, [*([(0, 0)] * axis), (0, width)], mode=mode)


# def dimensional_slices(t: np.ndarray, axis: int, thickness: int, step: int = None, pad_mode: str = "edge") -> Iterator[ndarray]:
#     length = t.shape[axis]
#     if step is None:
#         step = thickness
#     for start in range(0, length, step):
#         slice = narrow(t, axis, start, thickness)
#         width = thickness - slice.shape[axis]
#         if width > 0:
#             slice = pad(slice, axis, width, mode=pad_mode)
#         yield slice


# def slices(t: np.ndarray, axes: tuple[int, int, int], shape: tuple[int, int, int], step: tuple[int, int, int], pad_mode: str = "edge") -> Iterator[ndarray]:
#     for x in dimensional_slices(t, axes[0], shape[0], step[0], pad_mode):
#         for y in dimensional_slices(x, axes[1], shape[1], step[1], pad_mode):
#             for z in dimensional_slices(y, axes[2], shape[2], step[2], pad_mode):
#                 yield z


# def train_slices(dir_path: Path, shape: tuple[int, int, int], step: tuple[int, int, int]):
#     global scan
#     global segm
#     for case_path in cyclic_train_cases(dir_path):
#         slice_count = 0
#         load(case_path, train=True, clip=(-300, 400))
#         deformation()
#         # print(f"indices({scan.shape[-3:]}, {step}) are", list(indices(scan.shape[-3:], step)))
#         for (i, j, k) in indices(scan.shape[-3:], step):
#             slice_count += 1
#             yield dict(
#                 scan=torch.tensor(pad(scan[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
#                 segm=torch.tensor(pad(segm[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
#             )
#         print(f"Case {case_path} produced {slice_count} slices.")
#
#
# def valid_slices(dir_path: Path, shape: tuple[int, int, int], step: tuple[int, int, int]):
#     global scan
#     global segm
#     for case_path in valid_cases(dir_path):
#         load(case_path, train=True, clip=(-300, 400))
#         # deform()
#         for (i, j, k) in indices(scan.shape[-3:], step):
#             yield dict(
#                 scan=torch.tensor(pad(scan[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
#                 segm=torch.tensor(pad(segm[..., i:i + shape[0], j:j + shape[1], k:k + shape[2]], shape)),
#             )


# class FixedDataset(torch.utils.data.Dataset):
#     def __init__(self, generator: Iterator[dict]):
#         super().__init__()
#         self.items = list(generator)
#
#     def __len__(self):
#         return len(self.items)
#
#     def __getitem__(self, i: int):
#         return self.items[i]


### Global variables

scan = None
segm = None
