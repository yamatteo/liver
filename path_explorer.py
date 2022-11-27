from __future__ import annotations

import functools
import random
from pathlib import Path
from typing import Callable, Iterator

from rich import print


### Criteria
def contains(path: Path, filelist: list) -> bool:
    """True if path contains all listed files."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return all(f in files for f in filelist)

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
def iter_containing(path: Path, filelist: list) -> Iterator[Path]:
    """Iterates over subfolders containing listed files."""
    yield from discover(path, functools.partial(contains, filelist=filelist))

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


def split_trainables(path: Path, n: int = 10, shuffle=False, offset=0) -> tuple[list[Path], list[Path]]:
    """Lists of case_path for train and valid dataset."""
    trainables = sorted(list(iter_trainable(path)))
    if offset == "random":
        offset = random.randint(0, min(n-1, len(trainables)))
    if offset:
        trainables = [*trainables[offset:], *trainables[:offset]]
    train_cases = list(path / case for k, case in enumerate(trainables) if k % n != 0)
    valid_cases = list(path / case for k, case in enumerate(trainables) if k % n == 0)
    if shuffle:
        train_cases = random.sample(train_cases, len(train_cases))
    return train_cases, valid_cases