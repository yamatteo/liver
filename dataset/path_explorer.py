from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from rich.console import Console

console = Console()


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


def is_trainable(path: Path) -> bool:
    """True if path contains segmentation and registered nifti scans."""
    if not path.is_dir():
        return False
    files = [file_path.name for file_path in path.iterdir()]
    return "segmentation.nii.gz" in files and all(
        f"registered_phase_{phase}.nii.gz" in files
        for phase in ["b", "a", "v", "t"]
    )


### Iterators
def iter_dicom(path: Path) -> Iterator[Path]:
    """Iterates over DICOMDIR subfolders."""
    yield from discover(path, is_dicom)


def iter_original(path: Path) -> Iterator[Path]:
    """Iterates over subfolders contaning original nifti scans."""
    yield from discover(path, is_original)


def iter_trainable(path: Path) -> Iterator[Path]:
    """Iterates over subfolders contaning original nifti scans."""
    yield from discover(path, is_trainable)


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
    return selected_paths