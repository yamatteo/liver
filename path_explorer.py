from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from options import defaults

console = Console()


def is_dicomdir(path: Path) -> bool:
    """A directory is a dicom if it contains a 'DICOMDIR' file."""
    return "DICOMDIR" in [item.name for item in path.iterdir()]


def is_original_dir(path: Path) -> bool:
    """A directory is a dicom if it contains a 'DICOMDIR' file."""
    return all(
        f"original_phase_{phase}.nii.gz" in [item.name for item in path.iterdir()]
        for phase in ["b", "a", "v", "t"]
    )


def is_case_dir(case_path: Path | str, segmented: bool):
    if not case_path.is_dir():
        return False
    files = [path.name for path in case_path.iterdir()]
    case_respect_segmented = (
            ("segmentation.nii.gz" in files)
            or not segmented
    )
    case_respect_multiphase = all(
        f"registered_phase_{phase}.nii.gz" in files
        for phase in ["b", "a", "v", "t"]
    )
    return case_respect_multiphase and case_respect_segmented


def discover_dicomdirs(path: Path, relative_to_path: Path | None = None) -> list[Path]:
    """Collects paths of all dicom sub-folders."""
    if relative_to_path is None:
        relative_to_path = path
    if path.is_dir():
        if is_dicomdir(path):
            return [path.relative_to(relative_to_path)]
        else:
            return [
                subdir
                for subpath in path.iterdir()
                for subdir in discover_dicomdirs(subpath, relative_to_path)
            ]
    else:
        return []


def discover_original_dirs(path: Path, relative_to_path: Path | None = None) -> list[Path]:
    """Collects paths of all sub-folders containing original nifti."""
    if relative_to_path is None:
        relative_to_path = path
    if path.is_dir():
        if is_original_dir(path):
            return [path.relative_to(relative_to_path)]
        else:
            return [
                subdir
                for subpath in path.iterdir()
                for subdir in discover_original_dirs(subpath, relative_to_path)
            ]
    else:
        return []


def discover_cases_dirs(path: Path, segmented: bool, relative_to_path: Path | None = None) -> list[Path]:
    """Collects paths of all sub-folders containing cases."""
    if relative_to_path is None:
        relative_to_path = path
    if path.is_dir():
        if is_case_dir(path, segmented):
            return [path.relative_to(relative_to_path)]
        else:
            return [
                subdir
                for subpath in path.iterdir()
                for subdir in discover_cases_dirs(subpath, segmented, relative_to_path)
            ]
    else:
        return []


def get_args():
    parser = argparse.ArgumentParser()
    return dict(defaults, **vars(parser.parse_args()))


if __name__ == "__main__":
    opts = get_args()
    console.print(f"In the given path ({opts['sources']}) there are these dicom dirs:")
    for path in discover_dicomdirs(opts["sources"]):
        console.print("  ", path)
