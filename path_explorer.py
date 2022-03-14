from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from rich.console import Console

from options import defaults

console = Console()


# def is_dicomdir(path: Path) -> bool:
#     """A directory is a dicom if it contains a 'DICOMDIR' file."""
#     return "DICOMDIR" in [item.name for item in path.iterdir()]
#
#
# def is_original_dir(path: Path) -> bool:
#     """A directory is a dicom if it contains a 'DICOMDIR' file."""
#     return all(
#         f"original_phase_{phase}.nii.gz" in [item.name for item in path.iterdir()]
#         for phase in ["b", "a", "v", "t"]
#     )
#
#
# def is_case_dir(case_path: Path, segmented: bool):
#     if not case_path.is_dir():
#         return False
#     files = [path.name for path in case_path.iterdir()]
#     case_respect_segmented = (
#             ("segmentation.nii.gz" in files)
#             or not segmented
#     )
#     case_respect_multiphase = all(
#         f"registered_phase_{phase}.nii.gz" in files
#         for phase in ["b", "a", "v", "t"]
#     )
#     return case_respect_multiphase and case_respect_segmented
#
#
# def discover_dicomdirs(path: Path, relative_to_path: Path | None = None) -> list[Path]:
#     """Collects paths of all dicom sub-folders."""
#     if relative_to_path is None:
#         relative_to_path = path
#     if path.is_dir():
#         if is_dicomdir(path):
#             return [path.relative_to(relative_to_path)]
#         else:
#             return [
#                 subdir
#                 for subpath in path.iterdir()
#                 for subdir in discover_dicomdirs(subpath, relative_to_path)
#             ]
#     else:
#         return []
#
#
# def discover_original_dirs(path: Path, relative_to_path: Path | None = None) -> list[Path]:
#     """Collects paths of all sub-folders containing original nifti."""
#     if relative_to_path is None:
#         relative_to_path = path
#     if path.is_dir():
#         if is_original_dir(path):
#             return [path.relative_to(relative_to_path)]
#         else:
#             return [
#                 subdir
#                 for subpath in path.iterdir()
#                 for subdir in discover_original_dirs(subpath, relative_to_path)
#             ]
#     else:
#         return []
#
#
# def discover_cases_dirs(path: Path, segmented: bool, relative_to_path: Path | None = None) -> list[Path]:
#     """Collects paths of all sub-folders containing cases."""
#     if relative_to_path is None:
#         relative_to_path = path
#     if path.is_dir():
#         if is_case_dir(path, segmented):
#             return [path.relative_to(relative_to_path)]
#         else:
#             return [
#                 subdir
#                 for subpath in path.iterdir()
#                 for subdir in discover_cases_dirs(subpath, segmented, relative_to_path)
#             ]
#     else:
#         return []
#
#
# def is_case(case_path: Path):
#     if case_path.is_dir():
#         files = [file_path.name for file_path in case_path.iterdir()]
#         return all(
#             f"registered_phase_{phase}.nii.gz" in files
#             for phase in ["b", "a", "v", "t"]
#         )
#     else:
#         return False
#
#
# def is_segmented_case(case_path: Path):
#     if case_path.is_dir():
#         files = [file_path.name for file_path in case_path.iterdir()]
#         case_respect_segmented = "segmentation.nii.gz" in files
#         case_respect_multiphase = all(
#             f"registered_phase_{phase}.nii.gz" in files
#             for phase in ["b", "a", "v", "t"]
#         )
#         return case_respect_multiphase and case_respect_segmented
#     else:
#         return False
#
#
# def is_windowed_case(case_path: Path):
#     if case_path.is_dir():
#         files = [file_path.name for file_path in case_path.iterdir()]
#         case_respect_multiphase = all(
#             f"registered_phase_{phase}.nii.gz" in files
#             for phase in ["b", "a", "v", "t"]
#         )
#         case_respect_windowed = "zwindow.pt" in files
#         return case_respect_multiphase and case_respect_windowed
#     else:
#         return False
#
#
# def is_segmented_windowed_case(case_path: Path):
#     if case_path.is_dir():
#         files = [file_path.name for file_path in case_path.iterdir()]
#         case_respect_segmented = "segmentation.nii.gz" in files
#         case_respect_multiphase = all(
#             f"registered_phase_{phase}.nii.gz" in files
#             for phase in ["b", "a", "v", "t"]
#         )
#         case_respect_windowed = "zwindow.pt" in files
#         return case_respect_multiphase and case_respect_segmented and case_respect_windowed
#     else:
#         return False


def get_criterion(
        dicom: bool = False,
        original: bool = False,
        registered: bool = False,
        segmented: bool = False,
        windowed: bool = False
) -> Callable[[Path], bool]:
    """Make a callable to identify folders."""
    def criterion(case_path: Path) -> bool:
        if case_path.is_dir():
            files = [file_path.name for file_path in case_path.iterdir()]

            # Every condition is checked only if specified
            is_dicom = not dicom or "DICOMDIR" in files
            is_original = not original or all(
                f"original_phase_{phase}.nii.gz" in files
                for phase in ["b", "a", "v", "t"]
            )
            is_registered = not registered or all(
                f"registered_phase_{phase}.nii.gz" in files
                for phase in ["b", "a", "v", "t"]
            )
            is_segmented = not segmented or "segmentation.nii.gz" in files
            is_windowed = not windowed or "zwindow.pt" in files
            return is_dicom and is_original and is_registered and is_segmented and is_windowed
        else:
            return False

    return criterion


def discover(path: Path | str, select_dir: Callable) -> list[Path]:
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


def get_args():
    parser = argparse.ArgumentParser()
    return dict(defaults, **vars(parser.parse_args()))


if __name__ == "__main__":
    opts = get_args()
    console.print(f"In the given path ({opts['sources']}) there are these dicom dirs:")
    for path in discover(opts["sources"], get_criterion(dicom=True)):
        console.print("  ", Path("...") / path)
