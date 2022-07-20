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

# def get_criterion(
#         dicom: bool = False,
#         original: bool = False,
#         registered: bool = False,
#         segmented: bool = False,
#         predicted: bool = False,
#         windowed: bool = False
# ) -> Callable[[Path], bool]:
#     """Make a callable to identify folders."""
#
#     def criterion(case_path: Path) -> bool:
#         if case_path.is_dir():
#             files = [file_path.name for file_path in case_path.iterdir()]
#
#             # Every condition is checked only if specified
#             is_dicom = not dicom or "DICOMDIR" in files
#             is_original = not original or all(
#                 f"original_phase_{phase}.nii.gz" in files
#                 for phase in ["b", "a", "v", "t"]
#             )
#             is_registered = not registered or all(
#                 f"registered_phase_{phase}.nii.gz" in files
#                 for phase in ["b", "a", "v", "t"]
#             )
#             is_segmented = not segmented or "segmentation.nii.gz" in files
#             is_predicted = not predicted or "prediction.nii.gz" in files
#             is_windowed = not windowed or "zwindow.pt" in files
#             return is_dicom and is_original and is_registered and is_segmented and is_predicted and is_windowed
#         else:
#             return False
#
#     return criterion
#
#
# def criterion(
#         bundle: bool = False,
#         dicom: bool = False,
#         original: bool = False,
#         registered: bool = False,
#         segmented: bool = False,
#         predicted: bool = False
# ) -> Callable[[Path], bool]:
#     """Make a callable to identify folders."""
#
#     def _criterion(path: Path) -> bool:
#         if path.is_dir():
#             files = [file_path.name for file_path in path.iterdir()]
#             is_case_path = True
#
#             # Every condition is checked only if specified
#             if dicom:
#                 is_case_path = is_case_path and "DICOMDIR" in files
#             if original:
#                 is_case_path = is_case_path and all(
#                     f"original_phase_{phase}.nii.gz" in files
#                     for phase in ["b", "a", "v", "t"]
#                 )
#             if registered:
#                 is_case_path = is_case_path and all(
#                     f"registered_phase_{phase}.nii.gz" in files
#                     for phase in ["b", "a", "v", "t"]
#                 )
#             if segmented:
#                 is_case_path = is_case_path and "segmentation.nii.gz" in files
#             if predicted:
#                 is_case_path = is_case_path and "prediction.nii.gz" in files
#             if bundle:
#                 is_case_path = is_case_path and "train_bundle.nii.gz" in files
#             return is_case_path
#         else:
#             return False
#
#     return _criterion
