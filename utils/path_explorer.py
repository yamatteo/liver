from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from rich.console import Console

from options import defaults

console = Console()


def get_criterion(
        dicom: bool = False,
        original: bool = False,
        registered: bool = False,
        segmented: bool = False,
        predicted: bool = False,
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
            is_predicted = not predicted or "prediction.nii.gz" in files
            is_windowed = not windowed or "zwindow.pt" in files
            return is_dicom and is_original and is_registered and is_segmented and is_predicted and is_windowed
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
