import argparse
import importlib
import os
import re
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import dicom2nifti
import nibabel
from rich.console import Console

from path_explorer import discover, get_criterion



console = Console()

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
    console.print(f"[bold black]{case_name}.[/bold black] Converting dicom to nifti...")
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
                image = nibabel.load(temp_path / path.name)
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
                console.print(
                    f"{' ' * len(case_name)}  "
                    f"No image for phase [italic cyan]{phase}[/italic cyan]."
                )
                raise ValueError

        # If there is exactly one image per phase, save them as compressed nifti
        for phase in ["b", "a", "v", "t"]:
            nibabel.save(
                phases[phase],
                target_path / f"original_phase_{phase}.nii.gz",
            )
        console.print(
            f"{' ' * len(case_name)}  "
            f"Images saved in {target_path.absolute()}."
        )


def get_args():
    parser = argparse.ArgumentParser(description="Convert dicom to nifti.")
    parser.add_argument("--defaults", nargs="?", default="options.py", type=Path, help="path to defaults module")
    parser.add_argument("--outputs", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sources", type=Path, default=argparse.SUPPRESS)

    args = parser.parse_args()
    defaults_path, defaults_module = args.defaults.parent, args.defaults.stem
    sys.path.append(defaults_path)
    defaults = importlib.import_module(defaults_module).defaults
    return dict(defaults, **vars(args))


if __name__ == "__main__":
    opts = get_args()

    console.print("[bold orange3]Converting dicom to nifti:[/bold orange3]")
    for case_path in discover(opts["sources"], get_criterion(dicom=True)):
        target_path = opts["outputs"] / case_path
        target_path_is_complete = all(
            (target_path / f"original_phase_{phase}.nii.gz").exists()
            for phase in ["b", "a", "v", "t"]
        )
        if opts["overwrite"] or not target_path_is_complete:
            target_path.mkdir(parents=True, exist_ok=True)
            process_dicomdir(opts["sources"] / case_path, target_path)
        else:
            console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
