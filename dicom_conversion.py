import argparse
import re
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import dicom2nifti
import nibabel
from rich.console import Console

from options import defaults
from path_explorer import discover_dicomdirs

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
        with redirect_stderr(StringIO()):
            dicom2nifti.convert_directory(
                dicom_directory=str(source_path),
                output_folder=str(temp_path),
                compression=True
            )

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
    parser = argparse.ArgumentParser(description='Convert dicom to nifti.')
    parser.add_argument('--overwrite', action='store_true')

    return dict(defaults, **vars(parser.parse_args()))


if __name__ == "__main__":
    opts = get_args()

    for case_path in discover_dicomdirs(opts["sources"]):
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
