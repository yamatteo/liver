import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import nibabel
from rich.console import Console

from options import defaults
from path_explorer import discover_original_dirs

console = Console()


def register_case(source_path: Path, target_path: Path):
    """Register 4-phase scans with respect to phase v."""
    case_name = source_path.name
    console.print(f"[bold black]{case_name}.[/bold black] Working (it is a long process) ...")

    # Store the phase v as is, with its affine matrix
    v_phase = nibabel.load(source_path / "original_phase_v.nii.gz")

    nibabel.save(v_phase, str(target_path / "registered_phase_v.nii.gz"))
    with NamedTemporaryFile("wb", suffix=".nii") as cpp_file:
        for phase in ["b", "a", "t"]:
            # Run the registering library
            with open(target_path / f"registration_{phase}.log", "w") as logfile:
                niftireg_log = os.popen(
                    f"reg_f3d "
                    f"-ref {source_path / 'original_phase_v.nii.gz'} "
                    f"-flo {source_path / f'original_phase_{phase}.nii.gz'} "
                    f"-res {target_path / f'registered_phase_{phase}.nii.gz'} "
                    f"-cpp {cpp_file.name} "
                ).read()
                logfile.write(niftireg_log)


def get_args():
    parser = argparse.ArgumentParser(description='Convert dicom to tensor, registering phasesk')
    parser.add_argument('--overwrite', action='store_true')

    return dict(defaults, **vars(parser.parse_args()))


if __name__ == "__main__":
    opts = get_args()

    console.print("\n[bold orange3]Registering images:[/bold orange3]")
    for case_path in discover_original_dirs(opts["outputs"]):
        source_path = target_path = opts["outputs"] / case_path
        target_path_is_complete = all(
            (target_path / f"registered_phase_{phase}.nii.gz").exists()
            for phase in ["b", "a", "v", "t"]
        )
        if opts["overwrite"] or not target_path_is_complete:
            target_path.mkdir(parents=True, exist_ok=True)
            register_case(source_path, target_path)
        else:
            console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
