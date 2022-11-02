import argparse
from pathlib import Path

from rich.console import Console

from utils.path_explorer import iter_dicom, iter_original
from .dicom_conversion import process_dicomdir
from .niftyreg import register_case

console = Console()

parser = argparse.ArgumentParser(description="Convert dicom to nifti.")
parser.add_argument("--niftybin",   type=Path,  default="/usr/local/bin")
parser.add_argument("--overwrite",              action="store_true")
parser.add_argument("--outputs",    type=Path,  required=True)
parser.add_argument("--sources",    type=Path,  required=True)

opts = parser.parse_args()


console.print("[bold orange3]Converting dicom to nifti:[/bold orange3]")
for case_path in iter_dicom(opts.sources):
    target_path = opts.outputs / case_path
    target_path_is_complete = all(
        (target_path / f"original_phase_{phase}.nii.gz").exists()
        for phase in ["b", "a", "v", "t"]
    )
    if opts.overwrite or not target_path_is_complete:
        target_path.mkdir(parents=True, exist_ok=True)
        process_dicomdir(opts.sources / case_path, target_path)
    else:
        console.print(f"  [bold black]{case_path.name}.[/bold black] is already complete, skipping.")

console.print("[bold orange3]Registering images:[/bold orange3]")
for case_path in iter_original(opts.outputs):
    source_path = target_path = opts.outputs / case_path
    target_path_is_complete = all(
        (target_path / f"registered_phase_{phase}.nii.gz").exists()
        for phase in ["b", "a", "v", "t"]
    )
    if opts.overwrite or not target_path_is_complete:
        target_path.mkdir(parents=True, exist_ok=True)
        register_case(source_path, target_path, niftybin=opts.niftybin)
    else:
        console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
