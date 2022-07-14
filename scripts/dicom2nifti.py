import argparse
from pathlib import Path

from rich.console import Console

from utils.path_explorer import iter_dicom
from preprocessing import process_dicomdir

def main(opts):
    console = Console()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dicom to nifti.")
    parser.add_argument("--overwrite",              action="store_true")
    parser.add_argument("--outputs",    type=Path,  required=True)
    parser.add_argument("--sources",    type=Path,  required=True)
    opts = parser.parse_args()
    main(opts)