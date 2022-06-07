import os

import dotenv
from rich.console import Console

from preprocessing.dicom_conversion import process_dicomdir
from utils.path_explorer import discover, get_criterion

console = Console()
dotenv.load_dotenv()

console.print("[bold orange3]Converting dicom to nifti:[/bold orange3]")
for case_path in discover(os.getenv("SOURCES"), get_criterion(dicom=True)):
    target_path = os.getenv("OUTPUTS") / case_path
    target_path_is_complete = all(
        (target_path / f"original_phase_{phase}.nii.gz").exists()
        for phase in ["b", "a", "v", "t"]
    )
    if os.getenv("OVERWRITE", False) or not target_path_is_complete:
        target_path.mkdir(parents=True, exist_ok=True)
        process_dicomdir(os.getenv("SOURCES") / case_path, target_path)
    else:
        console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
