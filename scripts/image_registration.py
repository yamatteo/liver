import os
from pathlib import Path

import dotenv
from rich.console import Console

from preprocessing.registration import register_case
from utils.path_explorer import discover, get_criterion

console = Console()
dotenv.load_dotenv()

console.print("[bold orange3]Registering images:[/bold orange3]")
for case_path in discover(os.getenv("OUTPUTS"), get_criterion(original=True)):
    print(case_path)
    source_path = target_path = os.getenv("OUTPUTS") / case_path
    target_path_is_complete = all(
        (target_path / f"registered_phase_{phase}.nii.gz").exists()
        for phase in ["b", "a", "v", "t"]
    )
    if os.getenv("OVERWRITE") or not target_path_is_complete:
        target_path.mkdir(parents=True, exist_ok=True)
        register_case(source_path, target_path, niftybin=Path(os.getenv("NIFTY_BIN")))

    else:
        console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
