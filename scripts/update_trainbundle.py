import os
from pathlib import Path

import dotenv
import nibabel
import numpy as np
import torch
from rich.console import Console

from wrapped_tensors import Scan, Segm, Bundle
from utils.path_explorer import discover, get_criterion


def update_trainbundle(case: Path):
    console.print(f"[bold black]{case.name}.[/bold black] Updating bundle.")

    affine = nibabel.load(case / f"registered_phase_v.nii.gz").affine
    scan = Scan.from_niigz(case)
    segm = Segm.from_niigz(case)
    bundle = Bundle.from_join(scan, segm)

    nibabel.save(
        nibabel.Nifti1Image(
            bundle.cpu().numpy(),
            affine=affine
        ),
        case / "train_bundle.nii.gz",
    )


console = Console()
dotenv.load_dotenv()

console.print("[bold orange3]Updating train bundles[/bold orange3]")
for case_path in discover(os.getenv("OUTPUTS"), get_criterion(registered=True, segmented=True)):
    update_trainbundle(os.getenv("OUTPUTS") / case_path)
