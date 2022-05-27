import os
import sys
from pathlib import Path

import nibabel
import numpy as np
import torch

import dotenv
from rich.console import Console

sys.path.append(os.getcwd())
from utils.path_explorer import discover, get_criterion


def update_trainbundle(case: Path):
    console.print(f"[bold black]{case.name}.[/bold black] Updating bundle.")

    affine = nibabel.load(case / f"registered_phase_v.nii.gz").affine
    scan = torch.stack([
        torch.tensor(np.array(nibabel.load(
            case / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16))
        for phase in ["b", "a", "v", "t"]
    ])
    good_z = [not bool(torch.any(torch.all(torch.all(scan[:, :, :, z] == 0, dim=1), dim=1))) for z in
              range(scan.size(3))]
    a = 0
    for z in range(scan.size(3)):
        if any(good_z[:z]):
            break
        else:
            a = z
    b = scan.size(3)
    for z in reversed(range(scan.size(3))):
        if any(good_z[z:]):
            break
        else:
            b = z
    segm = torch.tensor(np.array(nibabel.load(
        case / f"segmentation.nii.gz"
    ).dataobj, dtype=np.int16)).unsqueeze(0)

    nibabel.save(
        nibabel.Nifti1Image(
            torch.cat([scan[:, :, :, a:b], segm[:, :, :, a:b]], dim=0).cpu().numpy(),
            affine=affine
        ),
        case / "train_bundle.nii.gz",
    )


console = Console()
dotenv.load_dotenv()

console.print("[bold orange3]Updating train bundles[/bold orange3]")
for case_path in discover(os.getenv("OUTPUTS"), get_criterion(registered=True, segmented=True)):
    update_trainbundle(os.getenv("OUTPUTS") / case_path)
