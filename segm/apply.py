from __future__ import annotations

import os
from pathlib import Path

import nibabel
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

import models
from options import defaults
from path_explorer import discover, get_criterion

console = Console()
classes = ["background", "liver", "tumor"]



@torch.no_grad()
def predict_case(case: Path, net882, net, device):
    print(f"Predicting {case}...")
    scan = torch.stack([
        torch.tensor(np.array(nibabel.load(
            case / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16)).float()
        for phase in ["b", "a", "v", "t"]
    ]).unsqueeze(0).to(device=device, dtype=torch.float32)
    dgscan = F.avg_pool3d(
        scan,
        kernel_size=(8, 8, 2)
    )
    dgpred = net882(dgscan)
    whole = torch.cat([
        scan,
        F.interpolate(dgpred, scan.shape[2:5], mode="trilinear"),
    ], dim=1)

    slices = [
        torch.zeros(2, 512, 512).to(dtype=torch.int64).cpu()
    ]
    for z in range(1 + scan.size(4) - opts["wafer_size"]):
        pred = net(whole[..., z: z + opts["wafer_size"]])
        slices.append(pred.argmax(dim=1).cpu())

    slices.append(torch.zeros(2, 512, 512).to(dtype=torch.int64))
    return torch.cat(slices).permute(1, 2, 0)


if __name__ == "__main__":
    opts = defaults
    opts["overwrite"] = False
    opts["sources"] = os.getenv("REGISTERED_FOLDER", opts["sources"])
    opts["outputs"] = os.getenv("PREDICTION_FOLDER", opts["outputs"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f'Using device {device}')

    net = models.get_model(**dict(opts, model="segm.2")).to(device=device, dtype=torch.float32)
    net.eval()

    net882 = models.get_model(**dict(opts, model="segm882.7")).to(device=device, dtype=torch.float32)
    net882.eval()

    console.print("[bold orange3]Segmenting:[/bold orange3]")
    for case_path in discover(opts["outputs"], get_criterion(registered=True)):
        source_path = target_path = opts["outputs"] / case_path
        target_path_is_complete = (target_path / f"prediction.nii.gz").exists()
        if opts["overwrite"] or not target_path_is_complete:
            target_path.mkdir(parents=True, exist_ok=True)
            our_best_guess = predict_case(case=source_path, net882=net882, net=net, device=device)

            affine = nibabel.load(target_path / f"registered_phase_v.nii.gz").affine
            nibabel.save(
                nibabel.Nifti1Image(
                    our_best_guess.cpu().numpy(),
                    affine=affine
                ),
                target_path / "prediction.nii.gz",
            )

        else:
            console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
