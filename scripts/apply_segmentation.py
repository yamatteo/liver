from __future__ import annotations

import os
import sys

import nibabel
import torch

sys.path.append(os.getcwd())
import models
from options import defaults as opts
from segm.apply import predict_case

import dotenv
from rich.console import Console

from utils.path_explorer import discover, get_criterion

console = Console()
dotenv.load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console.print(f'Using device {device}')

net = models.get_model(**dict(opts, model="segm.2")).to(device=device, dtype=torch.float32)
net.eval()

net882 = models.get_model(**dict(opts, model="segm882.7")).to(device=device, dtype=torch.float32)
net882.eval()

console.print("[bold orange3]Segmenting:[/bold orange3]")
for case_path in discover(os.getenv("OUTPUTS"), get_criterion(registered=True)):
    source_path = os.getenv("OUTPUTS") / case_path
    target_path = os.getenv("OUTPUTS") / case_path
    target_path_is_complete = (target_path / f"prediction.nii.gz").exists()
    if os.getenv("OVERWRITE") or not target_path_is_complete:
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