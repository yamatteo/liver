from __future__ import annotations

from pathlib import Path

import nibabel
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

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
    for z in range(1 + scan.size(4) - 5):
        pred = net(whole[..., z: z + 5])
        slices.append(pred.argmax(dim=1).cpu())

    slices.append(torch.zeros(2, 512, 512).to(dtype=torch.int64))
    return torch.cat(slices).permute(1, 2, 0)