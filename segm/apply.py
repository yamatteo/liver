from __future__ import annotations

from typing import Iterator

import nibabel
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import Tensor

import models
import segm3d882.models3d
import segm441.models
from dataset import scan_segm_tuples
from dataset.generators import get_scans, cases
from options import defaults
from segm.models import FunneledUNet

console = Console()
classes = ["background", "liver", "tumor"]

opts = defaults

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console.print(f'Using device {device}')

net = models.get_model(**dict(opts, model="segm.2")).to(device=device, dtype=torch.float32)
net.eval()

net882 = models.get_model(**dict(opts, model="segm882.7")).to(device=device, dtype=torch.float32)
net882.eval()


def process_case(net: FunneledUNet, case: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # case is shape (N=1, C=10, H=512, W=512, D=5)
    scan_aid = case[:, 0:7, :, :, :].to(device=device, dtype=torch.float32)

    pred = net(scan_aid).unsqueeze(-1)

    central_slice = case.size(4) // 2
    scan = scan_aid[:, 0:4, :, :, central_slice:central_slice + 1]
    aid = scan_aid[:, 4:7, :, :, central_slice:central_slice + 1]
    segm = case[:, 7:10, :, :, central_slice:central_slice + 1].to(device=device, dtype=torch.float32)

    return scan, pred, aid, segm  # shapes are (N=1, C=4|3|3|3, H=512, W=512, D=1)


for case in cases(opts["outputs"], segmented=False):
    print(f"Segmenting {case}...")

    print(f'Saving to {opts["outputs"] / case / "prediction.nii.gz"}')

    # affine = nibabel.load(opts["outputs"] / case / f"registered_phase_v.nii.gz").affine
    # nibabel.save(
    #     nibabel.Nifti1Image(
    #         our_best_guess.cpu().numpy(),
    #         affine=affine
    #     ), opts["outputs"] / case / "prediction.nii.gz",
    # )

def get_wafer() -> Iterator[Tensor]:
    for scan in get_scans(opts["outputs"]):
        dgscan = F.avg_pool3d(
            scan,
            kernel_size=(8, 8, 2)
        )

        with torch.no_grad():
            dgpred = net882(dgscan)

        whole = torch.cat([
            scan,
            F.interpolate(dgpred, scan.shape[2:5], mode="trilinear"),
        ], dim=1)
        for z in range(1 + scan.size(4) - opts["wafer_size"]):
            yield whole[..., z:z + opts["wafer_size"]]


    # pred882 = net882(F.avg_pool3d(scan, kernel_size=(8, 8, 2)).unsqueeze(0))
    # scan441 = F.avg_pool3d(scan, kernel_size=(4, 4, 1)).unsqueeze(0)
    # help441 = F.interpolate(pred882, scale_factor=2)
    # pad = scan441.size(4) - help441.size(4)
    # if pad > 0:
    #     help441 = torch.constant_pad_nd(help441, (0, pad))
    # pred441 = net441(torch.cat([scan441, help441], dim=1))
    #
    # input = torch.cat([
    #     scan.unsqueeze(0),
    #     F.interpolate(pred441, size=scan.shape[1:])
    # ], dim=1)
    #
    # slices = [
    #     torch.zeros(2, 512, 512).to(dtype=torch.int64)
    # ]
    # for z in range(input.size(4) - 5 + 1):
    #     pred = net(input[..., z: z + 5])
    #     slices.append(pred.argmax(dim=3).cpu())
    #
    # slices.append(torch.zeros(2, 512, 512).to(dtype=torch.int64))
    # our_best_guess = torch.cat(slices).permute(1, 2, 0)
