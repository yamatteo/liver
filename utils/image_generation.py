from __future__ import annotations

import importlib
import random

import torch

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None

from rich.console import Console
from tensors import *

console = Console()


def get_white(scan: ScanBatch, *, n, phase="v", z) -> Plane:
    return (50 + torch.clamp(scan.get_plane(n=n, phase=phase, z=z), -50, 250)) / 300


def get_color(scan: ScanBatch, pred: FloatSegmBatch, segm: FloatSegmBatch, *, n, z, mode: str = "none") -> Plane:
    if mode == "error":
        # color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True)
        color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True).get_plane(n=n, klass=0, z=z)
    elif mode in ("backg", "liver", "tumor"):
        color = segm.get_plane(n=n, klass=mode, z=z)
    elif mode in ("pred_backg", "pred_liver", "pred_tumor"):
        color = pred.get_plane(n=n, klass=mode[5:], z=z)
    else:
        color = 0
    return Plane(torch.clamp(get_white(scan, n=n, z=z) + color, 0, 1))


def rgb_sample(scan: ScanBatch, pred: FloatSegmBatch, segm: FloatSegmBatch, *, n, z: int | None = None, mode: tuple[str, str, str], format: str = "CHW"):
    if z is None:
        z = random.randint(0, scan.size(4) - 1)
    rgb = torch.stack(
        [
            get_color(scan, pred, segm, n=n, z=z, mode=color_mode)
            for color_mode in mode
        ],
        dim=2 if format=="HWC" else 0
    )
    return rgb  # shape is same as `format`


def wandb_sample(scan: Tensor, pred: Tensor, segm: Tensor):
    n = random.randint(0, scan.size(0) - 1)
    z = random.randint(0, scan.size(4) - 1)
    image = get_white(scan)[n, 0, :, :, z].unsqueeze(-1).numpy()
    class_labels = {
        0: "background",
        1: "liver",
        2: "tumor"
    }
    pred_mask = torch.argmax(pred, dim=1)[n, :, :, z].numpy()
    segm_mask = torch.argmax(segm, dim=1)[n, :, :, z].numpy()

    return wandb.Image(image, masks={
        "predictions": {
            "mask_data": pred_mask,
            "class_labels": class_labels
        },
        "ground_truth": {
            "mask_data": segm_mask,
            "class_labels": class_labels
        },
    })


def wandb_sample_debug(scan: Tensor, pred: Tensor, segm: Tensor):
    n = random.randint(0, scan.size(0) - 1)
    z = random.randint(0, scan.size(4) - 1)
    image = get_white(scan)[n, 0, :, :, z].unsqueeze(-1).numpy()
    class_labels = {
        0: "background",
        1: "liver",
        2: "tumor"
    }
    pred_mask = torch.argmax(pred, dim=1)[n, :, :, z].numpy()
    segm_mask = torch.argmax(segm, dim=1)[n, :, :, z].numpy()

    # console.print(f"pre  liver weight {torch.sum(segm[n, 1, :, :, z])}")
    # console.print(f"post liver weight {torch.sum(torch.argmax(segm, dim=1)[n, :, :, z] == 1)}")
    return wandb.Image(image, masks={
        "predictions": {
            "mask_data": pred_mask,
            "class_labels": class_labels
        },
    }), wandb.Image(image, masks={
        "ground_truth": {
            "mask_data": segm_mask,
            "class_labels": class_labels
        }
    })


@torch.no_grad()
def samples(net, ds, device=torch.device("cpu"), mode=("error", "tumor", "liver"), k=4, indices=None):
    if indices is None:
        indices = random.sample(range(ds.buffer_size), k)
    return torch.stack([rgb_sample(net, ds[i], mode=mode, device=device) for i in indices])
