from __future__ import annotations

import importlib
import random

import torch
from torch import Tensor

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None

def get_white(scan: Tensor) -> Tensor:
    return (50 + torch.clamp(scan[:, 2:3, :, :, :], -50, 250)) / 300


def get_color(scan: Tensor, pred: Tensor, segm: Tensor, mode: str = "none"):
    if mode == "error":
        color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True)
    elif mode == "background":
        color = segm[:, 0:1, :, :, :]
    elif mode == "liver":
        color = segm[:, 1:2, :, :, :]
    elif mode == "tumor":
        color = segm[:, 2:3, :, :, :]
    elif mode == "pred_background":
        color = pred[:, 0:1, :, :, :]
    elif mode == "pred_liver":
        color = pred[:, 1:2, :, :, :]
    elif mode == "pred_tumor":
        color = pred[:, 2:3, :, :, :]
    else:
        color = 0
    return torch.clamp(
        get_white(scan) + color,
        0, 1
    )


def rgb_sample(scan: Tensor, pred: Tensor, segm: Tensor, mode: tuple[str, str, str], z: int | None = None):
    rgb = torch.cat([
        get_color(scan, pred, segm, mode=color_mode)
        for color_mode in mode
    ], dim=1)
    if z is None:
        z = random.randint(0, scan.size(4) - 1)
    return rgb[0, :, :, :, z]  # shape is (C=3, H=512, W=512)

def wandb_sample(scan: Tensor, pred: Tensor, segm: Tensor):
    n = random.randint(0, scan.size(0)-1)
    z = random.randint(0, scan.size(4)-1)
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
@torch.no_grad()
def samples(net, ds, device=torch.device("cpu"), mode=("error", "tumor", "liver"), k=4, indices=None):
    if indices is None:
        indices = random.sample(range(ds.buffer_size), k)
    return torch.stack([rgb_sample(net, ds[i], mode=mode, device=device) for i in indices])
