from __future__ import annotations

import random

import torch
from rich.console import Console

from tensors import FloatScanBatch, FloatSegmBatch, Slice, Segm

console = Console()


def get_white(scan: FloatScanBatch, *, n: int, phase: int | str = "v", z: int) -> Slice:
    return Slice(torch.clamp(50 + scan.get_plane(n=n, phase=phase, z=z), 0, 255))


def get_color(scan: FloatScanBatch, pred: FloatSegmBatch, segm: FloatSegmBatch, *, n, z, mode: str = "none") -> Slice:
    # if mode == "error":
    #     # color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True)
    #     # color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True)[n, 0, :, :, z]
    # elif mode == "liver_error":
    #     color = torch.abs(segm - pred)[n, 1, :, :, z]
    # elif mode == "tumor_error":
    #     color = torch.abs(segm - pred)[n, 2, :, :, z]
    if mode in ("backg", "liver", "tumor"):
        color = segm.get_mask(n=n, channel=mode, z=z)
    elif mode in ("pred_backg", "pred_liver", "pred_tumor"):
        color = pred.get_mask(n=n, channel=mode[5:], z=z)
    else:
        color = 0
    return Slice(torch.clamp(get_white(scan, n=n, z=z) + color * 0.7, 0, 255).to(dtype=torch.int16))


def rgb_sample(scan: FloatScanBatch, pred: FloatSegmBatch, segm: FloatSegmBatch, *, n, z: int | None = None,
               mode: tuple[str, str, str], data_format: str = "CHW"):
    if z is None:
        z = random.randint(0, scan.size(4) - 1)
    rgb = torch.stack(
        [
            get_color(scan, pred, segm, n=n, z=z, mode=color_mode)
            for color_mode in mode
        ],
        dim=2 if data_format == "HWC" else 0
    )
    return rgb  # shape is same as `format`
