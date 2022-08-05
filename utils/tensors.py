from pathlib import Path

import torch
from torch.nn import functional

from . import ndarray


def load_scan(path: Path):
    return torch.as_tensor(ndarray.load_scan(path), dtype=torch.int16)


def load_segm(path: Path):
    return torch.as_tensor(ndarray.load_segm(path), dtype=torch.int16)


def load_floatscan(path: Path):
    return torch.as_tensor(ndarray.load_scan(path), dtype=torch.float32)


def load_floatsegm(path: Path):
    t = torch.as_tensor(ndarray.load_segm(path), dtype=torch.int64)
    return functional.one_hot(t, 3).permute(3, 0, 1, 2).float()

def scan_f2i(scan):
    return scan.to(dtype=torch.int16)

def segm_f2i(segm):
    return torch.argmax(segm, dim=0).to(dtype=torch.int16)
