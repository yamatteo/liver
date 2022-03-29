from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import nibabel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import Dataset882
from options import defaults
from segm3d882.models3d import UNet3d
from segmentation.dataset import GenericDataset
from segm3d882.utilities import dice_distance, jaccard_distance

console = Console()
classes = ["background", "liver", "tumor"]

opts = dict(defaults, model="segm882v0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNet3d(
    channels=[4, 16, 20],
    final_classes=3,
).to(device=device, dtype=torch.float32)

net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))

net.eval()


@torch.no_grad()
def apply_onthefly(scan, z_offset):
    total_z = scan.size(4)
    scan = F.avg_pool3d(
        scan[..., z_offset:z_offset + 32],
        kernel_size=(8, 8, 2)
    )
    predictions = net(scan)

    complete_predictions = torch.zeros((3, 512, 512, total_z))
    complete_predictions[:, :, :, z_offset:z_offset + 32] = torch.nn.functional.upsample_nearest(predictions,
                                                                                                 (512, 512, 32))

    return complete_predictions
