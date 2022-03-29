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
from torch import Tensor
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



def get_args():
    parser = argparse.ArgumentParser(description='Train the model', argument_default=argparse.SUPPRESS)
    parser.add_argument('--model', type=str, default=None, help='Use model from a .pth file')

    return dict(defaults, **vars(parser.parse_args()))


if __name__ == '__main__':
    opts = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f'Using device {device}')

    net = UNet3d(
        channels=[4, 16, 20],
        final_classes=3,
    ).to(device=device, dtype=torch.float32)

    net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
    console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")

    net.eval()
    with torch.no_grad():
        dataset = Dataset882(
            base_path=opts["outputs"],
            segmented=False,
        )
        for case in tqdm(dataset, total=len(dataset), desc='Predicting segmentation', unit='scan', leave=False):
            scan = case['scan'].to(device=device, dtype=torch.float32).unsqueeze(0)
            predictions = net(scan)

            complete_predictions = torch.zeros((3, 512, 512, case["total_z"]))
            complete_predictions[:, :, :, case["z_offset"]:case["z_offset"]+32] = torch.nn.functional.upsample_nearest(predictions, (512, 512, 32))

            np.savez_compressed(
                opts["outputs"] / case["case_dir"] / "prediction882.npz",
                pred882=complete_predictions.cpu().numpy(),
            )
