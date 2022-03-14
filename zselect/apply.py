from __future__ import annotations

import argparse

import torch
from rich.console import Console
from tqdm import tqdm

from dataset.dataset import QuantileDataset
from options import defaults
from .models import ZPredict

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

    net = ZPredict().to(device=device, dtype=torch.float32)

    net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
    console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")

    net.eval()
    with torch.no_grad():
        dataset = QuantileDataset(
            base_path=opts["outputs"],
            segmented=False,
        )
        for case in tqdm(dataset, total=len(dataset), desc='Predicting segmentation', unit='scan', leave=False):
            scan = case['scan'].to(device=device, dtype=torch.float32)

            predictions = net(scan)
            best_z = max(
                [z for z in range(len(predictions) - 32)],
                key=lambda z: torch.sum(predictions[z:z + 32]) - 1e-6 * z
            )
            torch.save(
                {
                    "z_offset": best_z,
                    "orig_affine": case["orig_affine"],
                    #"scan": case["orig_scan"][:, :, :, best_z:best_z + 32],
                    #"segmentation": case["orig_segm"][:, :, best_z:best_z + 32],
                },
                opts["outputs"] / case["case_dir"] / "zwindow.pt"
            )
