from __future__ import annotations

import argparse

import nibabel
import torch
from rich.console import Console
from tqdm import tqdm

from options import defaults
from .dataset import GenericDataset
from .models import FunneledUNet

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

    net = FunneledUNet(
        input_channels=4,
        wafer_size=5,
        internal_channels=[20, 24, 28, 32],
        classes=3,
    ).to(device=device, dtype=torch.float32)

    net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
    console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")

    net.eval()
    with torch.no_grad():
        dataset = GenericDataset(
            data_path=opts["outputs"],
            segmented=False,
            wafer=None,
            background_reduction=None
        )
        for case in tqdm(dataset, total=len(dataset), desc='Predicting segmentation', unit='scan', leave=False):
            predictions = [
                net(
                    case["scan"][:, :, :, z:z + net.wafer_size].unsqueeze(0).to(device=device, dtype=torch.float32)
                ).argmax(dim=3)
                for z in range(0, case["scan"].size(-1) - net.wafer_size + 1)
            ]
            predictions = torch.stack([
                *[torch.zeros(1, 512, 512).to(device=device, dtype=torch.float32), ] * ((net.wafer_size - 1) // 2),
                *predictions,
                *[torch.zeros(1, 512, 512).to(device=device, dtype=torch.float32), ] * (net.wafer_size // 2),
            ], dim=3).squeeze()
            nibabel.save(
                nibabel.Nifti1Image(predictions.cpu().numpy(), affine=nibabel.load(
                    opts["outputs"] / case["case"] / f"registered_phase_v.nii.gz"
                ).affine),
                opts["outputs"] / case["case"] / "prediction.nii.gz",
            )
