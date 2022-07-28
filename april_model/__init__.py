from pathlib import Path

import nibabel
import torch
from rich.console import Console
from torch import nn

from dataset.path_explorer import iter_registered
from .models import unet3dB
from .models import funet
from .segm.apply import predict_case

console = Console()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved_models = Path(__file__).parent / "saved_models"


def eval_all_folders(path: Path):
    console.print(f'Using device {device}')

    net = funet.FunneledUNet(
        channels=[7, 16, 32, 48, 64],
        wafer_size=5,
        final_classes=3,
        fullbypass=[4, 5, 6],
        final_activation=nn.Tanh(),
        clamp=(-100, 300),
    )
    net.load_state_dict(torch.load(saved_models / f'segm.2.pth'))
    net = net.to(device=device, dtype=torch.float32)
    net.eval()

    net882 = unet3dB.UNet3d(
        channels=[4, 32, 64, 128],
        final_classes=3,
        complexity=2,
        down_dropout=None,
        bottom_normalization=None,
        checkpointing=False,
    )
    net882.load_state_dict(torch.load(saved_models / f'segm882.7.pth'))
    net882 = net882.to(device=device, dtype=torch.float32)
    net882.eval()

    console.print("[bold orange3]Segmenting:[/bold orange3]")
    for case_path in iter_registered(path):
        source_path = path / case_path
        target_path = path / case_path
        target_path_is_complete = (target_path / f"prediction.nii.gz").exists()
        if not target_path_is_complete:
            target_path.mkdir(parents=True, exist_ok=True)
            our_best_guess = predict_case(case=source_path, net882=net882, net=net, device=device)

            affine = nibabel.load(target_path / f"registered_phase_v.nii.gz").affine
            nibabel.save(
                nibabel.Nifti1Image(
                    our_best_guess.cpu().numpy(),
                    affine=affine
                ),
                target_path / "prediction.nii.gz",
            )

        else:
            console.print(f"[bold black]{case_path}.[/bold black] is already complete, skipping.")


def eval_one_folder(path: Path):
    console.print(f'Using device {device}')

    net = funet.FunneledUNet(
        channels=[7, 16, 32, 48, 64],
        wafer_size=5,
        final_classes=3,
        fullbypass=[4, 5, 6],
        final_activation=nn.Tanh(),
        clamp=(-100, 300),
    )
    net.load_state_dict(torch.load(saved_models / f'segm.2.pth'))
    net = net.to(device=device, dtype=torch.float32)
    net.eval()

    net882 = unet3dB.UNet3d(
        channels=[4, 32, 64, 128],
        final_classes=3,
        complexity=2,
        down_dropout=None,
        bottom_normalization=None,
        checkpointing=False,
    )
    net882.load_state_dict(torch.load(saved_models / f'segm882.7.pth'))
    net882 = net882.to(device=device, dtype=torch.float32)
    net882.eval()

    console.print("[bold orange3]Segmenting:[/bold orange3]")

    source_path = path
    target_path = path
    target_path.mkdir(parents=True, exist_ok=True)
    our_best_guess = predict_case(case=source_path, net882=net882, net=net, device=device)

    affine = nibabel.load(target_path / f"registered_phase_v.nii.gz").affine
    nibabel.save(
        nibabel.Nifti1Image(
            our_best_guess.cpu().numpy(),
            affine=affine
        ),
        target_path / "prediction.nii.gz",
    )