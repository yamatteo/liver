from __future__ import annotations

import pickle
from pathlib import Path

import nibabel
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import nn

import dataset.path_explorer as px
import dataset.ndarray as nd
from distances import liverscore, tumorscore
from .models import funet
from .models import unet3dB

console = Console()
saved_models = Path(__file__).parent / "saved_models"


def eval_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    return net, net882, device


@torch.no_grad()
def apply(case_path: Path, net882, net, device):
    scan = torch.stack([
        torch.tensor(np.array(nibabel.load(
            case_path / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16)).float()
        for phase in ["b", "a", "v", "t"]
    ]).unsqueeze(0).to(device=device, dtype=torch.float32)
    dgscan = F.avg_pool3d(
        scan,
        kernel_size=(8, 8, 2)
    )
    dgpred = net882(dgscan)
    whole = torch.cat([
        scan,
        F.interpolate(dgpred, scan.shape[2:5], mode="trilinear"),
    ], dim=1)

    slices = [
        torch.zeros(2, 512, 512).to(dtype=torch.int64).cpu()
    ]
    for z in range(1 + scan.size(4) - 5):
        pred = net(whole[..., z: z + 5])
        slices.append(pred.argmax(dim=1).cpu())

    slices.append(torch.zeros(2, 512, 512).to(dtype=torch.int64))
    return torch.cat(slices).permute(1, 2, 0)


def predict_case(case_path: Path, net882, net, device):
    pred = apply(case_path=case_path, net882=net882, net=net, device=device)
    affine = nibabel.load(case_path / f"registered_phase_v.nii.gz").affine
    nibabel.save(
        nibabel.Nifti1Image(
            pred.cpu().numpy(),
            affine=affine
        ),
        case_path / "prediction.nii.gz",
    )


def evaluate_case(case_path: Path, net882, net, device):
    pred = apply(case_path=case_path, net882=net882, net=net, device=device)
    segm = torch.tensor(nd.load_segm(case_path))
    return {
        "liver": liverscore(pred, segm),
        "tumor": tumorscore(pred, segm)
    }


def evaluate(base_path: Path):
    net, net882, device = eval_setup()
    console.print("[bold orange3]Evaluating:[/bold orange3]")
    evaluate = lambda case_path: evaluate_case(case_path, net882, net, device)
    evaluations = px.recurse(
        base_path,
        px.is_trainable,
        case_in="  [bold black]{case}.[/bold black] Evaluating...",
        case_out="  ...completed.",
    )(evaluate)
    # for case_path in px.iter_trainable(base_path):
    #     source_path = base_path / case_path
    #     console.print(f"  [bold black]{case_path}.[/bold black] Evaluating...")
    #     pred = apply(case_path=source_path, net882=net882, net=net, device=device)
    #     segm = torch.tensor(nd.load_segm(source_path))
    #     evaluations[str(case_path)] = {
    #         "liver": liverscore(pred, segm),
    #         "tumor": tumorscore(pred, segm)
    #     }
    #     console.print(f"  {' ' * len(str(case_path))}  ...completed.")
    with open("april_evaluation.pickle", "wb") as f:
        pickle.dump(evaluations, f)
    console.print("Mean liver score:", sum(item["liver"] for item in evaluations.values()) / len(evaluations))
    console.print("Mean tumor score:", sum(item["tumor"] for item in evaluations.values()) / len(evaluations))


def apply_to_all_folders(path: Path):
    net, net882, device = eval_setup()

    console.print(f'Using device {device}')
    console.print("[bold orange3]Segmenting:[/bold orange3]")
    predict = lambda case_path: predict_case(case_path, net882, net, device)
    select = lambda p: px.is_registered(p) and not px.is_predicted(p)
    px.recurse(
        path,
        select,
        case_in="  [bold black]{case}.[/bold black] Predicting...",
        case_out="  ...completed."
    )(predict)
    # for case_path in px.iter_registered(path):
    #     source_path = path / case_path
    #     target_path = path / case_path
    #     target_path_is_complete = (target_path / f"prediction.nii.gz").exists()
    #     if not target_path_is_complete:
    #         target_path.mkdir(parents=True, exist_ok=True)
    #         console.print(f"  [bold black]{case_path}.[/bold black] Predicting...")
    #         our_best_guess = apply(case_path=source_path, net882=net882, net=net, device=device)
    #
    #         affine = nibabel.load(target_path / f"registered_phase_v.nii.gz").affine
    #         nibabel.save(
    #             nibabel.Nifti1Image(
    #                 our_best_guess.cpu().numpy(),
    #                 affine=affine
    #             ),
    #             target_path / "prediction.nii.gz",
    #         )
    #         console.print(f"  {' ' * len(str(case_path))}  ...completed.")
    #     else:
    #         console.print(f"  [bold black]{case_path}.[/bold black] is already complete, skipping.")


def apply_to_one_folder(case_path: Path):
    net, net882, device = eval_setup()
    console.print(f'Using device {device}')
    console.print(f"[bold orange3]Segmenting:[/bold orange3] {case_path.stem}...")
    predict_case(case_path, net882, net, device)
    console.print(f"             ...completed.")
