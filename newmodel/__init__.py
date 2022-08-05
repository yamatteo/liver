import pickle
from pathlib import Path

import nibabel
import numpy as np
import torch
from rich.console import Console

import utils.ndarray as nd
import utils.path_explorer as px
from utils.slices import overlapping_slices
from distances import liverscore, tumorscore
from .hunet import HunetNetwork, HalfUNet

console = Console()
saved_models = Path(__file__).parent / "saved_models"
saved_models.mkdir(exist_ok=True)


def setup_evaluation():
    raise NotImplemented
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HalfUNet()
    model.to(device=device)
    model.resume(saved_models, "last_checkpoint.pth", device)
    return model, device

def setup_train():
    pass


@torch.no_grad()
def apply(model, case_path, device):
    raise NotImplemented
    affine, bottom, top, height = nd.load_registration_data(case_path)
    scan = nd.load_scan_from_regs(case_path)
    scan = np.clip(scan, -1024, 1024)
    pad = 40 - scan.shape[3] % 40
    scan = np.pad(scan, ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=-1024)
    scan = np.reshape(scan, [1, *scan.shape])
    pred = np.concatenate([
        model(torch.tensor(piece, dtype=torch.float32, device=device)).cpu().numpy()
        for piece in overlapping_slices(scan, thickness=40, dim=4)
    ], axis=3)
    pred = np.argmax(pred, axis=1)[0]
    _pred = np.full([512, 512, height], -1024)
    _pred[..., bottom:top] = pred[..., :(top - bottom)]
    return _pred, affine


def predict_case(model, case_path, device):
    prediction, affine_matrix = apply(model, case_path, device)

    nibabel.save(
        nibabel.Nifti1Image(
            prediction,
            affine=affine_matrix
        ),
        case_path / "prediction.nii.gz",
    )


def evaluate_case(model, case_path, device):
    prediction, _ = apply(model, case_path, device)
    pred = torch.tensor(prediction)
    segm = torch.tensor(nd.load_segm(case_path))
    return {
        "liver": liverscore(pred, segm),
        "tumor": tumorscore(pred, segm)
    }


def predict_one_folder(case_path):
    model, device = setup_evaluation()
    console.print(f'Using device {device}')
    console.print(f"[bold orange3]Segmenting:[/bold orange3] {case_path.stem}...")
    predict_case(model, case_path, device)
    console.print(f"            ...completed.")


def predict_all_folders(path: Path):
    model, device = setup_evaluation()
    console.print(f'Using device {device}')
    console.print("[bold orange3]Segmenting:[/bold orange3]")
    predict = lambda case_path: predict_case(model, case_path, device)
    select = lambda p: px.is_registered(p) and not px.is_predicted(p)
    px.recurse(
        path,
        select,
        case_in="  [bold black]{case}.[/bold black] Predicting...",
        case_out="  ...completed."
    )(predict)


def evaluate_all_folders(base_path: Path):
    model, device = setup_evaluation()
    console.print(f'Using device {device}')
    console.print("[bold orange3]Evaluating:[/bold orange3]")
    evaluate = lambda case_path: evaluate_case(model, case_path, device)
    evaluations = px.recurse(
        base_path,
        px.is_trainable,
        case_in="  [bold black]{case}.[/bold black] Evaluating...",
        case_out="  ...completed.",
    )(evaluate)
    with open("july_evaluation.pickle", "wb") as f:
        pickle.dump(evaluations, f)
    console.print("Mean liver score:", sum(item["liver"] for item in evaluations.values()) / len(evaluations))
    console.print("Mean tumor score:", sum(item["tumor"] for item in evaluations.values()) / len(evaluations))
