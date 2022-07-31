import pickle
from pathlib import Path

import nibabel
import torch
from rich.console import Console

from dataset.path_explorer import iter_registered
import dataset.ndarray as nd
import dataset.path_explorer as px
from distances import liverscore, tumorscore
from .subclass_tensors import FloatScan, Segm
from .models.double_unet import DoubleUNet
from .tensor_loading import load_scan

console = Console()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved_models = Path(__file__).parent / "saved_models"

def setup_model():
    model = DoubleUNet()
    model.first_net.load_state_dict(torch.load(saved_models / "coarse400.pth", map_location=device))
    model.first_net.eval()
    model.first_net.to(device=device)
    model.second_net.load_state_dict(torch.load(saved_models / "finer400.pth", map_location=device))
    model.second_net.eval()
    model.second_net.to(device=device)
    model.up_sampler.to(device=device)
    model.down_sampler.to(device=device)
    return model


@torch.no_grad()
def predict_case(model, case_path):
    (a, b, original_d_size), affine_matrix, scan = load_scan(case_path)

    scan = scan.unsqueeze(0).to(dtype=torch.float32, device=device)
    dg_scan = model.down_sampler(scan)
    dg_pred = model.first_net.forward(dg_scan)
    dg_pred = model.up_sampler(dg_pred)

    pred = model.second_net.block_forward(torch.cat([scan, dg_pred], dim=1))
    pred = Segm(torch.argmax(pred, dim=1).to(dtype=torch.int16).squeeze(0))

    prediction = pred.cpu().numpy()

    nibabel.save(
        nibabel.Nifti1Image(
            prediction,
            affine=affine_matrix
        ),
        case_path / "prediction.nii.gz",
    )


@torch.no_grad()
def evaluate_case(model, case_path):
    (a, b, original_d_size), affine_matrix, scan = load_scan(case_path)

    scan = scan.unsqueeze(0).to(dtype=torch.float32, device=device)
    dg_scan = model.down_sampler(scan)
    dg_pred = model.first_net.forward(dg_scan)
    dg_pred = model.up_sampler(dg_pred)

    pred = model.second_net.block_forward(torch.cat([scan, dg_pred], dim=1))
    pred = Segm(torch.argmax(pred, dim=1).to(dtype=torch.int16).squeeze(0))
    segm = torch.tensor(nd.load_segm(case_path))
    return {
        "liver": liverscore(pred, segm),
        "tumor": tumorscore(pred, segm)
    }

def predict_one_folder(case_path):
    console.print(f"[bold orange3]Segmenting:[/bold orange3] {case_path.stem}...")
    model = setup_model()
    predict_case(model, case_path)
    console.print(f"            ...completed.")

def predict_all_folders(path: Path):
    console.print("[bold orange3]Segmenting:[/bold orange3]")
    model = setup_model()
    for case in iter_registered(path):
        case_path = path / case
        if (case_path / "prediction.nii.gz").exists():
            console.print(f" [bold black]{case_path}[/bold black] is already complete, skipping.")
        else:
            console.print(f"  [bold black]{case_path}.[/bold black] Predicting...")
            predict_case(model, case_path)
            console.print(f"  {' ' * len(str(case_path))}  ...completed.")

def evaluate_all_folders(base_path: Path):
    console.print("[bold orange3]Evaluating:[/bold orange3]")
    model = setup_model()
    evaluations = {}
    for case_path in px.iter_trainable(base_path):
        source_path = base_path / case_path
        console.print(f"  [bold black]{case_path}.[/bold black] Evaluating...")
        evaluations[str(case_path)] = evaluate_case(model, case_path)
        console.print(f"  {' ' * len(str(case_path))}  ...completed.")
    with open("july_evaluation.pickle", "wb") as f:
        pickle.dump(evaluations, f)
    console.print("Mean liver score:", sum(item["liver"] for item in evaluations.values()) / len(evaluations))
    console.print("Mean tumor score:", sum(item["tumor"] for item in evaluations.values()) / len(evaluations))
