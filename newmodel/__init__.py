import argparse
import pickle
from pathlib import Path

import nibabel
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from rich.progress import Progress
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

import report
import utils.ndarray as nd
import utils.path_explorer as px
from utils.slices import overlapping_slices, padded_nonoverlapping_scan_slices as scan_slices
from distances import liverscore, tumorscore
from .hunet import HunetNetwork, HalfUNet
from .data import store_441_dataset, Dataset, BufferDataset
from .models import UNet

console = Console()
saved_models = Path(__file__).parent / "saved_models"
saved_models.mkdir(exist_ok=True)


def setup_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_kwargs = torch.load(saved_models / "last_model_kwargs.pt")
    model = UNet(**model_kwargs)
    model.to(device=device)
    model.load_state_dict(
        torch.load(saved_models / "latest.pth", map_location=device)
    )
    console.print(f"Model loaded from {saved_models / 'latest.pth'}")
    return model, device


def setup_train(
        dataset_path: Path,
        buffer_size: int = 0,
        models_path: Path = "saved_models",
        model_name: str = "last.pth",
        device=torch.device("cpu"),
        batch_size: int = 50,
        lr: float = 1e-3,
        **kwargs):
    self = argparse.Namespace()
    self.device = device

    self.model = UNet(**kwargs)
    self.model_kwargs = kwargs
    self.model.to(device=device)

    try:
        self.model.load_state_dict(torch.load(models_path / model_name, map_location=device))
        console.print(f"Model loaded from {models_path / model_name}")
    except FileNotFoundError:
        console.print(f"Model {models_path / model_name} does not exist. Starting with a new model.")

    if buffer_size > 0:
        self.train_dataset = BufferDataset(dataset_path / "train", buffer_size, buffer_size // 10)
    else:
        self.train_dataset = Dataset(dataset_path / "train")
    self.valid_dataset = Dataset(dataset_path / "valid")

    self.tdl = DataLoader(
        self.train_dataset,
        pin_memory=True,
        batch_size=batch_size,
    )
    self.vdl = DataLoader(
        self.valid_dataset,
        pin_memory=True,
        batch_size=batch_size,
    )

    self.optimizer = AdaBelief(
        self.model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )

    self.loss_function = nn.CrossEntropyLoss(torch.tensor([1, 2, 5])).to(device=device)
    if buffer_size > 0:
        m = nn.CrossEntropyLoss(torch.tensor([1, 2, 5]), reduction="none").to(device=device)
        def score_function(pred, segm, keys):
            loss = m(pred, segm)
            loss = torch.mean(loss, dim=[1, 2, 3])
            scores = {k.item(): loss[i].item() for i, k in enumerate(keys)}
            return scores
        self.score_function = score_function
    return self


@torch.no_grad()
def evaluation_round(setup, epoch: int, epochs: int):
    round_loss = 0
    samples = []
    with Progress(transient=True) as progress:
        task = progress.add_task(
            f"Eval epoch {epoch + 1}/{epochs}.".ljust(50, ' '),
            total=len(setup.valid_dataset)
        )
        for batched_data in setup.vdl:
            scan = batched_data["scan"].to(device=setup.device)
            segm = batched_data["segm"].to(device=setup.device, dtype=torch.int64)
            batch_size = segm.size(0)

            pred = setup.model(scan)
            round_loss += setup.loss_function(pred, functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(
                dtype=torch.float32)).item() * batch_size
            samples.append(report.sample(
                scan.detach().cpu().numpy(),
                torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                segm.detach().cpu().numpy()
            ))

            progress.update(task, advance=batch_size)

    scan_loss = round_loss / len(setup.valid_dataset)
    return scan_loss, samples


def training_round(setup, epoch: int, epochs: int, use_buffer: bool = False):
    round_loss = 0
    samples = []
    with Progress(transient=True) as progress:
        task = progress.add_task(
            f"Training epoch {epoch + 1}/{epochs}. {len(setup.train_dataset)} to process.".ljust(50, ' '),
            total=len(setup.train_dataset)
        )
        for batched_data in setup.tdl:
            scan = batched_data["scan"].to(device=setup.device)
            segm = batched_data["segm"].to(device=setup.device, dtype=torch.int64)
            if use_buffer:
                keys = batched_data["keys"]
            batch_size = segm.size(0)

            setup.optimizer.zero_grad(set_to_none=True)

            pred = setup.model(scan)
            if use_buffer:
                scores = setup.score_function(pred, functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(dtype=torch.float32), keys=keys)
            loss = setup.loss_function(pred, functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(dtype=torch.float32))
            loss.backward()
            setup.optimizer.step()
            round_loss += loss.item() * batch_size
            samples.append(report.sample(
                scan.detach().cpu().numpy(),
                torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                segm.detach().cpu().numpy()
            ))

            progress.update(task, advance=batch_size)

    scan_loss = round_loss / len(setup.train_dataset)
    if use_buffer:
        return scan_loss, scores, samples
    return scan_loss, samples


def train(setup, *, epochs: int = 400, resume_from=0, models_path: Path, use_buffer: bool = False):
    run = report.init(project="liver-tumor-detection", entity="yamatteo", backend="wandb", level="debug")

    if isinstance(resume_from, str):
        model_name = resume_from
    elif isinstance(resume_from, int) and resume_from > 0:
        model_name = f"checkpoint{resume_from:03}.pth"
    else:
        model_name = None
    try:
        setup.model.load_state_dict(torch.load(models_path / model_name, map_location=setup.device))
        console.print(f"Model loaded from {models_path / model_name}")
    except FileNotFoundError:
        console.print(f"Model {models_path / model_name} does not exist. Starting with a new model.")
    try:
        for epoch in range(epochs):
            if epoch % 20 == 0:
                setup.model.eval()
                scan_loss, samples = evaluation_round(setup, epoch, epochs)
                console.print(
                    f"Evaluation epoch {epoch + 1}/{epochs}. "
                    f"Loss per scan: {scan_loss:.2e}"
                    f"".ljust(50, ' ')
                )
                report.append({"valid_epoch_loss": scan_loss, "samples": samples})
                torch.save(setup.model.state_dict(), models_path / "last_checkpoint.pth")
                torch.save(setup.model.state_dict(), models_path / f"checkpoint{epoch:03}.pth")
                torch.save(setup.model_kwargs, models_path / "last_model_kwargs.pt")
            else:
                setup.model.train()
                if use_buffer is True:
                    scan_loss, scores, samples = training_round(setup, epoch, epochs, use_buffer=True)
                    setup.train_dataset.drop(scores)
                else:
                    scan_loss, samples = training_round(setup, epoch, epochs, use_buffer=False)
                console.print(
                    f"Training epoch {epoch + 1}/{epochs}. "
                    f"Loss per scan: {scan_loss:.2e}"
                    f"".ljust(50, ' ')
                )
                report.append({"train_epoch_loss": scan_loss, "samples": samples})
    except:
        run.finish()
        console.print_exception()


@torch.no_grad()
def apply(model, case_path, device):
    pooler = nn.AvgPool3d(kernel_size=(4, 4, 1))
    unpooler = nn.Upsample(scale_factor=(4, 4, 1), mode='trilinear')
    affine, bottom, top, height = nd.load_registration_data(case_path)
    scan = nd.load_scan_from_regs(case_path)
    scan = np.clip(scan, -1024, 1024)
    predictions = []
    for slice in scan_slices(scan, (512, 512, 32)):
        slice = pooler(slice)
        pred = model(torch.tensor(slice, dtype=torch.float32, device=device).unsqueeze(0))
        pred = torch.argmax(pred, dim=1).to(dtype=torch.int16).cpu()
        pred = unpooler(pred)
        predictions.append(pred.cpu().numpy())
        del pred
        torch.cuda.empty_cache()
    prediction = np.concatenate(predictions, axis=3)
    pred = np.full([512, 512, height], fill_value=-1024, dtype=np.int16)
    pred[..., bottom:top] = prediction[..., :(top - bottom)]
    return pred, affine


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
    with open("newmodel_evaluation.pickle", "wb") as f:
        pickle.dump(evaluations, f)
    console.print("Mean liver score:", sum(item["liver"] for item in evaluations.values()) / len(evaluations))
    console.print("Mean tumor score:", sum(item["tumor"] for item in evaluations.values()) / len(evaluations))
