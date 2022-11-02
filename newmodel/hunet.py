from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from rich.progress import Progress
from torch.utils.data import DataLoader

import report
from .models import *
from .data import Dataset, store_441_dataset as _store_dataset

console = Console()


class HalfUNet(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale_factors = [(32, 32, 8), (16, 16, 4), (8, 8, 2), (4, 4, 1), (2, 2, 1)]
        self.poolers = nn.ModuleList([
            nn.AvgPool3d(kernel_shape)
            for kernel_shape in self.scale_factors
        ])
        self.dnblocks = nn.ModuleList([
            Block(
                in_channels=4 if n == 0 else 20 * n,
                out_channels=16 + 20 * n,
                complexity=1,
                actv="leaky",
                norm="batch",
                norm_momentum=0.1,
            )
            for n in range(len(self.scale_factors))
        ])
        self.upblocks = nn.ModuleList([
            Block(
                in_channels=16 + 20 * n,
                out_channels=16 + 20 * n,
                complexity=1,
                actv="relu",
                drop="drop",
                transpose=True,
            )
            for n in range(len(self.scale_factors))
        ])
        self.exits = nn.ModuleList([
            nn.Sequential(
                Block(16 + 20 * n, 3, complexity=2, kernel_size=(1, 1, 1)),
                nn.Upsample(scale_factor=scale_factor, mode='trilinear')
            )
            for n, scale_factor in enumerate(self.scale_factors)
        ])
        # self.pooler0 = nn.AvgPool3d((32, 32, 8))
        # self.block0 = Block(4, 16, complexity=1, actv="leaky", norm="instance")
        # self.unblock0 = Block(16, 16, complexity=1, actv="relu", transpose=True)
        # self.early_exit0 = nn.Sequential(
        #     Block(16, 3, complexity=2, kernel_size=(1, 1, 1)),
        #     nn.Upsample(scale_factor=(32, 32, 8), mode='trilinear')
        # )

    def forward_prep_exit(self, n: int, x: Tensor) -> Tensor:
        # x.shape is [N, 4 if n==0 else 20*n, 512, 512, 40] / scale_factors[n]
        return self.exits[n](self.upblocks[n](self.dnblocks[n](x)))

    def resume(self, models_path: Path, model_name="last_checkpoint.pth", device=torch.device("cpu")):
        try:
            self.load_state_dict(torch.load(models_path / model_name, map_location=device))
            console.print(f"Model loaded from {models_path / model_name}")
        except FileNotFoundError:
            console.print(f"Model {models_path / model_name} does not exist. Starting with a new model.")

    def save(self, models_path: Path, model_name="last_checkpoint.pth"):
        torch.save(self.state_dict(), models_path / model_name)


class HunetNetwork:
    net: HalfUNet
    store_dataset: Callable
    train_dataset: Dataset
    valid_dataset: Dataset
    tdl: DataLoader
    vdl: DataLoader
    loss_function: Module
    optimizer: AdaBelief

    @classmethod
    def init_0(cls, dataset_path: Path, device: torch.device, store: bool = False, source_path: Path = None):
        batch_size = 8
        self = HunetNetwork()
        self.net = HalfUNet()
        self.net.to(device=device)

        if store:
            _store_dataset(source_path, dataset_path, slice_shape=(512, 512, 40), min_slice_z=40)

        def format(array):
            return {
                "scan": self.net.poolers[0](torch.tensor(array[0:4], dtype=torch.float32)).to(device=device).detach(),
                "segm": torch.tensor(array[4], dtype=torch.int64, device=device).detach()
            }

        self.train_dataset = Dataset(dataset_path / "train", format)
        self.valid_dataset = Dataset(dataset_path / "valid", format)

        self.tdl = DataLoader(
            self.train_dataset,
            pin_memory=False,
            batch_size=batch_size,
        )
        self.vdl = DataLoader(
            self.valid_dataset,
            pin_memory=False,
            batch_size=batch_size,
        )

        parameters = [
            *self.net.dnblocks[0].parameters(),
            *self.net.upblocks[0].parameters(),
            *self.net.exits[0].parameters()
        ]

        self.optimizer = AdaBelief(
            parameters,
            lr=1e-3,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )

        self.loss_function = nn.CrossEntropyLoss().to(device=device)
        return self

    @torch.no_grad()
    def evaluation_round(self, n: int, epoch: int, epochs: int):
        round_loss = 0
        samples = []
        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Eval epoch {epoch + 1}/{epochs}.".ljust(50, ' '),
                total=len(self.valid_dataset)
            )
            for batched_data in self.vdl:
                scan = batched_data["scan"]
                segm = batched_data["segm"]
                batch_size = segm.size(0)

                pred = self.net.forward_prep_exit(n, scan)
                round_loss += self.loss_function(pred, segm).item() * batch_size
                samples.append(report.sample(
                    scan.detach().cpu().numpy(),
                    torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                    segm.detach().cpu().numpy()
                ))

                progress.update(task, advance=batch_size)

        scan_loss = round_loss / len(self.valid_dataset)
        return scan_loss, samples

    def training_round(self, n: int, epoch: int, epochs: int):
        round_loss = 0
        samples = []
        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Training epoch {epoch + 1}/{epochs}. {len(self.train_dataset)} to process.".ljust(50, ' '),
                total=len(self.train_dataset)
            )
            for batched_data in self.tdl:
                scan = batched_data["scan"]
                segm = batched_data["segm"]
                batch_size = segm.size(0)

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.net.forward_prep_exit(n, scan)
                loss = self.loss_function(pred, segm)
                loss.backward()
                self.optimizer.step()
                round_loss += loss.item() * batch_size
                samples.append(report.sample(
                    scan.detach().cpu().numpy(),
                    torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                    segm.detach().cpu().numpy()
                ))

                progress.update(task, advance=batch_size)

        scan_loss = round_loss / len(self.train_dataset)
        return scan_loss, samples

    def train(self, *, n: int, epochs: int = 400, models_path: Path):
        run = report.init(project="liver-tumor-detection", entity="yamatteo", backend="wandb", level="debug")
        try:
            for epoch in range(epochs):
                if epoch % 20 == 0:
                    self.net.eval()
                    scan_loss, samples = self.evaluation_round(n, epoch, epochs)
                    console.print(
                        f"Evaluation epoch {epoch + 1}/{epochs}. "
                        f"Loss per scan: {scan_loss:.2e}"
                        f"".ljust(50, ' ')
                    )
                    report.append({"valid_epoch_loss": scan_loss, "samples": samples})
                    self.net.save(models_path, "last_checkpoint.pth")
                    self.net.save(models_path, f"checkpoint{epoch:03}.pth")
                else:
                    self.net.train()
                    scan_loss, samples = self.training_round(n, epoch, epochs)
                    console.print(
                        f"Training epoch {epoch + 1}/{epochs}. "
                        f"Loss per scan: {scan_loss:.2e}"
                        f"".ljust(50, ' ')
                    )
                    report.append({"train_epoch_loss": scan_loss, "samples": samples})
        except:
            run.finish()
            console.print_exception()