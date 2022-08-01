from __future__ import annotations

from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from rich.progress import Progress
from torch.nn import Module, functional
from torch.utils.data import DataLoader

from .block import *
from .data import Dataset

console = Console()

class HalfUNet(Module):
    def __init__(
            self,
            # channels: list[int],
            # final_classes: int = 3,
            # complexity: int = 2,
            # down_activation: str = "leaky",
            # down_normalization: str = "",
            # down_dropout: str = "",
            # bottom_activation: str = "relu",
            # bottom_normalization: str = "",
            # bottom_dropout: str = "",
            # up_activation: str = "relu",
            # up_normalization: str = "",
            # up_dropout: str = "",
            # pool: str = "max222"
    ):
        super().__init__()
        self.pooler0 = nn.AvgPool3d((32, 32, 8))
        self.block0 = Block(4, 16, actv="leaky", norm="instance")
        self.unblock0 = Block(16, 3, actv="relu", transpose=True)
        self.unpooler0 = nn.Upsample(scale_factor=(32, 32, 8), mode='trilinear')

    def forward(self, x: Tensor) -> Tensor:
        # x.shape is [N, 4, 512, 512, Z]
        return self.unpooler0(self.unblock0(self.block0(self.pooler0(x))))


class HunetNetwork:
    def __init__(self, dataset_path: Path, device: torch.device):
        def format(array):
            return {
                "scan": torch.tensor(array[0:4], dtype=torch.float32, device=device),
                "segm": torch.tensor(array[4], dtype=torch.int64, device=device)
            }

        self.train_dataset = Dataset(dataset_path / "train", format)
        self.valid_dataset = Dataset(dataset_path / "valid", format)
        self.tdl = DataLoader(
            self.train_dataset,
            pin_memory=True,
            batch_size=20,
        )
        self.vdl = DataLoader(
            self.valid_dataset,
            pin_memory=True,
            batch_size=20,
        )

        self.net = HalfUNet()
        self.net.to(device=device)
        self.optimizer = AdaBelief(
            self.net.first_net.parameters(),
            lr=1e-3,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )

        self.loss_function = nn.CrossEntropyLoss(torch.tensor([1, 5, 20])).to(device=device)

    @torch.no_grad()
    def evaluation_round(self, epoch: int, epochs: int):
        round_loss = 0
        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Eval epoch {epoch + 1}/{epochs}.".ljust(50, ' '),
                total=len(self.valid_dataset)
            )
            for batched_data in self.vdl:
                scan = batched_data["scan"]
                segm = batched_data["segm"]
                batch_size = segm.size(0)

                pred = self.net.forward(scan)
                round_loss += self.loss_function(pred, segm).item() * batch_size

                progress.update(task, advance=batch_size)

        scan_loss = round_loss / len(self.valid_dataset)
        return scan_loss

    def training_round(self, epoch: int, epochs: int):
        round_loss = 0
        # samples = []
        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Training epoch {epoch + 1}/{epochs}.".ljust(50, ' '),
                total=len(self.train_dataset)
            )
            for batched_data in self.vdl:
                scan = batched_data["scan"]
                segm = batched_data["segm"]
                batch_size = segm.size(0)

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.net.forward(scan)
                loss = self.loss_function(pred, segm)
                loss.backward()
                self.optimizer.step()
                round_loss += loss.item() * batch_size
                # sample, _, _ = report.sample(scan.detach().cpu(), pred.detach().cpu(), segm.detach().cpu())
                # report.append({"training_sample": sample})

                progress.update(task, advance=batch_size)

        scan_loss = round_loss / len(self.train_dataset)
        return scan_loss  #, samples

    def train(self, epochs: int = 400):
        for epoch in range(epochs):
            if epoch % 20 == 0:
                self.net.eval()
                scan_loss = self.evaluation_round(epoch, epochs)
                console.print(
                    f"Evaluation epoch {epoch + 1}/{epochs}. "
                    f"Loss per scan: {scan_loss:.2e}"
                    f"".ljust(50, ' ')
                )
            else:
                self.net.train()
                scan_loss = self.training_round(epoch, epochs)
                console.print(
                    f"Training epoch {epoch + 1}/{epochs}. "
                    f"Loss per scan: {scan_loss:.2e}"
                    f"".ljust(50, ' ')
                )