from __future__ import annotations

from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from rich.progress import Progress
from torch.utils.data import DataLoader

import report
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
        self.block0 = Block(4, 16, complexity=1, actv="leaky", norm="instance")
        self.unblock0 = Block(16, 16, complexity=1, actv="relu", transpose=True)
        self.early_exit0 = nn.Sequential(
            Block(16, 3, complexity=2, kernel_size=(1, 1, 1)),
            nn.Upsample(scale_factor=(32, 32, 8), mode='trilinear')
        )

    def forward(self, x: Tensor) -> Tensor:
        # x.shape is [N, 4, 512, 512, 40]
        return self.early_exit0(self.unblock0(self.block0(self.pooler0(x))))
        

    def resume(self, models_path: Path, model_name="last_checkpoint.pth", device=torch.device("cpu")):
        try:
            self.load_state_dict(torch.load(models_path / model_name, map_location=device))
            console.print(f"Model loaded from {models_path / model_name}")
        except FileNotFoundError:
            console.print(f"Model {models_path / model_name} does not exist. Starting with a new model.")

    def save(self, models_path: Path, model_name="last_checkpoint.pth"):
        torch.save(self.state_dict(), models_path / model_name)


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
            pin_memory=False,
            batch_size=10,
        )
        self.vdl = DataLoader(
            self.valid_dataset,
            pin_memory=False,
            batch_size=10,
        )

        self.net = HalfUNet()
        self.net.to(device=device)
        self.optimizer = AdaBelief(
            self.net.parameters(),
            lr=1e-3,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )

        self.loss_function = nn.CrossEntropyLoss().to(device=device)

    @torch.no_grad()
    def evaluation_round(self, epoch: int, epochs: int):
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

                pred = self.net.forward(scan)
                round_loss += self.loss_function(pred, segm).item() * batch_size
                samples.append(report.sample(
                    scan.detach().cpu().numpy(),
                    torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                    segm.detach().cpu().numpy()
                ))

                progress.update(task, advance=batch_size)

        scan_loss = round_loss / len(self.valid_dataset)
        return scan_loss, samples

    def training_round(self, epoch: int, epochs: int):
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
                console.print("batch size", batch_size)

                self.optimizer.zero_grad(set_to_none=True)

                pred = self.net.forward(scan)
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

    def train(self, *, epochs: int = 400, models_path: Path):
        run = report.init(project="liver-tumor-detection", entity="yamatteo", backend="wandb", level="debug")
        try:
            for epoch in range(epochs):
                if epoch % 20 == 0:
                    self.net.eval()
                    scan_loss, samples = self.evaluation_round(epoch, epochs)
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
                    scan_loss, samples = self.training_round(epoch, epochs)
                    console.print(
                        f"Training epoch {epoch + 1}/{epochs}. "
                        f"Loss per scan: {scan_loss:.2e}"
                        f"".ljust(50, ' ')
                    )
                    report.append({"train_epoch_loss": scan_loss, "samples": samples})
        except:
            run.finish()
            console.print_exception()