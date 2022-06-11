from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import report
from buffer_dataset import BufferDataset
# from dataset import BufferDataset2 as BufferDataset
from models.multi_unet import UNet
from tensors import *

console = Console()


def train_step(scan: ScanBatch, segm: FloatSegmBatch, *, model: UNet, optimizer: Optimizer, keys: list[int]) \
        -> tuple[float, dict[int, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model.forward(scan)
    batch_losses, info_items = pred.distance_from(segm)  # TODO change to generic distance
    batch_losses_items = {k: batch_losses[i].item() for i, k in enumerate(keys)}

    loss = torch.sum(batch_losses)
    loss.backward()
    optimizer.step()
    info_items["train"] = loss.item() / len(batch_losses)
    report.append(info_items)
    return info_items["train"], batch_losses_items


def valid_step(scan: ScanBatch, segm: FloatSegmBatch, *, model: UNet, keys: list[int])\
        -> tuple[float, dict[int, float]]:
    model.eval()
    with torch.no_grad():
        pred = model.forward(scan)
        batch_losses, info_items = pred.distance_from(segm)
        batch_losses_items = {k: batch_losses[i].item() for i, k in enumerate(keys)}

        loss = torch.sum(batch_losses)
        info_items["train"] = loss.item() / len(batch_losses)
        info_items["sample"] = report.sample(scan=scan.cpu(), pred=pred.cpu(), segm=segm.cpu())
        # sample, debug_sample = wandb_sample_debug(scan=scan.cpu(), pred=pred.cpu(), segm=segm.cpu())
        report.append(info_items)
        return loss.item(), batch_losses_items


def train_cycle(model, *,
                epochs: int,
                # dataset: BufferDataset,
                # validation_dataset: BufferDataset,
                train_dataloader: DataLoader,
                valid_dataloader: DataLoader,
                optimizer: Optimizer,
                device: torch.device,
                models_path: Path):
    model.to(device=device, dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        losses = []
        with Progress(transient=False) as progress:
            task = progress.add_task(
                f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
                total=len(train_dataloader)
            )
            for scan_segm in train_dataloader:
                scan, segm = scan_segm.separate()
                scan = scan.to(device=device, dtype=torch.float32)
                segm = segm.to(device=device, dtype=torch.float32)

                loss, batch_losses = train_step(scan, segm, model=model, optimizer=optimizer, keys=list(range(len(scan))))

                epoch_loss += loss
                losses.extend(batch_losses.values())
                progress.update(
                    task,
                    advance=(train_dataloader.batch_size or 1),
                    description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
                )
            progress.update(
                task,
                description=(f"Epoch {epoch}/{epochs}. "
                             f"Total loss: {epoch_loss:.2e}. "
                             f"Loss per scan: {epoch_loss / len(train_dataloader):.2e}").ljust(50, ' ')
            )
        train_dataloader.dataset.update(losses)

        model.eval()
        losses = []
        with Progress(transient=False) as progress:
            task = progress.add_task(f"Validation Step.".ljust(50, ' '), total=len(valid_dataloader))
            with torch.no_grad():
                for scan_segm in valid_dataloader:
                    scan, segm = scan_segm.separate()
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)

                    keys = list(range(len(scan)))
                    loss, batch_losses = valid_step(scan, segm, model=model, keys=keys)

                    epoch_loss += loss
                    losses.extend(batch_losses.values())
                    progress.update(
                        task,
                        advance=(valid_dataloader.batch_size or 1),
                    )
            progress.update(
                task,
                description=f"Validation Step."
                            f"Loss per scan: {epoch_loss / len(valid_dataloader):.2e}".ljust(50, ' ')
            )
        valid_dataloader.dataset.update(losses)

        torch.save(model.state_dict(), models_path / "last_checkpoint.pth")
