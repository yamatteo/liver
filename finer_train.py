from __future__ import annotations

import heapq
import random
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import report
from buffer_dataset import StoredDataset
from distance import batch_cross_entropy, individual_cross_entropy, train_cross_entropy
from subclass_tensors import *

console = Console()

@torch.no_grad()
def eval_valid_round(model, *, dataset: StoredDataset, batch_size: int, device: torch.device, epoch: int, epochs: int):
    round_loss = 0
    with Progress(transient=True) as progress:
        task = progress.add_task(
            f"Eval epoch {epoch + 1}/{epochs}. Evaluating validation dataset.".ljust(50, ' '),
            total=len(dataset)
        )
        for _, bundle in dataset.batches(batch_size):
            bundle = bundle.to(device=device)
            scan = bundle[:, 0:4].float()
            aid = functional.one_hot(bundle[:, 4].to(dtype=torch.int64), 3).permute(0, 4, 1, 2, 3).float()
            segm = bundle[:, 5].long()

            pred = model.forward(torch.cat([scan, aid], dim=1))
            round_loss += batch_cross_entropy(pred, segm) * segm.size(0)

            progress.update(task, advance=batch_size)

    scan_loss = round_loss / len(dataset)
    report.append({"valid_dataset_per_scan_loss": scan_loss})
    return scan_loss

@torch.no_grad()
def eval_train_round(model, *, dataset: StoredDataset, batch_size: int, device: torch.device, epoch: int, epochs: int):
    round_loss = 0
    losses = {}
    with Progress(transient=True) as progress:
        task = progress.add_task(
            f"Epoch {epoch + 1}/{epochs}. Evaluating training dataset.".ljust(50, ' '),
            total=len(dataset)
        )
        for keys, bundle in dataset.batches(batch_size):
            bundle = bundle.to(device=device)
            scan = bundle[:, 0:4].float()
            aid = functional.one_hot(bundle[:, 4].to(dtype=torch.int64), 3).permute(0, 4, 1, 2, 3).float()
            segm = bundle[:, 5].long()

            pred = model.forward(torch.cat([scan, aid], dim=1))
            batch_losses_items = individual_cross_entropy(pred, segm, keys=keys)
            losses.update(batch_losses_items)

            round_loss += sum(batch_losses_items.values())

            progress.update(task, advance=batch_size)

    scan_loss = round_loss / len(dataset)
    report.append({"train_dataset_per_scan_loss": scan_loss})
    return scan_loss, losses


def training_round(model, *, dataloader: DataLoader, optimizer: Optimizer, device: torch.device, epoch: int, epochs: int):
    round_loss = 0
    with Progress(transient=True) as progress:
        task = progress.add_task(
            f"Training epoch {epoch + 1}/{epochs}.".ljust(50, ' '),
            total=len(dataloader)
        )
        for bundle in dataloader:
            bundle = bundle.to(device=device)
            scan = bundle[:, 0:4].float()
            aid = functional.one_hot(bundle[:, 4].to(dtype=torch.int64), 3).permute(0, 4, 1, 2, 3).float()
            segm = bundle[:, 5].long()

            optimizer.zero_grad(set_to_none=True)

            pred = model.forward(torch.cat([scan, aid], dim=1))

            loss = train_cross_entropy(pred, segm)

            loss.backward()
            optimizer.step()

            round_loss += loss.item() * segm.size(0)

            progress.update(task, advance=1)
    sample, _, _ = report.sample(
        FloatScanBatch(scan).detach().cpu(),
        FloatSegmBatch(pred).detach().cpu(),
        FloatSegmBatch.from_int(segm).detach().cpu()
    )
    scan_loss = round_loss / len(dataloader.dataset)
    report.append({"focused_per_scan_loss": scan_loss, "training_sample": sample})
    return scan_loss


def train_cycle(model, *,
                epochs: int,
                train_dataset: StoredDataset,
                valid_dataset: StoredDataset,
                optimizer: Optimizer,
                device: torch.device,
                models_path: Path,
                batch_size: int,
                buffer_size: int):
    model.to(device=device, dtype=torch.float32)
    report.debug("Training model:", model)

    dataloader = None
    for epoch in range(epochs):
        if epoch % 20 == 0:
            losses = {}
            model.eval()

            scan_loss = eval_valid_round(model, dataset=valid_dataset, batch_size=batch_size, device=device, epoch=epoch, epochs=epochs)
            console.print(
                f"Eval epoch {epoch + 1}/{epochs}. "
                f"Validation dataset. "
                f"Loss per scan: {scan_loss:.2e}"
                f"".ljust(50, ' ')
            )

            scan_loss, losses = eval_train_round(model, dataset=train_dataset, batch_size=batch_size, device=device, epoch=epoch, epochs=epochs)
            console.print(
                f"Eval epoch {epoch + 1}/{epochs}. "
                f"Training dataset. "
                f"Loss per scan: {scan_loss:.2e}"
                f"".ljust(50, ' ')
            )
            # losses = {k: random.random() for k in range(len(train_dataset))}  # For debug

            # largest = heapq.nlargest(int(buffer_size*1.1), list(losses.keys()), lambda k: losses[k])
            # reasonable = heapq.nsmallest(buffer_size, largest, lambda k: losses[k])
            reasonable = heapq.nlargest(buffer_size, list(losses.keys()), lambda k: losses[k])
            sub_dataset = train_dataset.subset(reasonable)
            dataloader = DataLoader(
                sub_dataset,
                pin_memory=True,
                batch_size=batch_size,
                # collate_fn=lambda l: FloatBundleBatch(torch.stack(l))
            )
            torch.save(model.state_dict(), models_path / "last_checkpoint.pth")
            torch.save(model.state_dict(), models_path / f"checkpoint{epoch:03}.pth")

        else:
            model.train()
            scan_loss = training_round(model, dataloader=dataloader, optimizer=optimizer, device=device, epoch=epoch, epochs=epochs)
            console.print(
                f"Training epoch {epoch + 1}/{epochs}. "
                f"Training dataset. "
                f"Loss per scan: {scan_loss:.2e}"
                f"".ljust(50, ' ')
            )
