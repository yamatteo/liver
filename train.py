from __future__ import annotations

import heapq
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import report
from buffer_dataset import BufferDataset, StoredDataset
# from dataset import BufferDataset2 as BufferDataset
from models.multi_unet import UNet
from tensors import *

console = Console()


def train_batch_step(scan: ScanBatch,
                     segm: FloatSegmBatch, *,
                     model: UNet,
                     optimizer: Optimizer,
                     keys: list[int]
                     ) -> tuple[float, dict[int, float]]:
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

def valid_batch_step(scan: ScanBatch,
                     segm: FloatSegmBatch, *,
                     model: UNet,
                     keys: list[int]
                     ) -> tuple[float, dict[int, float]]:
    model.eval()
    pred = model.forward(scan)
    batch_losses, info_items = pred.distance_from(segm)  # TODO change to generic distance
    batch_losses_items = {k: batch_losses[i].item() for i, k in enumerate(keys)}

    loss = torch.sum(batch_losses)
    info_items["train"] = loss.item() / len(batch_losses)
    report.append(info_items)
    return info_items["train"], batch_losses_items


def valid_step(scan: ScanBatch, segm: FloatSegmBatch, *, model: UNet, keys: list[int]) \
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


def eval_step(
        model, *,
        train_dataset: StoredDataset,
        valid_dataset: StoredDataset,
        optimizer: Optimizer,
        device: torch.device,
        batch_size: int,
        epoch: int,
        epochs: int
):
    model.to(device=device, dtype=torch.float32)
    model.eval()
    epoch_loss = 0
    losses = {}

    with Progress() as progress:
        task = progress.add_task(
            f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
            total=len(valid_dataset)
        )
        for keys, scan_segm in valid_dataset.batches(batch_size):
            scan, segm = scan_segm.separate()
            scan = scan.to(device=device, dtype=torch.float32)
            segm = segm.to(device=device, dtype=torch.float32)

            loss, batch_losses = train_batch_step(
                scan,
                segm,
                model=model,
                optimizer=optimizer,
                keys=keys
            )

            epoch_loss += loss
            progress.update(
                task,
                advance=batch_size,
                description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
            )
        progress.update(
            task,
            description=(f"Epoch {epoch}/{epochs}. "
                         f"Validation total loss: {epoch_loss:.2e}. "
                         f"Loss per scan: {epoch_loss / len(valid_dataset):.2e}").ljust(50, ' ')
        )

    with Progress() as progress:
        task = progress.add_task(
            f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
            total=len(train_dataset)
        )
        for keys, scan_segm in train_dataset.batches(batch_size):
            scan, segm = scan_segm.separate()
            scan = scan.to(device=device, dtype=torch.float32)
            segm = segm.to(device=device, dtype=torch.float32)

            loss, batch_losses = train_batch_step(
                scan,
                segm,
                model=model,
                optimizer=optimizer,
                keys=keys
            )

            epoch_loss += loss
            losses.update(batch_losses)
            progress.update(
                task,
                advance=batch_size,
                description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
            )
        progress.update(
            task,
            description=(f"Epoch {epoch}/{epochs}. "
                         f"Training total loss: {epoch_loss:.2e}. "
                         f"Loss per scan: {epoch_loss / len(train_dataset):.2e}").ljust(50, ' ')
        )
    return losses


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
    dataloader = None
    for epoch in range(epochs):
        if epoch % 20 == 0:
            # Evaluation
            epoch_loss = 0
            losses = {}

            with Progress() as progress:
                task = progress.add_task(
                    f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
                    total=len(valid_dataset)
                )
                for keys, scan_segm in valid_dataset.batches(batch_size):
                    scan, segm = scan_segm.separate()
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)

                    loss, batch_losses = valid_batch_step(
                        scan,
                        segm,
                        model=model,
                        keys=keys
                    )

                    epoch_loss += loss
                    progress.update(
                        task,
                        advance=batch_size,
                        description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
                    )
                progress.update(
                    task,
                    description=(f"Epoch {epoch}/{epochs}. "
                                 f"Validation total loss: {epoch_loss:.2e}. "
                                 f"Loss per scan: {epoch_loss / len(valid_dataset):.2e}").ljust(50, ' ')
                )

            with Progress() as progress:
                task = progress.add_task(
                    f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
                    total=len(train_dataset)
                )
                for keys, scan_segm in train_dataset.batches(batch_size):
                    scan, segm = scan_segm.separate()
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)

                    loss, batch_losses = valid_batch_step(
                        scan,
                        segm,
                        model=model,
                        keys=keys
                    )

                    epoch_loss += loss
                    losses.update(batch_losses)
                    progress.update(
                        task,
                        advance=batch_size,
                        description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
                    )
                progress.update(
                    task,
                    description=(f"Epoch {epoch}/{epochs}. "
                                 f"Training total loss: {epoch_loss:.2e}. "
                                 f"Loss per scan: {epoch_loss / len(train_dataset):.2e}").ljust(50, ' ')
                )
            smallest = heapq.nsmallest(buffer_size, list(losses.keys()), lambda k: losses[k])
            sub_dataset = train_dataset.subset(smallest)
            dataloader = DataLoader(
                sub_dataset,
                pin_memory=True,
                batch_size=batch_size,
                collate_fn=lambda l: FloatBatchBundle(torch.stack(l))
            )
        else:
            model.train()
            epoch_loss = 0
            with Progress(transient=True) as progress:
                task = progress.add_task(
                    f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' '),
                    total=len(dataloader)
                )
                for scan_segm in dataloader:
                    scan, segm = scan_segm.separate()
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)

                    loss, batch_losses = train_batch_step(scan, segm, model=model, optimizer=optimizer, keys=[])

                    epoch_loss += loss
                    # losses.update(batch_losses)
                    progress.update(
                        task,
                        advance=1,
                        description=f"Epoch {epoch + 1}/{epochs}. Loss so far: {epoch_loss:.2e}".ljust(50, ' ')
                    )
                # progress.update(
                #     task,
                #     description=(f"Epoch {epoch}/{epochs}. "
                #                  f"Total loss: {epoch_loss:.2e}. "
                #                  f"Loss per scan: {epoch_loss / len(dataset):.2e}").ljust(50, ' ')
                # )

        torch.save(model.state_dict(), models_path / "last_checkpoint.pth")
