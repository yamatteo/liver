from __future__ import annotations

import heapq
import os
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch import Tensor
from tqdm import tqdm

from dataset import BufferDataset2 as BufferDataset
from functions.distances import batch_jaccard_distance, batch_l1_loss
import report
from utils.image_generation import wandb_sample_debug as wandb_sample

console = Console()


def train_step(scan: Tensor, segm: Tensor, *, model, optimizer, keys) -> tuple[float, dict[int, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(scan)

    # pure_jaccard = jaccard_distance(pred, segm)
    #
    # jaccard = jaccard_distance(
    #     functional.softmax(pred, dim=1),
    #     segm
    # )
    #
    # jaccard2 = jaccard_distance(
    #     functional.softmax(pred, dim=1)[:, 1:, :, :, :],
    #     segm[:, 1:, :, :, :]
    # )
    #
    # pixel = functional.l1_loss(pred, segm)

    batch_jd = batch_jaccard_distance(pred, segm)
    batch_l1 = batch_l1_loss(pred, segm)
    batch_loss = batch_jd + batch_l1
    batch_losses_items = {k: batch_loss[i].item() for i, k in enumerate(keys)}

    loss = torch.mean(batch_loss)
    loss.backward()
    optimizer.step()
    report.append({
        "train_loss": loss.item(),
        "train_jd": torch.mean(batch_jd).item(),
        "train_l1": torch.mean(batch_l1).item(),
    })
    return loss.item(), batch_losses_items


def valid_step(scan: Tensor, segm: Tensor, *, model, keys) -> tuple[float, dict[int, float]]:
    model.eval()
    with torch.no_grad():
        pred = model(scan)

        # pure_jaccard = jaccard_distance(pred, segm)
        #
        # jaccard = jaccard_distance(
        #     functional.softmax(pred, dim=1),
        #     segm
        # )
        #
        # jaccard2 = jaccard_distance(
        #     functional.softmax(pred, dim=1)[:, 1:, :, :, :],
        #     segm[:, 1:, :, :, :]
        # )
        #
        # pixel = functional.l1_loss(pred, segm)

        batch_jd = batch_jaccard_distance(pred, segm)
        batch_l1 = batch_l1_loss(pred, segm)
        batch_loss = batch_jd + batch_l1
        batch_losses_items = {k: batch_loss[i].item() for i, k in enumerate(keys)}

        loss = torch.mean(batch_loss)
        report.append({
            "valid_loss": loss.item(),
            "valid_jd": torch.mean(batch_jd).item(),
            "valid_l1": torch.mean(batch_l1).item(),
            "sample": wandb_sample(scan=scan.cpu(), pred=pred.cpu(), segm=segm.cpu())
        })
        return loss.item(), batch_losses_items


def train_cycle(model, *, epochs: int, dataset: BufferDataset, optimizer: AdaBelief, device, train_drop: int = None):
    model.to(device=device, dtype=torch.float32)
    if train_drop is None:
        train_drop = len(dataset) // 10
    for epoch in range(epochs):
        epoch_loss = 0
        losses = {}
        with tqdm(
                total=len(dataset) + dataset.valid_len(),
                desc=f'Epoch {epoch + 1}/{epochs}',
                unit='scan'
        ) as pbar:
            for keys, (scan, segm) in dataset.train_batches():
                scan = scan.to(device=device, dtype=torch.float32)
                segm = segm.to(device=device, dtype=torch.float32)

                loss, batch_losses = train_step(scan, segm, model=model, optimizer=optimizer, keys=keys)

                pbar.update(len(keys))
                epoch_loss += loss
                losses.update(batch_losses)
                pbar.set_postfix(**{'loss (batch)': epoch_loss})

            smallest = heapq.nsmallest(train_drop, list(losses.keys()), lambda k: losses[k])
            dataset.drop(list(smallest))

            torch.save(model.cpu().state_dict(), Path(os.getenv("SAVED_MODELS")) / "last_checkpoint.pth")
            model.cuda()

            losses = {}
            for keys, (scan, segm) in dataset.valid_batches():
                scan = scan.to(device=device, dtype=torch.float32)
                segm = segm.to(device=device, dtype=torch.float32)
                loss, batch_losses = valid_step(scan, segm, model=model, keys=keys)

                pbar.update(len(keys))
                losses.update(batch_losses)

            smallest = heapq.nsmallest(1, list(losses.keys()), lambda k: losses[k])
            dataset.valid_drop(list(smallest))
