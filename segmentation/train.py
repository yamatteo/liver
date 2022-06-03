from __future__ import annotations

import heapq
import os
import shutil
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch import Tensor
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BufferDataset2 as BufferDataset
from report import Report
from utils.generators import train_slices
from utils.image_generation import rgb_sample

console = Console()
report = Report()
classes = ["background", "liver", "tumor"]


def train_step(model, scan: Tensor, segm: Tensor, global_step: int, optimizer, writer):
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(scan, segm)

    loss.backward()
    optimizer.step()
    writer.add_scalar(
        "training_loss",
        loss.item(),
        global_step=global_step,
    )
    return loss.item()


def valid_step(model, scan: Tensor, segm: Tensor, global_step: int, optimizer, writer):
    with torch.no_grad():
        loss, pred = model.loss_forward(scan, segm)

        writer.add_scalar(
            "validation_loss",
            loss.item(),
            global_step=global_step,
        )
        return loss.item(), pred


def train_cycle(model, epochs: int, dataset: BufferDataset, optimizer: AdaBelief, writer: SummaryWriter, device):
    # console.print(f"before model {next(model.parameters()).is_cuda}")
    model.to(device=device, dtype=torch.float32)
    # console.print(f"after model {next(model.parameters()).is_cuda}")
    model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        losses = {}
        samples = []
        if epoch % 20 != 0:

            with tqdm(
                    total=len(dataset),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    unit='img'
            ) as pbar:
                for k, (scan, segm) in dataset:
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)

                    loss_item = train_step(model, scan=scan, segm=segm, global_step=global_step, optimizer=optimizer,
                                           writer=writer)

                    global_step += 1
                    pbar.update(1)
                    epoch_loss += loss_item
                    losses[k] = loss_item
                    pbar.set_postfix(**{'loss (batch)': epoch_loss})

            smallest = heapq.nsmallest(10, list(losses.keys()), lambda k: losses[k])
            dataset.drop(list(smallest))

            torch.save(model.cpu().state_dict(), Path(os.getenv("SAVED_MODELS")) / "last_checkpoint.pth")
            model.cuda()
        else:

            with tqdm(
                    total=dataset.valid_len(),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    unit='img'
            ) as pbar:
                for k, (scan, segm) in dataset.valid_iter():
                    scan = scan.to(device=device, dtype=torch.float32)
                    segm = segm.to(device=device, dtype=torch.float32)
                    loss_item, pred = valid_step(model, scan=scan, segm=segm, global_step=global_step, optimizer=optimizer,
                                                 writer=writer)
                    samples.append(rgb_sample(
                        scan.cpu(),
                        pred.cpu(),
                        # functional.softmax(pred.cpu(), dim=1)[:, 1:, :, :, :],
                        segm.cpu(),
                        ("error", "tumor", "liver")
                    ))
                    global_step += 1
                    pbar.update(1)
                    epoch_loss += loss_item
                    losses[k] = loss_item
                    pbar.set_postfix(**{'loss (batch)': epoch_loss})

                    writer.add_images(
                        "samples",
                        torch.stack(samples),
                        global_step=global_step,
                        dataformats="NCHW"
                    )

            smallest = heapq.nsmallest(1, list(losses.keys()), lambda k: losses[k])
            dataset.valid_drop(list(smallest))

