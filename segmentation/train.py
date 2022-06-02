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
from utils.generators import train_slices
from utils.image_generation import rgb_sample

console = Console()
classes = ["background", "liver", "tumor"]


# @torch.no_grad()
# def evaluate(net: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.DeviceObjType) -> dict[str, float]:
#     net.eval()
#     num_val_batches = len(dataloader)
#     metrics = {
#         "dice": 0,
#         "cross_entropy": 0,
#         "jaccard": 0,
#     }
#
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         wafers = batch['wafer'].to(device=device, dtype=torch.float32)
#         segmentations = batch['segmentation'].to(device=device, dtype=torch.long)
#         segmentations_one_hot = one_hot(segmentations, len(classes)).float()
#
#         predictions_one_hot = net(wafers)
#         predictions = predictions_one_hot.argmax(dim=3)
#
#         metrics["dice"] += dice_distance(
#             predictions_one_hot[..., 1:],
#             segmentations_one_hot[..., 1:],
#             spatial_dims=[1, 2]
#         ).item()
#
#         metrics["cross_entropy"] += nn.functional.cross_entropy(
#             predictions_one_hot.permute(0, 3, 1, 2),
#             segmentations,
#         ).item()
#
#         metrics["jaccard"] += jaccard_distance(
#             predictions_one_hot,
#             segmentations_one_hot,
#             spatial_dims=[1, 2]
#         ).item()
#
#     net.train()
#     try:
#         return {
#             label: value / num_val_batches
#             for label, value in metrics.items()
#         }
#     except ZeroDivisionError:
#         return metrics


# @torch.no_grad()
# def samples(net, valid_ds, device):
#     k = 8
#     indices = random.sample(range(len(valid_ds)), k)
#     wafers = torch.stack([
#         valid_ds[i]["wafer"]
#         for i in indices
#     ]).to(device=device, dtype=torch.float32)
#     segmentation = torch.stack([
#         valid_ds[i]["segmentation"]
#         for i in indices
#     ]).to(device=device, dtype=torch.long)
#     # predict the mask
#     prediction = net(wafers).argmax(dim=3)
#     wafers = torch.clamp(wafers[:, 2, :, :, 2], 0, 256) / 256
#     # scan = scan[:, 0, :, :, 1]
#     prediction = prediction / 2
#     segmentation = segmentation / 2
#     return torch.stack([
#         image[i]
#         for image in [wafers, prediction, segmentation]
#         for i in range(k)
#     ]).unsqueeze(1)


def train_net(device, writer_path, data_path, model, slice_shape=(32, 32, 8)):
    shutil.rmtree(Path(writer_path), ignore_errors=True)
    dataset = BufferDataset(
        generator=train_slices(data_path, slice_shape),
        buffer_size=100,
        train_to_valid_odds=9,
        valid_buffer_size=20
    )
    optimizer = AdaBelief(
        model.parameters(),
        lr=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )

    writer = SummaryWriter(writer_path)
    train_cycle(model, epochs=1000, dataset=dataset, optimizer=optimizer, writer=writer, device=device)


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

            smallest = heapq.nsmallest(10, list(losses.keys()), lambda k: losses[k])
            dataset.valid_drop(list(smallest))

