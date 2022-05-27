from __future__ import annotations

import heapq
import shutil
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BufferDataset2 as BufferDataset
from utils.generators import train_bundles, train_slices

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


def train_net(device, writer_path, data_path, model, slice_side = 16):
    shutil.rmtree(Path(writer_path), ignore_errors=True)
    dataset = BufferDataset(
        generator=train_slices(data_path, slice_side),
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
    model.to(device)
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
        loss = model.loss(scan, segm)

        writer.add_scalar(
            "validation_loss",
            loss.item(),
            global_step=global_step,
        )
        return loss.item()



def train_cycle(model, epochs: int, dataset: BufferDataset, optimizer: AdaBelief, writer: SummaryWriter, device):
    model.train()
    model.to(device=device, dtype=torch.float32)
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        losses = {}
        if epoch % 20 != 0:

            with tqdm(
                    total=len(dataset),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    unit='img'
            ) as pbar:
                for k, (scan, segm) in dataset:
                    scan.to(device=device, dtype=torch.float32)
                    segm.to(device=device, dtype=torch.float32)
                    loss_item = train_step(model, scan=scan, segm=segm, global_step=global_step,optimizer=optimizer, writer=writer)

                    global_step += 1
                    pbar.update(1)
                    epoch_loss += loss_item
                    losses[k] = loss_item
                    pbar.set_postfix(**{'loss (batch)': epoch_loss})

            smallest = heapq.nsmallest(10, list(losses.keys()), lambda k: losses[k])
            dataset.drop(list(smallest))

        else:

            with tqdm(
                    total=dataset.valid_len(),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    unit='img'
            ) as pbar:
                for k, (scan, segm) in dataset.valid_iter():
                    scan.to(device=device, dtype=torch.float32)
                    segm.to(device=device, dtype=torch.float32)
                    loss_item = valid_step(model, scan=scan, segm=segm, global_step=global_step,optimizer=optimizer, writer=writer)

                    global_step += 1
                    pbar.update(1)
                    epoch_loss += loss_item
                    losses[k] = loss_item
                    pbar.set_postfix(**{'loss (batch)': epoch_loss})

                    # Evaluation round
                    # if global_step % 100 == 0:
                    #     # writer.add_scalars(
                    #     #     "evaluation",
                    #     #     evaluate(net, valid_dl, device),
                    #     #     global_step=global_step,
                    #     # )
                    #     writer.add_images(
                    #         "samples",
                    #         samples(net, dataset, device),
                    #         global_step=global_step,
                    #         dataformats="NCHW"
                    #     )

            smallest = heapq.nsmallest(10, list(losses.keys()), lambda k: losses[k])
            dataset.valid_drop(list(smallest))

