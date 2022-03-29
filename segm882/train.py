from __future__ import annotations

import argparse
import heapq
import random
import shutil
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.buffer import BufferDataset
from dataset.generators import slices882, scan_segm_tuples
from functions.distances import halfway_jaccard_distance, jaccard_distance
from models import get_model
from options import defaults

console = Console()
classes = ["background", "liver", "tumor"]


@torch.no_grad()
def evaluate(net: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.DeviceObjType) -> dict[str, float]:
    net.eval()
    num_val_batches = len(dataloader)
    metrics = {
        "dice": 0,
        "cross_entropy": 0,
        "jaccard": 0,
    }

    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        scans = batch['scan'].to(device=device, dtype=torch.float32)
        segmentations = batch['segmentation'].to(device=device, dtype=torch.long)
        segmentations_one_hot = one_hot(segmentations, len(classes)).float()

        predictions_one_hot = net(scans)
        predictions = predictions_one_hot.argmax(dim=1)

        metrics["dice"] += dice_distance(
            predictions_one_hot.permute(0, 2, 3, 4, 1)[..., 1:],
            segmentations_one_hot[..., 1:],
            spatial_dims=[1, 2]
        ).item()

        metrics["cross_entropy"] += nn.functional.cross_entropy(
            predictions_one_hot,
            segmentations,
        ).item()

        metrics["jaccard"] += jaccard_distance(
            predictions_one_hot.permute(0, 2, 3, 4, 1),
            segmentations_one_hot,
            spatial_dims=[1, 2, 3]
        ).item()

    net.train()
    try:
        return {
            label: value / num_val_batches
            for label, value in metrics.items()
        }
    except ZeroDivisionError:
        return metrics


@torch.no_grad()
def samples(net, valid_ds, device):
    k = 1
    cases = torch.cat([ case for (i, case) in random.sample(valid_ds.buffer.items(), k)])
    scan = torch.narrow(cases, 1, 0, 4).to(device=device, dtype=torch.float32)
    segm = torch.narrow(cases, 1, 4, 3).to(device=device, dtype=torch.float32)
    # predict the mask
    pred = net(scan)
    zs = [ int(i*cases.size(4) / 24) for i in range(24) ]


    scan = torch.clamp(torch.narrow(scan, 1, 2, 1), 0, 256) / 256  # Only phase v
    errors = torch.sum(torch.abs(segm-pred), dim=1, keepdim=True)
    liver = torch.narrow(segm, 1, 1, 1)
    tumor = torch.narrow(segm, 1, 2, 1)
    red = torch.clamp(torch.sum(torch.cat([scan, errors], dim=1), dim=1, keepdim=True), 0, 1)
    green = torch.clamp(torch.sum(torch.cat([scan, tumor], dim=1), dim=1, keepdim=True), 0, 1)
    blue = torch.clamp(torch.sum(torch.cat([scan, liver], dim=1), dim=1, keepdim=True), 0, 1)
    return torch.cat([red, green, blue], dim=1).permute(4, 0, 1, 2, 3)[zs].reshape(24, 3, 64, 64)


def train_net(net, device, case, **opts) -> float:
    scan = torch.narrow(case, 1, 0, 4).to(device=device, dtype=torch.float32)
    segmentations = torch.narrow(case, 1, 4, 3).to(device=device, dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)

    predictions = net(scan)

    with torch.no_grad():
        jaccard = jaccard_distance(predictions, segmentations)
    part_jaccard = jaccard_distance(torch.narrow(predictions, 1, 1, 2), torch.narrow(segmentations, 1, 1, 2))
    pixel = F.mse_loss(predictions, segmentations)
    (part_jaccard + pixel).backward()
    optimizer.step()

    return jaccard.item()







def get_args():
    parser = argparse.ArgumentParser(description='Train the model', argument_default=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float)
    parser.add_argument('--model', type=str, required=True, help='Use model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return dict(defaults, **vars(parser.parse_args()))

def batches(dataset: BufferDataset, n: int):
    """Yield successive n-sized chunks from lst."""
    items = list(dataset.buffer.items())
    for i in range(0, dataset.buffer_size, n):
        indices = [j for j, x in items[i:i + n]]
        xx = torch.cat([x for j, x in items[i:i + n]])
        yield indices, xx

if __name__ == '__main__':
    opts = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f'Using device {device}')

    net = get_model(**opts).to(device=device, dtype=torch.float32)

    shutil.rmtree(Path(opts["outputs"]) / opts["model"], ignore_errors=True)


    def get_slices() -> Iterator[Tensor]:
        return slices882(opts["outputs"], thick=32, step=4)

    def get_scansegm() -> Iterator[Tensor]:
        def _iter_() -> Iterator[Tensor]:
            for (scan, segm) in scan_segm_tuples(opts["outputs"]):
                yield F.avg_pool3d(
                    torch.cat([scan, segm], dim=1),
                    kernel_size=(8, 8, 2)
                )
        return _iter_()

    dataset = BufferDataset(
        tensor_generator_factory=get_scansegm,
        buffer_size=100,
    )

    optimizer = AdaBelief(
        net.parameters(),
        lr=opts["learning_rate"],
        eps=opts["adabelief_eps"],
        betas=(opts["adabelief_b1"], opts["adabelief_b2"]),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )

    global_step = 0
    writer = SummaryWriter(opts["outputs"] / opts["model"])

    # Begin training
    try:
        for epoch in range(opts["epochs"]):
            net.train()
            epoch_loss = 0
            buffer_losses = {}
            with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{opts["epochs"]}', unit='img') as pbar:
                for case in dataset.buffer.values():
                    loss = train_net(net, device, case, **opts)
                    # for i, loss in zip(indices, losses):
                    #     buffer_losses[i] = loss
                    global_step += 1
                    epoch_loss += loss
                    pbar.update(1)
                    pbar.set_postfix(**{'loss (epoch)': epoch_loss})

                writer.add_scalars(
                    "training",
                    {
                        "jaccard": epoch_loss/len(dataset),
                    },
                    global_step=global_step,
                )
                # Evaluation round
                division_step = 10
                if division_step > 0:
                    if epoch % division_step == 0:
                        # writer.add_scalars(
                        #     "evaluation",
                        #     evaluate(net, valid_dl, device),
                        #     global_step=global_step,
                        # )
                        net.eval()
                        writer.add_images(
                            "samples",
                            samples(net, dataset, device),
                            global_step=global_step,
                            dataformats="NCHW"
                        )
                        torch.cuda.empty_cache()

            # smallest = heapq.nsmallest(5, dataset.buffer.keys(), key=lambda i: losses[i])
            # smallest = list(dataset.buffer.keys())[:3]
            # dataset.drop(smallest)
    except KeyboardInterrupt:
        pass
    finally:
        console.print(f"Saving model at {opts['saved_models'] / (opts['model'] + '.pth')}")
        torch.save(net.cpu().state_dict(), opts['saved_models'] / (opts['model'] + '.pth'))
    sys.exit(0)
