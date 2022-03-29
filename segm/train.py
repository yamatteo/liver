from __future__ import annotations

import argparse
import heapq
import os
import random
import shutil
import sys
import time
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

import models
from dataset.buffer import BufferDataset
from dataset.generators import scan_segm_tuples
from functions.distances import jaccard_distance
from models import unet3dB
import segm3d882.models3d
import segm441.models
from options import defaults
from models.funet import FunneledUNet

console = Console()
classes = ["background", "liver", "tumor"]


def process_case(net: FunneledUNet, case: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # case is shape (N=1, C=10, H=512, W=512, D=5)
    scan_aid = case[:, 0:7, :, :, :].to(device=device, dtype=torch.float32)

    pred = net(scan_aid).unsqueeze(-1)

    central_slice = case.size(4) // 2
    scan = scan_aid[:, 0:4, :, :, central_slice:central_slice + 1]
    aid = scan_aid[:, 4:7, :, :, central_slice:central_slice + 1]
    segm = case[:, 7:10, :, :, central_slice:central_slice + 1].to(device=device, dtype=torch.float32)

    return scan, pred, aid, segm  # shapes are (N=1, C=4|3|3|3, H=512, W=512, D=1)


def get_white(scan: Tensor, phase: str = "v") -> Tensor:
    if phase == "b":
        return torch.clamp(scan[:, 0:1, :, :, :], 0, 256) / 256
    elif phase == "a":
        return torch.clamp(scan[:, 1:2, :, :, :], 0, 256) / 256
    elif phase == "v":
        return torch.clamp(scan[:, 2:3, :, :, :], 0, 256) / 256
    elif phase == "t":
        return torch.clamp(scan[:, 3:4, :, :, :], 0, 256) / 256


def get_color(scan: Tensor, pred: Tensor, aid: Tensor, segm: Tensor, mode: str = "none"):
    if mode == "error":
        color = torch.sum(torch.abs(segm - pred), dim=1, keepdim=True)
    elif mode == "aid_background":
        color = aid[:, 0:1, :, :, :]
    elif mode == "aid_liver":
        color = aid[:, 1:2, :, :, :]
    elif mode == "aid_tumor":
        color = aid[:, 2:3, :, :, :]
    elif mode == "background":
        color = segm[:, 0:1, :, :, :]
    elif mode == "liver":
        color = segm[:, 1:2, :, :, :]
    elif mode == "tumor":
        color = segm[:, 2:3, :, :, :]
    elif mode == "pred_background":
        color = pred[:, 0:1, :, :, :]
    elif mode == "pred_liver":
        color = pred[:, 1:2, :, :, :]
    elif mode == "pred_tumor":
        color = pred[:, 2:3, :, :, :]
    else:
        color = 0
    return torch.clamp(
        get_white(scan) + color,
        0, 1
    )


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
        wafers = batch['wafer'].to(device=device, dtype=torch.float32)
        segmentations = batch['segmentation'].to(device=device, dtype=torch.long)
        segmentations_one_hot = one_hot(segmentations, len(classes)).float()

        predictions_one_hot = net(wafers)
        predictions = predictions_one_hot.argmax(dim=3)

        metrics["dice"] += dice_distance(
            predictions_one_hot[..., 1:],
            segmentations_one_hot[..., 1:],
            spatial_dims=[1, 2]
        ).item()

        metrics["cross_entropy"] += nn.functional.cross_entropy(
            predictions_one_hot.permute(0, 3, 1, 2),
            segmentations,
        ).item()

        metrics["jaccard"] += jaccard_distance(
            predictions_one_hot,
            segmentations_one_hot,
            spatial_dims=[1, 2]
        ).item()

    net.train()
    try:
        return {
            label: value / num_val_batches
            for label, value in metrics.items()
        }
    except ZeroDivisionError:
        return metrics


def rgb_sample(net, case, mode, z: int | None = None, device=torch.device("cpu")):
    # case is shape (N=1, C=10, H=512, W=512, D=5)
    scan, pred, aid, segm = process_case(net, case, device)
    red_mode, green_mode, blue_mode = mode
    red = get_color(scan, pred, aid, segm, mode=red_mode)
    green = get_color(scan, pred, aid, segm, mode=green_mode)
    blue = get_color(scan, pred, aid, segm, mode=blue_mode)
    rgb = torch.cat([red, green, blue], dim=1)
    if z is None:
        z = random.randint(0, scan.size(4) - 1)
    return rgb[0, :, :, :, z]  # shape is (C=3, H=512, W=512)


@torch.no_grad()
def samples(net, ds, device=torch.device("cpu"), mode=("error", "tumor", "liver"), k=4, indices=None):
    if indices is None:
        indices = random.sample(range(ds.buffer_size), k)
    return torch.stack([rgb_sample(net, ds[i], mode=mode, device=device) for i in indices])


def train_step(case, net, optimizer, device, writer, global_step):
    optimizer.zero_grad(set_to_none=True)
    scan, pred, aid, segm = process_case(net, case, device)

    jaccard1 = jaccard_distance(pred, segm)
    jaccard2 = jaccard_distance(
        F.softmax(pred, dim=1)[:, 1:, :, :, :],
        F.softmax(segm, dim=1)[:, 1:, :, :, :]
    )
    jaccard3 = jaccard_distance(
        pred[:, 2:, :, :, :],
        segm[:, 2:, :, :, :]
    )
    pixel = F.l1_loss(
        pred[:, :, :, :, :],
        segm[:, :, :, :, :]
    )
    liver_weight = torch.sum(segm[:, 1])
    tumor_weight = torch.sum(segm[:, 2])
    liver_presence = liver_weight / (liver_weight+1)
    tumor_presence = tumor_weight / (tumor_weight + 1)
    # loss = (tumor_presence * jaccard3 + liver_presence * jaccard2 + pixel) / (tumor_presence + liver_presence + 1)
    loss = (tumor_presence + liver_presence + 1) * pixel + jaccard2

    loss.backward()
    optimizer.step()
    writer.add_scalars(
        "training",
        {
            "pixel": pixel.item(),
            "jaccard1": jaccard1.item(),
            "jaccard2": jaccard2.item(),
            "jaccard3": jaccard3.item(),
        },
        global_step=global_step,
    )
    return loss.item()


def train_net(net, device, **opts):
    shutil.rmtree(Path(opts["outputs"]) / opts["model"], ignore_errors=True)
    net882 = models.get_model(**dict(opts, model="segm882.7")).to(device=torch.device("cpu"), dtype=torch.float32)
    net882.eval()

    def get_wafer() -> Iterator[Tensor]:
        for (scan, segm) in scan_segm_tuples(opts["outputs"]):
            dgscan = F.avg_pool3d(
                scan,
                kernel_size=(8, 8, 2)
            )

            with torch.no_grad():
                dgpred = net882(dgscan)

            whole = torch.cat([
                scan,
                F.interpolate(dgpred, scan.shape[2:5], mode="trilinear"),
                segm,
            ], dim=1)
            for z in range(1 + scan.size(4) - opts["wafer_size"]):
                yield whole[..., z:z + opts["wafer_size"]]

    with torch.no_grad():
        dataset = BufferDataset(
            tensor_generator=get_wafer(),
            buffer_size=100,
        )
        samples(net, dataset, device)
    # n_valid = int(len(dataset) * opts["validation_percent"])
    # n_train = len(dataset) - n_valid
    # train_ds, valid_ds = random_split(
    #     dataset,
    #     [n_train, n_valid],
    #     generator=torch.Generator().manual_seed(0)
    # )
    # train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, batch_size=opts["batch_size"],
    #                       num_workers=opts["num_workers"])
    # valid_dl = DataLoader(valid_ds, shuffle=False, drop_last=True, batch_size=opts["batch_size"],
    #                       num_workers=opts["num_workers"])

    optimizer = AdaBelief(
        net.parameters(),
        lr=opts["learning_rate"],
        eps=opts["adabelief_eps"],
        betas=(opts["adabelief_b1"], opts["adabelief_b2"]),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )
    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     lr=opts["learning_rate"],
    #     momentum=0.9,
    #     nesterov=True,
    # )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion = nn.CrossEntropyLoss(torch.tensor([1.0, 1.5, 2], device=device))
    global_step = 0

    writer = SummaryWriter(opts["outputs"] / opts["model"])

    # Begin training
    for epoch in range(opts["epochs"]):
        net.train()
        epoch_loss = 0
        losses = {}
        with tqdm(total=len(dataset), desc=f'Epoch {epoch + 1}/{opts["epochs"]}', unit='img') as pbar:
            for i, case in enumerate(dataset):
                loss = train_step(case, net, optimizer, device, writer, global_step)

                global_step += 1
                pbar.update(1)
                epoch_loss += loss
                losses[i] = loss
                pbar.set_postfix(**{'loss (batch)': epoch_loss})

                # Evaluation round
                if global_step % 100 == 0:
                    # writer.add_scalars(
                    #     "evaluation",
                    #     evaluate(net, valid_dl, device),
                    #     global_step=global_step,
                    # )
                    writer.add_images(
                        "samples",
                        samples(net, dataset, device),
                        global_step=global_step,
                        dataformats="NCHW"
                    )


        writer.add_scalars(
            "training2d",
            {
                "jaccard": epoch_loss,
            },
            global_step=global_step,
        )

        n = 90 - min(10 * epoch, 80)
        smallest = heapq.nsmallest(n, list(losses.keys()), lambda i: losses[i])
        dataset.drop_by_position(list(smallest))
        # time.sleep(1)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model', argument_default=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--model', type=str, default=None, help='Use model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    # parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    # parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return dict(defaults, **vars(parser.parse_args()))


if __name__ == '__main__':
    opts = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f'Using device {device}')

    net = models.get_model(**opts).to(device=device, dtype=torch.float32)

    # summary(net, (4, 512, 512, 5), opts["batch_size"], device=str(device))

    if opts["model"]:
        try:
            net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
            console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")
        except FileNotFoundError as err:
            console.print(f"No model at {opts['saved_models'] / (opts['model'] + '.pth')}")
            console.print(f"Using a fresh one.")
            # for t in net.parameters():
            #     if t.dim() == 4 and t.size(3) == 3:
            #         torch.nn.init.normal_(t, mean=0.0, std=0.001)
            #         torch.nn.init.eye_(t[:, :, 1, 1])
            #     elif t.dim() == 4 and t.size(3) == 1:
            #         # torch.nn.init.normal_(t, mean=0.0, std=0.001)
            #         torch.nn.init.eye_(t[:, :, 0, 0])
            #     else:
            #         torch.nn.init.normal_(t, mean=0.0, std=0.001)
            # w = torch.zeros(16, 7, 5, 5, 5)
            # w[0, 4, 2, 2, 2] = 1
            # w[1, 5, 2, 2, 2] = 1
            # w[2, 6, 2, 2, 2] = 1
            # net.funnel.conv.weight = torch.nn.Parameter(w)
            #
            # w = torch.zeros(3, 32, 1, 1)
            # w[0, 0, 0, 0] = 1
            # w[1, 1, 0, 0] = 1
            # w[2, 2, 0, 0] = 1
            # net.head[0].weight = torch.nn.Parameter(w)
            # net.to(device=device, dtype=torch.float32)

    try:
        train_net(net=net, device=device, **opts)
    except KeyboardInterrupt:
        pass
    finally:
        if opts["model"]:
            console.print(f"Saving model at {opts['saved_models'] / (opts['model'] + '.pth')}")
            torch.save(net.cpu().state_dict(), opts['saved_models'] / (opts['model'] + '.pth'))
    sys.exit(0)
