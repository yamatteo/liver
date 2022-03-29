from __future__ import annotations

import argparse
import heapq
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from rich.console import Console
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import Dataset882
from dataset.twostage import BufferThickslice882, BufferCube441
from options import defaults
from segm441.models import UNet3d, get_model
import segm3d882.models3d
from segmentation.dataset import GenericDataset
from segm3d882.utilities import dice_distance, jaccard_distance

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
    case = random.choice(valid_ds)
    k = 8
    scan = case[:, 0:7].to(device=device, dtype=torch.float32)
    segmentations = case[:, 7:10].to(device=device, dtype=torch.float32)
    indices = [(i * (scan.size(-1) - 1)) // (k - 1) for i in range(k)]
    # predict the mask
    predictions = net(scan)

    wafers = torch.clamp(scan[0, 2, :, :, indices].permute(2, 0, 1), 0, 256) / 256  # Only phase v
    # scan = scan[:, 0, :, :, 1]
    prediction = predictions[0, 1, :, :, indices].permute(2, 0, 1)  # Only the liver
    segmentation = segmentations[0, 1, :, :, indices].permute(2, 0, 1)  # Only the liver
    prediction_t = predictions[0, 2, :, :, indices].permute(2, 0, 1)  # Only the tumor
    segmentation_t = segmentations[0, 2, :, :, indices].permute(2, 0, 1)  # Only the tumor
    return F.interpolate(torch.stack([
                                         image[i]
                                         for i in range(k)
                                         for image in [torch.clamp(wafers + prediction, 0, 1), wafers,
                                                       torch.clamp(wafers + segmentation, 0, 1)]
                                     ] + [
                                         image[i]
                                         for i in range(k)
                                         for image in [torch.clamp(wafers + prediction_t, 0, 1), wafers,
                                                       torch.clamp(wafers + segmentation_t, 0, 1)]
                                     ]).reshape(16, 3, 32, 32),
                         size=(128, 128))


def train_net(net, prenet, device, **opts):
    shutil.rmtree(Path(opts["outputs"]) / "last_run", ignore_errors=True)

    with torch.no_grad():
        dataset = BufferCube441(
            base_path=opts["outputs"],
            model882=prenet,
            buffer_size=100,
            batch_size=opts["batch_size"],
        )

    # n_valid = int(len(dataset) * opts["validation_percent"])
    # n_train = len(dataset) - n_valid
    # train_ds, valid_ds = random_split(
    #     dataset,
    #     [n_train, n_valid],
    #     generator=torch.Generator().manual_seed(0)
    # )
    # train_dl = DataLoader(train_ds, shuffle=True, drop_last=False, batch_size=opts["batch_size"],
    #                       num_workers=opts["num_workers"])
    # valid_dl = DataLoader(valid_ds, shuffle=False, drop_last=False, batch_size=opts["batch_size"],
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
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion = nn.CrossEntropyLoss(torch.tensor([1.0, 1.5, 2], device=device))
    global_step = 0

    writer = SummaryWriter(opts["outputs"] / "last_run")

    # Begin training
    for epoch in range(opts["epochs"]):
        net.train()
        epoch_loss = 0
        losses = []
        with tqdm(total=len(dataset)*opts["batch_size"], desc=f'Epoch {epoch + 1}/{opts["epochs"]}', unit='img') as pbar:
            for i, case in enumerate(dataset):
                scan = case[:, 0:7].to(device=device, dtype=torch.float32)
                segmentations = case[:, 7:10].to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)

                predictions = net(scan)

                # loss = jaccard_distance(
                #     predictions,
                #     segmentations,
                #     spatial_dims=[2, 3, 4]
                # )
                loss = F.l1_loss(
                    predictions,
                    segmentations,
                )
                loss.backward()
                optimizer.step()

                pbar.update(opts["batch_size"])
                global_step += 1
                losses.append(loss.item())
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (epoch)': epoch_loss})

                writer.add_scalars(
                    "training",
                    {
                        "jaccard": loss.item(),
                    },
                    global_step=global_step,
                )

                # Evaluation round
                division_step = 100
                if division_step > 0:
                    if global_step % division_step == 0:
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
                        torch.cuda.empty_cache()

        smallest = heapq.nsmallest(10, range(len(dataset)), key=lambda i: losses[i])
        dataset.drop(set(smallest))


def get_args():
    parser = argparse.ArgumentParser(description='Train the model', argument_default=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
    #                     help='Learning rate', dest='lr')
    parser.add_argument('--model', type=str, required=True, help='Use model from a .pth file')
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

    net = get_model(opts["model"]).to(device=device, dtype=torch.float32)
    prenet = segm3d882.models3d.get_model("segm882.0")
    prenet.eval()

    try:
        net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
        prenet.load_state_dict(torch.load(opts["saved_models"] / "segm882.0.pth"))
        console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")
    except FileNotFoundError as err:
        console.print(f"No model at {opts['saved_models'] / (opts['model'] + '.pth')}")
        console.print(f"Using a fresh one.")

    try:
        train_net(net=net, prenet=prenet, device=device, **opts)
    except KeyboardInterrupt:
        pass
    finally:
        console.print(f"Saving model at {opts['saved_models'] / (opts['model'] + '.pth')}")
        torch.save(net.cpu().state_dict(), opts['saved_models'] / (opts['model'] + '.pth'))
    sys.exit(0)
