from __future__ import annotations

import argparse
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

from options import defaults
from .dataset import GenericDataset
from .models import FunneledUNet
from .utilities import dice_distance, jaccard_distance, mj_distance

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


@torch.no_grad()
def samples(net, valid_ds, device):
    k = 8
    indices = random.sample(range(len(valid_ds)), k)
    wafers = torch.stack([
        valid_ds[i]["wafer"]
        for i in indices
    ]).to(device=device, dtype=torch.float32)
    segmentation = torch.stack([
        valid_ds[i]["segmentation"]
        for i in indices
    ]).to(device=device, dtype=torch.long)
    # predict the mask
    prediction = net(wafers).argmax(dim=3)
    wafers = torch.clamp(wafers[:, 2, :, :, 2], 0, 256) / 256
    # scan = scan[:, 0, :, :, 1]
    prediction = prediction / 2
    segmentation = segmentation / 2
    return torch.stack([
        image[i]
        for image in [wafers, prediction, segmentation]
        for i in range(k)
    ]).unsqueeze(1)


def train_net(net, device, **opts):
    shutil.rmtree(Path(opts["outputs"]) / "last_run", ignore_errors=True)

    dataset = GenericDataset(
        data_path=opts["outputs"],
        segmented=True,
        wafer=5,
        background_reduction=0.1,
    )

    n_valid = int(len(dataset) * opts["validation_percent"])
    n_train = len(dataset) - n_valid
    train_ds, valid_ds = random_split(
        dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(0)
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, batch_size=opts["batch_size"],
                          num_workers=opts["num_workers"])
    valid_dl = DataLoader(valid_ds, shuffle=False, drop_last=True, batch_size=opts["batch_size"],
                          num_workers=opts["num_workers"])

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
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{opts["epochs"]}', unit='img') as pbar:
            for batch in train_dl:
                wafers = batch['wafer'].to(device=device, dtype=torch.float32)
                segmentations = batch['segmentation'].to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                predictions = net(wafers)

                loss = mj_distance(
                    predictions.float(),
                    one_hot(segmentations, len(classes)).float(),
                    spatial_dims=[1, 2]
                )
                loss.backward()
                optimizer.step()

                pbar.update(wafers.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                writer.add_scalars(
                    "training",
                    {
                        "jaccard": loss.item(),
                    },
                    global_step=global_step,
                )

                # Evaluation round
                division_step = (n_train // (10 * opts["batch_size"]))
                if division_step > 0:
                    if global_step % division_step == 0:
                        writer.add_scalars(
                            "evaluation",
                            evaluate(net, valid_dl, device),
                            global_step=global_step,
                        )
                        writer.add_images(
                            "samples",
                            samples(net, valid_ds, device),
                            global_step=global_step,
                            dataformats="NCHW"
                        )
                        torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser(description='Train the model', argument_default=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    # parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
    #                     help='Learning rate', dest='lr')
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

    net = FunneledUNet(
        input_channels=4,
        wafer_size=5,
        internal_channels=[20, 24, 28, 32],
        classes=3,
    ).to(device=device, dtype=torch.float32)

    # summary(net, (4, 512, 512, 5), opts["batch_size"], device=str(device))

    if opts["model"]:
        try:
            net.load_state_dict(torch.load(opts["saved_models"] / (opts['model'] + '.pth')))
            console.print(f"Model loaded from {opts['saved_models'] / (opts['model'] + '.pth')}")
        except FileNotFoundError as err:
            console.print(f"No model at {opts['saved_models'] / (opts['model'] + '.pth')}")
            console.print(f"Using a fresh one.")

    try:
        train_net(net=net, device=device, **opts)
    except KeyboardInterrupt:
        pass
    finally:
        if opts["model"]:
            console.print(f"Saving model at {opts['saved_models'] / (opts['model'] + '.pth')}")
            torch.save(net.cpu().state_dict(), opts['saved_models'] / (opts['model'] + '.pth'))
    sys.exit(0)
