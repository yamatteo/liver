from __future__ import annotations

import shutil
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief
from torch import nn, Tensor
from torch.nn import Module, functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BufferDataset2 as BufferDataset
from functions.distances import jaccard_distance
from utils.generators import train_bundles


def actv_layer(actv: str, **_) -> Module | None:
    """Return required activation layer."""
    if actv == "relu":
        return nn.ReLU(True)
    if actv == "leaky":
        return nn.LeakyReLU(True)
    if actv == "sigmoid":
        return nn.Sigmoid()
    if actv == "tanh":
        return nn.Tanh()
    return None


def norm_layer(norm: str, channels: int, momentum: float = 0.9, affine: bool = False) -> Module | None:
    """Return required normalization layer."""
    if norm == "batch3d":
        return nn.BatchNorm3d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance3d":
        return nn.InstanceNorm3d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "batch2d":
        return nn.BatchNorm2d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance2d":
        return nn.InstanceNorm2d(num_features=channels, momentum=momentum, affine=affine)
    return None


def drop_layer(drop: str, drop_prob: float = 0.5) -> Module | None:
    """Return required dropout layer."""
    if drop == "drop2d":
        return nn.Dropout2d(p=drop_prob, inplace=True)
    if drop == "drop3d":
        return nn.Dropout3d(p=drop_prob, inplace=True)
    return None


def conv_layer(dims: str, in_channels: int, out_channels: int) -> Module:
    """Return required convolution layer."""
    if dims == "2d":
        layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
        )
    else:
        layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )
    return layer


def pool_layer(pool: str) -> Module | None:
    """Return required pooling layer."""
    if pool == "max22":
        return nn.MaxPool2d(kernel_size=2)
    if pool == "max222":
        return nn.MaxPool3d(kernel_size=2)
    if pool == "avg222":
        return nn.AvgPool3d(kernel_size=(2, 2, 2))
    if pool == "avg441":
        return nn.AvgPool3d(kernel_size=(4, 4, 1))
    return None


class Block(Module):
    """Base convolution block for 2D/3D Unet."""

    def __init__(
            self,
            dims: str,
            in_channels: int,
            out_channels: int,
            complexity: int = 2,
            actv: str = "relu",
            norm: str = "",
            drop: str = "",
    ):
        super().__init__()
        self.repr = f"Block(" \
                    f"dims={dims}, " \
                    f"in_channels={in_channels}, " \
                    f"out_channels={out_channels}, " \
                    f"complexity={complexity}, " \
                    f"actv={actv!r}, " \
                    f"norm={norm!r}, " \
                    f"drop={drop!r}" \
                    f")"

        layers = [
                     conv_layer(
                         dims=dims,
                         in_channels=in_channels,
                         out_channels=out_channels
                     ),
                 ] + [
                     actv_layer(actv=actv),
                     conv_layer(
                         dims=dims,
                         in_channels=in_channels,
                         out_channels=out_channels
                     )
                 ] * complexity + [
                     norm_layer(norm=norm + dims, channels=out_channels),
                     drop_layer(drop=drop + dims)
                 ]
        self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Layer(Module):
    """UNet convolution, down-sampling and skip connection layer."""

    def __init__(
            self,
            dims: str,
            channels: list[int],
            complexity: int = 2,
            down_activation: str = "leaky",
            down_normalization: str = "",
            down_dropout: str = "",
            bottom_activation: str = "relu",
            bottom_normalization: str = "",
            bottom_dropout: str = "",
            up_activation: str = "relu",
            up_normalization: str = "",
            up_dropout: str = "",
    ):
        super().__init__()
        self.repr = f"Layer(" \
                    f"dims={dims}, " \
                    f"channels={channels!r}, " \
                    f"complexity={complexity}, " \
                    f"down_activation={down_activation!r}, " \
                    f"down_normalization={down_normalization!r}, " \
                    f"down_dropout={down_dropout!r}" \
                    f"bottom_activation={bottom_activation!r}, " \
                    f"bottom_normalization={bottom_normalization!r}, " \
                    f"bottom_dropout={bottom_dropout!r}" \
                    f"up_activation={up_activation!r}, " \
                    f"up_normalization={up_normalization!r}, " \
                    f"up_dropout={up_dropout!r}" \
                    f")"
        self.block = Block(
            dims=dims,
            in_channels=channels[0],
            out_channels=channels[1],
            complexity=complexity,
            actv=down_activation if len(channels) > 2 else bottom_activation,
            norm=down_normalization if len(channels) > 2 else bottom_normalization,
            drop=down_dropout if len(channels) > 2 else bottom_dropout,
        )
        if len(channels) > 2:
            self.pool = pool_layer(pool="max22" if dims == "2d" else "max222")
            self.submodule = Layer(
                dims=dims,
                channels=channels[1:],
                complexity=complexity,
                down_activation=down_activation,
                down_normalization=down_normalization,
                down_dropout=down_dropout,
                bottom_activation=bottom_activation,
                bottom_normalization=bottom_normalization,
                bottom_dropout=bottom_dropout,
                up_activation=up_activation,
                up_normalization=up_normalization,
                up_dropout=up_dropout
            )
            self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
            self.upconv = Block(
                dims=dims,
                in_channels=channels[1] + channels[2],
                out_channels=channels[1],
                complexity=complexity,
                actv=up_activation,
                norm=up_normalization,
                drop=up_dropout
            )

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        try:
            z = self.unpool(self.submodule(self.pool(y)))
            zpad = y.size(-1) - z.size(-1)
            z = functional.pad(z, [0, zpad])
            return self.upconv(torch.cat([y, z], dim=1))
        except AttributeError:
            return y


class UNet(Module):
    """Complete UNet, 2d or 3d."""

    def __init__(
            self,
            dims: str,
            channels: list[int],
            final_classes: int = 3,
            complexity: int = 2,
            down_activation: str = "leaky",
            down_normalization: str = "",
            down_dropout: str = "",
            bottom_activation: str = "relu",
            bottom_normalization: str = "",
            bottom_dropout: str = "",
            up_activation: str = "relu",
            up_normalization: str = "",
            up_dropout: str = "",
    ):
        super().__init__()
        self.repr = f"UNet(" \
                    f"dims={dims}, " \
                    f"channels={channels!r}, " \
                    f"final_classes={final_classes}, " \
                    f"complexity={complexity}, " \
                    f"down_activation={down_activation!r}, " \
                    f"down_normalization={down_normalization!r}, " \
                    f"down_dropout={down_dropout!r}" \
                    f"bottom_activation={bottom_activation!r}, " \
                    f"bottom_normalization={bottom_normalization!r}, " \
                    f"bottom_dropout={bottom_dropout!r}" \
                    f"up_activation={up_activation!r}, " \
                    f"up_normalization={up_normalization!r}, " \
                    f"up_dropout={up_dropout!r}" \
                    f")"

        self.model = nn.Sequential(
            Layer(
                dims=dims,
                channels=channels,
                complexity=complexity,
                down_activation=down_activation,
                down_normalization=down_normalization,
                down_dropout=down_dropout,
                bottom_activation=bottom_activation,
                bottom_normalization=bottom_normalization,
                bottom_dropout=bottom_dropout,
                up_activation=up_activation,
                up_normalization=up_normalization,
                up_dropout=up_dropout,
            ),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=final_classes,
                kernel_size=(1, 1, 1),
            ),
            nn.Sigmoid()
        )
        self.dataset = None
        self.optimizer = None
        self.writer = None

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, scan: Tensor, segm: Tensor):
        pred = self.forward(scan)
        jaccard2 = jaccard_distance(
            functional.softmax(pred, dim=1)[:, 1:, :, :, :],
            functional.softmax(segm, dim=1)[:, 1:, :, :, :]
        )
        pixel = functional.l1_loss(
            pred[:, :, :, :, :],
            segm[:, :, :, :, :]
        )
        liver_weight = torch.sum(segm[:, 1])
        tumor_weight = torch.sum(segm[:, 2])
        liver_presence = liver_weight / (liver_weight + 1)
        tumor_presence = tumor_weight / (tumor_weight + 1)
        return (tumor_presence + liver_presence + 1) * pixel + jaccard2

    def train_step(self, scan: Tensor, segm: Tensor, global_step: int):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(scan, segm)

        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar(
            "training_loss",
            loss.item(),
            global_step=global_step,
        )
        return loss.item()

    def train_setup(self, data_path: str | Path, writer_path: str | Path):
        shutil.rmtree(Path(writer_path), ignore_errors=True)
        self.dataset = BufferDataset(
            generator=train_bundles(data_path),
            buffer_size=100,
        )
        self.optimizer = AdaBelief(
            self.parameters(),
            lr=1e-4,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )

        self.writer = SummaryWriter(writer_path)

    def train_teardown(self):
        self.dataset = None
        self.optimizer = None
        self.writer = None

    def train_cycle(self, epochs: int):
        self.train()
        global_step = 0
        for epoch in epochs:
            epoch_loss = 0
            losses = {}
            with tqdm(
                    total=len(self.dataset),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    unit='img'
            ) as pbar:
                for i, (k, (scan, segm)) in enumerate(self.dataset):
                    loss_item = self.train_step(scan, segm, global_step)

                    global_step += 1
                    pbar.update(1)
                    epoch_loss += loss_item
                    losses[i] = loss_item
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

            n = 90 - min(10 * epoch, 80)
            smallest = heapq.nsmallest(n, list(losses.keys()), lambda i: losses[i])
            dataset.drop_by_position(list(smallest))
            # time.sleep(1)