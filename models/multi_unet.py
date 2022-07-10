from __future__ import annotations

import math

import numpy as np
import torch
from rich.progress import Progress
from torch import nn, Tensor
from torch.nn import Module, functional

from wrapped_tensors import FloatSegmBatch, FloatScan, Segm, FloatScanBatch, FloatSegm


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
    if norm == "batch":
        return nn.BatchNorm3d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance":
        return nn.InstanceNorm3d(num_features=channels, momentum=momentum, affine=affine)
    return None


def drop_layer(drop: str, drop_prob: float = 0.5) -> Module | None:
    """Return required dropout layer."""
    if drop == "drop":
        return nn.Dropout3d(p=drop_prob, inplace=True)
    return None


def conv_layer(in_channels: int, out_channels: int) -> Module:
    """Return required convolution layer."""
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3, 3),
        padding=1,
    )


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
    """Base convolution block for 3D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            complexity: int = 2,
            actv: str = "relu",
            norm: str = "",
            drop: str = "",
    ):
        super().__init__()
        self.rebuild = f"Block(" \
                       f"in_channels={in_channels}, " \
                       f"out_channels={out_channels}, " \
                       f"complexity={complexity}, " \
                       f"actv={actv!r}, " \
                       f"norm={norm!r}, " \
                       f"drop={drop!r}" \
                       f")"

        self.repr = f"Block({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
                    f"{'>' + actv if actv else ''}" \
                    f"{'>' + norm if norm else ''}" \
                    f"{'>' + drop if drop else ''}" \
                    f")"

        layers = [
                     conv_layer(
                         in_channels=in_channels,
                         out_channels=out_channels
                     ),
                 ] + [
                     actv_layer(actv=actv),
                     conv_layer(
                         in_channels=out_channels,
                         out_channels=out_channels
                     )
                 ] * complexity + [
                     norm_layer(norm=norm, channels=out_channels),
                     drop_layer(drop=drop)
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
            in_channels=channels[0],
            out_channels=channels[1],
            complexity=complexity,
            actv=down_activation if len(channels) > 2 else bottom_activation,
            norm=down_normalization if len(channels) > 2 else bottom_normalization,
            drop=down_dropout if len(channels) > 2 else bottom_dropout,
        )
        if len(channels) > 2:
            self.pool = pool_layer("max222")
            self.submodule = Layer(
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
                in_channels=channels[1] + channels[2],
                out_channels=channels[1],
                complexity=complexity,
                actv=up_activation,
                norm=up_normalization,
                drop=up_dropout
            )

    # def __repr__(self):
    #     return self.repr

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
            nn.Hardsigmoid()
        )

    def forward(self, x: FloatScan | FloatScanBatch) -> FloatSegm | FloatSegmBatch:
        if isinstance(x, FloatScan):
            return FloatSegm(self.model(x.unsqueeze(0)).squeeze(0))
        else:
            return FloatSegmBatch(self.model(x))

    @torch.no_grad()
    def apply(self, x: FloatScan, thickness: int = 8) -> Segm:
        shape = (3, x.size("H"), x.size("W"), x.size("D"))
        base = torch.zeros(shape, device=x.device, dtype=torch.float32)
        size = x.size("D")
        assert x.boundaries() == (0, size)
        assert size >= thickness
        num_slices = math.ceil(size / thickness)

        with Progress(transient=True) as progress:
            task = progress.add_task(
                f"Segmenting scan.",
                total=num_slices
            )
            for j in range(num_slices):
                i = int(j * (size - thickness) / (num_slices - 1))
                slice = torch.narrow(x, -1, i, thickness)
                pred = self.forward(slice)
                base[..., i:i + thickness] += pred
                progress.update(task, advance=1)
        return FloatSegm(base).as_int()
