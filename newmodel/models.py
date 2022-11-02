from __future__ import annotations

import math

import numpy as np
import torch
from rich.progress import Progress
from torch import nn, Tensor
from torch.nn import Module, functional

from utils.debug import dbg, unique_debug


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
        return nn.MaxPool2d(kernel_size=(2, 2))
    if pool == "max222":
        return nn.MaxPool3d(kernel_size=(2, 2, 2))
    if pool == "max221":
        return nn.MaxPool3d(kernel_size=(2, 2, 1))
    if pool == "avg222":
        return nn.AvgPool3d(kernel_size=(2, 2, 2))
    if pool == "avg441":
        return nn.AvgPool3d(kernel_size=(4, 4, 1))
    return None


def unpool_layer(pool: str) -> Module | None:
    """Return required upsampling layer."""
    if pool == "max22":
        return nn.Upsample(scale_factor=(2, 2), mode='nearest')
    if pool == "max222":
        return nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
    if pool == "max221":
        return nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')
    if pool == "avg222":
        return nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
    if pool == "avg441":
        return nn.Upsample(scale_factor=(4, 4, 1), mode='trilinear')
    return None


class Block(Module):
    """Base convolution block for 3D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            complexity: int = 2,
            actv: str = "relu",
            norm: str = "",
            drop: str = "",
    ):
        super().__init__()
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
            pool: str = "max222"
    ):
        super().__init__()
        self.level = len(channels) - 2
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
            self.pool = pool_layer(pool)
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
                up_dropout=up_dropout,
                pool=pool,
            )
            self.unpool = unpool_layer(pool)
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
        x = self.block(x)
        try:
            original_z_size = x.size(-1)
            y = self.pool(x)
            y = self.submodule(y)
            y = self.unpool(y)
            z_pad = original_z_size - y.size(-1)
            y = functional.pad(y, [0, z_pad])
            y = torch.cat([y, x], dim=1)
            y = self.upconv(y)
            return y
        except AttributeError:
            return x
        # y = self.block(x)
        # try:
        #     z = self.unpool(self.submodule(self.pool(y)))
        #     zpad = y.size(-1) - z.size(-1)
        #     z = functional.pad(z, [0, zpad])
        #     return self.upconv(torch.cat([y, z], dim=1))
        # except AttributeError:
        #     return y


class UNet(Module):
    def __init__(self, channels: list[int], down_normalization="", up_dropout="", **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            Layer(channels=channels, down_normalization=down_normalization, up_dropout=up_dropout),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=3,
                kernel_size=(1, 1, 1),
            ),
        )

    def set_momentum(self, momentum: float):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                module.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
