from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import Module, functional

from utils.debug import unique_debug, dbg
from .models import Layer as Layer3d


def actv_layer(actv: str) -> Optional[Module]:
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


def norm_layer(norm: str, channels: int, momentum: float, affine: bool) -> Optional[Module]:
    """Return required normalization layer."""
    if norm == "batch":
        return nn.BatchNorm2d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance":
        return nn.InstanceNorm2d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "batch3d":
        return nn.BatchNorm3d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance3d":
        return nn.InstanceNorm3d(num_features=channels, momentum=momentum, affine=affine)
    return None


def drop_layer(drop: str, drop_prob: float = 0.5) -> Optional[Module]:
    """Return required dropout layer."""
    if drop == "drop":
        return nn.Dropout2d(p=drop_prob, inplace=True)
    return None


def conv_layer(in_channels: int, out_channels: int) -> Module:
    """Return required convolution layer."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        padding=1,
    )


def pool_layer(pool: str) -> Optional[Module]:
    """Return required pooling layer."""
    if pool == "max22":
        return nn.MaxPool2d(kernel_size=(2, 2))
    if pool == "avg22":
        return nn.AvgPool2d(kernel_size=(2, 2))
    if pool == "avg44":
        return nn.AvgPool2d(kernel_size=(4, 4))
    return None


def unpool_layer(pool: str) -> Optional[Module]:
    """Return required upsampling layer."""
    if pool == "max22":
        return nn.Upsample(scale_factor=(2, 2), mode='nearest')
    if pool == "avg22":
        return nn.Upsample(scale_factor=(2, 2), mode='bilinear')
    if pool == "avg44":
        return nn.Upsample(scale_factor=(4, 4), mode='bilinear')
    return None


class Funnel(Module):
    """Initial convolution block for a 3D -> 2D UNet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            complexity: int,
            funnel_size: int,
    ):
        super(Funnel, self).__init__()
        self.repr = f"Funnel({'>'.join([str(in_channels), *([str(out_channels)] * complexity), str(out_channels)])}" \
                    f"{'>' + 'leaky'}" \
                    f"{'>' + 'instance'}" \
                    f")"

        # noinspection PyTypeChecker
        layers = [
                     nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=1)
                 ] + [
                     nn.LeakyReLU(True),
                     nn.Conv3d(out_channels, out_channels, (3, 3, 3), padding=1)
                 ] * complexity + [
                     nn.LeakyReLU(True),
                     nn.Conv3d(out_channels, out_channels, (3, 3, funnel_size), padding=(1, 1, funnel_size))
                 ] + [
                     nn.InstanceNorm3d(out_channels, momentum=0.9),
                 ]
        self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        # shape goes from [N, inC, X, Y, Z] to [N, outC, X, Y]
        return self.model(x).squeeze(-1)


class Block(Module):
    """Base convolution block for 2D Unet."""

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
                     norm_layer(norm=norm, channels=out_channels, affine=False, momentum=0.9),
                     drop_layer(drop=drop)
                 ]
        self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Layer(Module):
    """UNet2D convolution, down-sampling and skip connection layer."""

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
            pool: str = "max22",
            funnel_size: int = 0,
    ):
        super().__init__()
        self.level = len(channels) - 2
        if funnel_size != 0:
            self.block = Funnel(
                in_channels=channels[0],
                out_channels=channels[1],
                complexity=complexity - 1,
                funnel_size=funnel_size,
            )
        else:
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

    def forward(self, x: Tensor) -> Tensor:
        with unique_debug(f"layer {self.level}"):
            dbg(x.shape)
            x = self.block(x)
            dbg("block", x.shape)
            try:
                original_z_size = x.size(-1)
                y = self.pool(x)
                dbg("pool", y.shape)
                y = self.submodule(y)
                dbg("submodule", y.shape)
                y = self.unpool(y)
                dbg("unpool", y.shape)
                z_pad = original_z_size - y.size(-1)
                y = functional.pad(y, [0, z_pad])
                dbg("pad", y.shape)
                y = torch.cat([y, x], dim=1)
                dbg("skip connection", y.shape)
                y = self.upconv(y)
                dbg("upconv", y.shape)
                return y
            except AttributeError:
                return x


class FUNet(Module):
    def __init__(
            self,
            funnel_channels: list[int],
            funnel_size: int,
            downsampled_channels: list[int],
            down_normalization="",
            up_dropout="",
            **kwargs
    ):
        super().__init__()

        self.funnel_model = nn.Sequential(
            Layer(
                channels=funnel_channels,
                down_normalization=down_normalization,
                up_dropout=up_dropout,
                funnel_size=funnel_size,
            ),
            nn.Conv2d(
                in_channels=funnel_channels[1],
                out_channels=3,
                kernel_size=(1, 1),
            ),
        )
        self.downsampled_model = Layer3d(
            channels=downsampled_channels,
            down_normalization=down_normalization,
            up_dropout=up_dropout,
        )
        self.downsampled_early_exit = nn.Conv3d(
            in_channels=downsampled_channels[1],
            out_channels=3,
            kernel_size=(1, 1, 1),
        )

    def set_momentum(self, momentum: float):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
                module.momentum = momentum

    def dx_forward(self, x: Tensor) -> Tensor:
        x = self.downsampled_model(x)
        x = self.downsampled_early_exit(x)
        return x
