from __future__ import annotations

import torch

from torch import nn
from torch import Tensor
from torch.nn import functional



## Single layers

def actv_layer(actv: str) -> nn.Module | None:
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


def drop_layer(drop: str, drop_prob: float = 0.5) -> nn.Module | None:
    """Return required dropout layer."""
    if drop == "drop" or drop == "drop2d":
        return nn.Dropout2d(p=drop_prob, inplace=True)
    if drop == "drop3d":
        return nn.Dropout3d(p=drop_prob, inplace=True)
    return None


def norm_layer(norm: str, channels: int, momentum: float = 0.9, affine: bool = False) -> nn.Module | None:
    """Return required normalization layer."""
    if norm == "batch" or norm == "batch2d":
        return nn.BatchNorm2d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance" or norm == "instance2d":
        return nn.InstanceNorm2d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "batch3d":
        return nn.BatchNorm3d(num_features=channels, momentum=momentum, affine=affine)
    if norm == "instance3d":
        return nn.InstanceNorm3d(num_features=channels, momentum=momentum, affine=affine)
    return None


def pool_layer(pool: str) -> nn.Module | None:
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


def unpool_layer(pool: str) -> nn.Module | None:
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


## Convolution blocks

class SeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super(SeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=3, padding=1,
                                   padding_mode="reflect", groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeparableConvolution3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1, kernel: tuple[int, int, int] = (3, 3, 3)):
        super(SeparableConvolution3d, self).__init__()
        padding = tuple(1 if k == 3 else 0 for k in kernel)

        self.depthwise = nn.Conv3d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel, padding=padding,
                                   padding_mode="reflect", groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SCBlock(nn.Module):
    """Base separable convolution block for 2D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernels_per_layer: int = 1,
            complexity: int = 2,
            actv: str = "relu",
            norm: str = "instance",
            drop: str = "drop",
    ):
        super().__init__()
        self.repr = f"SCBlock({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
                    f"{'>' + actv if actv else ''}" \
                    f"{'>' + norm if norm else ''}" \
                    f"{'>' + drop if drop else ''}" \
                    f")"

        layers = [
                     SeparableConvolution(
                         in_channels=in_channels,
                         out_channels=out_channels,
                     ),
                 ] + [
                     actv_layer(actv=actv),
                     SeparableConvolution(
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


class SCBlock3d(nn.Module):
    """Base separable convolution block for 3D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel: tuple[int, int, int] = (3, 3, 3),
            kernels_per_layer: int = 1,
            complexity: int = 2,
            actv: str = "relu",
            norm: str = "instance3d",
            drop: str = "drop3d",
    ):
        super().__init__()
        self.repr = f"SCBlock[{''.join(map(str, kernel))}]" \
                    f"({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
                    f"{'>' + actv if actv else ''}" \
                    f"{'>' + norm if norm else ''}" \
                    f"{'>' + drop if drop else ''}" \
                    f")"

        layers = [
                     SeparableConvolution3d(
                         in_channels=in_channels,
                         out_channels=out_channels,
                         kernel=kernel,
                         kernels_per_layer=kernels_per_layer,
                     ),
                 ] + [
                     actv_layer(actv=actv),
                     SeparableConvolution3d(
                         in_channels=out_channels,
                         out_channels=out_channels,
                         kernel=kernel,
                         kernels_per_layer=kernels_per_layer,
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


## UNet


class Unet3d(nn.Module):
    """UNet convolution, down-sampling and skip connection layer."""

    def __init__(
            self,
            channels: list[int],
            conv_kernels: list[tuple[int, int, int]] = None,
            pool_layers: list[str] = None,
            outermost: bool = True,
            complexity: int = 2,
            down_activation: str = "leaky",
            down_normalization: str = "instance3d",
            down_dropout: str = "drop3d",
            bottom_activation: str = "relu",
            bottom_normalization: str = "instance3d",
            bottom_dropout: str = "drop3d",
            up_activation: str = "relu",
            up_normalization: str = "instance3d",
            up_dropout: str = "drop3d",
            split: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.split = split
        self.level = level = len(channels) - 2
        if conv_kernels is None:
            conv_kernels = [(3, 3, 1) if i < 2 else (3, 3, 3) for i, _ in enumerate(channels[1:])]
        if pool_layers is None:
            pool_layers = ["max221" if i < 2 else "max222" for i, _ in enumerate(channels[2:])]
        self.block = SCBlock3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel=conv_kernels[0],
            complexity=complexity,
            actv=down_activation if level > 0 else bottom_activation,
            norm=down_normalization if level > 0 else bottom_normalization,
            drop=down_dropout if level > 0 else bottom_dropout,
        )
        if level > 0:
            self.pool = pool_layer(pool_layers[0])
            self.submodule = Unet3d(
                channels=channels[1:],
                conv_kernels=conv_kernels[1:],
                pool_layers=pool_layers[1:],
                outermost=False,
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
            )
            self.unpool = unpool_layer(pool_layers[0])
            self.upconv = SCBlock3d(
                in_channels=channels[1] + channels[2],
                out_channels=channels[1],
                kernel=conv_kernels[0],
                complexity=complexity,
                actv=up_activation,
                norm=up_normalization,
                drop=up_dropout
            )
        if outermost:
            self.final = nn.Conv3d(
                in_channels=channels[1],
                out_channels=3,
                kernel_size=(1, 1, 1),
            )

    def set_momentum(self, momentum: float):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
                module.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        # print(f"DEBUG: inside level {self.level} forward(x{x.shape})")
        x = self.block(x)
        if self.level > 0:
            original_z_size = x.size(-1)
            y = self.pool(x)
            y = self.submodule(y)
            y = self.unpool(y)
            z_pad = original_z_size - y.size(-1)
            y = functional.pad(y, [0, z_pad])
            y = torch.cat([y, x], dim=1)
            x = self.upconv(y)
            del y
        try:
            x = self.final(x)
        except AttributeError:
            pass
        return x

#
# class UnetLayer(nn.Module):
#     """UNet convolution, down-sampling and skip connection layer."""
#
#     def __init__(
#             self,
#             channels: list[int],
#             complexity: int = 2,
#             down_activation: str = "leaky",
#             down_normalization: str = "instance",
#             down_dropout: str = "drop",
#             bottom_activation: str = "relu",
#             bottom_normalization: str = "instance",
#             bottom_dropout: str = "drop",
#             up_activation: str = "relu",
#             up_normalization: str = "instance",
#             up_dropout: str = "drop",
#             pool: str = "max22",
#     ):
#         super().__init__()
#         self.level = level = len(channels) - 2
#         self.block = SCBlock(
#             in_channels=channels[0],
#             out_channels=channels[1],
#             complexity=complexity,
#             actv=down_activation if level > 0 else bottom_activation,
#             norm=down_normalization if level > 0 else bottom_normalization,
#             drop=down_dropout if level > 0 else bottom_dropout,
#         )
#         if level > 0:
#             self.pool = pool_layer(pool)
#             self.submodule = UnetLayer(
#                 channels=channels[1:],
#                 complexity=complexity,
#                 down_activation=down_activation,
#                 down_normalization=down_normalization,
#                 down_dropout=down_dropout,
#                 bottom_activation=bottom_activation,
#                 bottom_normalization=bottom_normalization,
#                 bottom_dropout=bottom_dropout,
#                 up_activation=up_activation,
#                 up_normalization=up_normalization,
#                 up_dropout=up_dropout,
#                 pool=pool,
#             )
#             self.unpool = unpool_layer(pool)
#             self.upconv = SCBlock(
#                 in_channels=channels[1] + channels[2],
#                 out_channels=channels[1],
#                 complexity=complexity,
#                 actv=up_activation,
#                 norm=up_normalization,
#                 drop=up_dropout
#             )
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.block(x)
#         if self.level > 0:
#             original_z_size = x.size(-1)
#             y = self.pool(x)
#             y = self.submodule(y)
#             y = self.unpool(y)
#             z_pad = original_z_size - y.size(-1)
#             y = functional.pad(y, [0, z_pad])
#             y = torch.cat([y, x], dim=1)
#             y = self.upconv(y)
#             return y
#         else:
#             return x
#
#
# class UNet(nn.Module):
#     def __init__(
#             self,
#             channels: list[int],
#             complexity: int = 2,
#             down_activation: str = "leaky",
#             down_normalization: str = "instance",
#             down_dropout: str = "drop",
#             bottom_activation: str = "relu",
#             bottom_normalization: str = "instance",
#             bottom_dropout: str = "drop",
#             up_activation: str = "relu",
#             up_normalization: str = "instance",
#             up_dropout: str = "drop",
#             pool: str = "max22",
#     ):
#         super().__init__()
#
#         self.model = nn.Sequential(
#             UnetLayer(
#                 channels=channels,
#                 complexity=complexity,
#                 down_activation=down_activation,
#                 down_normalization=down_normalization,
#                 down_dropout=down_dropout,
#                 bottom_activation=bottom_activation,
#                 bottom_normalization=bottom_normalization,
#                 bottom_dropout=bottom_dropout,
#                 up_activation=up_activation,
#                 up_normalization=up_normalization,
#                 up_dropout=up_dropout,
#                 pool=pool,
#             ),
#             nn.Conv2d(
#                 in_channels=channels[1],
#                 out_channels=3,
#                 kernel_size=(1, 1),
#             ),
#         )
#
#     def set_momentum(self, momentum: float):
#         for module in self.modules():
#             if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
#                 module.momentum = momentum
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)