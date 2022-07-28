from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, functional

from .block import Block


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
        y = self.block(x)
        try:
            z = self.unpool(self.submodule(self.pool(y)))
            zpad = y.size(-1) - z.size(-1)
            z = functional.pad(z, [0, zpad])
            return self.upconv(torch.cat([y, z], dim=1))
        except AttributeError:
            return y
