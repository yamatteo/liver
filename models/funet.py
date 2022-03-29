from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.nn.functional as F


class Funnel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bypass: list[int]):
        super().__init__()
        lpad, rpad = (kernel_size - 1) // 2, kernel_size // 2
        self.pad = nn.ConstantPad3d((0, 0, lpad, rpad, lpad, rpad), 0.0)
        self.conv = nn.Conv3d(
            in_channels=in_channels - len(bypass),
            out_channels=out_channels - len(bypass),
            kernel_size=(kernel_size,) * 3,
        )
        self.in_channels = in_channels
        self.bypass = bypass
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        processed_channels = [i for i in range(self.in_channels) if i not in self.bypass]

        y1 = self.conv(self.pad(x[:, processed_channels])).squeeze(-1)
        y2 = x[:, self.bypass, :, :, self.kernel_size // 2]
        return torch.cat([y1, y2], dim=1)

class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            complexity: int,
            activation: Module = nn.ReLU(True),
            normalization: Callable[[int], Module] = None,
            dropout: Module = None,
            checkpointing: bool = False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
        ]
        for _ in range(complexity):
            layers.append(activation)
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        if normalization:
            layers.append(normalization(out_channels))
        if dropout:
            layers.append(dropout)
        self.model = nn.Sequential(*layers)

        if checkpointing:
            def forward(x):
                if self.training:
                    return checkpoint_sequential(self.model, complexity, x)
                else:
                    return self.model(x)
        else:
            def forward(x):
                return self.model(x)
        self.forward = forward

    def forward(self, x):
        ...


class UNetLayer(nn.Module):
    def __init__(
            self,
            channels: list[int],
            complexity: int,
            down_activation: Module = nn.LeakyReLU(True),
            down_normalization: Module = None,
            down_dropout: Module = nn.Dropout2d(0.5, True),
            bottom_activation: Module = nn.ReLU(True),
            bottom_normalization: Callable[[int], Module] = lambda n: nn.BatchNorm2d(n, momentum=0.9, affine=False),
            bottom_dropout: Module = None,
            up_activation: Module = nn.ReLU(True),
            up_normalization: Module = None,
            up_dropout: Module = None,
            checkpointing: bool = False,
    ):
        super().__init__()
        self.block = Block(
            channels[0],
            channels[1],
            complexity=complexity,
            activation=down_activation if len(channels) > 2 else bottom_activation,
            normalization=down_normalization if len(channels) > 2 else bottom_normalization,
            dropout=down_dropout if len(channels) > 2 else bottom_dropout,
            checkpointing=checkpointing,
        )
        if len(channels) > 2:
            self.pool = nn.MaxPool2d(2)
            self.submodule = UNetLayer(
                channels[1:],
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
                checkpointing=checkpointing,
            )
            self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
            self.upconv = Block(
                channels[1] + channels[2],
                channels[1],
                complexity=complexity,
                activation=up_activation,
                normalization=up_normalization,
                dropout=up_dropout,
                checkpointing=checkpointing,
            )

        if len(channels) > 2:
            def forward(x):
                y = self.block(x)
                z = self.unpool(self.submodule(self.pool(y)))
                zpad = y.size(-1) - z.size(-1)
                z = F.pad(z, [0, zpad])
                return self.upconv(torch.cat([y, z], dim=1))
        else:
            def forward(x):
                return self.block(x)
        self.forward = forward

    def forward(self, x):
        ...


class FunneledUNet(nn.Module):
    def __init__(
            self,
            channels: list[int],
            final_classes: int,
            complexity: int = 1,
            down_activation: Module | None = nn.LeakyReLU(True),
            down_normalization: Module | None = None,
            down_dropout: Module | None = None,
            bottom_activation: Module | None = nn.ReLU(True),
            bottom_normalization: Callable[[int], Module] | None = None,
            bottom_dropout: Module | None = None,
            up_activation: Module | None = nn.ReLU(True),
            up_normalization: Module | None = None,
            up_dropout: Module | None = None,
            checkpointing: bool = False,
            wafer_size: int = 5,
            bypass: list[int] | None = None,
            fullbypass: list[int] | None = [],
            final_activation: Module | None = nn.Softmax(dim=1),
            clamp: tuple[int, int] | None = None
    ):
        super().__init__()
        self.wafer_size = wafer_size
        self.fullbypass = fullbypass
        self.clamp = clamp

        self.funnel = Funnel(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=wafer_size,
            bypass=[] if bypass is None else bypass,
        )

        self.body = UNetLayer(
                channels[1:],
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
                checkpointing=checkpointing,
            )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[2]+len(fullbypass),
                out_channels=final_classes,
                kernel_size=(1, 1),
            ),
            final_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.clamp:
            x = torch.clamp(x, *self.clamp)
        bx = x[:, self.fullbypass, :, :, self.wafer_size//2]
        x = self.funnel(x)
        x = self.body(x)
        return self.head(torch.cat([x, bx], dim=1))
