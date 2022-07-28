from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint_sequential


class Block3d(Module):
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
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1),
        ]
        for _ in range(complexity):
            layers.append(activation)
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=1))
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


class UNet3dLayer(nn.Module):
    def __init__(
            self,
            channels: list[int],
            complexity: int,
            down_activation: Module = nn.LeakyReLU(True),
            down_normalization: Module = None,
            down_dropout: Module = nn.Dropout3d(0.5, True),
            bottom_activation: Module = nn.ReLU(True),
            bottom_normalization: Callable[[int], Module] = lambda n: nn.BatchNorm3d(n, momentum=0.9, affine=False),
            bottom_dropout: Module = None,
            up_activation: Module = nn.ReLU(True),
            up_normalization: Module = None,
            up_dropout: Module = None,
            checkpointing: bool = False,
    ):
        super().__init__()
        self.block = Block3d(
            channels[0],
            channels[1],
            complexity=complexity,
            activation=down_activation if len(channels) > 2 else bottom_activation,
            normalization=down_normalization if len(channels) > 2 else bottom_normalization,
            dropout=down_dropout if len(channels) > 2 else bottom_dropout,
            checkpointing=checkpointing,
        )
        if len(channels) > 2:
            self.pool = nn.MaxPool3d(2)
            self.submodule = UNet3dLayer(
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
            self.upconv = Block3d(
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
                zpad = y.size(4) - z.size(4)
                z = functional.pad(z, [0, zpad])
                return self.upconv(torch.cat([y, z], dim=1))
        else:
            def forward(x):
                return self.block(x)
        self.forward = forward

    def forward(self, x):
        ...


class UNet3d(nn.Module):
    def __init__(
            self,
            channels: list[int],
            final_classes: int,
            complexity: int = 1,
            down_activation: Module | None = nn.LeakyReLU(True),
            down_normalization: Module | None = None,
            down_dropout: Module | None = nn.Dropout3d(0.5, True),
            bottom_activation: Module | None = nn.ReLU(True),
            bottom_normalization: Callable[[int], Module] | None = lambda n: nn.BatchNorm3d(n, momentum=0.9,
                                                                                            affine=False),
            bottom_dropout: Module | None = None,
            up_activation: Module | None = nn.ReLU(True),
            up_normalization: Module | None = None,
            up_dropout: Module | None = None,
            checkpointing: bool = False,
    ):
        super().__init__()

        self.model = nn.Sequential(
            UNet3dLayer(
                channels,
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
            ),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=final_classes,
                kernel_size=(1, 1, 1),
            ),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
