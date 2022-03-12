from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class Funnel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        lpad, rpad = (kernel_size - 1) // 2, kernel_size // 2
        self.pad = nn.ConstantPad3d((0, 0, lpad, rpad, lpad, rpad), 0.0)
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,) * 3,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x)).squeeze(-1)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = checkpoint(self.conv2, y)
        return self.norm(y)


class UNetLayer(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.block = Block(channels[0], channels[1])
        if len(channels) > 2:
            self.pool = nn.MaxPool2d(2)
            self.submodule = UNetLayer(channels[1:])
            self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
            self.upconv = Block(channels[1] + channels[2], channels[1])

    def forward(self, x):
        y = self.block(x)
        try:
            z = self.unpool(self.submodule(self.pool(y)))
            return self.upconv(torch.cat([y, z], dim=1))
        except AttributeError:
            return y


class FunneledUNet(nn.Module):
    def __init__(self, input_channels: int, wafer_size: int, internal_channels: list[int], classes: int):
        super().__init__()
        self.wafer_size = wafer_size

        self.funnel = Funnel(
            in_channels=input_channels,
            out_channels=internal_channels[0],
            kernel_size=wafer_size,
        )

        self.body = UNetLayer(internal_channels)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=internal_channels[1],
                out_channels=classes,
                kernel_size=(1, 1),
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.funnel(x)
        x = self.body(x)
        return self.head(x).permute(0, 2, 3, 1)
