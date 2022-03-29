from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class Block3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm3d(num_features=out_channels, momentum=0.9)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = checkpoint(self.conv2, y)
        return self.norm(y)


class UNet3dLayer(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.block = Block3d(channels[0], channels[1])
        if len(channels) > 2:
            self.pool = nn.MaxPool3d(2)
            self.submodule = UNet3dLayer(channels[1:])
            self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
            self.upconv = Block3d(channels[1] + channels[2], channels[1])

    def forward(self, x):
        y = self.block(x)
        try:
            z = self.unpool(self.submodule(self.pool(y)))
            zpad = y.size(4) - z.size(4)
            z = func.pad(z, [0, zpad])
            return self.upconv(torch.cat([y, z], dim=1))
        except AttributeError:
            return y


class UNet3d(nn.Module):
    def __init__(self, channels: list[int], final_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            UNet3dLayer(channels),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=final_classes,
                kernel_size=(1, 1, 1),
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)



def get_model(name: str) -> nn.Module:
    if name == "segm882.0":
        return UNet3d(
            channels=[4, 16, 20],
            final_classes=3,
        )