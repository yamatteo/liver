from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import Tensor
from torch.utils.checkpoint import checkpoint
# from batchrenorm import BatchRenorm3d

class Block3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, complexity: int, final, checkpointing: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.inner = nn.Sequential(
            *([
                  nn.ReLU(),
                  nn.Conv3d(out_channels, out_channels, 3, padding=1)
              ] * complexity)
        )
        if final == nn.BatchNorm3d:  # or final == BatchRenorm3d:
            self.final = final(num_features=out_channels, momentum=0.5, affine=False)
        elif final == nn.Dropout3d:
            self.final = final(p=0.5)
        else:
            self.final = nn.Identity()

        if checkpointing:
            def _forward(self, x):
                y = self.conv1(x)
                y = checkpoint(self.inner, y)
                return self.final(y)

            self.forward = _forward

    def forward(self, x):
        return self.final(self.inner(self.conv1(x)))


class UNet3dLayer(nn.Module):
    def __init__(self, channels: list[int], complexity: int, norm):
        super().__init__()
        self.block = Block3d(channels[0], channels[1], complexity=complexity, final=norm)
        if len(channels) > 2:
            self.pool = nn.MaxPool3d(2)
            self.submodule = UNet3dLayer(channels[1:], complexity=complexity, norm=norm)
            self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
            self.upconv = Block3d(channels[1] + channels[2], channels[1], complexity=complexity, final=norm)

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
    def __init__(self, channels: list[int], final_classes: int, complexity: int = 1, norm=None):
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm3d

        self.model = nn.Sequential(
            UNet3dLayer(channels, complexity=complexity, norm=norm),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=final_classes,
                kernel_size=(1, 1, 1),
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
