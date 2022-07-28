from __future__ import annotations

import math

import numpy as np
import torch
from rich.progress import Progress
from torch import nn, Tensor
from torch.nn import Module, functional

from ..subclass_tensors import *
from .skip_connection_layer import Layer, pool_layer, unpool_layer


class UNet(Module):
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
            pool: str = "max222"
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
                pool=pool
            ),
            nn.Conv3d(
                in_channels=channels[1],
                out_channels=final_classes,
                kernel_size=(1, 1, 1),
            ),
            # nn.Tanh()
        )

    def forward(self, x: Tensor) -> FloatSegmBatch:
        # dim = FloatSegmBatch.fixed_shape["C"]
        return FloatSegmBatch(self.model(x))

    def block_forward(self, x: Tensor):
        # Assuming x is shape [N, PH+KL, X, Y, Z]
        pieces = []
        next_x = x
        while next_x.size(4) > 15:
            current_x, next_x = next_x[..., 0:8], next_x[..., 8:]
            pieces.append(self.model(current_x))
        pieces.append(self.model(next_x))
        return FloatSegmBatch(torch.cat(pieces, dim=4))

    # @torch.no_grad()
    # def apply(self, x: FloatScan, thickness: int = 8) -> Segm:
    #     shape = (3, x.size("H"), x.size("W"), x.size("D"))
    #     base = torch.zeros(shape, device=x.device, dtype=torch.float32)
    #     size = x.size("D")
    #     assert x.boundaries() == (0, size)
    #     assert size >= thickness
    #     num_slices = math.ceil(size / thickness)
    #
    #     with Progress(transient=True) as progress:
    #         task = progress.add_task(
    #             f"Segmenting scan.",
    #             total=num_slices
    #         )
    #         for j in range(num_slices):
    #             i = int(j * (size - thickness) / (num_slices - 1))
    #             slice = torch.narrow(x, -1, i, thickness)
    #             pred = self.forward(slice)
    #             base[..., i:i + thickness] += pred
    #             progress.update(task, advance=1)
    #     return FloatSegm(base).as_int()

class DoubleUNet:
    def __init__(self):
        self.down_sampler = pool_layer("avg441")
        self.first_net = UNet(
            channels=[4, 32, 64, 96, 128],
            pool="max221"
        )
        self.up_sampler = unpool_layer("avg441")
        self.second_net = UNet(
            channels=[7, 32, 64, 96, 128],
            pool="max221"
        )