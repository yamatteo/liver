from __future__ import annotations

from torch import nn, Tensor
from torch.nn import Module


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


class Block(Module):
    """Base convolution block for 3D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
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
