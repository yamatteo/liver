from __future__ import annotations

from torch import nn, Tensor
from torch.nn import Module


def actv_layer(actv: str) -> Module | None:
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


def conv_layer(transpose: bool = False, **kwargs) -> Module:
    """Return required convolution layer."""
    if transpose:
        return nn.ConvTranspose3d(**kwargs)
    else:
        return nn.Conv3d(**kwargs)


class Block(Module):
    """Base convolution block for 3D Unet."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            complexity: int = 2,
            kernel_size: tuple[int, int, int] = (3, 3, 3),
            padding: tuple[int, int, int] = (0, 0, 0),
            actv: str = "relu",
            norm: str = "",
            norm_momentum: float = 0.9,
            drop: str = "",
            transpose: bool = False
    ):
        super().__init__()
        if transpose:
            self.repr = f"Block({'<'.join([*([str(in_channels)] * complexity), str(out_channels)])}" \
                        f"{', ' + actv if actv else ''}" \
                        f"{', ' + norm if norm else ''}" \
                        f"{', ' + drop if drop else ''}" \
                        f")"
        else:
            self.repr = f"Block({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
                        f"{', ' + actv if actv else ''}" \
                        f"{', ' + norm if norm else ''}" \
                        f"{', ' + drop if drop else ''}" \
                        f")"
        if transpose:
            layers = [
                         conv_layer(
                             in_channels=in_channels,
                             out_channels=in_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             transpose=transpose,
                         ),
                         actv_layer(actv=actv),
                     ] * complexity + [
                         conv_layer(
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             transpose=transpose,
                         ),
                     ]
        else:
            layers = [
                         conv_layer(
                             in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             transpose=transpose,
                         )
                     ] + [
                         actv_layer(actv=actv),
                         conv_layer(
                             in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             padding=padding,
                             transpose=transpose,
                         ),
                     ] * complexity

        layers += [
            norm_layer(norm=norm, channels=out_channels, momentum=norm_momentum),
            drop_layer(drop=drop)
        ]
        self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
