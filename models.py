from __future__ import annotations

import types
import typing
from pathlib import Path

import torch
import yaml
from torch import Tensor
from torch import nn

T = typing.TypeVar('T')
throuple = types.GenericAlias(tuple, (T,) * 3)
int3d = throuple[int]


def build(path: Path):
    with open(path, 'r') as f:
        archs = yaml.load(f.read(), yaml.Loader)
    return {name: build_module(arch) for name, arch in archs.items()}


def rebuild(path: Path):
    data = torch.load(path)
    modules = {name: build_module(arch) for name, arch in data["arch"].items()}
    for name, module in modules.items():
        module.load_state_dict(data[name])
    return modules


def build_module(arch: dict | list | str):
    # print("building:", arch)
    match arch:
        case None | "null" | "none" | {None: _} | {"null": _} | {"none": _}:
            return nn.Identity()
        case {"pipeline": [*streams]}:
            return Pipeline(*map(build_module, streams))
        case {"sequential": [*modules]}:
            return nn.Sequential(*map(build_module, modules))
        case {"skipcat": {**arch}}:
            return SkipCat(build_module(arch))
        case {"splitcat": [sub1, sub2]}:
            return SplitCat(build_module(sub1), build_module(sub2))
        case {"stream": {**args}}:
            args_iter = iter(args.items())
            mod = build_module(dict([next(args_iter)]))
            return Stream(module=mod, **dict(args_iter))

        case "elu":
            return nn.ELU(True)
        case "relu":
            return nn.ReLU(True)
        case "leaky":
            return nn.LeakyReLU(True)
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "drop2d":
            return nn.Dropout2d(p=0.5, inplace=True)
        case "drop3d":
            return nn.Dropout3d(p=0.5, inplace=True)

        case "dw_max_22":
            return nn.MaxPool2d(kernel_size=(2, 2))
        case "dw_max_222":
            return nn.MaxPool3d(kernel_size=(2, 2, 2))
        case "dw_max_221":
            return nn.MaxPool3d(kernel_size=(2, 2, 1))
        case "dw_avg_222":
            return nn.AvgPool3d(kernel_size=(2, 2, 2))
        case "dw_avg_441":
            return nn.AvgPool3d(kernel_size=(4, 4, 1))
        case "uw_max_22":
            return nn.Upsample(scale_factor=(2, 2), mode='nearest')
        case "uw_max_222":
            return nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        case "uw_max_221":
            return nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')
        case "uw_max_441":
            return nn.Upsample(scale_factor=(4, 4, 1), mode='nearest')
        case "uw_avg_222":
            return nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        case "uw_avg_441":
            return nn.Upsample(scale_factor=(4, 4, 1), mode='trilinear')

        case {"batch2d": dict(kwargs)}:
            return nn.BatchNorm2d(**kwargs)
        case {"batch3d": dict(kwargs)}:
            return nn.BatchNorm3d(**kwargs)
        case {"insta2d": dict(kwargs)}:
            return nn.InstanceNorm2d(**kwargs)
        case {"insta3d": dict(kwargs)}:
            return nn.InstanceNorm3d(**kwargs)
        case {"Conv3d": dict(kwargs)}:
            return nn.Conv3d(**kwargs)
        case {"SConv3d": [in_ch, out_ch]}:
            return SConv3d(in_ch, out_ch)
        case {"SConv3d_flat": [in_ch, out_ch]}:
            return SConv3d(in_ch, out_ch, kernel=(3, 3, 1))
        case {"SConv3d": dict(kwargs)}:
            return SConv3d(**kwargs)
        case {"SCBlock3d": dict(kwargs)}:
            return SCBlock3d(**kwargs)
        case [*modules]:
            return nn.Sequential(*map(build_module, modules))
    raise ValueError(f"Unrecognizable architecture {arch}")


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


class SkipCat(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return torch.cat([x, self.submodule(x)], dim=1)


class SplitCat(nn.Module):
    def __init__(self, sub1: nn.Module, sub2: nn.Module):
        super().__init__()
        self.sub1 = sub1
        self.sub2 = sub2

    def forward(self, x):
        return torch.cat([self.sub1(x), self.sub2(x)], dim=1)


class Pipeline(nn.Module):
    def __init__(self, *streams):
        super().__init__()
        self.streams = nn.ModuleList(streams)

    def to_cuda(self):
        for stream in self.streams:
            stream.to_cuda()

    def to_cpu(self):
        for stream in self.streams:
            stream.to_cpu()

    def forward(self, *inputs):
        return (stream(x) for x, stream in zip(inputs, self.streams))


class Stream(nn.Module):
    def __init__(self, module: nn.Module, cuda=None, use_grad=True, **kwargs):
        super().__init__()
        self.module = module
        self.module.requires_grad_(use_grad)
        self.cuda = cuda
        self.use_grad = use_grad
        self.device = torch.device("cpu") if cuda is None else torch.device(f"cuda:{cuda}")

    def to_cpu(self):
        self.device = torch.device("cpu")
        self.module.to(self.device)

    def to_cuda(self):
        self.device = torch.device(f"cuda:{self.cuda}")
        self.module.to(self.device)

    def forward(self, x):
        if x is None:
            return None
        if self.use_grad:
            return self.module(x.to(device=self.device))
        with torch.no_grad():
            return self.module(x.to(device=self.device))


## Convolution blocks

# class SeparableConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernels_per_layer=1):
#         super(SeparableConvolution, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=3, padding=1,
#                                    padding_mode="reflect", groups=in_channels)
#         self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


class SConv3d(nn.Module):
    def __init__(
            self,
            channels,
            kernels_per_layer=1,
            kernel: tuple[int, int, int] = (3, 3, 3),
            stride: tuple[int, int, int] = (1, 1, 1),
    ):
        super(SConv3d, self).__init__()
        padding = tuple(k // 2 for k in kernel)
        in_channels, out_channels = channels

        self.depthwise = nn.Conv3d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel, padding=padding,
                                   padding_mode="reflect", stride=stride, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels * kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


#
# class SCBlock(nn.Module):
#     """Base separable convolution block for 2D Unet."""
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             *,
#             kernels_per_layer: int = 1,
#             complexity: int = 2,
#             actv: str = "relu",
#             norm: str = "instance",
#             drop: str = "drop",
#     ):
#         super().__init__()
#         self.repr = f"SCBlock({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
#                     f"{'>' + actv if actv else ''}" \
#                     f"{'>' + norm if norm else ''}" \
#                     f"{'>' + drop if drop else ''}" \
#                     f")"
#
#         layers = [
#                      SeparableConvolution(
#                          in_channels=in_channels,
#                          out_channels=out_channels,
#                      ),
#                  ] + [
#                      actv_layer(actv=actv),
#                      SeparableConvolution(
#                          in_channels=out_channels,
#                          out_channels=out_channels
#                      )
#                  ] * complexity + [
#                      norm_layer(norm=norm, channels=out_channels),
#                      drop_layer(drop=drop)
#                  ]
#         self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])
#
#     def __repr__(self):
#         return self.repr
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)
#
#
# class SCBlock3d(nn.Module):
#     """Base separable convolution block for 3D Unet."""
#
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             *,
#             kernel: tuple[int, int, int] = (3, 3, 3),
#             kernels_per_layer: int = 1,
#             complexity: int = 2,
#             actv: str = "relu",
#             norm: str = "instance3d",
#             drop: str = "drop3d",
#     ):
#         super().__init__()
#         self.repr = f"SCBlock[{''.join(map(str, kernel))}]" \
#                     f"({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
#                     f"{'>' + actv if actv else ''}" \
#                     f"{'>' + norm if norm else ''}" \
#                     f"{'>' + drop if drop else ''}" \
#                     f")"
#
#         layers = [
#                      SeparableConvolution3d(
#                          in_channels=in_channels,
#                          out_channels=out_channels,
#                          kernel=kernel,
#                          kernels_per_layer=kernels_per_layer,
#                      ),
#                  ] + [
#                      actv_layer(actv=actv),
#                      SeparableConvolution3d(
#                          in_channels=out_channels,
#                          out_channels=out_channels,
#                          kernel=kernel,
#                          kernels_per_layer=kernels_per_layer,
#                      )
#                  ] * complexity + [
#                      norm_layer(norm=norm, channels=out_channels),
#                      drop_layer(drop=drop)
#                  ]
#         self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])
#
#     def __repr__(self):
#         return self.repr
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)


class SCBlock3d(nn.Module):
    """Base separable convolution block for 3D Unet."""

    def __init__(
            self,
            channels: list[int],
            *,
            kernel: int3d = (3, 3, 3),
            stride: int3d = (1, 1, 1),
            kernels_per_layer: int = 1,
            actv: str = "none",
            norm: str = "none",
            drop: str = "none",
    ):
        super().__init__()
        self.repr = f"SCBlock[" \
                    f"{''.join(map(str, kernel))}" \
                    f"{'/' + ''.join(map(str, stride)) if stride != (1, 1, 1) else ''}" \
                    f"]" \
                    f"(" \
                    f"{'>'.join(map(str, channels))}" \
                    f"{'>' + actv if actv else ''}" \
                    f"{'>' + norm if norm else ''}" \
                    f"{'>' + drop if drop else ''}" \
                    f")"

        layers = []
        for i in range(0, len(channels) - 1):
            if i > 0:
                layers.append(build_module(actv))
            layers.append(
                SConv3d(
                    channels=channels[i:i + 2],
                    kernel=kernel,
                    stride=stride if i == len(channels) - 2 else (1, 1, 1),
                    kernels_per_layer=kernels_per_layer,
                )
            )
        layers.append(build_module({norm: {'num_features': channels[-1]}}))
        layers.append(build_module(drop))
        self.model = nn.Sequential(*[lyr for lyr in layers if not isinstance(lyr, nn.Identity)])

    def __repr__(self):
        return self.repr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


## UNet
#
#
# class Unet3d(nn.Module):
#     """UNet convolution, down-sampling and skip connection layer."""
#
#     def __init__(
#             self, *,
#             bottom_activation: str = "relu",
#             bottom_dropout: str = "drop3d",
#             bottom_normalization: str = "instance3d",
#             channels: list[int],
#             complexity: int = 2,
#             conv_kernels: list[tuple[int, int, int]] = None,
#             down_activation: str = "leaky",
#             down_dropout: str = "drop3d",
#             down_normalization: str = "instance3d",
#             finals: int = 3,
#             outermost: bool = True,
#             pool_layers: list[str] = None,
#             up_activation: str = "relu",
#             up_dropout: str = "drop3d",
#             up_normalization: str = "instance3d",
#             **kwargs,
#     ):
#         super().__init__()
#         self.level = level = len(channels) - 2
#         if conv_kernels is None:
#             conv_kernels = [(3, 3, 1) if i < 2 else (3, 3, 3) for i, _ in enumerate(channels[1:])]
#         if pool_layers is None:
#             pool_layers = ["max221" if i < 2 else "max222" for i, _ in enumerate(channels[2:])]
#         self.block = SCBlock3d(
#             in_channels=channels[0],
#             out_channels=channels[1],
#             kernel=conv_kernels[0],
#             complexity=complexity,
#             actv=down_activation if level > 0 else bottom_activation,
#             norm=down_normalization if level > 0 else bottom_normalization,
#             drop=down_dropout if level > 0 else bottom_dropout,
#         )
#         if level > 0:
#             self.pool = pool_layer(pool_layers[0])
#             self.submodule = Unet3d(
#                 channels=channels[1:],
#                 conv_kernels=conv_kernels[1:],
#                 pool_layers=pool_layers[1:],
#                 outermost=False,
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
#             )
#             self.unpool = unpool_layer(pool_layers[0])
#             self.upconv = SCBlock3d(
#                 in_channels=channels[1] + channels[2],
#                 out_channels=channels[1],
#                 kernel=conv_kernels[0],
#                 complexity=complexity,
#                 actv=up_activation,
#                 norm=up_normalization,
#                 drop=up_dropout
#             )
#         if outermost:
#             self.final = nn.Conv3d(
#                 in_channels=channels[1],
#                 out_channels=finals,
#                 kernel_size=(1, 1, 1),
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
#             x = self.upconv(y)
#             del y
#         try:
#             x = self.final(x)
#         except AttributeError:
#             pass
#         return x


def set_momentum(model, momentum: float):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
            module.momentum = momentum
