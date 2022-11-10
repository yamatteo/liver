# from __future__ import annotations
#
# import traceback
# import types
# import typing
# from pathlib import Path
#
# import torch
# import yaml
# from rich import print
# from torch import Tensor
# from torch import nn
# from torch.nn import functional
# from torch.nn import *
#
# T = typing.TypeVar('T')
# throuple = types.GenericAlias(tuple, (T,) * 3)
# int3d = throuple[int]
#
#
# def build(path: Path, *, models_path: Path):
#     with open(path, 'r') as f:
#         archs = yaml.load(f.read(), yaml.Loader)
#
#     main, main_args = next(iter(archs.items()))
#     return build_module({main: main_args}, archs=archs, models_path=models_path)
#
#
# def build_module(arch, archs: dict, models_path: Path):
#     match arch:
#         case str(name) if name in archs:
#             return build_module(archs[name], archs, models_path)
#         case str() | int() | float():
#             return arch
#         case list(items):
#             return [build_module(item, archs, models_path) for item in items]
#     assert isinstance(arch, dict), f"Unrecognizable architecture of type {type(arch)}"
#     match list(iter(arch.items())):
#         case []:
#             return {}
#         case [(str(name), None)] if name[0].isupper():
#             return eval(name)()
#         case [(str(name), list(args))] if name[0].isupper():
#             return eval(name)(*build_module(args, archs, models_path))
#         case [(str(name), dict(kwargs))] if name[0].isupper():
#             return eval(name)(**build_module(kwargs, archs, models_path))
#         case [('module', str(name)), *targs]:
#             return Stream(build_module(archs[name], archs, models_path), **dict(targs, models_path=models_path))
#         case [('module', dict(module)), *targs]:
#             return Stream(build_module(module, archs, models_path), **dict(targs, models_path=models_path))
#         case list(targs):
#             return {key: build_module(value, archs, models_path) for key, value in targs}
#     raise ValueError(f"Unrecognizable architecture {arch}")
#
#
# class Pipeline(Module):
#     def __init__(self, *steps: list[Stream]):
#         super().__init__()
#         inputs = [{label for stream in streams for label in stream.inputs} for streams in steps]
#         finals = {label for streams in steps for stream in streams for label in stream.outputs if label[0].isupper()}
#         self.preserved_outputs = [
#             set.union(
#                 {label for requireds in inputs[i + 1:] for label in requireds},
#                 finals
#             )
#             for i, streams in enumerate(steps)
#         ]
#         self.steps: list[list[Stream]] = ModuleList(ModuleList(streams) for streams in steps)
#
#     def save(self, epoch=None):
#         for streams in self.steps:
#             for stream in streams:
#                 stream.save(epoch=epoch)
#
#     def to_cuda(self):
#         for streams in self.steps:
#             for stream in streams:
#                 stream.to_cuda()
#
#     def to_cpu(self):
#         for streams in self.steps:
#             for stream in streams:
#                 stream.to_cpu()
#
#     def single_forward(self, items: dict):
#         for streams in self.steps:
#             for stream in streams:
#                 items.update(stream(items))
#
#     def pipeline_forward(self, step_items: list[dict[str, Tensor | typing.Any]]):
#         # TODO test real parallelism
#         for step, (items, streams) in enumerate(zip(step_items, self.steps)):
#             for stream in streams:
#                 items.update(stream(items))
#             items.update({key: None for key in items if key not in self.preserved_outputs[step]})
#
#
# class Stream(nn.Module):
#     def __init__(self, module: nn.Module, *, load=False,
#                  storage=None,
#                  models_path=None, **kwargs):
#         super().__init__()
#         self.module = module
#
#         if kwargs.get("requires_grad", True) is False:
#             self.module.requires_grad_(False)
#             self.forward = torch.no_grad()(self.forward)
#
#         self.to_device = kwargs["to_device"] if torch.cuda.is_available() and "to_device" in kwargs else None
#
#         self.inputs = kwargs.get("inputs", None)
#         self.outputs = kwargs.get("outputs", None)
#
#         if (models_path := kwargs.get("models_path", None)) and (store_file := kwargs.get("store", None)):
#             self.storage = models_path / store_file
#         else:
#             self.storage = None
#
#         if self.storage:
#             try:
#                 device = torch.device(self.to_device) if self.to_device else torch.device("cpu")
#                 self.load_state_dict(torch.load(self.storage, map_location=device))
#             except (FileNotFoundError, RuntimeError) as err:
#                 print(f"Can't load from {self.storage}.", err)
#
#     def save(self, epoch=None):
#         if self.storage:
#             torch.save(self.state_dict(), self.storage)
#             if epoch:
#                 torch.save(self.state_dict(), self.storage.with_suffix(f".{epoch:03}.pth"))
#
#     def to_cpu(self):
#         self.module.to(device=torch.device("cpu"))
#
#     def to_cuda(self):
#         self.module.to(device=self.to_device)
#
#     def move(self, *inputs):
#         inputs = tuple(torch.as_tensor(data) for data in inputs)
#         if self.to_device:
#             inputs = tuple(to(x, self.to_device) for x in inputs)
#         return inputs
#
#     def forward(self, *args):
#         match args:
#             case (dict(items), ):
#                 try:
#                     inputs = tuple(items[i] for i in self.inputs)
#                 except KeyError:
#                     return {}
#             case tuple() if all(isinstance(a, Tensor) for a in args):
#                 if self.inputs:
#                     inputs = tuple(args[i] for i in self.inputs)
#                 else:
#                     inputs = args
#             case _:
#                 raise ValueError("Stream accept either a single dict or many tensors as input.")
#
#         inputs = self.move(*inputs)
#         outputs = self.module(*inputs)
#         outputs = wrap(outputs)
#         if self.outputs:
#             outputs = {name: unwrap(outputs[i]) for i, name in enumerate(self.outputs)}
#         return outputs
#
#
# # Structures
# class Cat(nn.Module):
#     def __init__(self, *, dim: int = 1):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, *inputs: Tensor):
#         return torch.cat(inputs, dim=self.dim),
#
#
# class Sequential(nn.Module):
#     def __init__(self, *modules: Module):
#         super().__init__()
#         self.streams = nn.ModuleList(modules)
#
#     def forward(self, *inputs):
#         for stream in self.streams:
#             # print(f"Sequential({', '.join([f'{input.shape}:{input.device}' for input in inputs])})")
#             inputs = stream(*inputs)
#             if isinstance(inputs, torch.Tensor):
#                 inputs = (inputs,)
#         return inputs
#
#
# class Split(nn.Module):
#     def __init__(self, *modules: Stream):
#         super().__init__()
#         self.streams = nn.ModuleList(modules)
#
#     def forward(self, *inputs):
#         outputs = []
#         for stream in self.streams:
#             out = stream(*inputs)
#             if isinstance(out, Tensor):
#                 outputs.append(out)
#             else:
#                 outputs.extend(out)
#         return outputs
#
#
# class Cascade(nn.Module):
#     def __init__(self, *modules):
#         super().__init__()
#         self.streams = ModuleList(modules)
#
#     def forward(self, x: Tensor):
#         outputs = []
#         for stream in self.streams:
#             x = stream(x)
#             outputs.append(x)
#         return outputs
#
#
# class Onehot(nn.Module):
#     def __init__(self, *, classes: int = 2):
#         super().__init__()
#         self.classes = classes
#
#     def forward(self, x: Tensor):
#         # Assume x is batch
#         return functional.one_hot(x.long(), num_classes=self.classes).permute(0, x.ndim, *range(1, x.ndim)).float()
#
#
# class CrossEntropy(nn.Module):
#     def __init__(self, *, weights: list[int | float] = None):
#         super().__init__()
#         if weights:
#             weights = torch.tensor(weights)  # , device=torch.device("cuda:0"))
#         self.raw = nn.CrossEntropyLoss(weights, reduction="none")
#
#     def forward(self, input, target):
#         loss = self.raw(input, target)
#         return torch.mean(loss, dim=list(range(1, loss.ndim)))
#
#
# class SkipCat(nn.Module):
#     def __init__(self, submodule: nn.Module):
#         super().__init__()
#         self.submodule = submodule
#
#     def forward(self, x: Tensor):
#         return torch.cat([x, self.submodule(x)], dim=1)
#
#
# class SplitCat(nn.Module):
#     def __init__(self, sub1: nn.Module, sub2: nn.Module):
#         super().__init__()
#         self.sub1 = sub1
#         self.sub2 = sub2
#
#     def forward(self, x):
#         return torch.cat(unwrap([self.sub1(x), self.sub2(x)]), dim=1)
#
#
# class Pool:
#     def __new__(cls, pool: str):
#         return pool_layer(pool)
#
#
# class Unpool:
#     def __new__(cls, pool: str):
#         return unpool_layer(pool)
#
#
# class SConv3d(nn.Module):
#     def __init__(
#             self,
#             channels,
#             kernels_per_layer=1,
#             kernel: tuple[int, int, int] = (3, 3, 3),
#             stride: tuple[int, int, int] = (1, 1, 1),
#     ):
#         super(SConv3d, self).__init__()
#         padding = tuple(k // 2 for k in kernel)
#         in_channels, out_channels = channels
#
#         self.depthwise = nn.Conv3d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel, padding=padding,
#                                    padding_mode="reflect", stride=stride, groups=in_channels)
#         self.pointwise = nn.Conv3d(in_channels * kernels_per_layer, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x
#
#
# class SCBlock3d(nn.Module):
#     """Base separable convolution block for 3D Unet."""
#
#     def __init__(
#             self,
#             channels: list[int],
#             *,
#             kernel: int3d = (3, 3, 3),
#             stride: int3d = (1, 1, 1),
#             kernels_per_layer: int = 1,
#             actv: str = "none",
#             norm: str = "none",
#             drop: str = "none",
#     ):
#         super().__init__()
#         self.repr = f"SCBlock[" \
#                     f"{''.join(map(str, kernel))}" \
#                     f"{'/' + ''.join(map(str, stride)) if stride != (1, 1, 1) else ''}" \
#                     f"]\n" \
#                     f"\t(" \
#                     f"{'>'.join(map(str, channels))}" \
#                     f"{'>' + actv if actv else ''}" \
#                     f"{'>' + norm if norm else ''}" \
#                     f"{'>' + drop if drop else ''}" \
#                     f")"
#
#         layers = []
#         for i in range(0, len(channels) - 1):
#             if i > 0:
#                 layers.append(actv_layer(actv))
#             layers.append(
#                 SConv3d(
#                     channels=channels[i:i + 2],
#                     kernel=kernel,
#                     stride=stride if i == len(channels) - 2 else (1, 1, 1),
#                     kernels_per_layer=kernels_per_layer,
#                 )
#             )
#         layers.append(norm_layer(norm, channels[-1]))
#         layers.append(drop_layer(drop))
#         self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None and not isinstance(lyr, nn.Identity)])
#
#     def __repr__(self):
#         return self.repr
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.model(x)
#
#
# def to(x: Tensor | tuple | list | dict, device):
#     if isinstance(device, str):
#         device = torch.device(device)
#     match x:
#         case Tensor():
#             return x.to(device=device)
#         case tuple():
#             return tuple(to(y, device) for y in x)
#         case list():
#             return list(to(y, device) for y in x)
#         case dict(d):
#             return {key: to(value, device) for key, value in d.items()}
#         case _:
#             return x
#
# def wrap(arg):
#     if isinstance(arg, Tensor):
#         return arg,
#     elif isinstance(arg, (tuple, list)):
#         return arg
#
# def unwrap(arg):
#     match arg:
#         case Tensor():
#             return arg
#         case (t, ) | [t, ]:
#             return t
#         case tuple() | list():
#             return tuple(map(unwrap, arg))
#
#
# # ## Single layers
# #
# # def actv_layer(actv: str) -> nn.Module | None:
# #     """Return required activation layer."""
# #     if actv == "relu":
# #         return nn.ReLU(True)
# #     if actv == "leaky":
# #         return nn.LeakyReLU(True)
# #     if actv == "sigmoid":
# #         return nn.Sigmoid()
# #     if actv == "tanh":
# #         return nn.Tanh()
# #     return None
# #
# #
# # def drop_layer(drop: str, drop_prob: float = 0.5) -> nn.Module | None:
# #     """Return required dropout layer."""
# #     if drop == "drop" or drop == "drop2d":
# #         return nn.Dropout2d(p=drop_prob, inplace=True)
# #     if drop == "drop3d":
# #         return nn.Dropout3d(p=drop_prob, inplace=True)
# #     return None
# #
# #
# # def norm_layer(norm: str, channels: int, momentum: float = 0.9, affine: bool = False) -> nn.Module | None:
# #     """Return required normalization layer."""
# #     if norm == "batch" or norm == "batch2d":
# #         return nn.BatchNorm2d(num_features=channels, momentum=momentum, affine=affine)
# #     if norm == "instance" or norm == "instance2d":
# #         return nn.InstanceNorm2d(num_features=channels, momentum=momentum, affine=affine)
# #     if norm == "batch3d":
# #         return nn.BatchNorm3d(num_features=channels, momentum=momentum, affine=affine)
# #     if norm == "instance3d":
# #         return nn.InstanceNorm3d(num_features=channels, momentum=momentum, affine=affine)
# #     return None
# #
# #
# # def pool_layer(pool: str) -> nn.Module | None:
# #     """Return required pooling layer."""
# #     if pool == "max22":
# #         return nn.MaxPool2d(kernel_size=(2, 2))
# #     if pool == "max222":
# #         return nn.MaxPool3d(kernel_size=(2, 2, 2))
# #     if pool == "max221":
# #         return nn.MaxPool3d(kernel_size=(2, 2, 1))
# #     if pool == "max441":
# #         return nn.MaxPool3d(kernel_size=(4, 4, 1))
# #     if pool == "max882":
# #         return nn.MaxPool3d(kernel_size=(8, 8, 2))
# #     if pool == "avg222":
# #         return nn.AvgPool3d(kernel_size=(2, 2, 2))
# #     if pool == "avg441":
# #         return nn.AvgPool3d(kernel_size=(4, 4, 1))
# #     return None
# #
# #
# # def unpool_layer(pool: str) -> nn.Module | None:
# #     """Return required upsampling layer."""
# #     if pool == "max22":
# #         return nn.Upsample(scale_factor=(2, 2), mode='nearest')
# #     if pool == "max222":
# #         return nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
# #     if pool == "max221":
# #         return nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')
# #     if pool == "avg222":
# #         return nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
# #     if pool == "avg441":
# #         return nn.Upsample(scale_factor=(4, 4, 1), mode='trilinear')
# #     return None
#
#
# ## Convolution blocks
#
# # class SeparableConvolution(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernels_per_layer=1):
# #         super(SeparableConvolution, self).__init__()
# #         self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=3, padding=1,
# #                                    padding_mode="reflect", groups=in_channels)
# #         self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, out_channels, kernel_size=1)
# #
# #     def forward(self, x):
# #         x = self.depthwise(x)
# #         x = self.pointwise(x)
# #         return x
#
#
# #
# # class SCBlock(nn.Module):
# #     """Base separable convolution block for 2D Unet."""
# #
# #     def __init__(
# #             self,
# #             in_channels: int,
# #             out_channels: int,
# #             *,
# #             kernels_per_layer: int = 1,
# #             complexity: int = 2,
# #             actv: str = "relu",
# #             norm: str = "instance",
# #             drop: str = "drop",
# #     ):
# #         super().__init__()
# #         self.repr = f"SCBlock({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
# #                     f"{'>' + actv if actv else ''}" \
# #                     f"{'>' + norm if norm else ''}" \
# #                     f"{'>' + drop if drop else ''}" \
# #                     f")"
# #
# #         layers = [
# #                      SeparableConvolution(
# #                          in_channels=in_channels,
# #                          out_channels=out_channels,
# #                      ),
# #                  ] + [
# #                      actv_layer(actv=actv),
# #                      SeparableConvolution(
# #                          in_channels=out_channels,
# #                          out_channels=out_channels
# #                      )
# #                  ] * complexity + [
# #                      norm_layer(norm=norm, channels=out_channels),
# #                      drop_layer(drop=drop)
# #                  ]
# #         self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])
# #
# #     def __repr__(self):
# #         return self.repr
# #
# #     def forward(self, x: Tensor) -> Tensor:
# #         return self.model(x)
# #
# #
# # class SCBlock3d(nn.Module):
# #     """Base separable convolution block for 3D Unet."""
# #
# #     def __init__(
# #             self,
# #             in_channels: int,
# #             out_channels: int,
# #             *,
# #             kernel: tuple[int, int, int] = (3, 3, 3),
# #             kernels_per_layer: int = 1,
# #             complexity: int = 2,
# #             actv: str = "relu",
# #             norm: str = "instance3d",
# #             drop: str = "drop3d",
# #     ):
# #         super().__init__()
# #         self.repr = f"SCBlock[{''.join(map(str, kernel))}]" \
# #                     f"({'>'.join([str(in_channels), *([str(out_channels)] * complexity)])}" \
# #                     f"{'>' + actv if actv else ''}" \
# #                     f"{'>' + norm if norm else ''}" \
# #                     f"{'>' + drop if drop else ''}" \
# #                     f")"
# #
# #         layers = [
# #                      SeparableConvolution3d(
# #                          in_channels=in_channels,
# #                          out_channels=out_channels,
# #                          kernel=kernel,
# #                          kernels_per_layer=kernels_per_layer,
# #                      ),
# #                  ] + [
# #                      actv_layer(actv=actv),
# #                      SeparableConvolution3d(
# #                          in_channels=out_channels,
# #                          out_channels=out_channels,
# #                          kernel=kernel,
# #                          kernels_per_layer=kernels_per_layer,
# #                      )
# #                  ] * complexity + [
# #                      norm_layer(norm=norm, channels=out_channels),
# #                      drop_layer(drop=drop)
# #                  ]
# #         self.model = nn.Sequential(*[lyr for lyr in layers if lyr is not None])
# #
# #     def __repr__(self):
# #         return self.repr
# #
# #     def forward(self, x: Tensor) -> Tensor:
# #         return self.model(x)
#
# ## UNet
# #
# #
# # class Unet3d(nn.Module):
# #     """UNet convolution, down-sampling and skip connection layer."""
# #
# #     def __init__(
# #             self, *,
# #             bottom_activation: str = "relu",
# #             bottom_dropout: str = "drop3d",
# #             bottom_normalization: str = "instance3d",
# #             channels: list[int],
# #             complexity: int = 2,
# #             conv_kernels: list[tuple[int, int, int]] = None,
# #             down_activation: str = "leaky",
# #             down_dropout: str = "drop3d",
# #             down_normalization: str = "instance3d",
# #             finals: int = 3,
# #             outermost: bool = True,
# #             pool_layers: list[str] = None,
# #             up_activation: str = "relu",
# #             up_dropout: str = "drop3d",
# #             up_normalization: str = "instance3d",
# #             **kwargs,
# #     ):
# #         super().__init__()
# #         self.level = level = len(channels) - 2
# #         if conv_kernels is None:
# #             conv_kernels = [(3, 3, 1) if i < 2 else (3, 3, 3) for i, _ in enumerate(channels[1:])]
# #         if pool_layers is None:
# #             pool_layers = ["max221" if i < 2 else "max222" for i, _ in enumerate(channels[2:])]
# #         self.block = SCBlock3d(
# #             in_channels=channels[0],
# #             out_channels=channels[1],
# #             kernel=conv_kernels[0],
# #             complexity=complexity,
# #             actv=down_activation if level > 0 else bottom_activation,
# #             norm=down_normalization if level > 0 else bottom_normalization,
# #             drop=down_dropout if level > 0 else bottom_dropout,
# #         )
# #         if level > 0:
# #             self.pool = pool_layer(pool_layers[0])
# #             self.submodule = Unet3d(
# #                 channels=channels[1:],
# #                 conv_kernels=conv_kernels[1:],
# #                 pool_layers=pool_layers[1:],
# #                 outermost=False,
# #                 complexity=complexity,
# #                 down_activation=down_activation,
# #                 down_normalization=down_normalization,
# #                 down_dropout=down_dropout,
# #                 bottom_activation=bottom_activation,
# #                 bottom_normalization=bottom_normalization,
# #                 bottom_dropout=bottom_dropout,
# #                 up_activation=up_activation,
# #                 up_normalization=up_normalization,
# #                 up_dropout=up_dropout,
# #             )
# #             self.unpool = unpool_layer(pool_layers[0])
# #             self.upconv = SCBlock3d(
# #                 in_channels=channels[1] + channels[2],
# #                 out_channels=channels[1],
# #                 kernel=conv_kernels[0],
# #                 complexity=complexity,
# #                 actv=up_activation,
# #                 norm=up_normalization,
# #                 drop=up_dropout
# #             )
# #         if outermost:
# #             self.final = nn.Conv3d(
# #                 in_channels=channels[1],
# #                 out_channels=finals,
# #                 kernel_size=(1, 1, 1),
# #             )
# #
# #     def forward(self, x: Tensor) -> Tensor:
# #         x = self.block(x)
# #         if self.level > 0:
# #             original_z_size = x.size(-1)
# #             y = self.pool(x)
# #             y = self.submodule(y)
# #             y = self.unpool(y)
# #             z_pad = original_z_size - y.size(-1)
# #             y = functional.pad(y, [0, z_pad])
# #             y = torch.cat([y, x], dim=1)
# #             x = self.upconv(y)
# #             del y
# #         try:
# #             x = self.final(x)
# #         except AttributeError:
# #             pass
# #         return x
#
#
# def set_momentum(model, momentum: float):
#     for module in model.modules():
#         if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
#             module.momentum = momentum
