from torch import nn, Tensor

from . import wrap
from .structures import Sequential
from .monostream import Stream


# class Conv(AbstractStream):
#     def __init__(self, in_channels, out_channels, kernel=(3, 3, 3), stride=(1, 1, 1)):
#         padding = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
#         super(Conv, self).__init__(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding,
#                                    padding_mode="reflect")
#         self.repr = f"Con({in_channels}>{out_channels})"
#
#
# class PWConv(Prepr, nn.Conv3d):
#     def __init__(self, in_channels, out_channels):
#         super(PWConv, self).__init__(in_channels, out_channels, kernel_size=1)
#         self.repr = f"PWCon({in_channels}>{out_channels})"
#
#
# class DWConv(Prepr, nn.Conv3d):
#     def __init__(self, in_channels, kernel=3, kernels_per_layer=1, padding=0, stride=1):
#         if isinstance(kernel, int):
#             kernel = (kernel, kernel, kernel)
#         if isinstance(padding, int):
#             padding = (padding, padding, padding)
#         if isinstance(stride, int):
#             stride = (stride, stride, stride)
#         super(DWConv, self).__init__(
#             in_channels,
#             in_channels * kernels_per_layer,
#             kernel_size=kernel,
#             padding=padding,
#             padding_mode="reflect",
#             stride=stride,
#             groups=in_channels
#         )
#         self.repr = f"DWConv[" \
#                     f"{''.join(map(str, kernel))}" \
#                     f"{'/' + ''.join(map(str, stride)) if stride != (1, 1, 1) else ''}" \
#                     f"]" \
#                     f"({in_channels}>{in_channels * kernels_per_layer})"
#
#
# class SConv(Prepr, nn.Sequential):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernels_per_layer=None,
#             kernel=(3, 3, 3),
#             stride=(1, 1, 1),
#     ):
#         if kernels_per_layer is None:
#             kernels_per_layer = max(1, out_channels // in_channels)
#         padding = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
#         self.repr = f"SConv[" \
#                     f"{''.join(map(str, kernel))}" \
#                     f"{'/' + ''.join(map(str, stride)) if stride != (1, 1, 1) else ''}" \
#                     f"]" \
#                     f"({in_channels}>{in_channels * kernels_per_layer})"
#
#         super(SConv, self).__init__(
#             DWConv(in_channels, kernels_per_layer=kernels_per_layer, kernel=kernel, padding=padding, stride=stride),
#             PWConv(in_channels * kernels_per_layer, out_channels)
#         )


# def convolutions(name, ich, och, kernel=(3, 3, 3), stride=(1, 1, 1)):
#     kwargs = dict(
#         in_channels=ich,
#         out_channels=och,
#         kernel_size=kernel,
#         stride=stride,
#         padding=(kernel[0] // 2, kernel[1] // 2, kernel[2] // 2),
#         padding_mode="reflect",
#         groups=1,
#     )
#     nn.Conv3d()
#     if name == "pconv":
#         return Stream("Conv3d", **dict(kwargs, kernel_size=(1, 1, 1))),
#     elif name == "dconv":
#         return Stream("Conv3d", **dict(kwargs, out_channels=ich * (och // ich), groups=ich)),
#     elif name == "sconv":
#         return Stream("Conv3d", **dict(kwargs, out_channels=ich * (och // ich), groups=ich)),\
#                Stream("Conv3d", **dict(kwargs, in_channels=ich * (och // ich), kernel_size=(1, 1, 1)))
#     else:
#         return Stream("Conv3d", **kwargs),


class ConvBlock(Stream):
    def __init__(
            self,
            channels: list[int],
            *,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            actv: str = None,
            norm: str = None,
            drop: str = None,
            momentum: float = 0.9,
    ):
        super(ConvBlock, self).__init__(None)
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            padding_mode="reflect",
        )
        layers: list[nn.Module] = [
            Stream("Conv3d", channels[0], channels[1], **kwargs)
        ]
        kwargs["stride"] = (1, 1, 1)
        for i in range(1, len(channels) - 1):
            if actv:
                layers.append(Stream(actv))
            layers.append(Stream("Conv3d", channels[i], channels[i + 1], **kwargs))
        if norm:
            layers.append(Stream(norm, num_features=channels[-1], momentum=momentum))
        if drop:
            layers.append(Stream(drop))
        self.mod = Sequential(*layers)

        self.repr_dict = dict(
            name="ConvBlock",
            args=(channels, ),
            kwargs=dict(
                kernel=kernel_size,
                stride=stride,
                actv=actv,
                norm=norm,
                drop=drop,
                momentum=momentum,
            )
        )

        type = f"ConvBlock"
        kernel_size = "".join(map(str, kernel_size))
        stride = "/" + "".join(map(str, stride)) if stride != (1, 1, 1) else ""
        channels = list(map(str, channels))
        actv = f" > {actv} > " if actv else " > "
        norm = " > " + (norm + (f"*{momentum}" if momentum != 0.9 else "")) if norm else ""
        drop = " > " + drop if drop else ""
        self.repr = f"{type}[{kernel_size}{stride}]({actv.join(channels)}{norm}{drop})"

    def __repr__(self):
        return self.repr

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return wrap(self.mod(*args))
