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


def convolutions(name, ich, och, kernel=(3, 3, 3), stride=(1, 1, 1)):
    padding = (kernel[0] // 2, kernel[1] // 2, kernel[2] // 2)
    match name.lower():
        case "conv":
            return Stream("Conv3d", ich, och, kernel_size=kernel, stride=stride, padding=padding, padding_mode="reflect"),
        case "pconv":
            return Stream("Conv3d", ich, och, kernel_size=1, stride=stride),
        case "dconv":
            kernels_per_layer = och // ich
            return Stream(
                "Conv3d",
                ich,
                ich * kernels_per_layer,
                kernel_size=kernel,
                padding=padding,
                padding_mode="reflect",
                stride=stride,
                groups=ich
            ),
        case "sconv":
            kernels_per_layer = och // ich
            return (
                Stream("Conv3d", ich, ich*kernels_per_layer, kernel_size=kernel, stride=stride, padding=padding, padding_mode="reflect", groups=ich),
                Stream("Conv3d", ich*kernels_per_layer, och, kernel_size=1)
            )


class ConvBlock(Stream):
    def __init__(
            self,
            type: str,
            channels: list[int],
            *,
            kernel=(3, 3, 3),
            stride=(1, 1, 1),
            actv: str = None,
            norm: str = None,
            drop: str = None,
            momentum: float = 0.9,
    ):
        super(ConvBlock, self).__init__(None)

        layers: list[nn.Module] = [
            *convolutions(type, channels[0], channels[1], kernel=kernel, stride=stride)
        ]
        for i in range(1, len(channels) - 1):
            if actv:
                layers.append(Stream(actv))
            layers.extend(
                convolutions(
                    type,
                    channels[i],
                    channels[i + 1],
                    kernel=kernel,
                )
            )
        if norm:
            layers.append(Stream(norm, num_features=channels[-1], momentum=momentum))
        if drop:
            layers.append(Stream(drop))
        self.mod = Sequential(*layers)

        self.repr_dict = dict(
            name="ConvBlock",
            args=(type, channels),
            kwargs=dict(
                kernel=kernel,
                stride=stride,
                actv=actv,
                norm=norm,
                drop=drop,
                momentum=momentum,
            )
        )

        type = f"{type.capitalize()}Block"
        kernel = "".join(map(str, kernel))
        stride = "/" + "".join(map(str, stride)) if stride != (1, 1, 1) else ""
        channels = list(map(str, channels))
        actv = f" > {actv} > " if actv else " > "
        norm = " > " + (norm+(f"*{momentum}" if momentum != 0.9 else "")) if norm else ""
        drop = " > " + drop if drop else ""
        self.repr = f"{type}[{kernel}{stride}]({actv.join(channels)}{norm}{drop})"

    def __repr__(self):
        return self.repr

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return wrap(self.mod(*args))
