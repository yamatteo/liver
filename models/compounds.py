from torch.nn import Conv3d

from .streams import Stream
from .structures import Sequential


class ConvBlock(Sequential):
    def __new__(cls, channels: list[int], *, kernel_size=(3, 3, 3), stride=(1, 1, 1), actv=None, norm=None, drop=None,
                momentum: float = 0.9):
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            padding_mode="reflect",
        )
        layers = [
            Stream(Conv3d, channels[0], channels[1], **kwargs)
        ]
        kwargs["stride"] = (1, 1, 1)
        for i in range(1, len(channels) - 1):
            if actv:
                layers.append(Stream(actv))
            layers.append(Stream(Conv3d, channels[i], channels[i + 1], **kwargs))
        if norm:
            layers.append(Stream(norm, num_features=channels[-1], momentum=momentum))
        if drop:
            layers.append(Stream(drop))

        type = f"ConvBlock"
        kernel_size = "".join(map(str, kernel_size))
        stride = "/" + "".join(map(str, stride)) if stride != (1, 1, 1) else ""
        channels = list(map(str, channels))
        actv = f" > {actv} > " if actv else " > "
        norm = " > " + (norm + (f"*{momentum}" if momentum != 0.9 else "")) if norm else ""
        drop = " > " + drop if drop else ""
        custom_repr = f"{type}[{kernel_size}{stride}]({actv.join(channels)}{norm}{drop})"
        return Sequential(*layers, custom_repr=custom_repr)
