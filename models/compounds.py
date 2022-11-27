from typing import Union

from .streams import *
from .structures import Sequential, Structure, Separate, Parallel


def build(*args, **kwargs) -> Union[Stream, Structure, None]:
    if not args or args[0] == None:
        return None
    mod, *args = args
    if isinstance(mod, str):
        mod = eval(mod)
    if isinstance(mod, type):
        mod = mod(*args, **kwargs)
    if isinstance(mod, (Structure, Stream)):
        return mod
    else:
        print(f"Error: can't build {(mod, *args)}")
        return None


class ConvBlock(Sequential):
    def __new__(cls, channels: list[int], *, kernel_size=(3, 3, 3), stride=(1, 1, 1), actv=None, norm=None, drop=None,
                momentum: float = 0.1):
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            # padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            padding_mode="replicate",
        )
        layers = [
            Conv3d(channels[0], channels[1], **kwargs)
        ]
        kwargs["stride"] = (1, 1, 1)

        for i in range(1, len(channels) - 1):
            if actv:
                layers.append(build(actv))
            layers.append(Conv3d(channels[i], channels[i + 1], **kwargs))
        if norm:
            layers.append(build(norm, num_features=channels[-1], momentum=momentum))
        if drop:
            layers.append(build(drop))

        if isinstance(actv, type):
            actv = actv.__name__
        if isinstance(norm, type):
            norm = norm.__name__
        if isinstance(drop, type):
            drop = drop.__name__
        _type = f"ConvBlock"
        kernel_size = "".join(map(str, kernel_size))
        stride = "_" + "".join(map(str, stride)) if stride != (1, 1, 1) else ""
        channels = list(map(str, channels))
        actv = f" > {actv} > " if actv else " > "
        norm = " > " + (norm + (f"*{momentum}" if momentum != 0.1 else "")) if norm else ""
        drop = " > " + drop if drop else ""
        custom_repr = f"{_type}_{kernel_size}{stride}({actv.join(channels)}{norm}{drop})"
        return Sequential(*layers, custom_repr=custom_repr)


class SkipConnection(Sequential):
    def __new__(cls, *modules: Union[Stream, Structure], dim=1):
        from .streams import Identity
        from .structures import Cat
        repr_head = "SkipConnection" + (f"(dim={dim}):" if dim != 1 else ":")
        content = str("\n").join([repr(mod) for mod in modules])
        custom_repr = str("\n  ").join([
            repr_head,
            *content.splitlines(),
        ])
        modules = [
            Separate(
                Identity(),
                modules[0] if len(modules) == 1 else Sequential(*modules)
            ),
            Cat(dim=dim)
        ]
        return Sequential(*modules, custom_repr=custom_repr)


class EncDecConnection(Sequential):
    def __new__(cls, down_mode="max", up_mode="up_n", scale_factor=(2, 2, 2), dim=1, cat=True):
        from .streams import Identity
        from .structures import Cat
        def factory(*modules: Union[Stream, Structure]):
            if down_mode == "max" and up_mode == "up_n":
                modules = [
                    MaxPool3d(kernel_size=scale_factor),
                    *modules,
                    Upsample(scale_factor=scale_factor, mode="nearest"),
                ]
            elif down_mode == "avg" and up_mode == "up_n":
                modules = [
                    AvgPool3d(kernel_size=scale_factor),
                    *modules,
                    Upsample(scale_factor=scale_factor, mode="nearest"),
                ]
            elif down_mode == "imax" and up_mode == "iunmax":
                modules = [
                    MaxPool3d(kernel_size=scale_factor, return_indices=True),
                    Parallel(
                        Sequential(*modules),
                        Identity(),
                    ),
                    MaxUnpool3d(kernel_size=scale_factor),
                ]
            if cat:
                modules = [
                    Separate(
                        Identity(),
                        Sequential(
                            *modules,
                        ),
                    ),
                    Cat(dim=dim)
                ]
            return Sequential(*modules)
        return factory

        # repr_head = f"EncDecConnection(down_mode={down_mode}, up_mode={up_mode}, " \
        #             f"scale_factor={scale_factor}, dim={dim}):"
        # content = str("\n").join([repr(mod) for mod in modules])
        # custom_repr = str("\n  ").join([
        #     repr_head,
        #     *content.splitlines(),
        # ])
