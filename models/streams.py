import torch
from torch import nn, Tensor
from . import wrap


class AbstractStream(nn.Module):
    def __init__(self, class_name, name, *args, **kwargs):
        super(AbstractStream, self).__init__()
        r_args = ', '.join(map(str, args))
        r_kwargs = ', '.join([key+'='+str(value) for key, value in kwargs.items()])
        self.repr = f"{name.capitalize()}({r_args}{', ' if args and kwargs else ''}{r_kwargs})"
        self.repr_dict = dict(
            class_name=class_name,
            name=name,
            args=args,
            kwargs=kwargs,
        )

    def __repr__(self):
        return self.repr
   

class Stream(AbstractStream):
    def __init__(self, name, *args, **kwargs):
        super(Stream, self).__init__("Stream", name, *args, **kwargs)
        match name.lower():
            case "argmax":
                self.mod = lambda x: torch.argmax(x, dim=kwargs.get("dim", None))
            case "as_tensor":
                batch_dims = kwargs.get("batch_dims", None)
                device = kwargs.get("device", "cpu") if torch.cuda.is_available() else "cpu"

                def mod(x):
                    x = torch.as_tensor(x, device=device)
                    if batch_dims is None or batch_dims == x.ndim:
                        return x
                    elif batch_dims == x.ndim + 1:
                        return x.unsqueeze(0)
                    else:
                        raise ValueError(f"Tensor with shape {x.shape} can't be batch of {batch_dims} dimesions.")
                self.mod = mod
            case "identity":
                self.mod = nn.Identity()
            case "elu":
                self.mod = nn.ELU()
            case "relu":
                self.mod = nn.ReLU()
            case "leaky":
                self.mod = nn.LeakyReLU()
            case "sigmoid":
                self.mod = nn.Sigmoid()
            case "tanh":
                self.mod = nn.Tanh()
            case "drop":
                self.mod = nn.Dropout3d()
            case "batch":
                self.mod = nn.BatchNorm3d(*args, **kwargs)
            case "insta":
                self.mod = nn.InstanceNorm3d(*args, **kwargs)
            case "splitbatch":
                self.mod = SplitBatch(*args, **kwargs)
            case "max":
                kernel = tuple(map(int, args[0]))
                self.mod = nn.MaxPool3d(kernel)
            case "avg":
                kernel = tuple(map(int, args[0]))
                self.mod = nn.AvgPool3d(kernel)
            case "unmax":
                kernel = tuple(map(int, args[0]))
                self.mod = nn.Upsample(scale_factor=kernel, mode='nearest')
            case "unavg":
                kernel = tuple(map(int, args[0]))
                self.mod = nn.Upsample(scale_factor=kernel, mode='trilinear')
            case "conv":
                self.mod = nn.Conv3d(*args, **kwargs)
            case _:
                raise NotImplemented(f"Name {name} is not implemented.")

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return wrap(self.mod(arg) for arg in args)


class BiStream(AbstractStream):
    def __init__(self, name, *args, **kwargs):
        super(BiStream, self).__init__("Stream", name, *args, **kwargs)
        match name.lower():
            case "crossentropy":
                self.mod = nn.CrossEntropyLoss(*args, **kwargs)
            case _:
                raise NotImplemented

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor]:
        return wrap(self.mod(x, y))



