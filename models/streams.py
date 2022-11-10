from types import GeneratorType

from torch import nn, Tensor


def wrap(*args) -> tuple | tuple[Tensor, ...]:
    match args:
        case ():
            return ()
        case (tuple() | list() | GeneratorType() as items, *rest):
            return *wrap(*items), *wrap(*rest)
        case (item, ):
            return item,
        case (item, *rest):
            return item, *wrap(*rest)


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


class SplitBatch(nn.BatchNorm3d):
    def __init__(self, shape=(8, 8, 8), *args, **kwargs):
        super(SplitBatch, self).__init__(*args, **kwargs)
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        n, c, x, y, z = input.shape
        sx, sy, sz = self.shape
        fx, fy, fz = x//sx, y//sy, z//sz
        input = input.view([n, c, fx, sx, fy, sy, fz, sz])\
            .permute(0, 2, 4, 6, 1, 3, 5, 7)\
            .reshape([n*fx*fy*fz, c, sx, sy, sz])
        input = super(SplitBatch, self).forward(input)
        input = input.view(n, fx, fy, fz, c, sx, sy, sz)\
            .permute(0, 4, 1, 5, 2, 6, 3, 7)\
            .reshape([n, c, x, y, z])
        return input



