import torch
from torch import nn, Tensor
from . import wrap


class Stream(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super(Stream, self).__init__()
        r_args = ', '.join(map(str, args))
        r_kwargs = ', '.join([key + '=' + str(value) for key, value in kwargs.items()])
        self.repr = f"{name}({r_args}{', ' if args and kwargs else ''}{r_kwargs})"
        self.repr_dict = dict(
            name=name,
            args=args,
            kwargs=kwargs,
        )
        self.mod = build(name, *args, **kwargs)

    def __repr__(self):
        return self.repr

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        # if isinstance(self.mod, nn.Conv3d):
        #     print("Conv3d weight is on", self.mod.weight.device)
        #     print("Args[0] is on", args[0].device)
        return wrap(self.mod(*args))


def build(name, *args, **kwargs):
    match name:
        case None:
            return None
        case "argmax":
            return lambda x: torch.argmax(x, dim=kwargs.get("dim", None))
        case "as_tensor":
            batch_dims = kwargs.get("batch_dims", None)
            device = kwargs.get("device", None) if torch.cuda.is_available() else "cpu"
            dtype = kwargs.get("dtype", None)
            kwargs = {}
            if device:
                kwargs["device"] = device
            if dtype:
                kwargs["dtype"] = dtype

            def mod(x):
                if batch_dims is None or batch_dims==x.ndim:
                    return torch.as_tensor(x, **kwargs)
                return torch.as_tensor(x, **kwargs).unsqueeze(0)

            return mod
        case "Recall":
            argmax_input_dim = kwargs.get("argmax_input_dim", None)
            def recall(input, target):
                if argmax_input_dim:
                    input = torch.argmax(input, dim=argmax_input_dim)
                return (0.1 + torch.sum(torch.minimum(input, target))) / (0.1 + torch.sum(target))
            return recall
        case "FoldBatchNorm3d":
            return FoldBatchNorm3d(*args, **kwargs)
        case "unmax":
            kernel = tuple(map(int, args[0]))
            return nn.Upsample(scale_factor=kernel, mode='nearest')
        case "unavg":
            kernel = tuple(map(int, args[0]))
            return nn.Upsample(scale_factor=kernel, mode='trilinear')
        case _:
            return eval(f"nn.{name}")(*args, **kwargs)


class FoldBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, shape=(8, 8, 8), *args, **kwargs):
        super(FoldBatchNorm3d, self).__init__(*args, **kwargs)
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        n, c, x, y, z = input.shape
        sx, sy, sz = self.shape
        fx, fy, fz = x // sx, y // sy, z // sz
        input = input.view([n, c, fx, sx, fy, sy, fz, sz]) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape([n * fx * fy * fz, c, sx, sy, sz])
        input = super(FoldBatchNorm3d, self).forward(input)
        input = input.view(n, fx, fy, fz, c, sx, sy, sz) \
            .permute(0, 4, 1, 5, 2, 6, 3, 7) \
            .reshape([n, c, x, y, z])
        return input
