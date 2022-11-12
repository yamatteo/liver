import torch
from torch import Tensor
from torch import nn

from . import wrap
from .monostream import Stream


class Structure(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super(Structure, self).__init__()
        r_args = ', '.join(map(str, args))
        r_kwargs = ', '.join([key + '=' + str(value) for key, value in kwargs.items()])
        self.mod = None
        self.mods = []
        self.repr = f"{name}({r_args}{', ' if args and kwargs else ''}{r_kwargs})"
        self.repr_dict = dict(
            name=name,
            args=args,
            kwargs=kwargs,
        )

    def __repr__(self):
        modules = [self.mod] if self.mod is not None else self.mods
        content = str("\n").join([repr(mod) for mod in modules])
        return str("\n    ").join([
            self.repr+(":" if modules else ""),
            *content.splitlines(),
        ])

class Cat(Structure):
    def __init__(self, dim=1):
        super(Cat, self).__init__("Cat", dim=dim)
        self.dim = dim

    def forward(self, *args):
        return torch.cat(args, dim=self.dim),


class Select(Structure):
    def __init__(self, indexes: list[int]):
        super(Select, self).__init__("Select", indexes)
        self.indexes = indexes

    def forward(self, *args):
        return wrap(args[i] for i in self.indexes)


class SkipCat(Structure):
    def __init__(self, module: Stream | Structure, dim=1):
        super(SkipCat, self).__init__("SkipCat", dim=dim)
        self.mod = module
        self.dim = dim

    def forward(self, *args):
        return wrap(torch.cat([x, *self.mod(x)], dim=self.dim) for x in args)


class Separated(Structure):
    def __init__(self, *modules: Stream | Structure):
        super(Separated, self).__init__("Separated")
        self.mods = nn.ModuleList(modules)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(arg) for module, arg in zip(self.mods, args))


class Split(Structure):
    def __init__(self, *modules: Stream | Structure):
        super(Split, self).__init__("Split")
        self.mods = nn.ModuleList(modules)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(*args) for module in self.mods)


class Sequential(Structure):
    def __init__(self, *modules: Stream | Structure):
        super(Sequential, self).__init__("Sequential")
        self.mods = nn.ModuleList(modules)

    def forward(self, *args) -> tuple[Tensor, ...]:
        for module in self.mods:
            args = wrap(module(*args))
        return args


class Wrapper(nn.Module):
    def __init__(self, stream, *, inputs, outputs, rank=None, storage=None):
        super(Wrapper, self).__init__()
        self.stream = stream
        self.inputs = inputs
        self.outputs = outputs
        self.rank = rank
        match (torch.cuda.is_available(), rank):
            case (False, _) | (True, None):
                # print("Defining cpu input_to")
                def input_to(x):
                    return torch.as_tensor(x, device=torch.device("cpu"))
            case (True, int(r)):
                # print(f"Defining cuda:{r} input_to")
                def input_to(x):
                    return torch.as_tensor(x, device=torch.device(f"cuda:{r}"))
            case _:
                raise ValueError(
                    f"Combination (torch.cuda.is_available(), rank)="
                    f"{(torch.cuda.is_available(), rank)} is not acceptable."
                )
        self.input_to = input_to
        self.to_device()

        self.storage = storage
        try:
            self.load()
            print(f"Stream loaded from {self.storage}")
        except (AttributeError, FileNotFoundError, RuntimeError) as err:
            print(err)

    def __repr__(self):
        return str("\n    ").join([
            f"Wrapper"
            f"{'' if self.storage is None else ' in:' + str(self.storage)}"
            f" {str(', '.join(self.inputs))}{'' if self.rank is None else ' @ cuda:' + str(self.rank)}"
            f" >> {str(', '.join(self.outputs))}",
            *repr(self.stream).splitlines()
        ])

    def to_device(self):
        if torch.cuda.is_available() and self.rank is not None:
            self.stream.cuda(self.rank)
        else:
            self.stream.cpu()

    def forward(self, items: dict) -> dict:
        tensors = tuple(self.input_to(items[key]) for key in self.inputs)
        # print(f"First input device is {tensors[0].device}")
        tensors = self.stream(*tensors)
        items.update({key: tensors[i] for i, key in enumerate(self.outputs)})
        return items

    def load(self):
        if self.storage is None:
            raise AttributeError("This stream does not have a storage file.")
        self.stream.load_state_dict(torch.load(
            self.storage,
            map_location="cpu" if self.rank is None or not torch.cuda.is_available() else f"cuda:{self.rank}"
        ))

    def save(self):
        if self.storage is None:
            raise AttributeError("This stream does not have a storage file.")
        torch.save(self.stream.state_dict(), self.storage)
