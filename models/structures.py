import torch
from torch import Tensor
from torch import nn

from models.streams import AbstractStream, wrap


class Cat(AbstractStream):
    def __init__(self, dim=1):
        super(Cat, self).__init__("Cat", "Cat", dim)
        self.dim = dim

    def forward(self, *args):
        return torch.cat(args, dim=self.dim),


class Select(AbstractStream):
    def __init__(self, indexes: list[int]):
        super(Select, self).__init__("Select", "Select", indexes)
        self.indexes = indexes

    def forward(self, *args):
        return tuple(args[i] for i in self.indexes)


class Split(AbstractStream):
    def __init__(self, *modules: AbstractStream):
        super(Split, self).__init__("Split", "Split", [mod.repr_dict for mod in modules])
        self.mods = nn.ModuleList(modules)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(*args) for module in self.mods)

    def __repr__(self):
        lines = ["    " + line for module in self.mods for line in repr(module).split("\n")]
        return "\n".join(["Split:", *lines])


class Sequential(AbstractStream):
    def __init__(self, *modules: AbstractStream):
        super(Sequential, self).__init__("Sequential", "Sequential", [mod.repr_dict for mod in modules])
        self.mods = nn.ModuleList(modules)

    def forward(self, *args) -> tuple[Tensor, ...]:
        for module in self.mods:
            args = wrap(module(*args))
        return args

    def __repr__(self):
        lines = ["    " + line for module in self.mods for line in repr(module).split("\n")]
        return "\n".join(["Sequential:", *lines])
