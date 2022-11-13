from typing import Union

import torch
from torch import Tensor
from torch import nn

from .utils import wrap
from .streams import Stream


class Structure(nn.ModuleList):
    def __init__(self, *args: nn.Module, custom_repr=None, **kwargs):
        super(Structure, self).__init__(args)

        name = type(self).__name__
        repr_kwargs = str(", ").join([
            k + '=' + str(a)
            for k, a in kwargs.items()
            if not isinstance(a, (Structure, Stream))
        ])
        repr_head = f"{name}{'('+repr_kwargs+')' if repr_kwargs or len(self)==0 else ''}"
        if custom_repr:
            self.custom_repr = custom_repr
        else:
            content = str("\n").join([repr(mod) for mod in self])
            self.custom_repr = str("\n  ").join([
                repr_head + (":" if len(self) > 0 else ""),
                *content.splitlines(),
                ])
        if len(self) == 0:
            self.summary = repr_head
        else:
            self.summary = {repr_head: [repr(mod) if isinstance(mod, Stream) else mod.summary for mod in self]}

        rd_args = tuple(arg.repr_dict for arg in args)
        rd_kwargs = {
            key: value.repr_dict if isinstance(value, (Structure, Stream)) else value
            for key, value in kwargs.items()
        }
        self.repr_dict = dict(
            name=name,
            args=rd_args,
            kwargs=rd_kwargs,
        )

    def __repr__(self):
        return self.custom_repr


class Parallel(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Parallel, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(arg) for module, arg in zip(self, args, strict=True))


class Separate(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Separate, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(*args) for module in self)


class Sequential(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Sequential, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        for module in self:
            args = wrap(module(*args))
        return args


class SkipConnection(Structure):
    def __init__(self, *modules: Union[Stream, Structure], dim=1, custom_repr=None):
        super(SkipConnection, self).__init__(*modules, custom_repr=custom_repr)
        self.dim = dim

    def forward(self, *args) -> tuple[Tensor, ...]:
        orig, = args
        for module in self:
            args = wrap(module(*args))
        return wrap(torch.cat([orig, *args], dim=self.dim))


class Wrapper(nn.Module):
    def __init__(self, stream, *, inputs, outputs, rank=None, storage=None):
        super(Wrapper, self).__init__()
        self.stream = stream
        self.inputs = inputs
        self.outputs = outputs
        self.rank = rank
        if torch.cuda.is_available() and isinstance(rank, int):
            self.input_to = lambda x: torch.as_tensor(x, device=torch.device(f"cuda:{rank}"))
        else:
            self.input_to = lambda x: torch.as_tensor(x, device=torch.device("cpu"))
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
