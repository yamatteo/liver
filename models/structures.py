import inspect
from types import GeneratorType
from typing import Union

import torch
from torch import Tensor
from torch import nn

from .streams import *
from .utils import wrap


class Structure(nn.ModuleList):
    def __init__(self, *args: nn.Module, custom_repr=None, **kwargs):
        super(Structure, self).__init__(args)
        params = inspect.signature(self.__init__).parameters
        self.relevant_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in params and value != params[key].default
        }
        self.custom_repr = custom_repr

    def __repr__(self):
        if self.custom_repr:
            return self.custom_repr
        kwargs = str(", ").join([key + "=" + repr(value) for key, value in self.relevant_kwargs.items()])
        repr_head = f"{self.__class__.__name__}{'(' + kwargs + ')' if kwargs or len(self) == 0 else ''}"
        content = str("\n").join([repr(mod) for mod in self])
        return str("\n  ").join([
            repr_head + (":" if len(self) > 0 else ""),
            *content.splitlines(),
        ])

    @property
    def repr_dict(self):
        return dict(
            class_name=self.__class__.__name__,
            args=tuple(mod.repr_dict for mod in self),
            kwargs=self.relevant_kwargs
        )

    @property
    def summary(self):
        kwargs = str(", ").join([key + "=" + repr(value) for key, value in self.relevant_kwargs.items()])
        repr_head = f"{self.__class__.__name__}{'(' + kwargs + ')' if kwargs or len(self) == 0 else ''}"
        if len(self) == 0:
            return repr_head
        else:
            return {repr_head: [mod.summary if isinstance(mod, Structure) else repr(mod) for mod in self]}

    @classmethod
    def rebuild(cls, data):
        if isinstance(data, dict) and "class_name" in data and "args" in data and "kwargs" in data:
            class_name = data["class_name"]
            args = data["args"]
            args = cls.rebuild(args)
            kwargs = data["kwargs"]
            kwargs = cls.rebuild(kwargs)
            return eval(class_name)(*args, **kwargs)
        elif isinstance(data, dict):
            return {key: cls.rebuild(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [cls.rebuild(item) for item in data]
        elif isinstance(data, (tuple, GeneratorType)):
            return tuple(cls.rebuild(item) for item in data)
        else:
            return data


class Parallel(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Parallel, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(arg) for module, arg in zip(self, args, strict=True))

    def shaper(self, *shapes: tuple) -> tuple[tuple, ...]:
        return tuple(mod.shaper(shape) for mod, shape in zip(self, shapes, strict=True))


class Separate(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Separate, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        return wrap(module(*args) for module in self)

    def shaper(self, *shapes: tuple) -> tuple[tuple, ...]:
        return tuple(mod.shaper(*shapes) for mod in self)


class Sequential(Structure):
    def __init__(self, *modules: Union[Stream, Structure], custom_repr=None):
        super(Sequential, self).__init__(*modules, custom_repr=custom_repr)

    def forward(self, *args) -> tuple[Tensor, ...]:
        for module in self:
            args = wrap(module(*args))
        return args

    def shaper(self, *shapes) -> tuple[tuple, ...]:
        for module in self:
            shapes = module.shaper(*shapes)
        return shapes


# class Wrapper(nn.Module):
#     def __init__(self, stream, *, inputs, outputs, rank=None, storage=None):
#         super(Wrapper, self).__init__()
#         self.stream = stream
#         self.inputs = inputs
#         self.outputs = outputs
#         self.rank = rank
#         if torch.cuda.is_available() and isinstance(rank, int):
#             self.input_to = lambda x: torch.as_tensor(x, device=torch.device(f"cuda:{rank}"))
#         else:
#             self.input_to = lambda x: torch.as_tensor(x, device=torch.device("cpu"))
#         self.to_device()
#
#         self.storage = storage
#         try:
#             self.load()
#             print(f"Stream loaded from {self.storage}")
#         except (AttributeError, FileNotFoundError, RuntimeError) as err:
#             print(err)
#
#     def __repr__(self):
#         return str("\n    ").join([
#             f"Wrapper"
#             f"{'' if self.storage is None else ' in:' + str(self.storage)}"
#             f" {str(', '.join(self.inputs))}{'' if self.rank is None else ' @ cuda:' + str(self.rank)}"
#             f" >> {str(', '.join(self.outputs))}",
#             *repr(self.stream).splitlines()
#         ])
#
#     def to_device(self):
#         if torch.cuda.is_available() and self.rank is not None:
#             self.stream.cuda(self.rank)
#         else:
#             self.stream.cpu()
#
#     def forward(self, items: dict) -> dict:
#         tensors = tuple(self.input_to(items[key]) for key in self.inputs)
#         # print(f"First input device is {tensors[0].device}")
#         tensors = self.stream(*tensors)
#         items.update({key: tensors[i] for i, key in enumerate(self.outputs)})
#         return items
#
#     def load(self):
#         if self.storage is None:
#             raise AttributeError("This stream does not have a storage file.")
#         self.stream.load_state_dict(torch.load(
#             self.storage,
#             map_location="cpu" if self.rank is None or not torch.cuda.is_available() else f"cuda:{self.rank}"
#         ))
#
#     def save(self):
#         if self.storage is None:
#             raise AttributeError("This stream does not have a storage file.")
#         torch.save(self.stream.state_dict(), self.storage)
