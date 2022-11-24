from dataclasses import dataclass
from typing import Union

import torch

from .streams import Stream
from .structures import Structure


@dataclass
class Input:
    name: str
    device: torch.device = None
    dtype: torch.dtype = None

    def __init__(self, name, device=None, dtype=None):
        super(Input, self).__init__()
        self.name = name
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype

    def __call__(self, data) -> torch.Tensor:
        if isinstance(data, dict):
            data = data[self.name]
        return torch.as_tensor(data, dtype=self.dtype, device=self.device)


class Architecture:
    def __init__(self, stream: Union[Stream, Structure], inputs, outputs, cuda_rank=None, storage=None):
        super(Architecture, self).__init__()
        self.stream = stream

        if torch.cuda.is_available() and cuda_rank is not None:
            self.cuda_rank = cuda_rank
        else:
            self.cuda_rank = None

        device = torch.device("cpu") if self.cuda_rank is None else torch.device(f"cuda:{self.cuda_rank}")
        self.inputs = []
        for _input in inputs:
            if isinstance(_input, str):
                self.inputs.append(Input(_input, device))
            elif isinstance(_input, tuple) and len(_input) == 1:
                name, = _input
                self.inputs.append(Input(name, device))
            elif isinstance(_input, tuple) and len(_input) == 2:
                name, dtype = _input
                self.inputs.append(Input(name, device, dtype))
        self.outputs = outputs
        self.to_device()

        self.storage = storage
        try:
            self.load()
            print(f"Stream loaded from {self.storage}")
        except (AttributeError, FileNotFoundError, RuntimeError) as err:
            print("Couls not load architecture.", err)

    def __repr__(self):
        return str("\n  ").join([
            f"Architecture"
            f" {str(', '.join([ _i.name for _i in self.inputs ]))}{'' if self.cuda_rank is None else ' @ cuda:' + str(self.cuda_rank)}"
            f" >> {str(', '.join(self.outputs))}"
            f"{'' if self.storage is None else ' stored in:' + str(self.storage)}",
            *repr(self.stream).splitlines()
        ])

    def forward(self, items: dict) -> dict:
        tensors = tuple(input__(items) for input__ in self.inputs)
        tensors = self.stream(*tensors)
        items.update(zip(self.outputs, tensors))
        return items

    @classmethod
    def rebuild(cls, storage):
        data = torch.load(storage)
        arch = data["arch"]
        arch["stream"] = Structure.rebuild(arch["stream"])
        return cls(**arch)

    def load(self):
        if self.storage is None:
            raise AttributeError("This stream does not have a storage file.")
        data = torch.load(
            self.storage,
            map_location="cpu" if self.cuda_rank is None else f"cuda:{self.cuda_rank}"
        )
        self.stream.load_state_dict(data["state_dict"])

    @property
    def repr_dict(self):
        return dict(
            stream=self.stream.repr_dict,
            inputs=[ (_i.name, _i.dtype) for _i in self.inputs ],
            outputs=self.outputs,
            cuda_rank=self.cuda_rank,
            storage=self.storage,
        )

    def save(self):
        if self.storage is None:
            raise AttributeError("This stream does not have a storage file.")
        torch.save({
            "arch": self.repr_dict,
            "state_dict":self.stream.state_dict(),
        }, self.storage)

    def to_device(self):
        if self.cuda_rank is not None:
            self.stream.cuda(self.cuda_rank)
        else:
            self.stream.cpu()
