from __future__ import annotations

import itertools
from typing import Iterator
from warnings import warn

from torch import Tensor
from torch.utils.data import Dataset


class BufferDataset(Dataset):
    def __init__(self, tensor_generator: Iterator[Tensor], buffer_size: int):
        cyclic_tensor_generator = itertools.cycle(enumerate(tensor_generator))
        buffer = {}

        while len(buffer) < buffer_size:
            i, tensor = next(cyclic_tensor_generator)
            if i in buffer:
                warn("Buffer is larger than population.")
                self.proper = False
                break
            else:
                buffer[i] = tensor
        else:
            self.proper = True

        self.keys = {i: key for (i, key) in enumerate(buffer.keys())}
        self.buffer = buffer
        self.buffer_size = len(self.buffer)
        self.generator = cyclic_tensor_generator

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i) -> Tensor:
        k = self.keys[i]
        return self.buffer[k]

    def getitem(self, i) -> tuple[int, Tensor]:
        k = self.keys[i]
        return k, self.buffer[k]

    def __iter__(self):
        return iter(self.buffer.values())

    def refill(self):
        count = 0
        while count <= self.buffer_size and len(self.buffer) < self.buffer_size:
            count += 1
            i, x = next(self.generator)
            self.buffer[i] = x
        self.keys = {i: key for (i, key) in enumerate(self.buffer.keys())}
        if len(self.buffer) < self.buffer_size:
            raise StopIteration(f"Can't refill buffer with {self.buffer_size} attempts")

    def drop(self, keys: list[int]):
        print("Dropping:", keys)
        if self.proper:
            for k in keys:
                del self.buffer[k]
            self.refill()
        else:
            # If the buffer is not proper there is no need to fetch tensors again, just reordering
            for k in keys:
                self.buffer[k] = self.buffer.pop(k)

    def drop_by_position(self, positions: list[int]):
        keys = [self.keys[i] for i in positions]
        self.drop(keys)
