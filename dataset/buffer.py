from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterator, Callable
from warnings import warn

import nibabel
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool3d, avg_pool3d, one_hot, interpolate
from torch.utils.data import Dataset

from path_explorer import discover, get_criterion


class BufferDataset(Dataset):
    def __init__(self, generator: Callable[[], Iterator[Tensor]], buffer_size: int):
        self.current_generator = None
        self.get_tensors = generator
        self.buffer_size = buffer_size
        self.buffer = {}
        self.propre_buffer = self.fill()
        self.buffer_size = len(self.buffer)

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i):
        return self.buffer[i].data

    def fill(self):
        gen = iter(enumerate(self.get_tensors()))
        proper_buffer = True
        while len(self.buffer) < self.buffer_size:
            try:
                i, x = next(gen)
                self.buffer[i] = x
            except StopIteration:
                warn("Buffer is larger than population.")
                proper_buffer = False
                break
        self.current_generator = gen
        return proper_buffer

    def refill(self):
        missing = self.buffer_size - len(self.buffer)
        count = 0
        while count < 100 * missing and len(self.buffer) < self.buffer_size:
            count += 1
            try:
                i, x = next(self.current_generator)
                self.buffer[i] = x
            except StopIteration:
                self.current_generator = iter(enumerate(self.get_tensors()))
                break
        while count < 100 * missing and len(self.buffer) < self.buffer_size:
            count += 1
            i, x = next(self.current_generator)
            self.buffer[i] = x
        if len(self.buffer) < self.buffer_size:
            raise StopIteration(f"Can't refill buffer ({count} of 100*{missing})")

    def drop(self, indices: list[int]):
        if self.propre_buffer:
            for i in indices:
                del self.buffer[i]
            self.refill()
        else:
            for i in indices:
                x = self.buffer.pop(i)
                self.buffer[i] = x
