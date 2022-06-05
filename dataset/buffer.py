from __future__ import annotations

import itertools
from typing import Iterator
from warnings import warn

import torch
from rich.console import Console
from torch import Tensor
from torch.utils.data import Dataset
from tensors import *

console = Console()


class BufferDataset(Dataset):
    def __init__(self, generator: Iterator, buffer_size: int):
        cyclic_generator = itertools.cycle(enumerate(generator))
        buffer = {}

        while len(buffer) < buffer_size:
            i, tensor = next(cyclic_generator)
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
        self.generator = cyclic_generator

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


class BufferDataset2(Dataset):
    def __init__(self, generator: Iterator, *, buffer_size: int, train_to_valid_odds: int, valid_buffer_size: int, batch_size:int = 1):
        cyclic_generator = itertools.cycle(enumerate(generator))
        buffer = {}
        self.batch_size = batch_size

        while len(buffer) < buffer_size:
            # console.print(f"Populating buffer {len(buffer)+1}/{buffer_size}")
            k, tensor = next(cyclic_generator)
            if k in buffer:
                warn("Buffer is larger than population.")
                self.proper = False
                break
            if k % train_to_valid_odds != 0:
                buffer[k] = tensor
        else:
            self.proper = True

        valid_buffer = {}

        while len(valid_buffer) < valid_buffer_size:
            k, tensor = next(cyclic_generator)
            if k in valid_buffer:
                warn("Valid_buffer is larger than population.")
                self.valid_proper = False
                break
            if k % train_to_valid_odds == 0:
                valid_buffer[k] = tensor
        else:
            self.valid_proper = True

        self.buffer = buffer
        self.valid_buffer = valid_buffer
        self.keys = list(buffer.keys())
        self.valid_keys = list(valid_buffer.keys())
        self.buffer_size = len(self.buffer)
        self.valid_buffer_size = len(self.valid_buffer)
        self.generator = cyclic_generator

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i) -> tuple[int, Tensor]:
        k = self.keys[i]
        return k, self.buffer[k]

    def train_batches(self) -> Iterator[tuple[list, tuple[ScanBatch, FloatSegmBatch]]]:
        batch_keys = [self.keys[i:i + self.batch_size] for i in range(0, len(self.keys), self.batch_size)]
        for keys in batch_keys:
            batch = FloatBatchBundle.cat([self.buffer[k] for k in keys])
            yield keys, batch.separate()

    def valid_batches(self):
        batch_keys = [self.valid_keys[i:i + self.batch_size] for i in range(0, len(self.valid_keys), self.batch_size)]
        for keys in batch_keys:
            batch = FloatBatchBundle.cat([self.valid_buffer[k] for k in keys])
            yield keys, batch.separate()

    def valid_iter(self):
        return self.valid_buffer.items()

    def valid_len(self):
        return self.valid_buffer_size

    def valid_getitem(self, i) -> tuple[int, Tensor]:
        k = self.valid_keys[i]
        return k, self.valid_buffer[k]

    def refill(self):
        count = 0
        while count <= self.buffer_size and len(self.buffer) < self.buffer_size:
            k, x = next(self.generator)
            self.buffer[k] = x
            count += 1
        self.keys = list(self.buffer.keys())
        if len(self.buffer) < self.buffer_size:
            raise StopIteration(f"Can't refill buffer with {self.buffer_size} attempts")

    def valid_refill(self):
        count = 0
        while count <= self.valid_buffer_size and len(self.valid_buffer) < self.valid_buffer_size:
            k, x = next(self.generator)
            self.valid_buffer[k] = x
            count += 1
        self.valid_keys = list(self.valid_buffer.keys())
        if len(self.valid_buffer) < self.valid_buffer_size:
            raise StopIteration(f"Can't refill buffer with {self.valid_buffer_size} attempts")

    def drop(self, keys: list[int]):
        # print("Dropping:", keys)
        if self.proper:
            for k in keys:
                del self.buffer[k]
            self.refill()
        else:
            # If the buffer is not proper there is no need to fetch tensors again, just reordering
            for k in keys:
                self.buffer[k] = self.buffer.pop(k)

    def valid_drop(self, keys: list[int]):
        # print("Dropping:", keys)
        if self.valid_proper:
            for k in keys:
                del self.valid_buffer[k]
            self.valid_refill()
        else:
            # If the buffer is not proper there is no need to fetch tensors again, just reordering
            for k in keys:
                self.valid_buffer[k] = self.valid_buffer.pop(k)
