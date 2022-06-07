from __future__ import annotations

import heapq
import itertools
from typing import Iterable, TypeVar

from rich.console import Console
from torch.utils.data import Dataset

import report
from tensors import *

console = Console()

T = TypeVar('T')


class BufferDataset(Dataset):
    def __init__(self, generator: Iterator[T] | Iterable[T], *, max_size: int, batch_size: int | None,
                 turnover: float = 0.1):
        assert 0.0 < turnover < 1.0
        cyclic_generator = itertools.cycle(enumerate(generator))
        buffer = {}

        while len(buffer) < max_size:
            k, tensor = next(cyclic_generator)
            if k in buffer:
                report.warn("Buffer is larger than population.")
                self.proper = False
                break
            buffer[k] = tensor
        else:
            self.proper = True

        self.batch_size = batch_size
        self.buffer = buffer
        self.buffer_size = len(buffer)
        self.generator = cyclic_generator
        self.item_type = type(next(iter(buffer.values())))
        self.keys = list(buffer.keys())
        self.turnover = int(turnover * len(buffer))

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i) -> tuple[int, T]:
        k = self.keys[i]
        return k, self.buffer[k]

    def batches(self) -> Iterator[tuple[list[int], T]]:
        batch_keys = [self.keys[i:i + self.batch_size] for i in range(0, len(self.keys), self.batch_size)]
        for keys in batch_keys:
            yield keys, self.item_type.batch([self.buffer[k] for k in keys])

    def iter(self) -> Iterator[tuple[int | list[int], T]]:
        if self.batch_size is None:
            return iter(self.buffer.items())
        else:
            return self.batches()

    def replacement(self, losses: dict[int, float]):
        if self.proper:
            smallest = heapq.nsmallest(self.turnover, list(losses.keys()), lambda k: losses[k])
            for key in smallest:
                del self.buffer[key]
        self.buffer = {k: x for k, x in sorted(self.buffer.items(), key=lambda item: losses[item[0]])}
        while len(self.buffer) < self.buffer_size:
            k, x = next(self.generator)
            self.buffer[k] = x
        self.keys = list(self.buffer.keys())
