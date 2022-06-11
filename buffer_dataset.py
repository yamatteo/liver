from __future__ import annotations

import heapq
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar, Callable

import nibabel
import numpy as np
from rich.console import Console
from torch.utils.data import Dataset

import report
from tensors import *
from utils import generators

console = Console()

T = TypeVar('T')


class PreBufferDataset(Dataset):
    def __init__(self, tmpdir: TemporaryDirectory, *, max_size: int, turnover: float = 0.1):
        f_list = list(Path(tmpdir.name).iterdir())

        self.buffer = {}
        self.buffer_size = min(max_size, len(f_list))
        self.gen = itertools.cycle(map(str, f_list))
        self.keys = []
        self.losses = {}
        self.turnover = min(int(max_size * turnover), max(0, len(f_list) - max_size))

        self.refill()

    def __getitem__(self, i: int):
        return self.buffer[self.keys[i]].detach()

    def __len__(self):
        return len(self.buffer)

    def refill(self):
        for _ in range(self.buffer_size):
            if len(self) >= self.buffer_size:
                break
            path = next(self.gen)
            self.buffer[path] = FloatBatchBundle(np.array(nibabel.load(path).dataobj, dtype=np.float32))
        self.keys = list(self.buffer.keys())

    def update(self, losses: list[float]):
        self.losses.update({path: loss for path, loss in zip(self.buffer, losses)})
        smallest = heapq.nsmallest(self.turnover, self.keys, lambda k: losses[k])
        for key in smallest:
            del self.buffer[key]
        self.refill()


class BufferDataset(Dataset):
    def __init__(self, generator: Iterator[tuple[int, T]], *, max_size: int, batch_size: int,
                 turnover: float = 0.1):
        assert 0.0 < turnover < 1.0
        buffer = {}

        while len(buffer) < max_size:
            k, item = next(generator)
            if k in buffer:
                report.warn("Buffer is larger than population.")
                self.proper = False
                break
            buffer[k] = item
        else:
            self.proper = True

        self.batch_size = batch_size
        self.buffer = buffer
        self.buffer_size = len(buffer)
        self.generator = generator
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

    def replacement(self, losses: dict[int, float]):
        if self.proper:
            smallest = heapq.nsmallest(self.turnover, list(losses.keys()), lambda k: losses[k])
            for key in smallest:
                del self.buffer[key]
        self.buffer = {k: x for k, x in sorted(self.buffer.items(), key=lambda item: -losses[item[0]])}
        while len(self.buffer) < self.buffer_size:
            k, x = next(self.generator)
            self.buffer[k] = x
        self.keys = list(self.buffer.keys())

    def warmup(self, eval: Callable[[T], float]):
        losses = {}
        key = None
        for key, item in self.buffer.items():
            losses[key] = eval(item)
        while True:
            if key == 0:
                break
            k_min, v_min = min(losses.items(), key=lambda kv: kv[1])
            key, item = next(self.generator)
            value = eval(item)
            if value > v_min:
                self.buffer[key] = item
                losses[key] = value
                del self.buffer[k_min]
                del losses[k_min]
        self.keys = self.keys = list(self.buffer.keys())

    # @classmethod
    # def warmup(cls,
    #            generator: Iterator[tuple[int, T]],
    #            evaluator: Callable[[T], float], *,
    #            max_size: int,
    #            batch_size: int | None,
    #            turnover: float = 0.1) -> BufferDataset:
    #     self = cls.__new__(cls)
    #     assert 0.0 < turnover < 1.0
    #     buffer = {}
    #     losses = {}
    #
    #     while True:
    #         k, tensor = next(generator)
    #         if k == 0 and len(buffer) > 0:
    #             break
    #         report.debug(f"Evaluating k={k}")
    #         buffer[k] = tensor
    #         losses[k] = evaluator(tensor)
    #         if len(buffer) > max_size:
    #             out = min(buffer.keys(), key=lambda k: losses[k])
    #             del buffer[out]
    #
    #     self.batch_size = batch_size
    #     self.buffer = buffer
    #     self.buffer_size = len(buffer)
    #     self.generator = generator
    #     self.item_type = type(next(iter(buffer.values())))
    #     self.keys = list(buffer.keys())
    #     self.proper = (len(self.buffer) == max_size)
    #     self.turnover = int(turnover * len(buffer))
    #     return self


class StoredDataset(Dataset):
    def __init__(self, tempdir: TemporaryDirectory, batch_size: int):
        self.tempdir = tempdir
        self.files = list(Path(tempdir.name).iterdir())
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        return FloatBatchBundle(np.array(nibabel.load(path).dataobj, dtype=np.float32))

    def batches(self) -> Iterator[FloatBatchBundle]:
        for i in range(0, len(self), self.batch_size):
            yield FloatBatchBundle.batch([self[j] for j in range(i, min(i + self.batch_size, len(self)))])


def split_datasets(*, data_path: Path, shape: tuple[int, int, int], max_size: int | tuple[int, int], turnover: float = 0.1) \
        -> tuple[PreBufferDataset, PreBufferDataset]:
    train_tempdir = TemporaryDirectory()
    valid_tempdir = TemporaryDirectory()
    k = 0
    for i, case in enumerate(generators.cases(data_path, generators.criterion(bundle=True))):
        fbb = IntBundle(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16)).to_float_batch_bundle()
        for t in fbb.slices(shape):
            nibabel.save(
                nibabel.Nifti1Image(
                    t.numpy(),
                    affine=np.eye(4)
                ),
                Path((valid_tempdir if i % 10 == 0 else train_tempdir).name)
                / f"{k}.nii.gz",
            )
            k += 1

    if isinstance(max_size, int):
        train_max_size = valid_max_size = max_size
    else:
        train_max_size, valid_max_size = max_size

    return (
        PreBufferDataset(tmpdir=train_tempdir, max_size=train_max_size, turnover=turnover),
        PreBufferDataset(tmpdir=valid_tempdir, max_size=valid_max_size, turnover=turnover),
    )
