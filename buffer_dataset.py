from __future__ import annotations

import heapq
from pathlib import Path
from typing import TypeVar, Callable

import nibabel
import numpy as np
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

import report
from tensors import *
from utils import generators

console = Console()

T = TypeVar('T')


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
    def __init__(self, path: Path):
        self.files = list(path.iterdir())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int) -> FloatBundle:
        path = self.files[i]
        return FloatBundle(np.array(nibabel.load(path).dataobj, dtype=np.float32))


def update_datasets(*, data_path: Path, output_path: Path, shape: tuple[int, int, int]):
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    k = 0
    for i, case in enumerate(generators.cases(data_path, generators.criterion(bundle=True))):
        fbb = IntBundle(np.array(nibabel.load(
            case / f"train_bundle.nii.gz"
        ).dataobj, dtype=np.int16)).to_float_bundle()
        for t in fbb.slices(shape):
            nibabel.save(
                nibabel.Nifti1Image(
                    t.numpy(),
                    affine=np.eye(4)
                ),
                (valid_dir if i % 10 == 0 else train_dir) / f"{k:06}.nii.gz",
            )
            k += 1


def get_loaders(path: Path, batch_size: int):
    return (
        DataLoader(StoredDataset(path / "train"), batch_size=batch_size, pin_memory=True),
        DataLoader(StoredDataset(path / "valid"), batch_size=batch_size, pin_memory=True)
    )
