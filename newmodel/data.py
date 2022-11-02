from __future__ import annotations

import heapq
import itertools
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.utils.data
from rich.console import Console
from torch import nn

import utils.ndarray as nd
import utils.path_explorer as px
from utils.slices import fixed_shape_slices, padded_overlapping_bundle_slices

console = Console()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, format=None):
        super(Dataset, self).__init__()
        self.files = list(path.iterdir())
        self.format = format

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        data = torch.load(path)
        if self.format:
            return self.format(data)
        else:
            return data


class BufferDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, buffer_size: int, staging_size: int):
        super(BufferDataset, self).__init__()
        self.path = path
        self.buffer_size = buffer_size
        self.staging_size = staging_size
        self.file_gen = itertools.cycle(enumerate(path.iterdir()))
        self.buffer = {}
        self.fill()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, i: int):
        k = list(self.buffer.keys())[i]
        return {"keys": k, **self.buffer[k]}

    def fill(self):
        size = self.buffer_size
        for _ in range(size):
            if len(self.buffer) >= size:
                break
            k, path = next(self.file_gen)
            if k not in self.buffer:
                self.buffer[k] = torch.load(path)

    def drop(self, scores: dict[int, float]):
        smallest = heapq.nsmallest(self.staging_size, list(scores.keys()), lambda i: scores[i])
        for k in smallest:
            del self.buffer[k]
        self.fill()


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator: Iterator[dict], buffer_size: int, staging_size: int):
        super(GeneratorDataset, self).__init__()
        self.generator = itertools.cycle(enumerate(generator))
        self.buffer = {}
        self.buffer_size = buffer_size
        self.staging_size = staging_size

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, i: int):
        k = list(self.buffer.keys())[i]
        return {"keys": k, **self.buffer[k]}

    def fill(self):
        size = self.buffer_size
        for _ in range(size):
            if len(self.buffer) >= size:
                break
            k, data = next(self.generator)
            self.buffer[k] = data

    def drop(self, scores: dict[int, float]):
        smallest = heapq.nsmallest(self.staging_size, list(scores.keys()), lambda i: scores[i])
        for k in smallest:
            del self.buffer[k]
        self.fill()


def store_441_dataset(source_path: Path, target_path: Path, slice_shape: tuple[int, int, int]):
    i, k = 0, 0
    console.print("Storing dataset:")
    console.print("  source_path =", source_path)
    console.print("  target_path =", target_path)
    console.print(f"  slice_shape = {slice_shape}")
    (target_path / "train").mkdir(exist_ok=True)
    (target_path / "valid").mkdir(exist_ok=True)
    for case in px.iter_trainable(source_path):
        console.print("  ...working on:", case)
        case_path = source_path / case
        registered_scans = [
            nd.load_registered(case_path, phase)
            for phase in ["b", "a", "v", "t"]
        ]
        segm = nd.load_segm(case_path)
        affine, bottom, top, height = nd.load_registration_data(case_path)
        bundle = np.stack([*registered_scans, segm])[..., bottom:top]
        bundle = np.clip(bundle, a_min=-1024, a_max=1024)
        for slice in padded_overlapping_bundle_slices(bundle, slice_shape):
            assert tuple(slice.shape[-3:]) == slice_shape, f"Can't fix slice shape ({slice.shape} vs {slice_shape})!"
            scan = torch.tensor(slice[0:4], dtype=torch.float32)
            segm = torch.tensor(slice[4:5], dtype=torch.float32)
            scan = nn.AvgPool3d((4, 4, 1))(scan)
            segm = nn.MaxPool3d((4, 4, 1))(segm).squeeze(0).to(dtype=torch.int64)
            assert 0 <= torch.min(segm) <= torch.max(segm) <= 2

            if k % 10 == 0:
                file_path = (target_path / "valid" / f"{i:06}.pt")
            else:
                file_path = (target_path / "train" / f"{i:06}.pt")
            torch.save({"scan": scan, "segm": segm}, file_path)
            i += 1
        k += 1