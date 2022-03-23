from __future__ import annotations

import itertools
from pathlib import Path

import nibabel
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import max_pool3d, avg_pool3d, one_hot, interpolate
from torch.utils.data import Dataset

from path_explorer import discover, get_criterion


def case_cycle(base_path: Path | str, segmented: bool):
    yield from (
        base_path / case
        for case in itertools.cycle(
        discover(base_path, get_criterion(registered=True, segmented=segmented))
    )
    )


def scanseg(case_cycle):
    for case in case_cycle:
        yield torch.cat([
            torch.stack([
                torch.tensor(np.array(nibabel.load(
                    case / f"registered_phase_{phase}.nii.gz"
                ).dataobj, dtype=np.int16)).float()
                for phase in ["b", "a", "v", "t"]]),
            one_hot(torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long()).permute(3, 0, 1, 2).float()
        ], dim=0)


def thickslices(t: Tensor, thickness: int, overlapping: float) -> Tensor:
    height = t.size(-1)
    offset = int(thickness * overlapping)
    for i in range(height // offset + 1):
        j = int((i / (height // offset)) * (height - thickness))
        yield t[..., j: j + thickness]


def std_slices(base_path: Path):
    for scan_seg in scanseg(case_cycle(base_path, True)):
        for slc in thickslices(scan_seg, 32, 0.6):
            yield slc


def slices882(base_path: Path, thickness: int = 32, overlapping: float = 0.5):
    for case in case_cycle(base_path, True):
        scan = torch.stack([
            torch.tensor(np.array(nibabel.load(
                case / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16)).float()
            for phase in ["b", "a", "v", "t"]
        ])
        segm = one_hot(
            torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long(),
            3
        ).permute(3, 0, 1, 2).float()

        height = scan.size(-1)
        offset = int(thickness * overlapping)
        for i in range(height // offset + 1):
            j = int((i / (height // offset)) * (height - thickness))
            yield hash((str(case), i)), torch.cat([
                avg_pool3d(
                    scan[..., j: j + thickness]
                    , kernel_size=(8, 8, 2)
                )
                , max_pool3d(
                    segm[..., j: j + thickness],
                    kernel_size=(8, 8, 2)
                )], dim=0)


class BufferThickslice882(Dataset):
    def __init__(self, base_path: Path, buffer_size: int, turnover: int):
        self.generator = slices882(base_path)
        self.buffer_size = buffer_size
        self.turnover = turnover
        self.buffer = [
            next(self.generator)
            for i in range(buffer_size)
        ]

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i):
        _, x = self.buffer[i]
        return x

    def turn(self):
        self.buffer = self.buffer[self.turnover:] + [
            next(self.generator)
            for i in range(self.turnover)
        ]

    def drop(self, indices: set[int]):
        _buffer = [x for j, x in enumerate(self.buffer) if j not in indices]
        _hashes = {h for x, h in _buffer}
        while len(_buffer) < self.buffer_size:
            h, x = next(self.generator)
            if h not in _hashes:
                _buffer.append((h, x))
                _hashes.add(h)
        self.buffer = _buffer


def cubes441(base_path: Path, model882, thickness: int = 32, step: int = 16):
    for case in case_cycle(base_path, True):
        scan = torch.stack([
            torch.tensor(np.array(nibabel.load(
                case / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16)).float()
            for phase in ["b", "a", "v", "t"]
        ])
        segm = one_hot(
            torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long(),
            3
        ).permute(3, 0, 1, 2).float()

        height = scan.size(-1)
        z_indices = [n * step for n in range(1 + (height - thickness) // step)] + [height - thickness]
        for z in z_indices:
            thickslice = scan[..., z: z + thickness]
            prediction = interpolate(
                model882(
                    avg_pool3d(thickslice, kernel_size=(8, 8, 2)).unsqueeze(0)
                ),
                size=(128, 128, thickness)
            ).squeeze(0)
            for x, y in itertools.product(range(4), range(4)):
                cubescan = avg_pool3d(
                    scan[:, 128 * x: 128 * (x + 1), 128 * y: 128 * (y + 1), z: z + thickness],
                    kernel_size=(4, 4, 1)
                )
                cubehelp = prediction[:, 32 * x: 32 * (x + 1), 32 * y: 32 * (y + 1), :]
                cubesegm = max_pool3d(
                    segm[:, 128 * x: 128 * (x + 1), 128 * y: 128 * (y + 1), z: z + thickness],
                    kernel_size=(4, 4, 1)
                )
                yield case, (x, y, z), torch.cat([cubescan, cubehelp, cubesegm], dim=0)


def batchcubes441(base_path: Path, model882, batch_size: int = 1, thickness: int = 32, step: int = 16):
    gen = cubes441(base_path, model882, thickness, step)
    while True:
        identity = ""
        cubes = []
        for n in range(batch_size):
            case, index, t = next(gen)
            identity = identity + f"<{case}@{index}>"
            cubes = cubes + [t]
        yield HashTensor(identity, torch.stack(cubes))


class HashTensor():
    def __init__(self, identity, tensor: Tensor):
        self.identity = identity
        self.data = tensor

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        try:
            return self.identity == other.identity
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return not (self.identity == other.identity)
        except AttributeError:
            return True


class BufferCube441(Dataset):
    def __init__(self, base_path: Path, model882, buffer_size: int, batch_size: int):
        self.generator = batchcubes441(base_path, model882, batch_size)
        self.buffer_size = buffer_size
        self.buffer = [
            next(self.generator)
            for i in range(buffer_size)
        ]

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i):
        return self.buffer[i].data.detach()

    def refill(self):
        _buffer = set(self.buffer)
        while len(_buffer) < self.buffer_size:
            x = next(self.generator)
            _buffer.add(x)
        self.buffer = list(_buffer)

    def drop(self, indices: set[int]):
        self.buffer = [x for j, x in enumerate(self.buffer) if j not in indices]
        self.refill()

@torch.no_grad()
def waffer_from(base_path: Path, model441, model882, device, waffer_size: int = 3):
    for case in case_cycle(base_path, True):
        scan = torch.stack([
            torch.tensor(np.array(nibabel.load(
                case / f"registered_phase_{phase}.nii.gz"
            ).dataobj, dtype=np.int16))
            for phase in ["b", "a", "v", "t"]
        ]).to(device=device, dtype=torch.float32)
        segm = one_hot(
            torch.tensor(np.array(nibabel.load(
                case / f"segmentation.nii.gz"
            ).dataobj, dtype=np.int16)).long(),
            3
        ).permute(3, 0, 1, 2).to(device=device, dtype=torch.float32)

        pred882 = model882(avg_pool3d(scan, kernel_size=(8, 8, 2)).unsqueeze(0))
        scan441 = avg_pool3d(scan, kernel_size=(4, 4, 1)).unsqueeze(0)
        help441 = interpolate(pred882, scale_factor=2)
        pad = scan441.size(4) - help441.size(4)
        if pad > 0:
            help441 = torch.constant_pad_nd(help441, (0, pad))
        pred441 = model441(torch.cat([scan441, help441], dim=1))

        inputtarget = torch.cat([
            scan.unsqueeze(0),
            interpolate(pred441, size=scan.shape[1:]),
            segm.unsqueeze(0)
        ], dim=1)

        del scan, segm, pred882, scan441, help441, pred441
        torch.cuda.empty_cache()
        for z in range(inputtarget.size(4) - waffer_size):
            yield HashTensor((case, (z,)), inputtarget[..., z: z + waffer_size])


class BufferWaffer(Dataset):
    def __init__(self, base_path: Path, model441, model882, device, buffer_size: int = 100, waffer_size=3):
        self.waffer_from = waffer_from(base_path, model441=model441, model882=model882, device=device,
                                       waffer_size=waffer_size)
        self.buffer_size = buffer_size
        self.buffer = [
            next(self.waffer_from)
            for i in range(buffer_size)
        ]

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, i):
        return self.buffer[i].data

    def refill(self):
        _buffer = set(self.buffer)
        while len(_buffer) < self.buffer_size:
            _buffer.add(next(self.waffer_from))
        self.buffer = list(_buffer)

    def drop(self, indices: set[int]):
        self.buffer = [x for j, x in enumerate(self.buffer) if j not in indices]
        self.refill()
