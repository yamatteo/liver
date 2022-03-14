from __future__ import annotations

import random
from pathlib import Path

import nibabel
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from path_explorer import discover_cases_dirs, is_segmented_case, discover, is_case


class GenericDataset(Dataset):
    def __init__(self,
                 data_path: Path | str,
                 segmented: bool,
                 wafer: int | None,
                 background_reduction: float | None):
        self.cases = [
            data_path / case
            for case in discover_cases_dirs(data_path, segmented)
        ]
        self.cases = [
            {
                "case": case.name,

                "path": str(case),

                "scan": torch.stack([
                    torch.tensor(np.array(nibabel.load(
                        case / f"registered_phase_{phase}.nii.gz"
                    ).dataobj, dtype=np.int16))
                    for phase in ["b", "a", "v", "t"]
                ]),

                "segmentation": torch.tensor(np.array(nibabel.load(
                    case / f"segmentation.nii.gz"
                ).dataobj, dtype=np.int16)) if segmented else None
            }
            for case in self.cases
        ]
        if wafer is not None:
            self.cases = [
                {
                    "case": case["case"],

                    "path": case["path"],

                    "wafer": case["scan"][:, :, :, z:z + wafer],

                    "segmentation": case["segmentation"][:, :, z + (wafer // 2)]
                    if segmented
                    else None
                }
                for case in self.cases
                for z in range(0, case["scan"].size(-1) - wafer + 1)
            ]
            if background_reduction is not None:
                self.cases = [
                    w
                    for w in self.cases
                    if (
                            {0} != set(w["segmentation"].flatten().numpy().tolist())
                            or random.random() < background_reduction
                    )
                ]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        return self.cases[i]


class QuantileDataset(Dataset):
    def __init__(self, base_path: Path | str, segmented: bool):
        self.cases = [
            case_dir
            for case_dir in discover(base_path, is_segmented_case if segmented else is_case)
        ]
        self.cases = [
            {
                "name": case_dir.name,
                "case_dir": str(case_dir),
                "path": str((base_path / case_dir).resolve()),

                "orig_affine": nibabel.load(
                        base_path / case_dir / f"registered_phase_v.nii.gz"
                    ).affine,

                # "orig_scan": torch.stack([
                #     torch.tensor(np.array(nibabel.load(
                #         base_path / case_dir / f"registered_phase_{phase}.nii.gz"
                #     ).dataobj, dtype=np.int16))
                #     for phase in ["b", "a", "v", "t"]
                # ]),
                #
                # "orig_segm": torch.tensor(np.array(nibabel.load(
                #     base_path / case_dir / f"segmentation.nii.gz"
                # ).dataobj, dtype=np.int16)),

                "scan": torch.stack([
                    torch.tensor(np.array(nibabel.load(
                        base_path / case_dir / f"registered_phase_{phase}.nii.gz"
                    ).dataobj, dtype=np.int16))
                    for phase in ["b", "a", "v", "t"]
                ]).permute(3, 0, 1, 2).reshape(-1, 4, 512 * 512).float(),

                "segmentation": torch.tensor(np.array(nibabel.load(
                    base_path / case_dir / f"segmentation.nii.gz"
                ).dataobj, dtype=np.int16)).permute(2, 0, 1).reshape(-1, 512 * 512).float() if segmented else None
            }
            for case_dir in self.cases
        ]
        self.cases = [
            {
                "name": case["name"],
                "case_dir": case["case_dir"],
                "path": case["path"],
                "orig_affine": case["orig_affine"],
                #"orig_scan": case["orig_scan"],
                #"orig_segm": case["orig_segm"],

                "scan": self.quantiles(case["scan"]).permute(1, 2, 0).reshape(-1, 40),

                "segmentation": torch.mean(case["segmentation"], dim=-1) if segmented else None
            }
            for case in self.cases
        ]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        return self.cases[i]

    @staticmethod
    def quantiles(input: Tensor):
        return torch.quantile(
            input,
            torch.linspace(0.1, 1, 10),
            dim=-1
        )
