from __future__ import annotations

import random
from pathlib import Path

import nibabel
import numpy as np
import torch
from torch.utils.data import Dataset

from path_explorer import discover


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
