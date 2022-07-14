import argparse
import os
from pathlib import Path

import nibabel
import numpy as np
from rich.console import Console

from in_out import load_affine, load_scan, load_segm, iter_trainable

console = Console()

parser = argparse.ArgumentParser(description="Store scan and segmentation as a single bundle, separated into slices.")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--outputs", type=Path, required=True)
parser.add_argument("--sources", type=Path, required=True)
parser.add_argument("--shape", type=eval, default=(128, 128, 8))

opts = parser.parse_args()

target_path = Path(opts.outputs)
target_path.mkdir(parents=True, exist_ok=True)
train_dir = target_path / "train"
valid_dir = target_path / "valid"
train_dir.mkdir(exist_ok=True)
valid_dir.mkdir(exist_ok=True)

k = 0
for i, case in enumerate(iter_trainable(opts.source_path)):
    scan = load_scan(case)
    segm = load_segm(case)
    bundle = np.stack([*scan, segm], axis=0)
for i, bundle in enumerate(generators.train_bundles(source_path)):
    if pooler is not None:
        scan, segm = bundle.separate()
        scan = Scan.from_float(pooler(FloatScan.from_int(scan)))
        segm = Segm.from_float(pooler(FloatSegm.from_int(segm)))
        bundle = Bundle.from_join(scan, segm)

    for t in generators.slices(bundle, shape):
        nibabel.save(
            nibabel.Nifti1Image(
                t.numpy(),
                affine=np.eye(4)
            ),
            (valid_dir if i % 10 == 0 else train_dir) / f"{k:06}.nii.gz",
        )
        k += 1





def update_trainbundle(case: Path):
    console.print(f"  [bold black]{case.name}.[/bold black] Writing bundle.")

    affine = load_affine(case)
    scan = load_scan(case)
    segm = load_segm(case)

    nibabel.save(
        nibabel.Nifti1Image(
            np.stack([*scan, segm], axis=0),
            affine=affine
        ),
        case / "train_bundle.nii.gz",
    )


console.print("[bold orange3]Updating train bundles.[/bold orange3]")
for case_path in iter_trainable():
    update_trainbundle(os.getenv("OUTPUTS") / case_path)


