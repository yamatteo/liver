from __future__ import annotations

import argparse
import pickle
import re
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import dicom2nifti
import nibabel
import numpy as np
import pyelastix
from scipy import ndimage as scipy_ndimage
import sys

def main():
    here = Path(__file__)
    sys.path.insert(0, str(here.parent.parent))

    import preprocessing
    parser = argparse.ArgumentParser(description="Convert from DICOMDIR to nifti.")
    parser.add_argument("--source", type=Path, required=True, help='folder where is the dicomdir')
    parser.add_argument("--target", type=Path, help="where to store the output")

    args = parser.parse_args()
    if not args.target:
        args.target = args.source

    process_dicomdir(args.source, args.target)
    preprocessing.register_case(args.target)

def process_dicomdir(source_path: Path, target_path: Path):
    """Convert scans in `source_path` from dicom to nifti."""
    import preprocessing

    case_name = input(f"Case name? (Press ENTER for default: {source_path.name})")
    if len(case_name) < 2:
        case_name = source_path.name

    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)

        # Convert the whole directory
        dicom2nifti.convert_directory(
            dicom_directory=str(source_path),
            output_folder=str(temp_path),
            compression=True
        )

        # List what generated nifti is a 512x512xD image for the appropriate phase
        print("Index  - Filename")
        choises = {}
        for index, path in enumerate(temp_path.iterdir()):
            print(f"{index:>6} - {path.name}")
            choises[index] = path
        phases = {}

        for phase in ["basale", "arteriosa", "venosa", "tardiva"]:
            index = int(input(f"Select {phase} phase (by index): "))
            phases[phase[0]] = preprocessing.load_niftiimage(choises[index])
            phases[phase[0]].header.set_sform(phases[phase[0]].affine)
            phases[phase[0]].header.set_qform(phases[phase[0]].affine)
            preprocessing.save_original(phases[phase[0]], target_path, phase[0])

        # If there is exactly one image per phase, save them as compressed nifti
        # for phase in ["b", "a", "v", "t"]:
        #     phases[phase].header.set_sform(phases[phase].affine)
        #     phases[phase].header.set_qform(phases[phase].affine)
        #     save_original(phases[phase], target_path, phase)
        # print(
        #     f"{' ' * len(case_name)}  "
        #     f"Original images saved in {target_path.absolute()}."
        # )

if __name__=="__main__":
    main()