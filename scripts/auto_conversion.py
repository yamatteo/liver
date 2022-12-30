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

    import preprocessing as pp
    import path_explorer as px
    parser = argparse.ArgumentParser(description="Convert from DICOMDIR to nifti.")
    parser.add_argument("--source", type=Path, required=True, help='folder where is the dicomdir')
    parser.add_argument("--target", type=Path, help="where to store the output")

    args = parser.parse_args()
    if not args.target:
        args.target = args.source

    for folder_name in px.iter_dicom(args.source):
        case_path = args.target / folder_name
        if px.is_original(case_path) or px.is_registered(case_path):
            pass
        else:
            (args.target / folder_name).mkdir(parents=True, exist_ok=True)
            # print("Want to process", sources / folder_name, ">>", targets / case_name)
            px.process_dicomdir(args.source / folder_name, args.target / folder_name)

    for case_name in px.iter_original(args.target):
        case_path = args.target / case_name
        if px.is_registered(case_path):
            print(case_name, "is already registered. Should clear?")
        else:
            try:
                print("Registering", case_name)
                pp.register_case(case_path)
            except AssertionError as err:
                print(err)

if __name__=="__main__":
    main()