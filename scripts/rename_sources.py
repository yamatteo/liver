import argparse
import shutil
from pathlib import Path

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from rich import print

import path_explorer as px
from pydrive_utils import split_trainables, Path as DrivePath

parser = argparse.ArgumentParser(description='Rename folders')
parser.add_argument('--sources', type=Path, help='folder where are the sources')
parser.add_argument("--target", type=Path, help="where to store the renamed sources")
parser.add_argument("--move", action="store_true", help="move the folders")
parser.add_argument("--copy", action="store_true", help="copy and rename the folders")

if __name__ == "__main__":
    args = parser.parse_args()

    sources = DrivePath.resolve("/COLAB")
    train_cases, valid_cases = split_trainables(sources, shuffle=True)
    # file_list = drive.ListFile({'q': "'COLAB' in parents and trashed=false"}).GetList()
    for case in valid_cases:
        print(f"title: {case.name}")
    assert False
    sources = args.sources
    if args.target is None:
        target = sources
    else:
        target = args.target
    for case in px.discover(sources, px.is_anything):
        if str(case)[0:3] == "Hum":
            new_name = f"HE016-{str(case)[3:6]}"
            if args.move:
                shutil.move(sources/case, target/new_name)
            elif args.copy:
                shutil.copytree(sources/case, target/new_name)

            print(case, " >> ", new_name)
