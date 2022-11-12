import argparse
import shutil
from pathlib import Path

from rich import print

import path_explorer as px

parser = argparse.ArgumentParser(description='Rename folders')
parser.add_argument('--sources', type=Path, help='folder where are the sources')
parser.add_argument("--move", action="store_true", help="really do the moving")

if __name__ == "__main__":
    args = parser.parse_args()
    for case in px.discover(args.sources, px.is_anything):
        if str(case)[0:3] == "Hum":
            new_name = f"HE016-{str(case)[3:6]}"
            if args.move:
                shutil.move(args.sources/case, args.sources/new_name)
            print(case, " >> ", new_name)
