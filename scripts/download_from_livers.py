import argparse
from pathlib import Path

import httplib2
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
print(sys.path)
import frontend_liver.pydrive_utils as pu
import path_explorer as px

def main():
    pu.connect()

    parser = argparse.ArgumentParser(description='Rename folders')
    parser.add_argument("--sources", type=str, default="sources", help="where sources are stored on LIVERS")
    parser.add_argument('--target', type=Path, help='folder where to put sources')
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    root = pu.DrivePath([], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq")
    sources = root / args.sources
    target = args.target
    for case in pu.iter_registered(sources):
        # if str(case)[0:3] == "Hum":
        #     new_name = f"HE016-{str(case)[3:6]}"
        # print("Considering", target / new_name)
        source_case: pu.DrivePath = sources/case
        target_case = target / case.name
        target_case.mkdir(exist_ok=True)
        existing_files = [ file.name for file in target_case.iterdir()]
        print(case, ">>", target_case)
        for source_file in source_case.iterdir():
            target_file = target_case/source_file.name
            if not source_file.exists():
                print(" ", source_file, "do not exists")
                continue
            if target_file.name not in existing_files:
                print("  Downloading", source_file.name)
            elif args.overwrite:
                print("  Overwriting", source_file.name)
            else:
                print("  File", source_file.name, "already there.")
                continue
            source_file.obj.GetContentFile(target_file)




if __name__ == "__main__":
    main()