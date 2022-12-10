import argparse
from pathlib import Path

import httplib2

import frontend.pydrive_utils as pu
import path_explorer as px

def main():
    root = pu.DrivePath([], root="1N5UQx2dqvWy1d6ve1TEgEFthE8tEApxq").resolve()

    parser = argparse.ArgumentParser(description='Rename folders')
    parser.add_argument("--sources", type=str, default="sources", help="where sources are stored on LIVERS")
    parser.add_argument('--target', type=Path, help='folder where to put sources')
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    sources = root / args.sources
    target = args.target
    for case in pu.iter_registered(sources):
        print(case)
        # if str(case)[0:3] == "Hum":
        #     new_name = f"HE016-{str(case)[3:6]}"
        # print("Considering", target / new_name)
        # target_case = (target / new_name).mkdir()
        # existing_files = [ file.name for file in target_case.iterdir()]
        # for filename in ["segmentation.nii.gz", "registration_data.pickle", *[f"registered_phase_{ph}.nii.gz" for ph in "bavt"]]:
        #     source_file = sources/case/filename
        #     target_file = target_case/filename
        #     if not source_file.exists():
        #         print(" ", source_file, "do not exists")
        #         continue
        #     if target_file.name not in existing_files:
        #         f = pu.drive.CreateFile(dict(title=source_file.name, parents=[{"id": target_case.id}]))
        #         print("  Uploading", source_file.name)
        #     elif args.overwrite:
        #         f = pu.drive.CreateFile(
        #             dict(id=target_file.id, title=source_file.name, parents=[{"id": target_case.id}])
        #         )
        #         print("  Overwriting", source_file.name)
        #     else:
        #         print("  File", source_file.name, "already there.")
        #         continue
        #     f.SetContentFile(str(source_file))
        #     try:
        #         f.Upload()
        #     except httplib2.error.RedirectMissingLocation:
        #         print("Error while uploading: RedirectMissingLocation")




if __name__ == "__main__":
    main()