from __future__ import annotations
import argparse
from pathlib import Path
from rich.console import Console

import utils.ndarray
import preprocessing.pyelastix
from utils.path_explorer import iter_original


def main(opts):
    console = Console()
    console.print("[bold orange3]Registering images:[/bold orange3]")
    completed = []
    for case_path in iter_original(opts.sources):
        source_path = opts.sources / case_path
        target_path = opts.outputs / case_path
        target_path_is_complete = all(
            (target_path / f"registered_phase_{phase}.nii.gz").exists()
            for phase in ["b", "a", "v", "t"]
        )
        if opts.overwrite or not target_path_is_complete:
            target_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[bold black]{case_path}.[/bold black] Registering with pyelastix...")
            regs, matrix, bottom, top, height = preprocessing.pyelastix.regs_dict_from(source_path)
            utils.ndarray.save_registereds(
                regs,
                path=target_path,
                affine=matrix,
                bottom=bottom,
                top=top,
                height=height
            )
            completed.append(case_path)
        else:
            console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register phases to the fixed 'v' phase.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    main(parser.parse_args())
