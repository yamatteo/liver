import argparse
from pathlib import Path
from typing import List, Optional

from options import defaults
from rich.console import Console
from rich.progress import track
from rich.pretty import pprint

from options import defaults

console = Console()


def is_dicomdir(path: Path) -> bool:
    return "DICOMDIR" in [item.name for item in path.iterdir()]


def discover_dicomdirs(path: Path, relative_to_path: Optional[Path] = None) -> List[Path]:
    if relative_to_path is None:
        relative_to_path = path
    if path.is_dir():
        if is_dicomdir(path):
            return [path.relative_to(relative_to_path)]
        else:
            return [
                subdir
                for subpath in path.iterdir()
                for subdir in discover_dicomdirs(subpath, relative_to_path)
            ]
    else:
        return []


def get_args():
    parser = argparse.ArgumentParser()
    return dict(defaults, **vars(parser.parse_args()))


if __name__ == "__main__":
    opts = get_args()
    console.print(f"In the given path ({opts['sources']}) there are these dicom dirs:")
    for path in discover_dicomdirs(opts["sources"]):
        console.print("  ", path)
