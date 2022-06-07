from __future__ import annotations

import os
import shutil

import dotenv
from rich.console import Console

from utils.generators import cases
from utils.path_explorer import criterion

console = Console()
dotenv.load_dotenv()

target = os.getenv("TARGET")

for case in cases(os.getenv("OUTPUTS"), criterion(bundle=True)):
    target_case = target / case.relative_to(os.getenv("OUTPUTS"))
    target_case.mkdir()
    shutil.copy(case / "train_bundle.nii.gz", target_case / "train_bundle.nii.gz")
