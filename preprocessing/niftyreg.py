import os
from pathlib import Path
from rich.console import Console
from tempfile import NamedTemporaryFile

import dataset.ndarray

console = Console()


def register_case(source_path: Path, target_path: Path, niftybin: Path):
    """Register 4-phase scans with respect to phase v."""
    case_name = source_path.name
    niftybin = Path(niftybin)
    console.print(f"[bold black]{case_name}.[/bold black] Working (it is a long process) ...")

    # Store the phase v as is, with its affine matrix
    v_phase = dataset.ndarray.load_niftiimage(path=source_path / "original_phase_v.nii.gz")
    # v_phase = nibabel.load(source_path / "original_phase_v.nii.gz")

    dataset.ndarray.save_original(image=v_phase, path=source_path, phase='v')
    # nibabel.save(v_phase, str(target_path / "registered_phase_v.nii.gz"))
    with NamedTemporaryFile("wb", suffix=".nii") as cpp_file:
        for phase in ["b", "a", "t"]:
            # Run the registering library
            with open(target_path / f"registration_{phase}.log", "w") as logfile:
                niftireg_log = os.popen(
                    f"{niftybin / 'reg_f3d'} "
                    f"-ref {source_path / 'original_phase_v.nii.gz'} "
                    f"-flo {source_path / f'original_phase_{phase}.nii.gz'} "
                    f"-res {target_path / f'registered_phase_{phase}.nii.gz'} "
                    f"-cpp {cpp_file.name} "
                    f"-maxit 50 "
                ).read()
                logfile.write(niftireg_log)
        console.print(
            f"{' ' * len(case_name)}  "
            f"Registered images saved in {target_path.absolute()}."
        )
