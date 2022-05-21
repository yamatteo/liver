from __future__ import annotations

from pathlib import Path
from typing import Iterator, Callable


def cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    """Iterate over 'base_path' folders (recursively) that satisfy 'accepted_dir'"""
    base_path = Path(base_path).expanduser().resolve()
    yield from _cases(base_path, accepted_dir=accepted_dir)


def _cases(base_path: Path | str, accepted_dir: Callable[[Path], bool]) -> Iterator[Path]:
    if base_path.is_dir():
        if accepted_dir(base_path):
            yield base_path
        else:
            for sub_path in base_path.iterdir():
                yield from _cases(base_path / sub_path, accepted_dir=accepted_dir)
