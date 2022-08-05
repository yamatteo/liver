import tempfile
import unittest
from pathlib import Path

from utils.path_explorer import iter_dicom, iter_original, iter_trainable


class TestPathExplorerIterators(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = tempfile.TemporaryDirectory()
        base = Path(self.base_dir.name)
        for n in range(12):
            (base / f"Mock{n:03}").mkdir()
            if n % 2 == 0:
                (base / f"Mock{n:03}" / "DICOMDIR").touch()
            if n % 3 == 0:
                for phase in ["b", "a", "v", "t"]:
                    (base / f"Mock{n:03}" / f"original_phase_{phase}.nii.gz").touch()
            if n % 4 == 0:
                for phase in ["b", "a", "v", "t"]:
                    (base / f"Mock{n:03}" / f"registered_phase_{phase}.nii.gz").touch()
                (base / f"Mock{n:03}" / f"segmentation.nii.gz").touch()

    def tearDown(self) -> None:
        self.base_dir.cleanup()

    def test_iter_dicom(self):
        base = Path(self.base_dir.name)
        set_a = set(iter_dicom(base))
        set_b = {Path(f"Mock{n:03}") for n in range(12) if n % 2 == 0}
        self.assertSetEqual(set_a, set_b)

    def test_iter_original(self):
        base = Path(self.base_dir.name)
        set_a = set(iter_original(base))
        set_b = {Path(f"Mock{n:03}") for n in range(12) if n % 3 == 0}
        self.assertSetEqual(set_a, set_b)

    def test_iter_trainable(self):
        base = Path(self.base_dir.name)
        set_a = set(iter_trainable(base))
        set_b = {Path(f"Mock{n:03}") for n in range(12) if n % 4 == 0}
        self.assertSetEqual(set_a, set_b)
