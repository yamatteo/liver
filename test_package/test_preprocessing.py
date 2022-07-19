import argparse
import tempfile
import unittest
import warnings
from pathlib import Path

from scripts import dicom2nifti, register_phases
from utils.path_explorer import iter_dicom, iter_original, iter_trainable


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing(self):
        source = Path(__file__).parent / "sources"
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir)
            N = len(list(iter_dicom(source)))
            self.assertNotEqual(N, 0)
            self.assertEqual(len(list(iter_original(target))), 0)
            self.assertEqual(len(list(iter_trainable(target))), 0)

            opts = argparse.Namespace(
                sources=source,
                outputs=target,
                overwrite=False
            )
            try:
                dicom2nifti.main(opts)
            except ValueError as err:
                raise ValueError("Is libgdcm-tools installed?") from err
            self.assertEqual(len(list(iter_original(target))), N)

            opts = argparse.Namespace(
                niftybin="/usr/local/bin",
                sources=target,
                outputs=target,
                overwrite=False
            )
            self.assertTrue(Path("/usr/local/bin/reg_f3d").exists())
            register_phases.main(opts)
            for case in iter_original(target):
                (target / case / "segmentation.nii.gz").touch()
            self.assertEqual(len(list(iter_trainable(target))), N)

