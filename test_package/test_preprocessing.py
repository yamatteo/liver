import argparse
import tempfile
import pytest
import unittest
from pathlib import Path

from scripts import dicom2nifti, register_phases
from dataset.path_explorer import iter_dicom, iter_original, iter_trainable


@pytest.mark.skipif(not (Path(__file__).parent / "sources").exists(), reason="requires test_package/sources")
class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @pytest.mark.skipif(not Path("/usr/bin/gdcmconv").exists(), reason="requires libgdcm-tools")
    def test_conversion(self):
        source = Path(__file__).parent / "sources"
        target = self.tmp_path
        N = len(list(iter_dicom(source)))
        self.assertNotEqual(N, 0)
        self.assertEqual(len(list(iter_original(target))), 0)
        self.assertEqual(len(list(iter_trainable(target))), 0)

        opts = argparse.Namespace(
            sources=source,
            outputs=target,
            overwrite=False
        )
        dicom2nifti.main(opts)
        self.assertEqual(len(list(iter_original(target))), N)

    @pytest.mark.skipif(not Path("/usr/local/bin/reg_f3d").exists(), reason="requires nifty-reg")
    def test_preprocessing(self):
        target = self.tmp_path
        opts = argparse.Namespace(
            niftybin="/usr/local/bin",
            sources=target,
            outputs=target,
            overwrite=False
        )
        N = len(list(iter_original(target)))
        register_phases.main(opts)
        for case in iter_original(target):
            (target / case / "segmentation.nii.gz").touch()
        self.assertEqual(len(list(iter_trainable(target))), N)
