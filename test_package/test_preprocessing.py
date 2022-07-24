from __future__ import annotations

import argparse
import tempfile
import pytest
import unittest
from pathlib import Path

import scripts.dicom2nifti
import scripts.niftyreg
import scripts.pyelastix
from dataset.path_explorer import iter_dicom, iter_original, iter_registered


@pytest.mark.skipif(not (Path(__file__).parent / "sources").exists(), reason="requires test_package/sources")
class TestPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @pytest.mark.skipif(not Path("/usr/bin/gdcmconv").exists(), reason="requires libgdcm-tools")
    def test_conversion(self):
        self.assertNotEqual(list(iter_dicom(Path(__file__).parent / "sources")), [])
        source = Path(__file__).parent / "sources" / list(iter_dicom(Path(__file__).parent / "sources"))[0]
        target = self.tmp_path

        opts = argparse.Namespace(
            sources=source,
            outputs=target,
            overwrite=False
        )
        completed = scripts.dicom2nifti.main(opts)
        self.assertEqual(completed, [source])

    @pytest.mark.skipif(not Path("/usr/local/bin/reg_f3d").exists(), reason="requires nifty-reg")
    def test_niftyreg(self):
        source = Path(__file__).parent / "sources"
        target = self.tmp_path
        self.assertNotEqual(list(iter_original(source)), [])
        opts = argparse.Namespace(
            niftybin="/usr/local/bin",
            sources=source,
            outputs=target,
            overwrite=True
        )
        completed = scripts.niftyreg.main(opts)
        self.assertEqual(completed, list(iter_original(source)))

    @pytest.mark.skipif(not Path("/usr/bin/elastix").exists(), reason="requires elastix")
    def test_pyelastix(self):
        source = Path(__file__).parent / "sources"
        target = self.tmp_path
        self.assertNotEqual(list(iter_original(source)), [])

        opts = argparse.Namespace(
            sources=source,
            outputs=target,
            overwrite=True
        )

        completed = scripts.pyelastix.main(opts)
        self.assertEqual(completed, list(iter_original(source)))
