from pathlib import Path

import nibabel
import numpy as np

case = Path(input("Insert case dir:"))

image = nibabel.load(case / f"segmentation.nii.gz")
nibabel.save(
    nibabel.Nifti1Image(
        np.flip(np.array(image.dataobj), axis=1),
        affine=image.affine @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0,  0, 1]])
    ),
    case / f"segmentation.nii.gz",
)