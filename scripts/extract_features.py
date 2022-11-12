import io
import multiprocessing
from contextlib import redirect_stdout

import nibabelio
from pathlib import Path

import numpy as np
import yaml
import tempfile
import nibabel
from radiomics import featureextractor, getTestCase

import torch
from torch import Tensor
from torch.nn import functional, Parameter

params = {
    'imageType': {
        'Original': {},
        'Wavelet': {},
        'LoG': {},
        'Square': {},
        'SquareRoot': {},
        'Logarithm': {},
        'Exponential': {},

    },
    'featureClass': {
        'firstorder': [],
        'glcm': [
            # 'Autocorrelation',
            # 'JointAverage',
            # 'ClusterProminence',
            # 'ClusterShade',
            # 'ClusterTendency',
            # 'Contrast',
            # 'Correlation',
            # 'DifferenceAverage',
            # 'DifferenceEntropy',
            # 'DifferenceVariance',
            # 'JointEnergy',
            # 'JointEntropy',
            # 'Imc1',
            # 'Imc2',
            # 'Idm',
            # 'Idmn',
            # 'Id',
            # 'Idn',
            # 'InverseVariance',
            # 'MaximumProbability',
            # 'SumEntropy',
            # 'SumSquares'
        ],
        'gldm': [],
        'glrlm': [],
        'glszm': [],
        'shape': [],
    },
    'setting': {
        'binWidth': 25,
        'interpolator': 'sitkBSpline',
        'label': 1,
        'resampledPixelSpacing': None,
        'weightingNorm': None
    }
}

# with open("/content/radiomics/temp_params.yaml", 'w') as f:
#     f.write(yaml.dump(params))


def with_neighbours(x: torch.Tensor, minimum=1, kernel_size=(9, 9, 3)):
    kx, ky, kz = kernel_size
    assert all(k % 2 == 1 for k in kernel_size)
    kernel = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=(kx // 2, ky // 2, kz // 2),
        device=x.device,
        dtype=torch.float32,
    )
    kernel.bias = Parameter(torch.tensor([0.1 - minimum]), requires_grad=False)
    kernel.weight = Parameter(torch.ones((1, 1, *kernel_size)), requires_grad=False)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    return torch.clamp(kernel(x.to(dtype=torch.float32)).squeeze(), 0, 1).to(dtype=x.dtype)


def set_difference(self, other):
    return torch.clamp((self - other), 0, 1).to(dtype=torch.int16)


def masks(segm: np.ndarray):
    segm = torch.tensor(segm, dtype=torch.int16)
    orig_liver = (segm == 1).to(dtype=torch.int16)
    tumor = (segm == 2).to(dtype=torch.int16)
    ext_tumor = with_neighbours(tumor, 3, (11, 11, 3))
    liver = set_difference(orig_liver, ext_tumor)
    perit = set_difference(orig_liver, liver)
    return liver.cpu().numpy(), perit.cpu().numpy(), tumor.cpu().numpy()

def extract(case_path):
    affine, _, _, _ = nibabelio.load_registration_data(case_path)
    # scan = utils.ndarray.load_registered(case_path, phase="v")
    pred = nibabelio._load_ndarray(case_path/"nnunet_prediction.nii.gz")
    # scans = load_registered_case()
    liver, perit, tumor = masks(pred)

    # params_file = "/content/radiomics/temp_params.yaml"

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)

        nibabel.save(
            nibabel.Nifti1Image(
                liver,
                affine=affine
            ),
            tmpdir / "liver_mask.nii.gz",
        )

        nibabel.save(
            nibabel.Nifti1Image(
                perit,
                affine=affine
            ),
            tmpdir / "perit_mask.nii.gz",
        )

        nibabel.save(
            nibabel.Nifti1Image(
                tumor,
                affine=affine
            ),
            tmpdir / "tumor_mask.nii.gz",
        )

        # tumor_mask = (pred == 2).astype(float)
        # nibabel.save(
        #     nibabel.Nifti1Image(
        #         tumor_mask,
        #         affine=affine
        #     ),
        #     tmpdir / "tumor_mask.nii.gz",
        # )
        extractor = featureextractor.RadiomicsFeatureExtractor(params)

        with open(case_path / "nnunet_features.csv", 'w') as feat:
            for phase in ['b', 'a', 'v', 't']:
                for mask in ['liver', 'perit', 'tumor']:
                    print(f"Extracting features in phase {phase} for area {mask}.")

                    imageName = case_path / f"registered_phase_{phase}.nii.gz"
                    maskName = tmpdir / f"{mask}_mask.nii.gz"

                    result = extractor.execute(str(imageName), str(maskName))
                    for key, val in result.items():
                        print("\t%s: %s" % (key, val))
                        feat.write(f"{phase},{mask},{key},{val}\n")

if __name__=="__main__":
    import path_explorer as px

    def work(case_name):
        if px.contains(sources/case_name, ["nnunet_features.csv"]):
            print(case_name, "already extrated.")
            return
        print(case_name, "working on.")
        try:
            with redirect_stdout(io.StringIO()):
                extract(sources / case_name)
        except Exception as err:
            print(case_name, err)

    sources = Path("../sources")

    with multiprocessing.Pool(20) as p:
        p.map(work, list(px.iter_registered(sources)))