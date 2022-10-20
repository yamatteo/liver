import numpy as np
import seaborn
import seaborn_image
import torch
from ipywidgets import Dropdown, IntSlider, Output
from matplotlib import pyplot as plt
from torch.nn import Parameter
from torch.nn.functional import l1_loss

import utils.ndarray


def build_tv(state, inject, project, biject):
    seaborn.set(rc={'figure.figsize': (10, 10)})
    channels = Dropdown(
        description="What to see:",
        value=None,
        options=[
            ("Features masks", ("masks",)),
            ("Rgb scan", ("rgb",)),
            ("Segmentation", ("segm",)),
            ("Prediction", ("pred",)),
            ("LiverError", ("lerr",)),
            ("TumorError", ("terr",)),
            ("Check registration phase b", ("rcheck", "b")),
            ("Check registration phase a", ("rcheck", "a")),
            ("Check registration phase t", ("rcheck", "t")),
            ("Original phase b", ("orig", "b")),
            ("Original phase a", ("orig", "a")),
            ("Original phase v", ("orig", "v")),
            ("Original phase t", ("orig", "t")),
            ("Registered phase b", ("reg", "b")),
            ("Registered phase a", ("reg", "a")),
            ("Registered phase v", ("reg", "v")),
            ("Registered phase t", ("reg", "t")),
        ]
    )
    z_slider = IntSlider(description="Z slice:")
    tv = Output()

    def loaded_content(_event):
        seaborn.set(rc={'figure.figsize': (10, 10)})
        value = channels.value
        try:
            command, *args = value
            if command == "orig":
                phase = args[0]
                content = load_original(state.case_path, phase)
            elif command == "reg":
                phase = args[0]
                content = load_registered(state.case_path, phase)
            elif command == "rcheck":
                phase = args[0]
                content = load_regcheck(state.case_path, phase)
            elif command == "rgb":
                content = load_rgbscan(state.case_path)
            elif command == "segm":
                content = load_segm(state.case_path, "segmentation")
            elif command == "pred":
                content = load_segm(state.case_path, "prediction")
            elif command == "lerr":
                content = load_error(state.case_path, 1)
            elif command == "terr":
                content = load_error(state.case_path, 2)
            elif command == "masks":
                content = load_masks(state.case_path)
            z_slider.max = content.shape[-1] - 1
        except:
            content = None
        state.loaded_content = content

    state.observe(loaded_content, "case_path")
    channels.observe(loaded_content, "value")

    @tv.capture(clear_output=True, wait=True)
    def func(event):
        try:
            z = z_slider.value
            content = state.loaded_content
            plot(content, z)
        except Exception as err:
            print("No output.")
            print(err)

    z_slider.observe(func, "value")
    channels.observe(func, "value")

    return channels, z_slider, tv


def load_original(case_path, phase):
    data, _ = utils.ndarray.load_original(case_path, phase=phase)
    # orig.shape is [X, Y, Z]
    data = data.clip(0, 255) / 255
    return np.stack([data, data, data])  # shape is [RGB, X, Y, Z]


def load_regcheck(case_path, phase):
    reg = utils.ndarray.load_registered(case_path, phase=phase)
    orig, _ = utils.ndarray.load_original(case_path, phase=phase)
    origv, _ = utils.ndarray.load_original(case_path, phase="v")
    # shapes are [X, Y, Z], possibly different Z for orig and origv
    zmax = min(orig.shape[2], origv.shape[2])
    reg = reg[..., :zmax].clip(0, 255) / 255
    orig = orig[..., :zmax].clip(0, 255) / 255
    origv = origv[..., :zmax].clip(0, 255) / 255
    red = (0.6*origv + 1.4*orig).clip(0, 1)
    green = (0.6*origv + 1.4*reg).clip(0, 1)
    blue = origv
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]


def load_registered(case_path, phase):
    data = utils.ndarray.load_registered(case_path, phase=phase)
    # orig.shape is [X, Y, Z]
    data = data.clip(0, 255) / 255
    return np.stack([data, data, data])  # shape is [RGB, X, Y, Z]


def load_rgbscan(case_path):
    white = utils.ndarray.load_registered(case_path, phase="b")
    red = utils.ndarray.load_registered(case_path, phase="a")
    green = utils.ndarray.load_registered(case_path, phase="v")
    blue = utils.ndarray.load_registered(case_path, phase="t")
    white = white.clip(0, 255) / 255
    red = red.clip(0, 255) / 255
    green = green.clip(0, 255) / 255
    blue = blue.clip(0, 255) / 255
    red = (white + red).clip(0, 1)
    green = (white + green).clip(0, 1)
    blue = (white + blue).clip(0, 1)
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]


def load_segm(case_path, what: str = "segmentation"):
    white = utils.ndarray.load_registered(case_path, phase="v")
    segm = utils.ndarray.load_segm(case_path, what)
    assert white.shape[2] == segm.shape[2], "Segmentation and registered phase v have different height"
    white = white.clip(0, 255) / 255
    red = (segm == 1).astype(float)
    green = (segm == 2).astype(float)
    blue = np.zeros_like(segm)
    red = (white + red).clip(0, 1)
    green = (white + green).clip(0, 1)
    blue = (white + blue).clip(0, 1)
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]

# Masks

def with_neighbours(x: torch.Tensor, minimum = 1, kernel_size = (9, 9, 3)):
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
    return torch.clamp(kernel(x.unsqueeze(0).to(dtype=torch.float32)).squeeze(0), 0, 1).to(dtype=x.dtype)

def set_difference(self, other):
    return torch.clamp((self - other), 0, 1).to(dtype=torch.int16)

def masks(segm: torch.Tensor):
    orig_liver = (segm == 1).to(dtype=torch.int16)
    tumor = (segm == 2).to(dtype=torch.int16)
    ext_tumor = with_neighbours(tumor, 1, (7, 7, 1))
    liver = set_difference(orig_liver, ext_tumor)
    perit = set_difference(orig_liver, liver)
    return liver, perit, tumor

def load_masks(case_path):
    white = utils.ndarray.load_registered(case_path, phase="v")
    pred = utils.ndarray.load_segm(case_path, "prediction")
    liver, perit, tumor = masks(pred)
    assert white.shape[2] == pred.shape[2], "Segmentation and registered phase v have different height"
    white = white.clip(0, 255) / 255
    red = 0.6 * liver.cpu().numpy().astype(float)
    green = 0.6 * tumor.cpu().numpy().astype(float)
    blue = 0.6 * perit.cpu().numpy().astype(float)
    red = (white + red).clip(0, 1)
    green = (white + green).clip(0, 1)
    blue = (white + blue).clip(0, 1)
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]


def load_error(case_path, klass: int):
    white = utils.ndarray.load_registered(case_path, phase="v")
    segm = utils.ndarray.load_segm(case_path)
    pred = utils.ndarray.load_segm(case_path, "prediction")
    assert white.shape[2] == segm.shape[2], "Segmentation and registered phase v have different height"
    white = white.clip(0, 255) / 255
    red = l1_loss((segm == klass).astype(float), (pred == klass).astype(float))
    green = (segm == klass).astype(float)
    blue = np.zeros_like(segm)
    red = (white + red).clip(0, 1)
    green = (white + green).clip(0, 1)
    blue = (white + blue).clip(0, 1)
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]


def plot(rgb_x_y_z: np.ndarray, z: int):
    # shape is [RGB, X, Y, Z]
    y_x_rgb = np.transpose(rgb_x_y_z[..., z])
    plt.show(seaborn_image.imgplot(
        y_x_rgb,
        cbar=False,
    ))
