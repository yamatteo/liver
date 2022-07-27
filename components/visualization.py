import numpy as np
import seaborn
import seaborn_image
from ipywidgets import Dropdown, IntSlider, Output
from matplotlib import pyplot as plt
import dataset.ndarray


def build_tv(state, inject, project, biject):
    seaborn.set(rc={'figure.figsize': (10, 10)})
    channels = Dropdown(
        description="What to see:",
        value=None,
        options=[
            ("Rgb scan", ("rgb",)),
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
            ("Segmentation", ("segm",))
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
                content = load_segm(state.case_path)
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
    data, _ = dataset.ndarray.load_original(case_path, phase=phase)
    # orig.shape is [X, Y, Z]
    data = data.clip(0, 255) / 255
    return np.stack([data, data, data])  # shape is [RGB, X, Y, Z]


def load_regcheck(case_path, phase):
    reg, _ = dataset.ndarray.load_registered(case_path, phase=phase)
    orig, _ = dataset.ndarray.load_original(case_path, phase=phase)
    origv, _ = dataset.ndarray.load_original(case_path, phase="v")
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
    data, _ = dataset.ndarray.load_registered(case_path, phase=phase)
    # orig.shape is [X, Y, Z]
    data = data.clip(0, 255) / 255
    return np.stack([data, data, data])  # shape is [RGB, X, Y, Z]


def load_rgbscan(case_path):
    white, _ = dataset.ndarray.load_registered(case_path, phase="b")
    red, _ = dataset.ndarray.load_registered(case_path, phase="a")
    green, _ = dataset.ndarray.load_registered(case_path, phase="v")
    blue, _ = dataset.ndarray.load_registered(case_path, phase="t")
    white = white.clip(0, 255) / 255
    red = red.clip(0, 255) / 255
    green = green.clip(0, 255) / 255
    blue = blue.clip(0, 255) / 255
    red = (white + red).clip(0, 1)
    green = (white + green).clip(0, 1)
    blue = (white + blue).clip(0, 1)
    return np.stack([red, green, blue])  # shape is [RGB, X, Y, Z]


def load_segm(case_path):
    white, _ = dataset.ndarray.load_registered(case_path, phase="v")
    segm = dataset.ndarray.load_segm(case_path)
    assert white.shape[2] == segm.shape[2], "Segmentation and registered phase v have different height"
    white = white.clip(0, 255) / 255
    red = (segm == 1).astype(float)
    green = (segm == 2).astype(float)
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
