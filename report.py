import random

import numpy as np
import rich
import wandb

# from subclass_tensors import *
# from utils.image_generation import get_white, rgb_sample


def __mute(*_, **__):
    pass


def __warn(*args, **kwargs):
    kwargs["style"] = "bold underline" + kwargs.get("style", "")
    console.print(*args, **kwargs)


def __print(*args, **kwargs):
    console.print(*args, **kwargs)


def init(*, backend: str = "wandb", level: str = "info", project: str, entity: str):
    global append, debug, info, sample
    if level == "debug":
        info = debug = __print
    elif level == "info":
        info = __print
    if backend == "wandb":
        append = __append
        sample = __sample
        return wandb.init(project=project, entity=entity)
    else:
        return type('mock', (object,), {"finish": lambda: None})


def __append(items: dict):
    wandb.log(items)


def __sample(scan: np.ndarray, pred: np.ndarray, segm: np.ndarray):
    # scan.shape is [N, C, X, Y, Z], segm and pred are [N, X, Y, Z]
    n = random.randint(0, scan.shape[0] - 1)
    z = random.randint(0, scan.shape[4] - 1)

    scan = np.clip(scan[n, 2, :, :, z], 0, 255)
    segm = segm[n, :, :, z]
    pred = pred[n, :, :, z]

    class_labels = {
        0: "background",
        1: "liver",
        2: "tumor"
    }

    return wandb.Image(scan, masks={
        "predictions": {
            "mask_data": pred,
            "class_labels": class_labels
        },
        "ground_truth": {
            "mask_data": segm,
            "class_labels": class_labels
        },
    })


def status(*args, **kwargs):
    return console.status(*args, **kwargs)


console = rich.console.Console()
info = __mute
debug = __mute
warn = __warn
sample = __mute
append = __mute
