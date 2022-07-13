import random

import rich
import wandb

from subclass_tensors import *
from utils.image_generation import get_white, rgb_sample


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


def __sample(scan: FloatScanBatch, pred: FloatSegmBatch, segm: FloatSegmBatch):
    n = random.randint(0, scan.size(0) - 1)
    z = random.randint(0, scan.size(4) - 1)
    image = get_white(scan, n=n, z=z).unsqueeze(2)
    liver_image = rgb_sample(scan, pred, segm, mode=("liver_error", "liver", "background"), n=n, z=z, data_format="HWC")
    tumor_image = rgb_sample(scan, pred, segm, mode=("tumor_error", "tumor", "background"), n=n, z=z, data_format="HWC")

    image = (image / 255).numpy()
    liver_image = (liver_image / 255).numpy()
    tumor_image = (tumor_image / 255).numpy()

    class_labels = {
        0: "background",
        1: "liver",
        2: "tumor"
    }
    pred_slice = pred.get_slice(n, z).numpy()
    segm_slice = segm.get_slice(n, z).numpy()

    return (
        wandb.Image(image, masks={
            "predictions": {
                "mask_data": pred_slice,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": segm_slice,
                "class_labels": class_labels
            },
        }),
        wandb.Image(liver_image),
        wandb.Image(tumor_image)
    )


def status(*args, **kwargs):
    return console.status(*args, **kwargs)


console = rich.console.Console()
info = __mute
debug = __mute
warn = __warn
sample = __mute
append = __mute
