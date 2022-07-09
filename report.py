import random

import rich
import wandb

from tensors import FloatScanBatch, FloatSegmBatch
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
    n = random.randint(0, scan.size("N") - 1)
    z = random.randint(0, scan.size("D") - 1)
    image = get_white(scan, n=n, z=z).numpy()
    liver_image = rgb_sample(scan, pred, segm, mode=("liver_error", "liver", "background"), n=n, z=z, data_format="HWC")
    tumor_image = rgb_sample(scan, pred, segm, mode=("tumor_error", "tumor", "background"), n=n, z=z, data_format="HWC")
    class_labels = {
        0: "background",
        1: "liver",
        2: "tumor"
    }
    pred_mask = pred.as_int().get_plane(n=n, z=z).numpy()
    segm_mask = segm.as_int().get_plane(n=n, z=z).numpy()

    return (
        wandb.Image(image, masks={
            "predictions": {
                "mask_data": pred_mask,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": segm_mask,
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
