# -*- coding: utf-8 -*-
import random

# import idr_torch as idr
import numpy as np
import rich
import wandb

muted = False


def init(*, project: str = "liver-tumor-detection", entity: str = "yamatteo", id: str = None, config: dict = None,
         group: str = None, mute: bool = False):
    global muted
    if mute:
        muted = True
        return type('mock', (object,), {"finish": lambda: None})
    else:
        return wandb.init(project=project, entity=entity, mode="offline", id=id, group=group, config=config)


def print(*args, once=False):
    # if idr.rank == 0 or not once:
    #     rich.print(f"{idr.rank:02}> ", *args)
    rich.print(*args)


def append(items: dict, commit=True):
    if muted:
        return
    wandb.log(items, commit=commit)


def sample(scan: np.ndarray, pred: np.ndarray, segm: np.ndarray):
    if muted:
        return
    # scan.shape is [N, C, X, Y, Z], segm and pred are [N, X, Y, Z]
    n = random.randint(0, scan.shape[0] - 1)
    z = random.randint(0, scan.shape[4] - 1)

    white = np.clip(scan[n, 2, :, :, z], 0, 255)
    red = white + 60 * (pred[n, :, :, z] == 1)
    green = white + 60 * (pred[n, :, :, z] == 2)
    blue = white
    img = np.clip(np.stack([red, green, blue], axis=-1), 0, 255)

    return wandb.Image(img)

# def log_sample(scan: np.ndarray, pred: np.ndarray, segm: np.ndarray):
#     # scan.shape is [N, C, X, Y, Z], segm and pred are [N, X, Y, Z]
#     n = random.randint(0, scan.shape[0] - 1)
#     z = random.randint(0, scan.shape[4] - 1)
#
#     white = np.clip(scan[n, 2, :, :, z], 0, 255)
#     red = white + 60*(pred[n, :, :, z] == 1)
#     green = white + 60*(pred[n, :, :, z] == 2)
#     blue = white
#     img = np.clip(np.stack([red, green, blue], axis=-1), 0, 255)
#
#     wandb.log({"samples": wandb.Image(img)}, commit=False)
# segm = segm[n, :, :, z]
# pred = pred[n, :, :, z]

# class_labels = {
#     0: "background",
#     1: "liver",
#     2: "tumor"
# }

# return wandb.Image(scan, masks={
#     "predictions": {
#         "mask_data": pred,
#         "class_labels": class_labels
#     },
#     "ground_truth": {
#         "mask_data": segm,
#         "class_labels": class_labels
#     },
# })

# def append_deformation_samples(scan: np.ndarray, segm: np.ndarray, def_scan: np.ndarray, def_segm: np.ndarray, k:int = 1):
#     # scan.shape is [C, X, Y, Z], segm is [X, Y, Z]
#     if env.backend != "wandb":
#         return None
#     table = wandb.Table(columns=['original', 'deformed'])
#     class_labels = {
#         0: "background",
#         1: "liver",
#         2: "tumor"
#     }
#
#     for z in random.sample(range(scan.shape[-1]), k):
#         slice_scan = np.clip(scan[2, :, :, z], 0, 255)
#         slice_segm = segm[:, :, z]
#         slice_def_scan = np.clip(def_scan[2, :, :, z], 0, 255)
#         slice_def_segm = def_segm[:, :, z]
#
#         table.add_data(
#             wandb.Image(slice_scan, masks = {
#                 "segmentation" : {
#                     "mask_data" : slice_segm,
#                     "class_labels" : class_labels
#                 },
#             }),
#             wandb.Image(slice_def_scan, masks = {
#                 "segmentation" : {
#                     "mask_data" : slice_def_segm,
#                     "class_labels" : class_labels
#                 },
#             }),
#         )
#
#     wandb.log({"deformation_samples" : table}, commit=False)
