import importlib
import random

from utils.image_generation import get_white, rgb_sample

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None

from tensors import *

module_backend = "none"


def init(backend: str = "wandb"):
    global module_backend
    if backend == "wandb":
        assert wandb is not None, "Can't import wandb"
    module_backend = backend


def append(items: dict):
    if module_backend == "wandb":
        wandb.log(items)
    else:
        print(items)


def sample(scan: Tensor, pred: Tensor, segm: Tensor):
    if module_backend == "wandb":
        n = random.randint(0, scan.size(0) - 1)
        z = random.randint(0, scan.size(4) - 1)
        image = get_white(scan)[n, 0, :, :, z].unsqueeze(-1).numpy()
        liver_image = rgb_sample(scan, pred, segm, mode=("pred_liver", "liver", "background")).permute(1, 2, 0)
        tumor_image = rgb_sample(scan, pred, segm, mode=("pred_tumor", "tumor", "background")).permute(1, 2, 0)
        class_labels = {
            0: "background",
            1: "liver",
            2: "tumor"
        }
        pred_mask = torch.argmax(pred, dim=1)[n, :, :, z].numpy()
        segm_mask = torch.argmax(segm, dim=1)[n, :, :, z].numpy()

        # console.print(f"pre  liver weight {torch.sum(segm[n, 1, :, :, z])}")
        # console.print(f"post liver weight {torch.sum(torch.argmax(segm, dim=1)[n, :, :, z] == 1)}")
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
    else:
        return "sample_image"
