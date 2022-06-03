import importlib

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None

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
