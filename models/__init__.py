import torch
from batchrenorm import BatchRenorm3d
from torch import nn

from models import unet3dB
from models.funet import FunneledUNet
from models.unet3d import UNet3d


def get_model(model: str, **opts) -> nn.Module:
    if model == "segm882.2":
        net = UNet3d(
            channels=[4, 16, 32, 64],
            final_classes=3,
            complexity=2,
            norm=BatchRenorm3d,
        )
    elif model == "segm882.3":
        net = UNet3d(
            channels=[4, 16, 32, 64],
            final_classes=3,
            complexity=2,
            norm=nn.BatchNorm3d,
        )
    elif model == "segm882.4":
        net = UNet3d(
            channels=[4, 16, 32, 64],
            final_classes=3,
            complexity=2,
            norm=nn.Identity,
        )
    elif model == "segm882.5":
        net = UNet3d(
            channels=[4, 16, 32, 64],
            final_classes=3,
            complexity=2,
            norm=nn.Dropout3d,
        )
    elif model == "segm882.6":
        net = unet3dB.UNet3d(
            channels=[4, 32, 64, 128],
            final_classes=3,
            complexity=2,
            down_dropout=None,
            checkpointing=False,
        )
    elif model == "segm882.7":
        net = unet3dB.UNet3d(
            channels=[4, 32, 64, 128],
            final_classes=3,
            complexity=2,
            down_dropout=None,
            bottom_normalization=None,
            checkpointing=False,
        )
    elif model == "segm882.8":
        net = unet3dB.UNet3d(
            channels=[4, 32, 64, 128],
            final_classes=3,
            complexity=2,
            down_dropout=None,
            bottom_normalization=lambda n: nn.InstanceNorm3d(n, momentum=0.9, affine=False),
            checkpointing=False,
        )
    elif model == "segm441.0":
        net = UNet3d(
            channels=[7, 20, 40, 80],
            complexity=1,
            final_classes=3,
        )
    elif model == "segm441.1":
        net = UNet3d(
            channels=[7, 20, 40, 80],
            complexity=2,
            final_classes=3,
        )
    elif model == "segm.0":
        net = FunneledUNet(
            channels=[7, 16, 32, 48, 64],
            wafer_size=opts["wafer_size"],
            final_classes=3,
        )
    elif model == "segm.1":
        net = FunneledUNet(
            channels=[7, 16, 32, 48, 64],
            wafer_size=opts["wafer_size"],
            final_classes=3,
            bypass=[4, 5, 6],
        )
    elif model == "segm.2":
        net = FunneledUNet(
            channels=[7, 16, 32, 48, 64],
            wafer_size=opts["wafer_size"],
            final_classes=3,
            fullbypass=[4, 5, 6],
            final_activation=nn.Tanh(),
            clamp=(-100, 300),
        )
    else:
        net = None

    try:
        net.load_state_dict(torch.load(opts["saved_models"] / f'{model}.pth'))
    except FileNotFoundError:
        pass
    return net
