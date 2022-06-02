import os
import sys

import dotenv

sys.path.append(os.getcwd())
from pathlib import Path

import torch
from rich.console import Console

from models.multi_unet import UNet
from segmentation.train import train_net

console = Console()
dotenv.load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
console.print(f"Running on {device}")
writer_path = Path(os.getenv("TENSORBOARD"))
data_path = Path(os.getenv("OUTPUTS"))
model = UNet(
    dims="3d",
    channels=[4, 32, 64, 128]
)
try:
    model.load_state_dict(torch.load(Path(os.getenv("SAVED_MODELS")) / "last_checkpoint.pth"))
except FileNotFoundError:
    pass

train_net(device=device, writer_path=writer_path, data_path=data_path, model=model, slice_shape=eval(os.getenv("SLICE_SHAPE")))
