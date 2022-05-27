import os
import sys

import dotenv

sys.path.append(os.getcwd())
from pathlib import Path

import torch

from models.multi_unet import UNet
from segmentation.train import train_net

dotenv.load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer_path = Path(os.getenv("TENSORBOARD"))
data_path = Path(os.getenv("OUTPUTS"))
model = UNet(
    dims="3d",
    channels=[4, 32, 64, 128]
)

train_net(device=device, writer_path=writer_path, data_path=data_path, model=model)