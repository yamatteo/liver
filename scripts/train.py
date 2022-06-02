import os
import shutil
import sys

import dotenv
from adabelief_pytorch import AdaBelief
from torch.utils.tensorboard import SummaryWriter

from utils.generators import train_slices

sys.path.append(os.getcwd())
from pathlib import Path

import torch
from rich.console import Console

from models.multi_unet import UNet
from segmentation.train import train_cycle

from dataset import BufferDataset2 as BufferDataset

console = Console()
dotenv.load_dotenv()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
console.print(f"Running on {device}")
writer_path = Path(os.getenv("TENSORBOARD"))
data_path = Path(os.getenv("OUTPUTS"))
model = UNet(
    dims="3d",
    channels=[4, 16, 32, 64, 128]
)
try:
    model.load_state_dict(torch.load(Path(os.getenv("SAVED_MODELS")) / "last_checkpoint.pth"))
except FileNotFoundError:
    pass

slice_shape = (512, 512, 8)
shutil.rmtree(Path(writer_path), ignore_errors=True)
dataset = BufferDataset(
    generator=train_slices(data_path, slice_shape),
    buffer_size=50,
    train_to_valid_odds=10,
    valid_buffer_size=5
)
optimizer = AdaBelief(
    model.parameters(),
    lr=1e-4,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
    print_change_log=False,
)

writer = SummaryWriter(str(writer_path))
train_cycle(model, epochs=1000, dataset=dataset, optimizer=optimizer, writer=writer, device=device)

