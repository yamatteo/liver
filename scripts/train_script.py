import importlib
import os
import shutil

import dotenv
import torch

import  report
from train import train_cycle

try:
    wandb = importlib.import_module("wandb")
except ImportError:
    wandb = None
from adabelief_pytorch import AdaBelief
from pathlib import Path
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from dataset import BufferDataset2 as BufferDataset
from models.multi_unet import UNet
from utils.generators import train_slices


# wandb.init(project="liver-tumor-detecton", entity="yamatteo")
# wandb.config = {
#   "learning_rate": 1e-4,
#   "epochs": 10,
#   "batch_size": 1
# }

console = Console()
report.init(backend="none")
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

slice_shape = (64, 64, 8)

shutil.rmtree(Path(writer_path), ignore_errors=True)

dataset = BufferDataset(
    generator=train_slices(data_path, slice_shape),
    buffer_size=50,
    train_to_valid_odds=10,
    valid_buffer_size=5,
    batch_size=5
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

# writer = SummaryWriter(str(writer_path))
epochs = 10

train_cycle(model, epochs=epochs, dataset=dataset, optimizer=optimizer, device=device, train_drop=5)
