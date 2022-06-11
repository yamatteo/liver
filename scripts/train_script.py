import argparse
import os
from pathlib import Path

import dotenv
import torch
import wandb
from adabelief_pytorch import AdaBelief
from torch.utils.data import DataLoader

import report
# from dataset import BufferDataset2 as BufferDataset
from buffer_dataset import BufferDataset, split_datasets
from models.multi_unet import UNet
from train import train_cycle
from utils import generators

run = report.init(project="liver-tumor-detecton", entity="yamatteo", backend="none", level="debug")

dotenv.load_dotenv()
report.debug("Loaded enviromental variables:", dict(dotenv.dotenv_values()))
data_path = Path(os.getenv("OUTPUTS"))
models_path = Path(os.getenv("SAVED_MODELS"))

opts = argparse.Namespace(
    batch_size=4,
    buffer_size=20,
    epochs=400,
    learning_rate=1e-5,
    channels=[4, 32, 64, 128],
    resume=True,
    slice_shape=(16, 16, 8),
    train_to_valid_ratio=10,
)
wandb.config = vars(opts)  # Maybe these values are stored by wandb, maybe useful for hyperparameters search
report.debug("Using config options:", vars(opts))

# Ensure directory is present
if not models_path.exists():
    report.warn(f"Directory to load/save models does not exist. Making a new one at {models_path}.")
    models_path.mkdir()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
report.info(f"Running on {device}")

model = UNet(
    channels=opts.channels
)
report.debug("Model is:", model)

if opts.resume is False:
    report.info("Starting with a new model.")
elif opts.resume is True:
    model_name = "last_checkpoint.pth"
    try:
        model.load_state_dict(torch.load(models_path / model_name))
        report.info(f"Model loaded from {models_path / model_name}")
    except FileNotFoundError:
        report.info(f"Model {models_path / model_name} does not exist. Starting with a new model.")
else:
    model_name = opts.resume
    model.load_state_dict(torch.load(models_path / model_name))
    report.info(f"Model loaded from {models_path / model_name}")

with report.status("Loading dataset..."):
    train_dataset, valid_dataset = split_datasets(
        data_path=data_path,
        shape=opts.slice_scape,
        max_size=(opts.buffer_size, opts.buffer_size // opts.train_to_valid_ratio)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opts.batch_size, pin_memory=True)

optimizer = AdaBelief(
    model.parameters(),
    lr=opts.learning_rate,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
    print_change_log=False,
)

try:
    train_cycle(model,
                epochs=opts.epochs,
                # dataset=train_dataset,
                # validation_dataset=valid_dataset,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                optimizer=optimizer,
                device=device,
                models_path=models_path)
except Exception as err:
    run.finish()
    raise err
