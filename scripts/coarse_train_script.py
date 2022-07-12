import argparse
import os
from pathlib import Path

import torch
import wandb
from adabelief_pytorch import AdaBelief

import report
from buffer_dataset import store_datasets, get_datasets
from models.double_unet import DoubleUNet, pool_layer
from train import train_cycle

run = report.init(project="liver-tumor-detecton", entity="yamatteo", backend="none", level="debug")

# dotenv.load_dotenv()
# report.debug("Loaded enviromental variables:", dict(dotenv.dotenv_values()))
data_path = Path("/home/yamatteo/storage/hepato_outputs")
dataset_path = Path("/home/yamatteo/storage/dataset")
coarse_dataset_path = Path("/home/yamatteo/storage/coarse_dataset")
models_path = Path("/home/yamatteo/storage/liver/saved_models")

opts = argparse.Namespace(
    batch_size=4,
    buffer_size=20,
    epochs=400,
    learning_rate=1e-5,
    # channels=[4, 32, 64, 128],
    resume=False,
    # slice_shape=(16, 16, 8),
    train_to_valid_ratio=10,
)
# wandb.config = vars(opts)  # Maybe these values are stored by wandb, maybe useful for hyperparameters search
report.debug("Using config options:", vars(opts))

# Ensure directory is present
# if not models_path.exists():
#     report.warn(f"Directory to load/save models does not exist. Making a new one at {models_path}.")
#     models_path.mkdir()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
report.info(f"Running on {device}")

# ### Store coarse dataset
# coarse_dataset_path.mkdir(exist_ok=True)
# store_datasets(source_path=data_path, target_path=coarse_dataset_path, pooler=pool_layer("avg441"), shape=(128, 128, 8))

model = DoubleUNet()


# if opts.resume is False:
#     report.info("Starting with a new model.")
# elif opts.resume is True:
#     model_name = "last_checkpoint.pth"
#     try:
#         model.load_state_dict(torch.load(models_path / model_name))
#         report.info(f"Model loaded from {models_path / model_name}")
#     except FileNotFoundError:
#         report.info(f"Model {models_path / model_name} does not exist. Starting with a new model.")
# else:
#     model_name = opts.resume
#     model.load_state_dict(torch.load(models_path / model_name))
#     report.info(f"Model loaded from {models_path / model_name}")

# COARSE training
with report.status("Loading dataset..."):
    train_dataset, valid_dataset = get_datasets(coarse_dataset_path)

optimizer = AdaBelief(
    model.first_net.parameters(),
    lr=opts.learning_rate,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
    print_change_log=False,
)

try:
    losses = train_cycle(
        model.first_net,
        epochs=opts.epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        optimizer=optimizer,
        device=device,
        models_path=models_path,
        batch_size=opts.batch_size,
        buffer_size=opts.buffer_size)
except Exception as err:
    run.finish()
    raise err
