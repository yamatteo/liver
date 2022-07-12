import argparse
from pathlib import Path

import torch
from adabelief_pytorch import AdaBelief

import report
from buffer_dataset import get_finer_datasets
from finer_train import train_cycle
from models.double_unet import DoubleUNet

run = report.init(project="liver-tumor-detecton", entity="yamatteo", backend="none", level="debug")

# dotenv.load_dotenv()
# report.debug("Loaded enviromental variables:", dict(dotenv.dotenv_values()))
data_path = Path("/home/yamatteo/storage/hepato_outputs")
dataset_path = Path("/home/yamatteo/storage/dataset")
models_path = Path("/home/yamatteo/storage/liver/saved_models")

opts = argparse.Namespace(
    batch_size=4,
    buffer_size=20,
    epochs=400,
    learning_rate=1e-5,
    # channels=[4, 32, 64, 128],
    resume=False,
    coarse_model="coarse400.pth",
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
model.first_net.load_state_dict(torch.load(models_path / opts.coarse_model, map_location=device))
report.info(f"coarse model loaded from {models_path / opts.coarse_model}")

if opts.resume is False:
    report.info("Starting with a new model.")
elif opts.resume is True:
    model_name = "last_checkpoint.pth"
    try:
        model.second_net.load_state_dict(torch.load(models_path / model_name))
        report.info(f"Model loaded from {models_path / model_name}")
    except FileNotFoundError:
        report.info(f"Model {models_path / model_name} does not exist. Starting with a new model.")
else:
    model_name = opts.resume
    model.second_net.load_state_dict(torch.load(models_path / model_name))
    report.info(f"Model loaded from {models_path / model_name}")

# FINER training
with report.status("Loading dataset..."):
    train_dataset, valid_dataset = get_finer_datasets(dataset_path)
    report.info(f"Training dataset: {len(train_dataset)} items.")
    report.info(f"Validation dataset: {len(valid_dataset)} items.")

optimizer = AdaBelief(
    model.second_net.parameters(),
    lr=opts.learning_rate,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
    print_change_log=False,
)

try:
    losses = train_cycle(
        model.second_net,
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
