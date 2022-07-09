import argparse
import os
from pathlib import Path

import torch
import wandb
from adabelief_pytorch import AdaBelief

import report
from buffer_dataset import store_datasets
from models.double_unet import DoubleUNet, pool_layer

# run = report.init(project="liver-tumor-detecton", entity="yamatteo", backend="none", level="debug")

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
    resume=True,
    # slice_shape=(16, 16, 8),
    train_to_valid_ratio=10,
)
# wandb.config = vars(opts)  # Maybe these values are stored by wandb, maybe useful for hyperparameters search
# report.debug("Using config options:", vars(opts))

# Ensure directory is present
# if not models_path.exists():
#     report.warn(f"Directory to load/save models does not exist. Making a new one at {models_path}.")
#     models_path.mkdir()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# report.info(f"Running on {device}")

### Store coarse dataset
coarse_dataset_path.mkdir(exist_ok=True)
store_datasets(source_path=data_path, target_path=coarse_dataset_path, pooler=pool_layer("avg441"), shape=(128, 128, 8))

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
    dataset, valid_dataset = split_datasets(data_path=data_path,  shape=opts.slice_shape, batch_size=opts.batch_size)
    # model.eval()
    # train_cases, valid_cases = [], []
    # for i, case in enumerate(generators.cases(data_path, generators.criterion(bundle=True))):
    #     if i % 10 == 0:
    #         valid_cases.append(case)
    #     else:
    #         train_cases.append(case)
    #
    # report.info("Populating training dataset.")
    # dataset = BufferDataset(
    #     generator=generators.cycle_enum_slices(train_cases, opts.slice_shape),
    #     max_size=opts.buffer_size,
    #     batch_size=opts.batch_size
    # )
    # report.info("Populating validation dataset.")
    # valid_dataset = BufferDataset(
    #     generator=generators.cycle_enum_slices(valid_cases, opts.slice_shape),
    #     max_size=opts.buffer_size // opts.train_to_valid_ratio,
    #     batch_size=opts.batch_size
    # )

# if opts.resume:
#     with torch.no_grad():
#         model.to(device=device)
#         def evaluate(fbb):
#             scan, segm = fbb.separate()
#             scan = scan.to(device=device, dtype=torch.float32)
#             segm = segm.to(device=device, dtype=torch.float32)
#             pred = model.forward(scan)
#             batch_losses, _ = pred.distance_from(segm)
#             return torch.sum(batch_losses).item()
#
#         with report.status("Warming up datasets..."):
#             dataset.warmup(evaluate)
#             valid_dataset.warmup(evaluate)

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
                dataset=dataset,
                validation_dataset=valid_dataset,
                optimizer=optimizer,
                device=device,
                models_path=models_path)
except Exception as err:
    run.finish()
    raise err
