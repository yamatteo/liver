import argparse
import os
from pathlib import Path

import dotenv
import nibabel
import numpy as np
import torch
from rich.console import Console

from models.double_unet import DoubleUNet
from wrapped_tensors import Scan, Segm, Bundle
from utils.path_explorer import discover, get_criterion


from buffer_dataset import store_finer_datasets

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DoubleUNet()
model.first_net.load_state_dict(torch.load(models_path / opts.coarse_model, map_location=torch.device('cpu')))
model.up_sampler.to(device=device)
model.first_net.to(device=device)
model.down_sampler.to(device=device)

store_finer_datasets(source_path=data_path, target_path=dataset_path, shape=(128, 128, 8), model=model, device=device)
