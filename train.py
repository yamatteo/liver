import time

import torch
from adabelief_pytorch import AdaBelief
from rich import print
from torch.utils.data import DataLoader

import report
from dataset import GeneratorDataset
from models import Architecture
from slicing import slices


