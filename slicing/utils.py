import numpy as np
import torch

from typing import TypeVar, Union

Array = TypeVar("Array", np.ndarray, torch.Tensor)
opt_int_tuple = tuple[Union[int, None], ...]