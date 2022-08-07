from torch import Tensor, sum
from torch.nn.functional import l1_loss


def liverscore(i: Tensor, t: Tensor):
    i = (i == 1).to(dtype=float)
    t = (t == 1).to(dtype=float)
    return 1 - (l1_loss(i, t, reduction="sum") / (sum(t)+1)).item()


def tumorscore(i: Tensor, t: Tensor):
    i = (i == 2).to(dtype=float)
    t = (t == 2).to(dtype=float)
    return 1 - (l1_loss(i, t, reduction="sum") / (sum(t)+1)).item()
