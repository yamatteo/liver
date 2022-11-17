import torch
from torch import nn
from torch.nn import functional


class Argmax(nn.Module):
    def __init__(self, dim=1):
        super(Argmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.argmax(input, dim=self.dim)


class AsTensor(nn.Module):
    def __init__(self, batch_ndim=None, device=None, dtype=None):
        super(AsTensor, self).__init__()
        self.batch_ndim = batch_ndim
        self.kwargs = {}
        if device:
            self.kwargs["device"] = device
        if dtype:
            self.kwargs["dtype"] = dtype

    def forward(self, input):
        if self.batch_ndim is None or self.batch_ndim == input.ndim:
            return torch.as_tensor(input, **self.kwargs)
        return torch.as_tensor(input, **self.kwargs).unsqueeze(0)


class Cat(nn.Module):
    def __init__(self, dim=1):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.cat(args, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


class FoldNorm3d(nn.BatchNorm3d):
    def __init__(self, shape=(8, 8, 8), *args, **kwargs):
        super(FoldNorm3d, self).__init__(*args, **kwargs)
        self.shape = shape

    def forward(self, input):
        n, c, x, y, z = input.shape
        sx, sy, sz = self.shape
        fx, fy, fz = x // sx, y // sy, z // sz
        input = input.view([n, c, fx, sx, fy, sy, fz, sz]) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape([n * fx * fy * fz, c, sx, sy, sz])
        input = super(FoldNorm3d, self).forward(input)
        input = input.view(n, fx, fy, fz, c, sx, sy, sz) \
            .permute(0, 4, 1, 5, 2, 6, 3, 7) \
            .reshape([n, c, x, y, z])
        return input


class Recall(nn.Module):
    def __init__(self, argmax_input_dim=None):
        super(Recall, self).__init__()
        self.argmax_input_dim = argmax_input_dim

    def forward(self, input, target):
        if self.argmax_input_dim:
            input = torch.argmax(input, dim=self.argmax_input_dim)
        return (0.1 + torch.sum(torch.minimum(input, target))) / (0.1 + torch.sum(target))


class SoftRecall(nn.Module):
    def __init__(self, dim=1, klass=1, num_classes=None):
        super(SoftRecall, self).__init__()
        self.dim = dim
        self.klass=klass
        self.num_classes = num_classes

    def forward(self, input, target):
        input = functional.softmax(input, dim=self.dim)
        if self.num_classes:
            num_classes = self.num_classes
        else:
            num_classes = input.size(self.dim)
        dims = list(range(target.ndim))
        target = functional.one_hot(target, num_classes).float()
        dims.insert(self.dim, -1)
        target = target.permute(dims)
        input = input[:, self.klass]
        target = target[:, self.klass]
        return (0.1 + torch.sum(torch.minimum(input, target))) / (0.1 + torch.sum(target))
