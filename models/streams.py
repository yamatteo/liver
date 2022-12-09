import inspect
from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional

from .utils import wrap


class Stream(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Stream, self).__init__(*args, **kwargs)
        params = inspect.signature(self.__init__).parameters
        self.relevant_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in params and (
                        value != params[key].default or params[key].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        params = list(params.values())
        self.relevant_args = tuple(arg for arg, param in zip(args, params) if arg != param.default)

    def __repr__(self):
        args = str(", ").join(
            [repr(arg) for arg in self.relevant_args]
            + [key + "=" + repr(value) for key, value in self.relevant_kwargs.items()]
        )
        return f"{self.__class__.__name__}({args})"

    def forward(self, *args: Tensor) -> Tuple[Tensor, ...]:
        return wrap(super(Stream, self).forward(*args))

    @property
    def repr_dict(self):
        return dict(
            class_name=self.__class__.__name__,
            args=self.relevant_args,
            kwargs=self.relevant_kwargs
        )

    def shaper(self, *shapes) -> tuple[tuple[int, ...], ...]:
        return shapes


class Argmax(Stream, nn.Module):
    def __init__(self, dim=1):
        super(Argmax, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> tuple[Tensor]:
        return torch.argmax(input, dim=self.dim),  # Returns a tuple

    def shaper(self, shape: tuple[int]) -> tuple[tuple[int, ...]]:
        return tuple(d for i, d in enumerate(shape) if i != self.dim),


class AvgPool3d(Stream, nn.AvgPool3d):
    def __init__(self, kernel_size=(2, 2, 2)):
        super(AvgPool3d, self).__init__(kernel_size=kernel_size)


class BatchNorm3d(Stream, nn.BatchNorm3d):
    def __init__(self, num_features, momentum=0.1):
        super(BatchNorm3d, self).__init__(num_features, momentum=momentum)


class Cat(Stream, nn.Module):
    def __init__(self, dim=1):
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return torch.cat(args, dim=self.dim),

    def shaper(self, *shapes: tuple[int, ...]) -> tuple[tuple[int, ...]]:
        assert len(shapes) > 1
        base = shapes[0]
        ndim = len(base)
        assert all(len(shape) == ndim for shape in shapes)
        assert all(
            all(shape[n] == base[n] for shape in shapes)
            for n in range(ndim)
            if n != self.dim
        )
        return tuple(base[n] if n != self.dim else sum(shape[n] for shape in shapes) for n in range(ndim)),


class Clamp(Stream, nn.Module):
    def __init__(self, min, max):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input: Tensor) -> tuple[Tensor]:
        return torch.clamp(input, self.min, self.max),


class Conv3d(Stream, nn.Conv3d):
    def __init__(self, in_, out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=..., padding_mode="replicate"):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if padding is None or padding is ...:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding, padding)
        super(Conv3d, self).__init__(in_, out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     padding_mode=padding_mode)

        def shaper(shape: tuple) -> tuple[tuple]:
            n, c, x, y, z = shape
            assert c == in_, f"Input channels should be {in_}, got {c}."
            return (n, out, *[
                (s + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1)) // stride[i]
                for i, s in enumerate([x, y, z])
            ]),  # Returns a tuple[tuple], as Stream.shaper

        self.shaper = shaper


class CrossEntropyLoss(Stream, nn.CrossEntropyLoss):
    pass


class Dropout3d(Stream, nn.Dropout3d):
    pass


class Expression(Stream, nn.Module):
    def __init__(self, expression: str):
        super(Expression, self).__init__()
        self.expression = expression

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return wrap(eval(self.expression))  # Returns a tuple of Tensor

class Flatten(Stream, nn.Module):
    def __init__(self, start_dim = 1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, *args: Tensor) -> tuple[Tensor, ...]:
        return wrap(torch.flatten(a, start_dim=self.start_dim) for a in args)


class _FoldNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, *, folded_shape, momentum):
        super(_FoldNorm3d, self).__init__(num_features, momentum=momentum)


class FoldNorm3d(Stream, _FoldNorm3d):
    def __init__(self, num_features, *, folded_shape=None, momentum):
        super(FoldNorm3d, self).__init__(num_features, folded_shape=folded_shape, momentum=momentum)
        self.folded_shape = folded_shape

    def forward(self, input: Tensor) -> Tuple[Tensor]:
        n, c, x, y, z = input.shape
        if self.folded_shape is None:
            self.set_shape(input.shape)
        sx, sy, sz = self.folded_shape
        fx, fy, fz = x // sx, y // sy, z // sz
        input = input.view([n, c, fx, sx, fy, sy, fz, sz]) \
            .permute(0, 2, 4, 6, 1, 3, 5, 7) \
            .reshape([n * fx * fy * fz, c, sx, sy, sz])
        input = nn.BatchNorm3d.forward(self, input)
        input = input.view(n, fx, fy, fz, c, sx, sy, sz) \
            .permute(0, 4, 1, 5, 2, 6, 3, 7) \
            .reshape([n, c, x, y, z])
        return input,  # Returns a tuple, as in Stream.forward

    def set_shape(self, input_shape):
        n, c, x, y, z = input_shape
        top = (n * x * y * z) ** 0.5
        sx, sy, sz = 1, 1, 1
        while sx * sy * sz <= top:
            fx, fy, fz = x // sx, y // sy, z // sz
            if fx == max(fx, fy, fz):
                sx *= 2
            elif fy == max(fx, fy, fz):
                sy *= 2
            else:
                sz *= 2
        print(f"FoldNorm3d{input_shape} set shape to {(sx, sy, sz)}")
        self.folded_shape = sx, sy, sz


class Identity(Stream, nn.Identity):
    pass


class InstanceNorm3d(Stream, nn.InstanceNorm3d):
    def __init__(self, num_features, momentum=0.9):
        super(InstanceNorm3d, self).__init__(num_features, momentum=momentum)


class Jaccard(Stream, nn.Module):
    def __init__(self, index=1):
        super(Jaccard, self).__init__()
        self.index = index

    def forward(self, input: Tensor, target: Tensor) -> Tuple[Tensor, ...]:
        index = self.index
        intersection = ((input == index) * (target == index)).sum() + 0.1
        union = ((input == index) + (target == index)).sum() + 0.1
        return wrap(intersection / union)


class LeakyReLU(Stream, nn.LeakyReLU):
    pass


class Linear(Stream, nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features)

class MaskOf(Stream, nn.Module):
    def __init__(self, index):
        super(MaskOf, self).__init__()
        self.index = index

    def forward(self, input: Tensor) -> tuple[Tensor]:
        return torch.as_tensor(input == self.index, dtype=torch.int64),


class MaxPool3d(Stream, nn.MaxPool3d):
    def __init__(self, kernel_size=(2, 2, 2), return_indices=False):
        super(MaxPool3d, self).__init__(kernel_size=kernel_size, return_indices=return_indices)


class MaxUnpool3d(Stream, nn.MaxUnpool3d):
    def __init__(self, kernel_size=(2, 2, 2)):
        super(MaxUnpool3d, self).__init__(kernel_size=kernel_size)


class Precision(Stream, nn.Module):
    def __init__(self, index=1):
        super(Precision, self).__init__()
        self.index = index

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor]:
        index = self.index
        true_positive = 0.1 + ((input == index) * (target == index)).sum()
        all_positives = 0.1 + (input == index).sum()
        return wrap(true_positive / all_positives)


class Recall(Stream, nn.Module):
    def __init__(self, index=1):
        super(Recall, self).__init__()
        self.index = index

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor]:
        index = self.index
        true_positive = 0.1 + ((input == index) * (target == index)).sum()
        ground_truth = 0.1 + (target == index).sum()
        return wrap(true_positive / ground_truth)


class _IRNorm3d(nn.modules.instancenorm._InstanceNorm):
    def _apply_instance_norm(self, input):
        if self.training:
            nn.functional.instance_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                True, self.momentum, self.eps)
        return nn.functional.instance_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            False, self.momentum, self.eps)

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError('expected 4D or 5D input (got {}D input)'.format(input.dim()))

class IRNorm3d(Stream, _IRNorm3d):
    def __init__(self, num_features: int, momentum=0.1, affine=True, track_running_stats=True):
        super(IRNorm3d, self).__init__(num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class _LIRNorm3d(nn.modules.instancenorm._LazyNormBase, nn.modules.instancenorm._InstanceNorm):
    cls_to_become = _IRNorm3d  # type: ignore[assignment]

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError('expected 4D or 5D input (got {}D input)'.format(input.dim()))

class LIRNorm3d(Stream, _LIRNorm3d):
    def __init__(self, momentum=0.1, track_running_stats=True):
        super(LIRNorm3d, self).__init__(momentum=momentum, track_running_stats=track_running_stats)


class SoftPrecision(Stream, nn.Module):
    def __init__(self, dim=1, index=1, num_classes=None):
        super(SoftPrecision, self).__init__()
        self.dim = dim
        self.index = index
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor]:
        input = functional.softmax(input, dim=self.dim)
        if self.num_classes:
            num_classes = self.num_classes
        else:
            num_classes = input.size(self.dim)
        dims = list(range(target.ndim))
        target = functional.one_hot(target, num_classes).float()
        dims.insert(self.dim, -1)
        target = target.permute(dims)
        input = input[:, self.index]
        target = target[:, self.index]
        return wrap((0.1 + torch.sum(torch.minimum(input, target))) / (0.1 + torch.sum(input)))


class SoftRecall(Stream, nn.Module):
    def __init__(self, dim=1, index=1, num_classes=None):
        super(SoftRecall, self).__init__()
        self.dim = dim
        self.index = index
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> tuple[Tensor]:
        input = functional.softmax(input, dim=self.dim)
        if self.num_classes:
            num_classes = self.num_classes
        else:
            num_classes = input.size(self.dim)
        dims = list(range(target.ndim))
        target = functional.one_hot(target, num_classes).float()
        dims.insert(self.dim, -1)
        target = target.permute(dims)
        input = input[:, self.index]
        target = target[:, self.index]
        return wrap((0.1 + torch.sum(torch.minimum(input, target))) / (0.1 + torch.sum(target)))


class Upsample(Stream, nn.Upsample):
    def __init__(self, *, scale_factor, mode):
        super(Upsample, self).__init__(scale_factor=scale_factor, mode=mode)
