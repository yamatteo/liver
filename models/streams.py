import inspect

from torch import Tensor
from torch import nn

from .utils import wrap


class Stream(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Stream, self).__init__(*args, **kwargs)
        params = inspect.signature(self.__init__).parameters
        self.relevant_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in params and value != params[key].default
        }
        params = list(params.values())
        self.relevant_args = [arg for arg, param in zip(args, params) if arg != param.default]

    def __repr__(self):
        args = str(", ").join(
            [repr(arg) for arg in self.relevant_args]
            + [key + "=" + repr(value) for key, value in self.relevant_kwargs.items()]
        )
        return f"{self.__class__.__name__}({args})"

    def forward(self, *args: Tensor):
        return wrap(super(Stream, self).forward(*args))

    def repr_dict(self):
        return dict(
            class_name=self.__class__.__name__,
            args=self.relevant_args,
            kwargs=self.relevant_kwargs
        )


class Conv3d(Stream, nn.Conv3d):
    def __init__(self, in_, out, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=...):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        elif isinstance(padding, int):
            padding = (padding, padding, padding)
        super(Conv3d, self).__init__(in_, out, kernel_size=kernel_size, stride=stride, padding=padding)

        def shaper(shape):
            n, c, x, y, z = shape
            assert c == in_, f"Input channels should be {in_}, got {c}."
            return n, out, *[
                (s + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1)) // stride[i]
                for i, s in enumerate([x, y, z])
            ]
        self.shaper = shaper
