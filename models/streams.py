from torch import Tensor
from torch.nn import *

from .utils import wrap
from .custom_modules import *


class Stream(Module):
    def __new__(cls, klass, *args, **kwargs):
        if isinstance(klass, str):
            klass = eval(klass)

        name = klass.__name__
        r_args = ', '.join(map(str, args))
        r_kwargs = ', '.join([key + '=' + str(value) for key, value in kwargs.items()])

        class StreamMixin(Stream, klass):
            def __repr__(self):
                return f"{name}({r_args}{', ' if args and kwargs else ''}{r_kwargs})"

            repr_dict = dict(
                name=name,
                args=args,
                kwargs=kwargs,
            )

            def forward(self, *args: Tensor):
                return wrap(klass.forward(self, *args))

        obj = object.__new__(StreamMixin)

        return obj

    def __init__(self, klass, *args, **kwargs):
        if isinstance(klass, str):
            klass = eval(klass)
        klass.__init__(self, *args, **kwargs)
