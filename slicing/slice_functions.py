from typing import Literal, Iterator

from .utils import *
from .indices import *
from .narrow import spatial_narrow, dimensional_narrow
from .padding import spatial_pad


def slices(args: Union[list, tuple, Array], *, shape: opt_int_tuple, pad: bool = True,
           drop_below: opt_int_tuple = None, mode: Literal["consecutive", "overlapping"] = "consecutive",
           indices: bool = False) -> Iterator[Union[Array, tuple[tuple[int], Array, ...]]]:
    if isinstance(args, (list, tuple)):
        ndim = len(args[0].shape)
    else:
        ndim = len(args.shape)
    if drop_below:
        drop_below = (None,) * (ndim - len(shape)) + drop_below
    for starts, *_args in _slices(args, shape, mode=mode):
        if drop_below:
            if any(drop_below[n] is not None and _arg.shape[n] < drop_below[n] for n in range(ndim) for _arg in _args):
                continue
        if pad:
            _args = spatial_pad(_args, shape)
        if indices:
            yield starts, *_args
        elif isinstance(args, (list, tuple)):
            yield _args
        else:
            yield _args[0]


def _slices(args: Union[list, tuple, Array], shape: opt_int_tuple,
            mode: Literal["consecutive", "overlapping"] = "consecutive"):
    if isinstance(args, (list, tuple)):
        for starts, slice in __slices(args[0], shape, mode=mode):
            yield starts, slice, *(spatial_narrow(arg, starts, shape) for arg in args[1:])
    else:
        return __slices(args, shape, mode=mode)


def __slices(input: Array, shape: opt_int_tuple, mode: Literal["consecutive", "overlapping"] = "consecutive"):
    if len(shape) < 1:
        yield (), input
        return
    active_dim = len(input.shape) - len(shape)
    original = input.shape[active_dim]
    requested = shape[0] or original
    if mode == "consecutive":
        starts = range_s(original, requested)
    elif mode == "overlapping":
        starts = range_o(original, requested)
    else:
        raise ValueError(f"No mode named {mode}.")
    for s in starts:
        slice = dimensional_narrow(input, active_dim, s, requested)
        for _starts, _slice in __slices(slice, shape[1:], mode=mode):
            yield (s, *_starts), _slice
