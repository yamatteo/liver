from typing import Union

from .utils import Array, np, opt_int_tuple, torch


def spatial_pad(inputs: Union[list, tuple, Array], shape: opt_int_tuple):
    if isinstance(inputs, (tuple, list)):
        return tuple(_spatial_pad(input, shape) for input in inputs)
    else:
        return _spatial_pad(inputs, shape)


def _spatial_pad(input: Array, shape: opt_int_tuple):
    ndim = len(input.shape)
    nspatialdim = len(shape)
    for original, requested in zip(input.shape[-nspatialdim:], shape):
        assert requested is None or original <= requested
    if isinstance(input, np.ndarray):
        shape = (None,) * (ndim - nspatialdim) + shape
        pad_width = tuple(
            (0, 0) if requested is None else (0, requested - original)
            for requested, original in zip(shape, input.shape)
        )
        return np.pad(input, pad_width, mode="edge")
    elif isinstance(input, torch.Tensor):
        pad = tuple(
            (0, 0) if requested is None else (0, requested - original)
            for requested, original in zip(reversed(shape), reversed(input.shape))
        )
        pad = tuple(x for couple in pad for x in couple)
        return torch.nn.functional.pad(input, pad, mode="replicate")
