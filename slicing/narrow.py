from .utils import Array, opt_int_tuple

def dimensional_narrow(input: Array, dim: int, start:int, length: int) -> Array:
    selection = tuple(
        slice(start, start + length) if n==dim else slice(None, None)
        for n in range(len(input.shape))
    )
    return input.__getitem__(selection)

def non_spatial_narrow(input: Array, starts: opt_int_tuple, lengths: opt_int_tuple, strict=False) -> Array:
    assert len(input.shape) >= len(starts) == len(lengths)
    if strict and any(
            start + length > input.shape[dim]
            for dim, (start, length) in enumerate(zip(starts, lengths))
    ):
        raise ValueError(f"Violates strict requirement: input.shape={input.shape}, starts={starts}, lengths={lengths}.")
    selection = tuple(
        slice(None, None) if s is None or l is None else slice(s, s + l)
        for s, l in zip(starts, lengths)
    )
    return input.__getitem__(selection)


def spatial_narrow(input: Array, starts: opt_int_tuple, lengths: opt_int_tuple, strict=False) -> Array:
    assert len(input.shape) >= len(starts) == len(lengths)
    nonspatial_ndim = len(input.shape) - len(starts)
    if strict and any(
            start + length > input.shape[nonspatial_ndim + dim]
            for dim, (start, length) in enumerate(zip(starts, lengths))
            if start is not None and length is not None
    ):
        raise ValueError(f"Violates strict requirement: input.shape={input.shape}, starts={starts}, lengths={lengths}.")
    selection = (...,) + tuple(
        slice(None, None) if s is None or l is None else slice(s, s + l)
        for s, l in zip(starts, lengths)
    )
    return input.__getitem__(selection)

