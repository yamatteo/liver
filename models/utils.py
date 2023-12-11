from types import GeneratorType

from torch import Tensor


def wrap(*args) -> tuple[Tensor, ...]:
    """Returns all arguments as a plain tuple, flattening it if there is nesting."""
    if args == ():
        return ()
    first, *rest = args
    if isinstance(first, (tuple, list, GeneratorType)):
        return *wrap(*first), *wrap(*rest)
    return first, *wrap(*rest)