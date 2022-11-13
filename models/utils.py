from types import GeneratorType

from torch import Tensor


def wrap(*args) -> tuple[Tensor, ...]:
    if args == ():
        return ()
    first, *rest = args
    if isinstance(first, (tuple, list, GeneratorType)):
        return *wrap(*first), *wrap(*rest)
    return first, *wrap(*rest)