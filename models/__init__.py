from types import GeneratorType

from torch import Tensor


def wrap(*args) -> tuple[Tensor, ...]:
    match args:
        case ():
            return ()
        case (tuple() | list() | GeneratorType() as items, *rest):
            return *wrap(*items), *wrap(*rest)
        case (item, ):
            return item,
        case (item, *rest):
            return item, *wrap(*rest)


from .convolutions import ConvBlock
from .monostream import Stream
from .structures import Cat, Select, Separated, Sequential, SkipCat, Split, Wrapper
