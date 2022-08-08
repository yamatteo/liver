from types import SimpleNamespace
from utils.debug import dbg


class Namespace(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        for arg in args:
            self.__dict__.update(arg)
        self.__dict__.update(kwargs)

    def __getattribute__(self, item):
        try:
            return SimpleNamespace.__getattribute__(self, item)
        except AttributeError:
            # if item[0] == '_' or item in [
            #     'aihwerij235234ljsdnp34ksodfipwoe234234jlskjdf',
            #     'awehoi234_wdfjwljet234_234wdfoijsdfmmnxpi492',
            # ]:
            #     raise AttributeError
            dbg("DEBUG:", "Namespace does not have a", repr(item), "attribute. Returning None.")
            return None

    def __getitem__(self, item):
        return SimpleNamespace.__getattribute__(self, item)

    def __repr__(self):
        return repr(vars(self))

    def __rich__(self):
        return vars(self)

    def keys(self):
        return vars(self).keys()
