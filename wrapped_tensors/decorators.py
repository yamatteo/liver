import torch


def short(original_class):
    _init = original_class.__init__

    def init(self, data, **kwargs):
        kwargs["fix_dtype"] = torch.int16
        _init(self, data, **kwargs)

    original_class.__init__ = init
    return original_class


def floating(original_class):
    _init = original_class.__init__

    def init(self, data, **kwargs):
        kwargs["fix_dtype"] = torch.float32
        _init(self, data, **kwargs)

    original_class.__init__ = init
    return original_class


def bidimensional(original_class):
    _init = original_class.__init__

    def init(self, data, **kwargs):
        kwargs["fix_shape"] = {"X": ..., "Y": ...}
        _init(self, data, **kwargs)

    original_class.__init__ = init
    return original_class


def tridimensional(original_class):
    _init = original_class.__init__

    def init(self, data, **kwargs):
        kwargs["fix_shape"] = {"X": ..., "Y": ..., "Z": ...}
        _init(self, data, **kwargs)

    original_class.__init__ = init
    return original_class


def channels(fixed_channels):
    def _channels(original_class):
        _init = original_class.__init__

        def init(self, data, **kwargs):
            kwargs["fix_channels"] = fixed_channels
            _init(self, data, **kwargs)

        original_class.__init__ = init
        return original_class
    return _channels


def batch(original_class):
    _init = original_class.__init__

    def init(self, data, **kwargs):
        kwargs["fix_batch"] = ...
        _init(self, data, **kwargs)

    original_class.__init__ = init
    return original_class