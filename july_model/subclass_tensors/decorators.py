import torch


def integer(original_class):
    original_class.fixed_dtype = torch.int16
    return original_class


def floating(original_class):
    original_class.fixed_dtype = torch.float32
    return original_class


def bidimensional(original_class):
    original_class.fixed_shape = {"X": None, "Y": None}
    return original_class


def tridimensional(original_class):
    original_class.fixed_shape = {"X": None, "Y": None, "Z": None}
    return original_class


def channels(fixed_channels):
    def _channels(original_class):
        fixed_shape = original_class.fixed_shape
        if fixed_shape is ...:
            fixed_shape = {}

        original_class.fixed_shape = {"C": len(fixed_channels), **fixed_shape}
        original_class.fixed_channels = fixed_channels
        return original_class
    return _channels


def batch(original_class):
    fixed_shape = original_class.fixed_shape
    if fixed_shape is ...:
        fixed_shape = {}

    original_class.fixed_shape = {"N": None, **fixed_shape}
    return original_class