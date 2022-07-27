import inspect
from typing import Union, Tuple

from ipywidgets import Button
from traitlets import HasTraits, TraitError

trait_ref = Union[
    Tuple[str, HasTraits, str],
    Tuple[HasTraits, str],
    Tuple[HasTraits, str, str],
    Tuple[HasTraits, str, str, str],
]


def reaction(source: HasTraits, execute=False):
    def __reaction_decorator(__func):
        traits = inspect.signature(__func).parameters.keys()

        def func(_event):
            params = {trait: getattr(source, trait) for trait in traits}
            __func(**params)

        source.observe(func, list(traits))

        if execute:
            func(None)

    return __reaction_decorator


def injection(source: HasTraits, widget: HasTraits, source_traits=None, widget_trait=None, except_to=None):
    def _injection_decorator(__func):
        target = widget_trait or __func.__name__
        args = list(inspect.signature(__func).parameters.keys())
        traits = source_traits or args

        def func(_event):
            try:
                params = {arg: getattr(source, trait) for arg, trait in zip(args, traits)}
                setattr(widget, target, __func(**params))
            except:
                setattr(widget, target, except_to)

        source.observe(func, list(traits))

    return _injection_decorator

def onclick(button: Button):
    return button.on_click
