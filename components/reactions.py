import inspect

from traitlets import HasTraits


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


def state_reactions(state: HasTraits):
    react = reaction(state)

    def inject(widget: HasTraits, except_to=None):
        return injection(source=state, widget=widget, except_to=except_to)

    def project(widget: HasTraits, except_to=None):
        return injection(source=widget, widget=state, except_to=except_to)

    def biject(state_trait: str, widget: HasTraits, widget_trait: str = "value", except_to=None):
        f = lambda x: x
        injection(state, widget, (state_trait,), widget_trait, except_to=except_to)(f)
        injection(widget, state, (widget_trait,), state_trait, except_to=except_to)(f)

    return react, inject, project, biject