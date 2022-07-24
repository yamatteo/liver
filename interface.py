import functools
import inspect
from pathlib import Path

from ipywidgets import *
from traitlets import *

import dataset.ndarray
import dataset.path_explorer


class Interface(Tab):
    def __init__(self, **kwargs):
        Tab.__init__(self, **kwargs)
        self.trait_tree = {}

    def inject(self, widget):
        def _depend(func):
            trait = func.__name__
            states = {name for name in inspect.signature(func).parameters}

            def _func(event):
                params = {name: getattr(self, name) for name in states}
                setattr(widget, trait, func(**params))

            flagged_states = {name for state in states for name in self.trait_tree[state]}
            self.observe(_func, flagged_states)

        return _depend

    def ensure_state(self, name, value=...):
        if not self.has_trait(name):
            if hasattr(self, name):
                raise ValueError(f"Interface.{name} is not a trait but a {type(getattr(self, name))}.")
            self.add_traits(**{name: Any()})
            self.trait_tree[name] = {name}
        if value is not ...:
            setattr(self, name, value)

    def identity(self, state: str, widget: HasTraits, trait: str = "value", value=...):
        self.ensure_state(state, value=value)
        link((self, state), (widget, trait))
        return widget

    def new_tab(self, title, widget):
        self.children = (*self.children, widget)
        self.set_title(len(self.children) - 1, title)

    def on_click(self, widget: Button, *args, **kwargs):
        def _on_click(func):
            def _func(event):
                params = {state: getattr(self, state) for state in args}
                params.update(kwargs)
                return func(**params)
            widget.on_click(_func)
        return _on_click

    def property(self, func):
        out_state = func.__name__
        if hasattr(self, out_state):
            raise ValueError(f"Interface.{out_state} already exists.")
        in_states = {name for name in inspect.signature(func).parameters}

        def _func(_self):
            params = {name: getattr(_self, name) for name in in_states}
            return func(**params)

        setattr(type(self), out_state, property(_func))
        base_states = {name for state in in_states for name in self.trait_tree[state]}
        if out_state in self.trait_tree:
            raise ValueError(f"interface.trait_tree already has {out_state}")
        self.trait_tree[out_state] = base_states

print()


class HiddenState(HasTraits):
    drive_folder = Unicode()
    case = Any()
    tv_info = Any()


def old():
    hs = HiddenState()

    def react(*traits):
        def _react(func):
            hs.observe(func, *traits)
            return func

        return _react

    def observe(widget):
        def _observe(func):
            state = func.__name__
            if not hs.has_trait(state):
                hs.add_traits(**{state: Any()})
            traits = {name for name in inspect.signature(func).parameters}

            def _func(event):
                params = {name: getattr(widget, name) for name in traits}
                setattr(hs, state, func(**params))

            widget.observe(_func, *traits)

        return _observe

    # def dependant(widget, **functions):
    #     traits = functions.keys()
    #     states = {name for func in functions.values() for name in inspect.signature(func).parameters}
    #
    #     @react(*states)
    #     def __function(event):
    #         with widget.hold_trait_notifications():
    #             print("$ Dependency", widget)
    #             for trait, func in functions.items():
    #                 params = {name: getattr(interface_state, name) for name in inspect.signature(func).parameters}
    #                 print(f"  set {trait} to {func(**params)!r}")
    #                 setattr(widget, trait, func(**params))
    #
    #     return widget

    b01 = Button(button_style='info', layout=Layout(width='auto'))

    @depend(b01)
    def description(case):
        if case:
            return f"Convert only {str(case)} (overwrite)"
        else:
            return "No case selected!"

    b01.on_click(partial(dicom_callback, path=hs.case_path, overwrite=True))

    tv_dropdown = Dropdown(
        options=[
            ("Original phase b", ("orig", "b")),
            ("Original phase a", ("orig", "a")),
            ("Original phase v", ("orig", "v")),
            ("Original phase t", ("orig", "t")),
        ],
        layout=Layout(width='auto'),
        style={'description_width': 'initial'},
    )

    @depend(tv_dropdown)
    def description(case):
        if case:
            return f"What to visualize (about case {case}):"
        else:
            return "No case selected!"

    establish_identity("tv_info", tv_dropdown)

    tv = Output()

    @react("tv_info")
    def switch_channel(event):
        channel, *args = event.new
        if channel == "orig":
            phase = args[0]
            with tv:
                orig, matrix = dataset.ndarray.load_original(hs.case_path, phase=phase)

                def show(phase, z):
                    white = orig[:, :, z].T
                    white = white.clip(0, 255) / 255
                    seaborn_image.imgplot(
                        white,
                        cbar=False,
                    )

                w = interactive(show, z=(0, orig.shape[2]))
                display(w)

    tabs = {

        "Preprocess": VBox(
            [
                b01
            ]
        ),
        "Visualize": VBox(
            [
                tv_dropdown,
                tv
            ]
        )
    }
    tab = Tab()
    tab.children = list(tabs.values())
    for i, label in enumerate(tabs):
        tab.set_title(i, label)
    display(tab)

    hs.drive_folder = "COLAB"
