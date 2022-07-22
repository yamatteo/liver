import inspect

from ipywidgets import *
from traitlets import *

import dataset.ndarray
from functools import partial


class HiddenState(HasTraits):
    drive_folder = Unicode()
    case = Any()
    tv_info = Any()

    @property
    def base_path(self):
        return Path("/content/drive/MyDrive") / self.drive_folder

    @property
    def case_path(self):
        return Path("/content/drive/MyDrive") / self.drive_folder / self.case


hs = HiddenState()


def establish_identity(state: str, widget: HasTraits, trait: str = "value"):
    if not hs.has_trait(state):
        hs.add_traits(**{state: Any()})
    link((hs, state), (widget, trait))
    return widget


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


def depend(widget):
    def _depend(func):
        trait = func.__name__
        states = {name for name in inspect.signature(func).parameters}

        def _func(event):
            params = {name: getattr(hs, name) for name in states}
            setattr(widget, trait, func(**params))

        hs.observe(_func, *states)

    return _depend


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
i01 = Text(description="Drive folder:")
establish_identity("drive_folder", i01)

i02 = Dropdown(description="Select case:")


@depend(i02)
def options(drive_folder):
    return discover(hs.base_path)


establish_identity("case", i02)

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
    "Setup": VBox(
        [
            HTML("<h3>Set some variables</h3>"),
            i01,
            i02
        ]
    ),
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
