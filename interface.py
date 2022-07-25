from __future__ import annotations

import argparse
import functools
import inspect
from pathlib import Path

import seaborn_image
from IPython.core.display import display
from ipywidgets import *
from traitlets import *

import dataset.ndarray
import dataset.path_explorer
import scripts


def debug(*msg):
    print("%%%", *msg)


def reaction(widget: HasTraits, **kwargs):
    def _injection(__func):
        input_traits = set(inspect.signature(__func).parameters.keys())
        debug("Set up reaction", widget, input_traits, "->", __func.__name__)

        def func(event):
            params = {trait: getattr(widget, trait) for trait in input_traits}
            params.update(kwargs)
            __func(**params)

        widget.observe(func, input_traits)

    return _injection





def complex_reaction(*__inputs: tuple[HasTraits, str], **kwargs):
    def __reaction(__func):
        args = inspect.signature(__func).parameters.keys()

        def func(event):
            params = {arg: getattr(input, input_trait) for arg, (input, input_trait) in zip(args, __inputs)}
            params.update(kwargs)
            __func(**params)

        for input, input_trait in __inputs:
            input.observe(func, input_trait)

    return __reaction


def injection(input: HasTraits, target: HasTraits, **kwargs):
    def _injection(__func):
        target_trait = __func.__name__
        set_state(target, target_trait)
        input_traits = set(inspect.signature(__func).parameters.keys())
        debug("Set up injection", input, input_traits, "->", target, target_trait)

        def func(event):
            params = {trait: getattr(input, trait) for trait in input_traits}
            params.update(kwargs)
            value = __func(**params)
            # debug("Injecting", value, "into", target, target_trait)
            setattr(target, target_trait, value)

        input.observe(func, input_traits)

    return _injection


def complex_injection(__target: tuple[HasTraits, str], __inputs: list[tuple[HasTraits, str]], **kwargs):
    def _injection(__func):
        target, target_trait = __target
        args = inspect.signature(__func).parameters.keys()

        def func(event):
            params = {arg: getattr(input, input_trait) for arg, (input, input_trait) in zip(args, __inputs)}
            params.update(kwargs)
            value = __func(**params)
            setattr(target, target_trait, value)

        for input, input_trait in __inputs:
            input.observe(func, input_trait)

    return _injection

def injection_dlink(input: HasTraits, input_trait: str, target: HasTraits, target_trait: str):
    def func(event):
        try:
            setattr(target, target_trait, getattr(input, input_trait))
        except TraitError as err:
            debug(err)
            pass
    input.observe(func, input_trait)


def on_click(button: Button, *args: tuple[HasTraits, str], **kwargs):
    def _on_click(func):
        debug("Setup on_click", args)
        def _func(event):
            params = {trait: getattr(widget, trait) for widget, trait in args}
            params.update(kwargs)
            return func(**params)

        button.on_click(_func)

    return _on_click


def set_property(widget: HasTraits):
    def _property(func):
        target_trait = func.__name__
        if hasattr(widget, target_trait):
            raise ValueError(f"{type(widget)}.{target_trait} already exists.")
        input_traits = {name for name in inspect.signature(func).parameters}

        def _func(_self):
            params = {name: getattr(_self, name) for name in input_traits}
            return func(**params)

        setattr(type(widget), target_trait, property(_func))

    return _property


def set_state(widget: HasTraits, trait: str, value=...):
    if not widget.has_trait(trait):
        if hasattr(widget, trait):
            raise ValueError(f"Interface.{trait} is not a trait but a {type(getattr(widget, trait))}.")
        widget.add_traits(**{trait: Any()})
    if value is not ...:
        setattr(widget, trait, value)


def new_tab(tab, title, widget):
    tab.children = (*tab.children, widget)
    tab.set_title(len(tab.children) - 1, title)


def linked_text(cs: HasTraits, trait: str, description: str, value=...):
    set_state(cs, trait)
    input = Text(description=description)
    link((input, "value"), (cs, trait))
    if value is not ...:
        input.value = value
    return input


def linked_dropdown(options_widget, options_trait, value_widget, value_trait, description):
    dropdown = Dropdown(description=description)
    injection_dlink(options_widget, options_trait, dropdown, "options")
    # link((options_widget, options_trait), (dropdown, "options"))
    set_state(value_widget, value_trait)
    injection_dlink(dropdown, "value", value_widget, value_trait)
    injection_dlink(value_widget, value_trait, dropdown, "value")
    return dropdown


def add_setup_tab(interface, cs):
    input__drive_folder = linked_text(cs, "drive_folder", description="Drive folder:")

    @injection(cs, cs)
    def base_path(drive_mount, drive_folder, environment):
        if environment == "local":
            return Path(drive_mount) / drive_folder
        elif environment == "colab":
            return Path(drive_mount) / "MyDrive" / drive_folder

    @injection(cs, cs)
    def cases(base_path):
        try:
            return dataset.path_explorer.discover(base_path)
        except Exception as err:
            debug(err)
            return []

    @injection(cs, cs)
    def case(cases):
        if cases:
            return cases[0]
        else:
            return None

    dropdown__case = linked_dropdown(cs, "cases", cs, "case", description="Select case:")

    @injection(cs, cs)
    def case_path(base_path, case):
        try:
            return base_path / case
        except Exception as err:
            debug(err)
            return None

    label__isdicom = HTML()

    @injection(cs, label__isdicom)
    def value(case, case_path):
        if case:
            if dataset.path_explorer.is_dicom(case_path):
                return f"<i class='fa fa-check-square-o' aria-hidden='true'></i> Case {case} is a dicomdir."
            else:
                return f"<i class='fa fa-square-o' aria-hidden='true'></i> Case {case} is NOT a dicomdir."

    label__original = HTML()

    @injection(cs, label__original)
    def value(case, case_path):
        if case:
            if dataset.path_explorer.is_original(case_path):
                return f"<i class='fa fa-check-square-o' aria-hidden='true'></i> Case {case} has the original scans converted to nifti."
            else:
                return f"<i class='fa fa-square-o' aria-hidden='true'></i> Case {case} does not have the original scans in nifti format."

    label__registered = HTML()

    @injection(cs, label__registered)
    def value(case, case_path):
        if case:
            if dataset.path_explorer.is_registered(case_path):
                return f"<i class='fa fa-check-square-o' aria-hidden='true'></i> Case {case} is registered."
            else:
                return f"<i class='fa fa-square-o' aria-hidden='true'></i> Case {case} does not have the registered scans."

    new_tab(
        interface, "Setup",
        HBox([
            VBox([
                HTML("<h3>Set these variables</h3>"),
                input__drive_folder,
                dropdown__case,
            ]),
            VBox([
                HTML("<h3>About the selected case:</h3>"),
                label__isdicom,
                label__original,
                label__registered,
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )


def add_preprocess_tab(interface, cs):
    dropdown__case = linked_dropdown(cs, "cases", cs, "case", description="Select case:")
    button__convert = Button(layout=Layout(width='auto'))

    @injection(cs, button__convert)
    def description(case, case_path):
        if case:
            if dataset.path_explorer.is_dicom(case_path):
                if dataset.path_explorer.is_original(case_path):
                    return f"Convert only {str(case)} (overwrite!)"
                else:
                    return f"Convert only {str(case)}"
            else:
                return f"Can't convert {case} because it is not a dicomdir."
        else:
            return "No case selected!"

    @injection(cs, button__convert)
    def disabled(case, case_path):
        if case:
            if dataset.path_explorer.is_dicom(case_path):
                return False
        return True

    @injection(cs, button__convert)
    def button_style(case, case_path):
        if case and dataset.path_explorer.is_dicom(case_path) and dataset.path_explorer.is_original(case_path):
            return "warning"
        return "info"

    @on_click(button__convert, (cs, "case_path"), overwrite=True)
    def callback(*, case_path, overwrite):
        opts = argparse.Namespace(
            sources=case_path,
            outputs=case_path,
            overwrite=overwrite
        )
        scripts.dicom2nifti.main(opts)

    button__niftyreg = Button(layout=Layout(width='auto'))

    @injection(cs, button__niftyreg)
    def description(case, case_path):
        if case:
            if dataset.path_explorer.is_original(case_path):
                if dataset.path_explorer.is_registered(case_path):
                    return f"Register {str(case)} with NiftyReg (overwrite!)"
                else:
                    return f"Register {str(case)} with NiftyReg"
            else:
                return f"Can't register {case} because there are no originals."
        else:
            return "No case selected!"

    @injection(cs, button__niftyreg)
    def disabled(case, case_path):
        if case:
            if dataset.path_explorer.is_original(case_path):
                return False
        return True

    @injection(cs, button__niftyreg)
    def button_style(case, case_path):
        if case and dataset.path_explorer.is_original(case_path) and dataset.path_explorer.is_registered(case_path):
            return "warning"
        return "info"

    @on_click(button__niftyreg, (cs, "case_path"), overwrite=True)
    def callback(*, case_path, overwrite):
        opts = argparse.Namespace(
            sources=case_path,
            outputs=case_path,
            overwrite=overwrite,
            niftybin="/usr/local/bin"
        )
        scripts.niftyreg.main(opts)

    button__pyelastix = Button(layout=Layout(width='auto'))

    @injection(cs, button__pyelastix)
    def description(case, case_path):
        if case:
            if dataset.path_explorer.is_original(case_path):
                if dataset.path_explorer.is_registered(case_path):
                    return f"Register {str(case)} with PyElastix (overwrite!)"
                else:
                    return f"Register {str(case)} with PyElastix"
            else:
                return f"Can't register {case} because there are no originals."
        else:
            return "No case selected!"

    @injection(cs, button__pyelastix)
    def disabled(case, case_path):
        if case:
            if dataset.path_explorer.is_original(case_path):
                return False
        return True

    @injection(cs, button__pyelastix)
    def button_style(case, case_path):
        if case and dataset.path_explorer.is_original(case_path) and dataset.path_explorer.is_registered(case_path):
            return "warning"
        return "info"

    @on_click(button__pyelastix, (cs, "case_path"), overwrite=True)
    def callback(*, case_path, overwrite):
        opts = argparse.Namespace(
            sources=case_path,
            outputs=case_path,
            overwrite=overwrite,
        )
        scripts.pyelastix.main(opts)

    new_tab(
        interface, "Preprocessing",
        HBox([
            VBox([
                HTML("<h3>Single case operations</h3>"),
                dropdown__case,
                button__convert,
                button__niftyreg,
                button__pyelastix,
                HTML("<h3>Operations on the whole dataset</h3>"),

            ])
        ])
    )

def add_visualization_tab(interface, cs):
    content_holder = HasTraits()
    set_state(content_holder, "content")
    # content = None
    channels = Dropdown(
        description="What to see:",
        options=[
            ("Original phase b", ("orig", "b")),
            ("Original phase a", ("orig", "a")),
            ("Original phase v", ("orig", "v")),
            ("Original phase t", ("orig", "t")),
        ]
    )
    z_slider = IntSlider(description="Z slice:")

    @reaction(cs)
    def func(case_path):
        channels.value = None

    @complex_reaction((cs, "case_path"), (channels, "value"))
    def func(case_path, channel_choice):
        if channel_choice and cs.case_path and isinstance(channel_choice, tuple) and len(channel_choice)>0:
            command = channel_choice[0]
            try:
                if command == "orig":
                    phase = channel_choice[1]
                    orig, matrix = dataset.ndarray.load_original(case_path, phase=phase)
                    content_holder.content = orig.clip(0, 255) / 255
                    z_slider.max = content_holder.content.shape[-1]
            except Exception as err:
                debug("EXCEPTION!", err)

    tv_control = VBox([
        linked_dropdown(cs, "cases", cs, "case", description="Select case:"),
        channels,
        z_slider,
    ])
    tv_output = Output()

    new_tab(
        interface, "Visualize",
        HBox([
            tv_control,
            tv_output
        ])
    )
    return tv_output

def build_interface(cs: HasTraits):
    interface = Tab()
    set_state(cs, "Drive_mount")
    set_state(cs, "Environment")

    add_setup_tab(interface, cs)
    add_preprocess_tab(interface, cs)
    tv_output = add_visualization_tab(interface, cs)

    return interface, tv_output

    # interface.new_tab(
    #     "Developer's stuff",
    #     HBox([
    #         VBox([
    #             HTML("<h3>This is not the tab you are looking for</h3>"),
    #             input__drive_mount,
    #             test_ndarray,
    #             test_path_explorer,
    #             test_preprocessing,
    #             test_slicing
    #         ])
    #     ])
    # )


class CommonState(HasTraits):
    def __repr__(self):
        return "<Common state>"


common_state = CommonState()


#######################################################################################################################

def old():
    class Interface(Tab):
        def __init__(self, **kwargs):
            Tab.__init__(self, **kwargs)
            self.traits_tree = {}

        def inject(self, widget):
            def _depend(func):
                trait = func.__name__
                states = {name for name in inspect.signature(func).parameters}

                def _func(event):
                    params = {name: getattr(self, name) for name in states}
                    setattr(widget, trait, func(**params))

                flagged_states = {name for state in states for name in self.traits_tree[state]}
                self.observe(_func, flagged_states)

            return _depend

        def ensure_state(self, name, value=...):
            if not self.has_trait(name):
                if hasattr(self, name):
                    raise ValueError(f"Interface.{name} is not a trait but a {type(getattr(self, name))}.")
                self.add_traits(**{name: Any()})
                self.traits_tree[name] = {name}
            if value is not ...:
                setattr(self, name, value)

        def identity(self, state: str, widget: HasTraits, trait: str = "value", value=...):
            self.ensure_state(state, value=value)
            link((self, state), (widget, trait))
            return widget

        def on_click(self, widget: Button, *args, **kwargs):
            def _on_click(func):
                def _func(event):
                    params = {state: getattr(self, state) for state in args}
                    params.update(kwargs)
                    return func(**params)

                widget.on_click(_func)

            return _on_click

        def set_property(self, func):
            out_state = func.__name__
            if hasattr(self, out_state):
                raise ValueError(f"Interface.{out_state} already exists.")
            in_states = {name for name in inspect.signature(func).parameters}

            def _func(_self):
                params = {name: getattr(_self, name) for name in in_states}
                return func(**params)

            setattr(type(self), out_state, property(_func))
            base_states = {name for state in in_states for name in self.traits_tree[state]}
            if out_state in self.traits_tree:
                raise ValueError(f"interface.traits_tree already has {out_state}")
            self.traits_tree[out_state] = base_states

    class HasTraitsTree(HasTraits):
        def __init__(self, *args, **kwargs):
            super(HasTraitsTree, self).__init__(*args, **kwargs)
            self.traits_tree = {}

        def set_state(self, name, value=...):
            if not self.has_trait(name):
                if hasattr(self, name):
                    raise ValueError(f"{type(self)}.{name} is not a trait but a {type(getattr(self, name))}.")
                self.add_traits(**{name: Any()})
            if not name in self.traits_tree:
                self.traits_tree[name] = {name}
            if value is not ...:
                setattr(self, name, value)

    class TvOutput(HasTraits):
        def __init__(self, state, *args, **kwargs):
            super(TvOutput, self).__init__(*args, **kwargs)
            self.state = state


    class TvDropdown(HasTraitsTree, Dropdown):
        def __init__(self, *args, **kwargs):
            super(TvDropdown, self).__init__(*args, **kwargs)
            self.set_state("options")
            self.set_state("value")

        def add_option(self, label, func, *args, **kwargs):
            params = {trait: getattr(self, trait) for trait in args}
            params.update(kwargs)
            self.options = (*self.options, (label, functools.partial(func, **params)))

    tv_dropdown = interface_module.TvDropdown(
        layout=Layout(width='auto'),
        style={'description_width': 'initial'},
    )

    tv_dropdown.set_state("case_path")

    @interface_module.injection(interface, tv_dropdown)
    def case_path(case_path):
        return case_path

    @interface_module.injection(tv_dropdown, tv_output)
    def content(value):
        print("Change in dropwdown", value)
        try:
            return value()
        except Exception as err:
            print("Error", err)
            return None

    def load_original(case_path, phase):
        print("Case path", case_path)
        orig, matrix = dataset.ndarray.load_original(case_path, phase=phase)
        return orig.clip(0, 255) / 255

    tv_dropdown.add_option("Original phase b", load_original, "case_path", phase='b')
    tv_dropdown.add_option("Original phase a", load_original, "case_path", phase='a')
    tv_dropdown.add_option("Original phase v", load_original, "case_path", phase='v')
    tv_dropdown.add_option("Original phase t", load_original, "case_path", phase='t')

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
