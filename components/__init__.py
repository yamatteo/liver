import argparse

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Text, Output, Tab, HBox, VBox, HTML, Layout, Dropdown, Button, IntSlider

import dataset.ndarray
import dataset.path_explorer
import scripts.dicom2nifti
import scripts.niftyreg
import scripts.pyelastix
from .card_console import Console
from .commonstate import get_new_state
from .visualization import build_tv


def new_tab(tab, title, widget):
    tab.children = (*tab.children, widget)
    tab.set_title(len(tab.children) - 1, title)


def get_new_interface():
    state, react, inject, project, biject = get_new_state()
    tab = Tab()

    ######################################################## Setup ########################################################

    drive_folder_input = Text(description="Drive folder:")
    biject("drive_folder", drive_folder_input)

    console = Console(layout=Layout(max_height="10cm"))


    @react
    def print_cases(cases):
        with console.new_card():
            if cases:
                print("In the selected folder there are these cases:")
                for case in cases:
                    print(" ", case)
            else:
                print(f"No case in the selected directory: {state.base_path}")


    new_tab(
        tab, "Setup",
        HBox([
            VBox([
                HTML("<h3>Set these variables</h3>"),
                drive_folder_input,
            ]),
            VBox([
                console
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )

    #################################################### Preprocessing ####################################################

    case_dropdown = Dropdown(description="Select case:")


    @inject(case_dropdown, except_to=[])
    def options(cases):
        return cases


    biject("case", case_dropdown, except_to=None)

    button__convert = Button(layout=Layout(width='auto'))


    @inject(button__convert, except_to="---")
    def description(case, case_path):
        if dataset.path_explorer.is_dicom(case_path):
            if dataset.path_explorer.is_original(case_path):
                return f"Convert only {str(case)} (overwrite!)"
            else:
                return f"Convert only {str(case)}"
        else:
            return f"Can't convert {case} because it is not a dicomdir."


    @inject(button__convert, except_to=True)
    def disabled(case_path):
        if dataset.path_explorer.is_dicom(case_path):
            return False
        return True


    @inject(button__convert, except_to="info")
    def button_style(case_path):
        if dataset.path_explorer.is_dicom(case_path) and dataset.path_explorer.is_original(case_path):
            return "warning"
        return "info"


    @button__convert.on_click
    def callback(*args, **kwargs):
        opts = argparse.Namespace(
            sources=state.case_path,
            outputs=state.case_path,
            overwrite=True
        )
        with console.new_card():
            scripts.dicom2nifti.main(opts)


    button__niftyreg = Button(layout=Layout(width='auto'))


    @inject(button__niftyreg, except_to="---")
    def description(case, case_path):
        if dataset.path_explorer.is_original(case_path):
            if dataset.path_explorer.is_registered(case_path):
                return f"Register {str(case)} with NiftyReg (overwrite!)"
            else:
                return f"Register {str(case)} with NiftyReg"
        else:
            return f"Can't register {case}: no originals."


    @inject(button__niftyreg, except_to=True)
    def disabled(case_path):
        if dataset.path_explorer.is_original(case_path):
            return False
        return True


    @inject(button__niftyreg, except_to="info")
    def button_style(case_path):
        if dataset.path_explorer.is_original(case_path) and dataset.path_explorer.is_registered(case_path):
            return "warning"
        return "info"


    @button__niftyreg.on_click
    def callback(*args, **kwargs):
        opts = argparse.Namespace(
            sources=state.case_path,
            outputs=state.case_path,
            overwrite=True,
            niftybin="/usr/local/bin"
        )
        with console.new_card():
            scripts.niftyreg.main(opts)


    button__pyelastix = Button(layout=Layout(width='auto'))


    @inject(button__pyelastix, except_to="---")
    def description(case, case_path):
        if dataset.path_explorer.is_original(case_path):
            if dataset.path_explorer.is_registered(case_path):
                return f"Register {str(case)} with PyElastix (overwrite!)"
            else:
                return f"Register {str(case)} with PyElastix"
        else:
            return f"Can't register {case}: no originals."


    @inject(button__pyelastix, except_to=True)
    def disabled(case_path):
        if dataset.path_explorer.is_original(case_path):
            return False
        return True


    @inject(button__pyelastix, except_to="info")
    def button_style(case_path):
        if dataset.path_explorer.is_original(case_path) and dataset.path_explorer.is_registered(case_path):
            return "warning"
        return "info"


    @button__pyelastix.on_click
    def callback(*args, **kwargs):
        opts = argparse.Namespace(
            sources=state.case_path,
            outputs=state.case_path,
            overwrite=True,
        )
        with console.new_card():
            scripts.pyelastix.main(opts)


    new_tab(
        tab, "Preprocessing",
        HBox([
            VBox([
                HTML("<h3>Single case operations</h3>"),
                case_dropdown,
                button__convert,
                button__niftyreg,
                button__pyelastix,
                HTML("<h3>Operations on the whole dataset</h3>"),

            ]),
            VBox([
                console
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )

    ################################################### Visualization #####################################################
    channels, z_slider, tv = build_tv(state, inject, project, biject)

    new_tab(
        tab, "Visualization",
        HBox([
            VBox([
                HTML("<h3>Select what to see</h3>"),
                case_dropdown,
                channels,
                z_slider,
            ]),
            VBox([
                tv
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )

    return state, tab