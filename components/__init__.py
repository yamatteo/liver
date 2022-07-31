import argparse

from ipywidgets import Text, Tab, HBox, VBox, HTML, Layout, Dropdown, Button

import april_model
import dataset.ndarray
import dataset.path_explorer
import july_model
import scripts.dicom2nifti
import scripts.niftyreg
import scripts.pyelastix
from .buttons import ButtonGenerator
from .card_console import Console
from .commonstate import get_new_state
from .visualization import build_tv


def new_tab(tab, title, widget):
    tab.children = (*tab.children, widget)
    tab.set_title(len(tab.children) - 1, title)


def get_new_interface():
    state, react, inject, project, biject = get_new_state()
    console = Console(layout=Layout(max_height="10cm"))
    buttons = ButtonGenerator(state, console)
    tab = Tab()

    ######################################################## Setup ########################################################

    drive_folder_input = Text(description="Drive folder:")
    biject("drive_folder", drive_folder_input)



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

    # button__convert = Button(layout=Layout(width='auto'))
    #
    #
    # @inject(button__convert, except_to="---")
    # def description(case, case_path):
    #     if dataset.path_explorer.is_dicom(case_path):
    #         if dataset.path_explorer.is_original(case_path):
    #             return f"Convert only {str(case)} (overwrite!)"
    #         else:
    #             return f"Convert only {str(case)}"
    #     else:
    #         return f"Can't convert {case} because it is not a dicomdir."
    #
    #
    # @inject(button__convert, except_to=True)
    # def disabled(case_path):
    #     if dataset.path_explorer.is_dicom(case_path):
    #         return False
    #     return True
    #
    #
    # @inject(button__convert, except_to="info")
    # def button_style(case_path):
    #     if dataset.path_explorer.is_dicom(case_path) and dataset.path_explorer.is_original(case_path):
    #         return "warning"
    #     return "info"
    #
    #
    # @button__convert.on_click
    # def callback(*args, **kwargs):
    #     opts = argparse.Namespace(
    #         sources=state.case_path,
    #         outputs=state.case_path,
    #         overwrite=True
    #     )
    #     with console.new_card():
    #         scripts.dicom2nifti.main(opts)


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
                buttons.convert,
                # button__niftyreg,
                buttons.register,
                HTML("<h3>Operations on the whole dataset</h3>"),
                buttons.convert_all,
                buttons.register_all,

            ]),
            VBox([
                console
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )
    
    ################################################ Segmentation #####################################################

    button__april = Button(layout=Layout(width='auto'))

    @inject(button__april, except_to="---")
    def description(case, case_path):
        if dataset.path_explorer.is_registered(case_path):
            if dataset.path_explorer.is_predicted(case_path):
                return f"Predict {str(case)} with APRIL's model (overwrite!)"
            else:
                return f"Predict {str(case)} with APRIL's model"
        else:
            return f"{case} is not registered."

    @inject(button__april, except_to=True)
    def disabled(case_path):
        if dataset.path_explorer.is_registered(case_path):
            return False
        return True

    @inject(button__april, except_to="info")
    def button_style(case_path):
        if dataset.path_explorer.is_registered(case_path) and dataset.path_explorer.is_predicted(case_path):
            return "warning"
        return "info"

    @button__april.on_click
    def callback(*args, **kwargs):
        with console.new_card():
            april_model.eval_one_folder(state.case_path)
            
    button__july = Button(layout=Layout(width='auto'))

    @inject(button__july, except_to="---")
    def description(case, case_path):
        if dataset.path_explorer.is_registered(case_path):
            if dataset.path_explorer.is_predicted(case_path):
                return f"Predict {str(case)} with JULY's model (overwrite!)"
            else:
                return f"Predict {str(case)} with JULY's model"
        else:
            return f"{case} is not registered."

    @inject(button__july, except_to=True)
    def disabled(case_path):
        if dataset.path_explorer.is_registered(case_path):
            return False
        return True

    @inject(button__july, except_to="info")
    def button_style(case_path):
        if dataset.path_explorer.is_registered(case_path) and dataset.path_explorer.is_predicted(case_path):
            return "warning"
        return "info"

    @button__july.on_click
    def callback(*args, **kwargs):
        with console.new_card():
            july_model.eval_one_folder(state.case_path)
            
    new_tab(
        tab, "Segmentation",
        HBox([
            VBox([
                HTML("<h3>Single case operations</h3>"),
                case_dropdown,
                button__april,
                button__july,
                HTML("<h3>Operations on the whole dataset</h3>"),

            ]),
            VBox([
                console
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )

    ################################################# Visualization ###################################################
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


# Cite
# ITK-SNAP  # Paul A. Yushkevich, Joseph Piven, Heather Cody Hazlett, Rachel Gimpel Smith, Sean Ho, James C. Gee, and Guido Gerig. User-guided 3D active contour segmentation of anatomical structures: Significantly improved efficiency and reliability. Neuroimage. 2006 Jul 1; 31(3):1116-28.