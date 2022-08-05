from ipywidgets import Text, Tab, HBox, VBox, HTML, Layout, Dropdown

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

    new_tab(
        tab, "Preprocessing",
        HBox([
            VBox([
                HTML("<h3>Single case operations</h3>"),
                case_dropdown,
                buttons.convert,
                # button__niftyreg,
                buttons.register,
                HTML("<h3>Operations on the whole utils</h3>"),
                buttons.convert_all,
                buttons.register_all,

            ]),
            VBox([
                console
            ], layout=Layout(width='auto', margin="0px 20px")),
        ], layout=Layout(width='auto'))
    )

    ################################################ Segmentation #####################################################

    new_tab(
        tab, "Segmentation",
        HBox([
            VBox([
                HTML("<h4>Single case operations</h4>"),
                case_dropdown,
                buttons.april_one,
                buttons.july_one,
                HTML("<h4>Operations on the whole utils</h4>"),
                buttons.april_all,
                buttons.july_all,
                HTML("<h4>Evaluation</h4>"),
                buttons.april_evaluate,
                buttons.july_evaluate,
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
