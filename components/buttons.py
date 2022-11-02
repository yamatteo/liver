from ipywidgets import Button, Layout
from rich.console import Console

import april_model
import utils.path_explorer as px
import july_model
import newmodel as newmodel

from preprocessing import process_dicomdir
from utils import extractor
from . import reactions
from preprocessing import pyelastix

console = Console()


class ButtonGenerator:
    convert = Button(layout=Layout(width='auto'))
    convert_all = Button(layout=Layout(width='auto'))
    register = Button(layout=Layout(width='auto'))
    register_all = Button(layout=Layout(width='auto'))

    april_one = Button(
        description="Apply april's model",
        layout=Layout(width='auto')
    )
    april_all = Button(
        description="Apply april's model to all",
        layout=Layout(width='auto')
    )
    april_evaluate = Button(
        description="Evaluate april's model",
        layout=Layout(width='auto')
    )

    july_one = Button(
        description="Apply july's model",
        layout=Layout(width='auto')
    )
    july_all = Button(
        description="Apply july's model to all",
        layout=Layout(width='auto')
    )
    july_evaluate = Button(
        description="Evaluate july's model",
        layout=Layout(width='auto')
    )

    newmodel_one = Button(
        description="Apply newmodel",
        layout=Layout(width='auto')
    )
    newmodel_all = Button(
        description="Apply newmodel to all",
        layout=Layout(width='auto')
    )
    newmodel_evaluate = Button(
        description="Evaluate newmodel",
        layout=Layout(width='auto')
    )

    extract_features = Button(
        description="Extract features",
        layout=Layout(width='auto')
    )

    def __init__(self, state, output_console):
        _, inject, _, _ = reactions.state_reactions(state)

        # Convert one button
        @inject(self.convert, except_to="---")
        def description(case, case_path):
            if px.is_dicom(case_path):
                if px.is_original(case_path):
                    return f"Convert {str(case)} (overwrite!)"
                else:
                    return f"Convert {str(case)}"
            else:
                return f"Can't convert {case} because it is not a dicomdir."

        @inject(self.convert, except_to=True)
        def disabled(case_path):
            if px.is_dicom(case_path):
                return False
            return True

        @inject(self.convert, except_to="info")
        def button_style(case_path):
            if px.is_dicom(case_path) and px.is_original(case_path):
                return "warning"
            return "info"

        @self.convert.on_click
        def callback(*args, **kwargs):
            case_path = state.case_path
            with output_console.new_card():
                console.print(f"[bold orange]Converting:[/bold orange] {case_path.name}...")
                process_dicomdir(case_path, case_path)
                console.print(f"            ...completed.")

        # Convert all button
        @inject(self.convert_all, except_to="---")
        def description(cases):
            if cases:
                return f"Convert all cases (without overwriting)"
            else:
                return f"No case in the selected folder."

        @inject(self.convert_all, except_to=True)
        def disabled(cases):
            if len(cases) > 0:
                return False
            return True

        self.convert_all.button_style = "warning"

        @self.convert_all.on_click
        def callback(*args, **kwargs):
            base_path = state.base_path
            with output_console.new_card():
                console.print(f"[bold orange]Converting:[/bold orange]")
                for case_path in px.iter_dicom(base_path):
                    target_path = base_path / case_path
                    target_path_is_complete = all(
                        (target_path / f"original_phase_{phase}.nii.gz").exists()
                        for phase in ["b", "a", "v", "t"]
                    )
                    if not target_path_is_complete:
                        target_path.mkdir(parents=True, exist_ok=True)
                        console.print(f"  [bold black]{target_path.name}.[/bold black] converting...")
                        process_dicomdir(base_path / case_path, target_path)
                        console.print(f"   {' ' * len(target_path.name)}  ...completed.")
                    else:
                        console.print(f"  [bold black]{case_path.name}.[/bold black] is already complete, skipping.")

        # Register one
        @inject(self.register, except_to="---")
        def description(case, case_path):
            if px.is_original(case_path):
                if px.is_registered(case_path):
                    return f"Register {str(case)} with PyElastix (overwrite!)"
                else:
                    return f"Register {str(case)} with PyElastix"
            else:
                return f"Can't register {case}: no originals."

        @inject(self.register, except_to=True)
        def disabled(case_path):
            if px.is_original(case_path):
                return False
            return True

        @inject(self.register, except_to="info")
        def button_style(case_path):
            if px.is_original(case_path) and px.is_registered(case_path):
                return "warning"
            return "info"

        @self.register.on_click
        def callback(*args, **kwargs):
            case_path = state.case_path
            with output_console.new_card():
                console.print(f"[bold orange3]Registering:[/bold orange3] {case_path.stem}...")
                pyelastix.register_case(case_path)
                console.print(f"[bold orange3]            [/bold orange3] ...completed.")

        # Register all
        @inject(self.register_all, except_to="---")
        def description(cases):
            if cases:
                return f"Register all cases (without overwriting)"
            else:
                return f"No case in the selected folder."

        @inject(self.register_all, except_to=True)
        def disabled(cases):
            if len(cases) > 0:
                return False
            return True

        self.register_all.button_style = "warning"

        @self.register_all.on_click
        def callback(*args, **kwargs):
            base_path = state.base_path
            with output_console.new_card():
                console.print(f"[bold orange3]Registering:[/bold orange3]")
                for case_path in px.iter_original(base_path):
                    target_path = base_path / case_path
                    target_path_is_complete = (
                            (target_path / "registration_data.pickle").exists()
                            and all(
                        (target_path / f"registered_phase_{phase}.nii.gz").exists()
                        for phase in ["b", "a", "v", "t"]
                    ))
                    if not target_path_is_complete:
                        target_path.mkdir(parents=True, exist_ok=True)
                        console.print(f"  [bold black]{target_path.name}.[/bold black] registering...")
                        pyelastix.register_case(target_path)
                        console.print(f"   {' ' * len(target_path.name)}  ...completed.")
                    else:
                        console.print(f"  [bold black]{target_path.name}.[/bold black] is already complete, skipping.")

        @self.april_one.on_click
        def callback(event):
            case_path = state.case_path
            with output_console.new_card():
                april_model.apply_to_one_folder(case_path)

        @self.april_all.on_click
        def callback(*args, **kwargs):
            base_path = state.base_path
            with output_console.new_card():
                april_model.apply_to_all_folders(base_path)

        @self.april_evaluate.on_click
        def callback(event):
            base_path = state.base_path
            with output_console.new_card():
                april_model.evaluate(base_path)

        @self.july_one.on_click
        def callback(event):
            case_path = state.case_path
            with output_console.new_card():
                july_model.predict_one_folder(case_path)

        @self.july_all.on_click
        def callback(*args, **kwargs):
            base_path = state.base_path
            with output_console.new_card():
                july_model.predict_all_folders(base_path)

        @self.july_evaluate.on_click
        def callback(event):
            base_path = state.base_path
            with output_console.new_card():
                july_model.evaluate_all_folders(base_path)

        @self.newmodel_one.on_click
        def callback(event):
            case_path = state.case_path
            with output_console.new_card():
                newmodel.predict_one_folder(case_path)

        @self.newmodel_all.on_click
        def callback(*args, **kwargs):
            base_path = state.base_path
            with output_console.new_card():
                newmodel.predict_all_folders(base_path)

        @self.newmodel_evaluate.on_click
        def callback(event):
            base_path = state.base_path
            with output_console.new_card():
                newmodel.evaluate_all_folders(base_path)

        @self.extract_features.on_click
        def callback(event):
            case_path = state.case_path
            with output_console.new_card():
                extractor.extract(case_path)
