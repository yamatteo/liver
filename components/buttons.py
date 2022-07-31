from ipywidgets import Button, Layout
from rich.console import Console
import dataset.path_explorer as px

from preprocessing import process_dicomdir
from . import reactions
from preprocessing import pyelastix

console = Console()


class ButtonGenerator:
    convert = Button(layout=Layout(width='auto'))
    convert_all = Button(layout=Layout(width='auto'))
    register = Button(layout=Layout(width='auto'))
    register_all = Button(layout=Layout(width='auto'))

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
                    print(target_path, list(target_path.iterdir()))
                    print("@pickle", (target_path / "registration_data.pickle").exists())
                    print("phases", [
                            (target_path / f"registered_phase_{phase}.nii.gz").exists()
                            for phase in ["b", "a", "v", "t"]
                    ])
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

