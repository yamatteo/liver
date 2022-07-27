from pathlib import Path

from traitlets import HasTraits, Any
import dataset.path_explorer

from . import reactions

def get_new_state():
    state = HasTraits()

    traits = ["environment", "drive_mount", "drive_folder", "base_path", "cases", "case", "case_path", "loaded_content"]
    state.add_traits(**{trait: Any() for trait in traits})


    @reactions.injection(state, state, except_to=None)
    def base_path(environment, drive_mount, drive_folder):
        if environment == "local":
            return Path(drive_mount) / drive_folder
        elif environment == "colab":
            return Path(drive_mount) / "MyDrive" / drive_folder


    @reactions.injection(state, state, except_to=[])
    def cases(base_path):
        return dataset.path_explorer.discover(base_path)


    @reactions.injection(state, state, except_to=None)
    def case_path(base_path, case):
        return base_path / case


    react = reactions.reaction(state)


    def inject(widget: HasTraits, except_to=None):
        return reactions.injection(source=state, widget=widget, except_to=except_to)


    def project(widget: HasTraits, except_to=None):
        return reactions.injection(source=widget, widget=state, except_to=except_to)


    def biject(state_trait: str, widget: HasTraits, widget_trait: str = "value", except_to=None):
        f = lambda x: x
        reactions.injection(state, widget, (state_trait,), widget_trait, except_to=except_to)(f)
        reactions.injection(widget, state, (widget_trait,), state_trait, except_to=except_to)(f)

    return state, react, inject, project, biject
