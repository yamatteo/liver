from pathlib import Path

from traitlets import HasTraits, Any
import dataset.path_explorer

from . import reactions


def get_new_state():
    state = HasTraits()

    state.add_traits(**{
        trait: Any()
        for trait in [
            "base_path",
            "case",
            "cases",
            "case_path",
            "drive_folder",
            "drive_mount",
            "environment",
            "loaded_content",
        ]
    })

    @reactions.injection(state, state, except_to=None)
    def base_path(drive_mount, drive_folder):
        return Path(drive_mount) / "MyDrive" / drive_folder

    @reactions.injection(state, state, except_to=[])
    def cases(base_path):
        return dataset.path_explorer.discover(base_path)

    @reactions.injection(state, state, except_to=None)
    def case_path(base_path, case):
        return base_path / case

    return (state, *reactions.state_reactions(state))
