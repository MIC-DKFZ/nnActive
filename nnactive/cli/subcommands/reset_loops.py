import os.path
import shutil
from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.cli.subcommands.update_data import update_step
from nnactive.config import ActiveConfig
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder

"""
Resets the experiment with the given dataset id to loop 0.
Attention: This will delete all loop files from the already performed loops!
"""


@register_subcommand(
    "reset_loops",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    config = ActiveConfig.get_from_id(dataset_id)
    raw_dataset_path = get_raw_path(dataset_id)
    loop_val = len(get_sorted_loop_files(raw_dataset_path))
    state = State.get_id_state(dataset_id)
    if os.path.isfile(raw_dataset_path / f"{config.uncertainty}_{loop_val:03d}.json"):
        print(
            "Deleting file ",
            str(raw_dataset_path / f"{config.uncertainty}_{loop_val:03d}.json"),
        )
        os.remove(raw_dataset_path / f"{config.uncertainty}_{loop_val:03d}.json")
    elif os.path.isfile(
        raw_dataset_path / f"{config.uncertainty}_{state.loop:03d}.json"
    ):
        print(
            "Deleting file ",
            str(raw_dataset_path / f"{config.uncertainty}_{state.loop:03d}.json"),
        )
        os.remove(raw_dataset_path / f"{config.uncertainty}_{state.loop:03d}.json")

    update_step(dataset_id, loop_val=0, annotated=True, force=True)

    for loop_file in get_sorted_loop_files(raw_dataset_path)[1:]:
        print("Deleting file ", str(raw_dataset_path / loop_file))
        os.remove(raw_dataset_path / loop_file)

    nnactive_results_folder = get_nnactive_results_folder(dataset_id)
    for folder in os.listdir(nnactive_results_folder):
        if folder.startswith("loop"):
            print("Deleting folder ", str(nnactive_results_folder / folder))
            shutil.rmtree(nnactive_results_folder / folder)

    print("Resetting State file...")
    state.reset()
    state.save_state()
