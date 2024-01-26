import os.path
import shutil
from argparse import Namespace

from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.cli.subcommands.nnunet_preprocess import preprocess
from nnactive.cli.subcommands.update_data import update_step
from nnactive.config import ActiveConfig
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder

"""
Resets the experiment with the given dataset id to the given loop number (0 by default).
Attention: This will delete all loop files from the already performed loops until the specified loop!
If reset to loop 0, no preprocessing is performed to completely reset the dataset.
Otherwise, the preprocessing is performed with the do_all flag to do preprocessing until that loop.
"""


@register_subcommand(
    "reset_loops",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
        (
            "--loop_nr",
            {
                "type": int,
                "default": 0,
                "help": "Loop number to reset to. Default is 0.",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    reset_loop_nr = args.loop_nr
    config = ActiveConfig.get_from_id(dataset_id)
    raw_dataset_path = get_raw_path(dataset_id)
    state = State.get_id_state(dataset_id)
    if reset_loop_nr > state.loop:
        raise AttributeError(
            "Loop number to reset to is higher than the current loop number."
        )

    for file in os.listdir(raw_dataset_path):
        if file.startswith(f"{config.uncertainty}_") and file.endswith(".json"):
            if int(file.split("_")[-1].split(".")[0]) > reset_loop_nr:
                print("Deleting file ", str(raw_dataset_path / file))
                os.remove(raw_dataset_path / file)

    update_step(dataset_id, loop_val=reset_loop_nr, annotated=True, force=True)

    for loop_file in get_sorted_loop_files(raw_dataset_path)[reset_loop_nr + 1 :]:
        logger.info("Deleting file ", str(raw_dataset_path / loop_file))
        os.remove(raw_dataset_path / loop_file)

    nnactive_results_folder = get_nnactive_results_folder(dataset_id)
    for folder in os.listdir(nnactive_results_folder):
        if folder.startswith("loop"):
            if int(folder.split("_")[-1]) >= reset_loop_nr:
                logger.info("Deleting folder ", str(nnactive_results_folder / folder))
                shutil.rmtree(nnactive_results_folder / folder)

    logger.info("Resetting State file...")
    state.reset()
    if reset_loop_nr > 0:
        state.loop = reset_loop_nr
    state.save_state()

    # Preprocess with do_all is esp. needed for loop larger 0
    if reset_loop_nr > 0:
        preprocess(
            [dataset_id],
            configurations=[config.model_config],
            num_processes=[config.num_processes],
            do_all=True,
        )
