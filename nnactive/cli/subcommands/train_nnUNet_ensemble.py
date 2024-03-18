import os
import time
from argparse import Namespace

import torch
from nnunetv2.run.run_training import run_training

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.logger import monitor
from nnactive.results.state import State


@register_subcommand(
    "train_nnUNet_ensemble",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-f", "--force"),
            {"action": "store_true", "help": "Ignores the internal State."},
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    force = args.force

    train_nnUNet_ensemble(dataset_id, force)


def train_nnUNet_ensemble(dataset_id: int, force: bool = False):
    # ensure that set_num_interop is not executed twice
    # multithreading in torch doesn't help nnU-Net if run on GPU
    try:
        os.environ["torchset"]
    except KeyError:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["torchset"] = "True"

    device = torch.device("cuda")
    config = ActiveConfig.get_from_id(dataset_id)
    num_folds = config.working_folds
    state = State.get_id_state(dataset_id, verify=not force)

    for fold in range(num_folds):
        run_training(
            str(
                dataset_id
            ),  # TODO: fix this bug in nnU-Net requiring input to be string.
            config.model_config,
            fold,
            trainer_class_name=config.trainer,
            device=device,
            logger=monitor.get_logger(),
        )

    if not force:
        state.training = True
        state.save_state()
