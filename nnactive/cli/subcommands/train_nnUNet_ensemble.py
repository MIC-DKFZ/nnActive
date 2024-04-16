import functools
import math
import multiprocessing
import os
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Iterable

import torch
from loguru import logger
from nnunetv2.run.run_training import run_training
from nnunetv2.training.dataloading.utils import unpack_dataset

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_preprocessed_path
from nnactive.results.state import State


@register_subcommand(
    "train_nnUNet_ensemble",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-f", "--force"),
            {"action": "store_true", "help": "Ignores the internal State."},
        ),
        (("--n_gpus"), {"default": 1, "type": int}),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    force = args.force
    n_gpus = args.n_gpus
    config = ActiveConfig.get_from_id(dataset_id)

    with monitor.active_run(config=config.to_dict()):
        logger.info(config)

        train_nnUNet_ensemble(dataset_id, n_gpus, force)


def wrap_training(
    dataset_id: int,
    config: ActiveConfig,
    folds: Iterable[int],
    device: torch.device,
):
    # ensure that each fold/fork is mapped onto one gpu
    torch.cuda.set_device(device)
    for fold in folds:
        logger.info(
            f"Running training fold '{fold}' in process '{multiprocessing.current_process()}' with device '{device}'"
        )
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


def train_nnUNet_ensemble(dataset_id: int, n_gpus=1, force: bool = False):
    # ensure that set_num_interop is not executed twice
    # multithreading in torch doesn't help nnU-Net if run on GPU
    try:
        os.environ["torchset"]
    except KeyError:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["torchset"] = "True"

    config = ActiveConfig.get_from_id(dataset_id)
    num_folds = config.working_folds
    state = State.get_id_state(dataset_id, verify=not force)

    unpack_dataset(
        folder=str(
            get_preprocessed_path(dataset_id)
            / "_".join([config.model_plans, config.model_config])
        ),
        unpack_segmentation=True,
        overwrite_existing=False,
        num_processes=4 * n_gpus,
        verify_npy=False,
    )

    if n_gpus == 1:
        device = torch.device("cuda:0")
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
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
        folds = [
            [fold for fold in range(num_folds) if fold % n_gpus == d]
            for d in range(n_gpus)
        ]
        try:
            with ProcessPoolExecutor(max_workers=n_gpus) as executor:
                for _ in executor.map(
                    wrap_training,
                    [dataset_id] * num_folds,
                    [config] * num_folds,
                    folds,
                    devices,
                ):
                    pass
        except BrokenProcessPool as exc:
            raise MemoryError(
                "One of the worker processes died. "
                "This usually happens because you run out of memory. "
                "Try running with less processes."
            ) from exc

    if not force:
        state.training = True
        state.save_state()
