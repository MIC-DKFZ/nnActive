import functools
import math
import multiprocessing
import multiprocessing as mp
import os
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Iterable

import nnunetv2.paths
import torch
import wandb
from loguru import logger
from nnunetv2.run.run_training import run_training
from nnunetv2.training.dataloading.utils import unpack_dataset

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_preprocessed_path
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder


def wrap_training(
    dataset_id: int,
    config: ActiveConfig,
    folds: Iterable[int],
    device: torch.device,
    wandbgroup: str,
):
    config.set_nnunet_env()
    with monitor.active_run(group=wandbgroup):
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


@register_subcommand("train_nnUNet_ensemble")
def train_nnUNet_ensemble(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    force: bool = False,
):
    config.set_nnunet_env()

    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)

    # ensure that set_num_interop is not executed twice
    # multithreading in torch doesn't help nnU-Net if run on GPU
    try:
        os.environ["torchset"]
    except KeyError:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["torchset"] = "True"

    num_folds = config.train_folds

    # Custom preprocessor handles deleting of old _seg.npy files
    npp = (
        runtime_config.num_processes
        if runtime_config.n_gpus == 0
        else runtime_config.num_processes * runtime_config.n_gpus
    )
    unpack_dataset(
        folder=str(
            get_preprocessed_path(state.dataset_id)
            / "_".join([config.model_plans, config.model_config]),
        ),
        unpack_segmentation=True,
        overwrite_existing=False,
        num_processes=npp,
        verify_npy=False,
    )

    if runtime_config.n_gpus == 0:
        device = torch.device("cuda:0")
        for fold in range(num_folds):
            run_training(
                str(
                    state.dataset_id
                ),  # TODO: fix this bug in nnU-Net requiring input to be string.
                config.model_config,
                fold,
                trainer_class_name=config.trainer,
                device=device,
                logger=monitor.get_logger(),
            )
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(runtime_config.n_gpus)]
        folds = [
            [fold for fold in range(num_folds) if fold % runtime_config.n_gpus == d]
            for d in range(runtime_config.n_gpus)
        ]
        try:
            with ProcessPoolExecutor(
                max_workers=runtime_config.n_gpus, mp_context=mp.get_context("spawn")
            ) as executor:
                for _ in executor.map(
                    wrap_training,
                    [state.dataset_id] * num_folds,
                    [config] * num_folds,
                    folds,
                    devices,
                    [wandb.run.group] * num_folds,
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
