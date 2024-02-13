import os
import subprocess
from argparse import Namespace

import torch
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.logger import monitor
from nnactive.query_pool import query_pool
from nnactive.results.state import State

from .get_performance import get_performance
from .nnunet_preprocess import preprocess
from .train_nnUNet_ensemble import train_nnUNet_ensemble
from .update_data import update_step


@register_subcommand(
    "run_al_loops",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
        (
            ("--verbose"),
            {
                "action": "store_true",
                "help": "Disables progress bars and get more explicit print statements.",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    verbose = args.verbose

    config = ActiveConfig.get_from_id(dataset_id)
    state = State.get_id_state(dataset_id)

    with monitor.active_run(config=config.to_dict()):
        logger.info(config)

        try:
            os.environ["nnUNet_compile"]
        except KeyError:
            # torch.compile is only available from torch 2.0 onwards
            # see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
            if torch.__version__ >= "2.0":
                os.environ["nnUNet_compile"] = "True"

        for al_iteration in range(config.query_steps):
            if al_iteration < state.loop:
                continue
            if al_iteration > state.loop:
                raise ValueError("A loop has not been executed!")
            if state.preprocess is False:
                monitor.log("task", "preprocess", epoch=al_iteration)
                # Preprocess only images that are annotated
                do_all = al_iteration == 0

                preprocess(
                    [dataset_id],
                    configurations=[config.model_config],
                    num_processes=[config.num_processes],
                    verbose=verbose,
                    do_all=do_all,
                )

                state = State.get_id_state(dataset_id)

            if state.training is False:
                # verbose not necessary here.
                monitor.log("task", "training", epoch=al_iteration)
                train_nnUNet_ensemble(dataset_id, monitor=monitor)
                state = State.get_id_state(dataset_id)
            if state.get_performance is False:
                monitor.log("task", "get_performance", epoch=al_iteration)
                get_performance(dataset_id, verbose=verbose)
                state = State.get_id_state(dataset_id)
            if al_iteration < config.query_steps - 1:
                if state.pred_tr is False and state.query is False:
                    monitor.log("task", "query_pool", epoch=al_iteration)
                    query_pool(dataset_id, verbose=verbose)
                    state = State.get_id_state(dataset_id)
                if state.update_data is False:
                    monitor.log("task", "update_step", epoch=al_iteration)
                    update_step(dataset_id, annotated=True)
                    state = State.get_id_state(dataset_id)
