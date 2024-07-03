import os

import nnunetv2.paths
import torch
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.config.struct import RuntimeConfig
from nnactive.logger import monitor
from nnactive.query_pool import query_pool
from nnactive.results.state import State

from .steps import preprocess, step_performance, step_train, step_update


@register_subcommand("run_experiment")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    verbose: bool = False,
) -> None:
    config.set_nnunet_env()

    print(f"{continue_id=}")
    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)

    dataset_id = state.dataset_id

    # TODO: update nnUNet env vars based on config

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
                    config,
                    runtime_config,
                    continue_id=continue_id,
                    verbose=verbose,
                    do_all=do_all,
                )

                state = State.get_id_state(dataset_id)

            if state.training is False:
                # verbose not necessary here.
                monitor.log("task", "training", epoch=al_iteration)
                step_train(config, runtime_config, continue_id=continue_id)
                state = State.get_id_state(dataset_id)
            if state.get_performance is False:
                monitor.log("task", "get_performance", epoch=al_iteration)
                step_performance(
                    config,
                    runtime_config,
                    continue_id=continue_id,
                    verbose=verbose,
                )
                state = State.get_id_state(dataset_id)
            if al_iteration < config.query_steps - 1:
                if state.pred_tr is False and state.query is False:
                    monitor.log("task", "query_pool", epoch=al_iteration)
                    query_pool(
                        config,
                        runtime_config,
                        continue_id=continue_id,
                        verbose=verbose,
                    )
                    state = State.get_id_state(dataset_id)
                if state.update_data is False:
                    monitor.log("task", "update_step", epoch=al_iteration)
                    step_update(config, continue_id=continue_id, annotated=True)
                    state = State.get_id_state(dataset_id)
