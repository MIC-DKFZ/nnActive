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
from nnactive.results.utils import get_results_folder
from nnactive.utils.io import save_json
from nnactive.utils.timer import Timer

from .steps import preprocess, step_performance, step_train, step_update


@register_subcommand("run_experiment")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    verbose: bool = False,
    benchmark: bool = False,
    force_run: bool = False,
) -> None:
    config.set_nnunet_env()
    timer_dict: dict[str, Timer] = {}
    timer_dict["Loop Time"] = Timer()
    timer_dict["Runtime"] = Timer()
    timer_dict["Query Time"] = Timer()
    timer_dict["Preprocess Timer"] = Timer()
    timer_dict["Train Time"] = Timer()
    timer_dict["Val Time"] = Timer()
    timer_dict["Data-Update Time"] = Timer()

    timer_dict["Runtime"].start()

    print(f"{continue_id=}")
    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)

    if state.in_progress and not force_run:
        raise RuntimeError(
            f"Training already in progress for experiment {config.name()}. Check "
            "the current trainings or set up a new nnActive experiment."
        )

    continue_id = state.dataset_id
    # Update config file if needed
    existing_config = ActiveConfig.get_from_id(continue_id)
    if config != existing_config:
        logger.info(
            f"Overwriting {ActiveConfig.filename()} with updated configuration."
        )
        config.save_id(continue_id)

    loop_budget = (
        runtime_config.max_loops
        if runtime_config.max_loops is not None
        else config.query_steps
    )

    with monitor.active_run(
        config=config.to_dict(), state=state, state_tag="run_experiment"
    ):
        logger.info(config)

        try:
            os.environ["nnUNet_compile"]
        except KeyError:
            # torch.compile is only available from torch 2.0 onwards
            # see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
            if torch.__version__ >= "2.0":
                os.environ["nnUNet_compile"] = "True"
        state.in_progress = True
        state.save_state()
        try:
            for al_iteration in range(config.query_steps):
                # If loop_budget is 0, we have reached the maximum number of loops
                if loop_budget == 0:
                    break

                timer_dict["Loop Time"].start()
                time_loop = False
                if al_iteration < state.loop:
                    continue
                if al_iteration > state.loop:
                    raise ValueError("A loop has not been executed!")

                if state.preprocess is False:
                    time_loop = True
                    monitor.log("task", "preprocess", epoch=al_iteration)
                    timer_dict["Preprocess Timer"].start()
                    # Preprocess only images that are annotated
                    do_all = al_iteration == 0
                    preprocess(
                        config,
                        runtime_config,
                        continue_id=continue_id,
                        verbose=verbose,
                        do_all=do_all,
                    )
                    preprocess_time = timer_dict["Preprocess Timer"].stop()
                    state = State.get_id_state(continue_id)

                if state.training is False:
                    # verbose not necessary here.
                    monitor.log("task", "training", epoch=al_iteration)
                    timer_dict["Train Time"].start()
                    step_train(
                        config,
                        runtime_config,
                        continue_id=continue_id,
                        raise_on_in_progress=False,
                    )
                    train_time = timer_dict["Train Time"].stop()
                    state = State.get_id_state(continue_id)
                if state.get_performance is False:
                    monitor.log("task", "get_performance", epoch=al_iteration)
                    timer_dict["Val Time"].start()
                    step_performance(
                        config,
                        runtime_config,
                        continue_id=continue_id,
                        verbose=verbose,
                    )
                    performance_time = timer_dict["Val Time"].stop()
                    state = State.get_id_state(continue_id)
                if al_iteration < config.query_steps - 1:
                    if state.pred_tr is False and state.query is False:
                        monitor.log("task", "query_pool", epoch=al_iteration)
                        timer_dict["Query Time"].start()
                        query_pool(
                            config,
                            runtime_config,
                            continue_id=continue_id,
                            verbose=verbose,
                        )
                        query_time = timer_dict["Query Time"].stop()
                        state = State.get_id_state(continue_id)
                    if state.update_data is False:
                        monitor.log("task", "update_step", epoch=al_iteration)
                        timer_dict["Data-Update Time"].start()
                        step_update(config, continue_id=continue_id, annotated=True)
                        update_time = timer_dict["Data-Update Time"].stop()
                        state = State.get_id_state(continue_id)

                    # time loop only if all tasks are completed
                    if time_loop:
                        loop_time = timer_dict["Loop Time"].stop()
                        monitor.write_metric(loop_time, "Loop Time", epoch=al_iteration)

                    # reduce loop budget
                    loop_budget -= 1
                if benchmark:
                    break

        except Exception as err:
            state.in_progress = False
            state.save_state()
            raise RuntimeError("An error occured in 'run_experiment'.") from err

        state.in_progress = False
        state.save_state()

        loop_time = timer_dict["Runtime"].stop()
        monitor.write_metric(loop_time, "Runtime")
        b_times = {}
        time_dict = {}
        for key, timer in timer_dict.items():
            time_dict[key] = timer.average()

        b_times["times"] = time_dict
        b_times["config"] = config.to_dict()
        b_times["runtime_config"] = runtime_config.to_dict()
        save_json(b_times, get_results_folder(continue_id) / "benchmark_times.json")
