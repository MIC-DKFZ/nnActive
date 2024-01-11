import subprocess
from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
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
        (("-v", "--verbose"), {"action": "store_true"}),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    verbose = args.verbose

    config = ActiveConfig.get_from_id(dataset_id)
    state = State.get_id_state(dataset_id)

    print(config)

    for al_iteration in range(config.query_steps):
        if al_iteration < state.loop:
            continue
        if al_iteration > state.loop:
            raise ValueError("A loop has not been executed!")
        if state.preprocess is False:
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

            # ex_call = f"nnactive nnunet_preprocess -d {dataset_id} -c {config.model_config} -np {config.num_processes}"
            # if do_all:
            #     ex_call += " --do_all"
            # subprocess.run(
            #     ex_call,
            #     shell=True,
            #     check=True,
            # )
            # state = State.get_id_state(dataset_id)
            # state.preprocess = True
            # state.save_state()

        if state.training is False:
            train_nnUNet_ensemble(dataset_id)
            state = State.get_id_state(dataset_id)
        if state.get_performance is False:
            get_performance(dataset_id)
            state = State.get_id_state(dataset_id)
        if al_iteration < config.query_steps - 1:
            if state.pred_tr is False and state.query is False:
                query_pool(dataset_id)
                state = State.get_id_state(dataset_id)
            if state.update_data is False:
                update_step(dataset_id, annotated=True)
                state = State.get_id_state(dataset_id)
