import subprocess
from argparse import ArgumentParser, Namespace

from get_performance import get_performance
from predict_nnUNet_ensemble import predict_nnUNet_ensemble
from query_step import query_step
from train_nnUNet_ensemble import train_nnUNet_ensemble
from update_data import update_step

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.results.state import State


@register_subcommand(
    "multi_loop_AL", [(("-d", "--dataset_id"), {"type": int, "required": True})]
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id

    config = ActiveConfig.get_from_id(dataset_id)
    state = State.get_id_state(dataset_id)

    print(config)
    # print(state)

    for al_iteration in range(config.query_steps):
        if al_iteration < state.loop:
            continue
        if al_iteration > state.loop:
            raise ValueError("A loop has not been executed!")
        if state.preprocess is False:
            # Preprocessing can require a lot of time
            subprocess.call(
                f"nnUNetv2_preprocess -d {dataset_id} -c {config.model_config} -np {config.num_processes}",
                shell=True,
            )
            state = State.get_id_state(dataset_id)
            state.preprocess = True
            state.save_state()

        if state.training is False:
            train_nnUNet_ensemble(dataset_id)
            state = State.get_id_state(dataset_id)
        if state.get_performance is False:
            get_performance(dataset_id)
            state = State.get_id_state(dataset_id)
        if state.query is False:
            # This might be the place where multiprocessing fails us!
            predict_nnUNet_ensemble(dataset_id)
            state = State.get_id_state(dataset_id)
        if al_iteration < config.query_steps - 1:
            if state.query is False:
                query_step(
                    dataset_id,
                    config.patch_size,
                    config.uncertainty,
                    config.query_size,
                    seed=config.seed,
                )
                state = State.get_id_state(dataset_id)
        if al_iteration < config.query_steps - 1:
            if state.update_data is False:
                update_step(dataset_id)
                state = State.get_id_state(dataset_id)
