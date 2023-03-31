import subprocess
from argparse import ArgumentParser

from get_performance import get_performance
from predict_nnUNet_ensemble import predict_nnUNet_ensemble
from query_step import query_step
from train_nnUNet_ensemble import train_nnUNet_ensemble
from update_data import update_step

from nnactive.config import ActiveConfig
from nnactive.results.state import State

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-id", type=int, required=True)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    config = ActiveConfig.get_from_id(dataset_id)
    state = State.get_id_state(dataset_id)

    print(config)

    for al_iteration in range(config.query_steps):
        if al_iteration < state.loop:
            continue
        if state.training is False:
            subprocess.call(
                f"nnUNetv2_preprocess -d {dataset_id} -c {config.model_config} -np {config.num_processes}",
                shell=True,
            )
            train_nnUNet_ensemble(dataset_id)
            state = State.get_id_state(dataset_id)
        if state.get_performance is False:
            get_performance(dataset_id)
            state = State.get_id_state(dataset_id)
        if state.query is False:
            predict_nnUNet_ensemble(dataset_id)
            if al_iteration < config.query_steps - 1:
                query_step(
                    dataset_id,
                    config.patch_size,
                    config.uncertainty,
                    config.query_size,
                    seed=config.seed,
                )
                state = State.get_id_state(dataset_id)
            if state.update_data is False:
                update_step(dataset_id)
                state = State.get_id_state(dataset_id)
