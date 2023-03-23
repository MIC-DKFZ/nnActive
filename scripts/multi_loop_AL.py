import subprocess
from argparse import ArgumentParser
from pathlib import Path

from get_performance import get_performance
from predict_nnUNet_ensemble import predict_nnUNet_ensemble
from query_step import query_step
from train_nnUNet_ensemble import train_nnUNet_ensemble
from update_data import update_step

from nnactive.config import ActiveConfig

SCRIPTPATH = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-id", type=int, required=True)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    config = ActiveConfig.get_from_id(dataset_id)

    print(config)

    for al_iteration in range(config.query_steps):
        subprocess.call(
            f"nnUNetv2_preprocess -d {dataset_id} -c {config.model_config} -np 4",
            shell=True,
        )
        train_nnUNet_ensemble(dataset_id)
        get_performance(dataset_id)
        predict_nnUNet_ensemble(dataset_id)
        if al_iteration == config.query_steps - 1:
            query_step(
                dataset_id, config.patch_size, config.uncertainty, config.query_size
            )
            update_step(dataset_id)
