import subprocess
import time
from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.results.state import State


@register_subcommand("train_nnUNet_ensemble", [(("-d", "--dataset_id"), {"type": int})])
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id

    train_nnUNet_ensemble(dataset_id)


def train_nnUNet_ensemble(dataset_id):
    config = ActiveConfig.get_from_id(dataset_id)
    num_folds = config.working_folds
    state = State.get_id_state(dataset_id)

    for fold in range(num_folds):
        ex_command = f"nnUNetv2_train {dataset_id} {config.model_config} {fold} -tr {config.trainer}"
        print(ex_command)
        subprocess.run(ex_command, shell=True, check=True)

    state.training = True
    state.save_state()
