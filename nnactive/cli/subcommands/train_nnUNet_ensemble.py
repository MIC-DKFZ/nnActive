import subprocess
from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.results.state import State


@register_subcommand("train_nnUNet_ensemble", [(("-d", "--dataset_id"), {"type": int})])
def main(args: Namespace) -> None:
    # TODO: help
    dataset_id = args.dataset_id

    num_folds = 5

    train_nnUNet_ensemble(dataset_id, num_folds)


def train_nnUNet_ensemble(dataset_id, num_folds=5):
    config = ActiveConfig.get_from_id(dataset_id)

    for fold in range(num_folds):
        ex_command = f"nnUNetv2_train {dataset_id} {config.model_config} {fold} -tr {config.trainer}"
        # print(ex_command)
        subprocess.call(ex_command, shell=True)

    state = State.get_id_state(dataset_id)
    state.training = True
    state.save_state()
