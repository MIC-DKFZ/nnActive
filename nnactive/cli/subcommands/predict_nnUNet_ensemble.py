import subprocess
from argparse import ArgumentParser, Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.nnunet.utils import get_raw_path, get_results_path

TIMEOUT_S = 60 * 60


@register_subcommand(
    "predict_nnUNet_ensemble", [(("-d", "--dataset_id"), {"type": int})]
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id

    num_folds = 5
    predict_nnUNet_ensemble(dataset_id, num_folds)


def predict_nnUNet_ensemble(dataset_id, num_folds=5):
    config = ActiveConfig.get_from_id(dataset_id)

    images_path = get_raw_path(dataset_id) / "imagesTr"
    output_path = get_results_path(dataset_id) / "predTr"

    for fold in range(num_folds):
        output_fold_path = output_path / f"fold_{fold}"
        ex_command = f"nnUNetv2_predict -d {dataset_id} -c {config.model_config} -i {images_path} -o {output_fold_path} -tr {config.trainer} -f {fold} --save_probabilities {config.add_uncertainty}"
        print(ex_command)
        subprocess.run(
            ex_command, shell=True, check=True
        )  # timeout=TIMEOUT_S is not longer required due to better multiprocessing
