import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import (
    convert_id_to_dataset_name,
    get_preprocessed_path,
    get_raw_path,
    get_results_path,
)

# from nnactive.paths import nnActive_results
from nnactive.paths import get_nnActive_results

nnActive_results = get_nnActive_results()


def main():
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    trainer = "nnUNetDebugTrainer"
    images_path = get_raw_path(dataset_id) / "imagesVal"
    labels_path = get_raw_path(dataset_id) / "labelsVal"
    loop_val = len(get_sorted_loop_files(get_raw_path(dataset_id))) - 1
    pred_path = get_results_path(dataset_id) / "predVal"
    dataset_json_path = get_raw_path(dataset_id) / "dataset.json"
    plans_path = get_preprocessed_path(dataset_id) / "nnUNetPlans.json"

    loop_results_path: Path = (
        nnActive_results
        / convert_id_to_dataset_name(dataset_id)
        / f"loop_{loop_val:03d}"
        / "summary.json"
    )

    ex_command = f"nnUNetv2_predict -d {dataset_id} -c 3d_fullres -i {images_path} -o {pred_path} -tr {trainer}"
    subprocess.call(ex_command, shell=True)

    os.makedirs(loop_results_path.parent, exist_ok=True)
    ex_command = f"nnUNetv2_evaluate_folder -djfile {dataset_json_path} -pfile {plans_path} -o {loop_results_path} {labels_path} {pred_path}"
    subprocess.call(ex_command, shell=True)


if __name__ == "__main__":
    main()
