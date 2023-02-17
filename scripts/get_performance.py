import json
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from nnunetv2.utilities.file_path_utilities import get_output_folder

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
    configuration = "3d_fullres"
    images_path = get_raw_path(dataset_id) / "imagesVal"
    labels_path = get_raw_path(dataset_id) / "labelsVal"
    loop_val = len(get_sorted_loop_files(get_raw_path(dataset_id))) - 1
    pred_path = get_results_path(dataset_id) / "predVal"
    dataset_json_path = get_raw_path(dataset_id) / "dataset.json"
    plans_identifier = "nnUNetPlans"
    plans_path = get_preprocessed_path(dataset_id) / f"{plans_identifier}.json"

    loop_results_path: Path = (
        nnActive_results
        / convert_id_to_dataset_name(dataset_id)
        / f"loop_{loop_val:03d}"
    )

    loop_summary_json = loop_results_path / "summary.json"
    loop_summary_cross_val_json = loop_results_path / "summary_cross_val.json"

    ex_command = f"nnUNetv2_predict -d {dataset_id} -c {configuration} -i {images_path} -o {pred_path} -tr {trainer}"
    subprocess.call(ex_command, shell=True)

    os.makedirs(loop_results_path, exist_ok=True)
    ex_command = f"nnUNetv2_evaluate_folder -djfile {dataset_json_path} -pfile {plans_path} -o {loop_summary_json} {labels_path} {pred_path}"
    subprocess.call(ex_command, shell=True)

    print("Creating a summary of the cross validation results from training...")
    summary_cross_val_dict = {}
    for fold in [0, 1, 2, 3, 4]:
        trained_model_path = get_output_folder(
            dataset_id, trainer, plans_identifier, configuration, fold
        )
        summary_json_train = Path(trained_model_path) / "validation" / "summary.json"
        with open(summary_json_train, "r") as f:
            summary_dict_train = json.load(f)
        summary_cross_val_dict[f"fold_{fold}"] = summary_dict_train
    with open(loop_summary_cross_val_json, "w") as f:
        json.dump(summary_cross_val_dict, f, indent=2)


if __name__ == "__main__":
    main()
