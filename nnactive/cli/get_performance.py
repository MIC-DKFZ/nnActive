import json
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from nnunetv2.utilities.file_path_utilities import get_output_folder

from nnactive.config import ActiveConfig
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import (
    convert_id_to_dataset_name,
    get_preprocessed_path,
    get_raw_path,
    get_results_path,
)
from nnactive.paths import get_nnActive_results
from nnactive.results.state import State

nnActive_results = get_nnActive_results()
TIMEOUT_S = 60 * 60


def get_mean_foreground_cv(summary_cross_val_dict, n_folds):
    """
    Get the mean over the foreground means across folds.
    Each fold has entry "foreground_mean" representing the mean over all foreground classes across images.
    Args:
        summary_cross_val_dict: Dictionary with the individual metrics per fold
        n_folds: number of folds

    Returns:
        Dict: mean dict containing the mean foreground metrics across fold
    """
    all_foreground_mean = []
    for fold in range(n_folds):
        all_foreground_mean.append(
            summary_cross_val_dict[f"fold_{fold}"]["foreground_mean"]
        )
    # Iterate over each metric (e.g. Dice, FN, FP, ...) and take the mean
    mean_dict = {}
    for key in all_foreground_mean[0].keys():
        mean_dict[key] = np.array([d[key] for d in all_foreground_mean]).mean()
    return mean_dict


def get_mean_cv(summary_cross_val_dict, n_folds):
    """
    Get the mean of the individual classes across folds.
    Each fold has entry "mean" representing the mean over the individual classes across images.
    Structure "mean": {'1': Dice: .., FN ..., ... '2': Dice:..., ...}
    Args:
        summary_cross_val_dict: Dictionary with the individual metrics per fold
        n_folds: number of folds

    Returns:
        Dict: per class dict containing the mean metrics per class across folds
    """
    mean_dicts_list = []
    for fold in range(n_folds):
        mean_dicts_list.append(summary_cross_val_dict[f"fold_{fold}"]["mean"])
    class_dicts = {}

    # First iterate over class indices
    for class_idx in mean_dicts_list[0].keys():
        class_dicts[class_idx] = {}
        # Iterate over each metric (e.g. Dice, FN, FP, ...) and take the mean for each class
        for key in mean_dicts_list[0][class_idx].keys():
            class_dicts[class_idx][key] = np.array(
                [d[class_idx][key] for d in mean_dicts_list]
            ).mean()
    return class_dicts


def main():
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id

    get_performance(dataset_id)


def get_performance(dataset_id):
    config = ActiveConfig.get_from_id(dataset_id)
    images_path = get_raw_path(dataset_id) / "imagesVal"
    labels_path = get_raw_path(dataset_id) / "labelsVal"
    loop_val = len(get_sorted_loop_files(get_raw_path(dataset_id))) - 1
    pred_path = get_results_path(dataset_id) / "predVal"
    dataset_json_path = get_raw_path(dataset_id) / "dataset.json"
    plans_identifier = "nnUNetPlans"
    plans_path = get_preprocessed_path(dataset_id) / f"{plans_identifier}.json"
    splits_path = get_preprocessed_path(dataset_id) / "splits_final.json"

    loop_results_path: Path = (
        nnActive_results
        / convert_id_to_dataset_name(dataset_id)
        / f"loop_{loop_val:03d}"
    )

    loop_summary_json = loop_results_path / "summary.json"
    loop_summary_cross_val_json = loop_results_path / "summary_cross_val.json"
    ex_command = f"nnUNetv2_predict -d {dataset_id} -c {config.model_config} -i {images_path} -o {pred_path} -tr {config.trainer}"
    subprocess.call(ex_command, shell=True, timeout=TIMEOUT_S)

    os.makedirs(loop_results_path, exist_ok=True)
    ex_command = f"nnUNetv2_evaluate_folder -djfile {dataset_json_path} -pfile {plans_path} -o {loop_summary_json} {labels_path} {pred_path}"
    subprocess.call(ex_command, shell=True)

    # Summarize the cross validation performance as json. Might be interesting to track across loops
    print("Creating a summary of the cross validation results from training...")
    with open(splits_path, "r") as f:
        n_folds = len(json.load(f))
    summary_cross_val_dict = {}

    # first save the individual cross val dicts by simply appending them with key fold_X
    for fold in range(n_folds):
        trained_model_path = get_output_folder(
            dataset_id, config.trainer, plans_identifier, config.model_config, fold
        )
        summary_json_train = Path(trained_model_path) / "validation" / "summary.json"
        with open(summary_json_train, "r") as f:
            summary_dict_train = json.load(f)
        summary_cross_val_dict[f"fold_{fold}"] = summary_dict_train

    # get foreground mean across folds
    foreground_mean_cv = get_mean_foreground_cv(summary_cross_val_dict, n_folds)
    # get the per class mean across folds
    per_class_mean_cv = get_mean_cv(summary_cross_val_dict, n_folds)
    summary_cross_val_dict["mean"] = {}
    summary_cross_val_dict["mean"]["foreground_mean"] = foreground_mean_cv
    summary_cross_val_dict["mean"]["mean"] = per_class_mean_cv

    # save the cv results
    with open(loop_summary_cross_val_json, "w") as f:
        json.dump(summary_cross_val_dict, f, indent=2)

    state = State.get_id_state(dataset_id)
    state.get_performance = True
    state.save_state()


if __name__ == "__main__":
    main()
