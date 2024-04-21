import json
import multiprocessing
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2

# from nnunetv2.inference.predict_from_raw_data import predict
from nnunetv2.utilities.file_path_utilities import get_output_folder

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.logger import monitor
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.predict import predict_entry_point
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


@register_subcommand(
    "get_performance",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-f", "--force"),
            {"action": "store_true", "help": "Ignores the internal State."},
        ),
        (
            ("--verbose"),
            {
                "action": "store_true",
                "help": "Disables progress bars and get more explicit print statements.",
            },
        ),
        (("--n_gpus"), {"default": 1, "type": int}),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    force = args.force
    verbose = args.verbose
    n_gpus = args.n_gpus
    config = ActiveConfig.get_from_id(dataset_id)
    with monitor.active_run(config=config.to_dict()):
        get_performance(dataset_id, force, verbose, n_gpus)


def wrap_prediction(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    config: ActiveConfig,
    verbose: bool,
    num_parts: int,
    part_id: int,
    device: torch.device,
):
    logger.info(
        f"Running prediction in process '{multiprocessing.current_process()}' with device '{device}'"
    )
    folds = [fold for fold in range(config.working_folds)]
    torch.cuda.set_device(device)
    predict_entry_point(
        input_folder=input_folder,
        output_folder=output_folder,
        dataset_id=dataset_id,
        train_identifier=config.trainer,
        configuration_identifier=config.model_config,
        folds=folds,
        verbose=verbose,
        num_parts=num_parts,
        part_id=part_id,
        disable_progress_bar=verbose,
    )


def get_performance(
    dataset_id: int, force: bool = False, verbose: bool = False, n_gpus: int = 1
):
    state = State.get_id_state(dataset_id, verify=not force)
    config = ActiveConfig.get_from_id(dataset_id)
    images_path = get_raw_path(dataset_id) / "imagesVal"
    labels_path = get_raw_path(dataset_id) / "labelsVal"
    loop_val = len(get_sorted_loop_files(get_raw_path(dataset_id))) - 1
    pred_path = get_results_path(dataset_id) / "predVal"
    dataset_json_path = get_raw_path(dataset_id) / "dataset.json"
    plans_path = get_preprocessed_path(dataset_id) / f"{config.model_plans}.json"

    num_folds = config.working_folds
    loop_results_path: Path = (
        nnActive_results
        / convert_id_to_dataset_name(dataset_id)
        / f"loop_{loop_val:03d}"
    )

    loop_summary_json = loop_results_path / "summary.json"
    loop_summary_cross_val_json = loop_results_path / "summary_cross_val.json"

    # TODO: redo add_validation in config!
    if n_gpus == 1:
        device = torch.device("cuda:0")
        predict_entry_point(
            input_folder=str(images_path),
            output_folder=str(pred_path),
            dataset_id=dataset_id,
            train_identifier=config.trainer,
            configuration_identifier=config.model_config,
            folds=[i for i in range(num_folds)],
            verbose=verbose,
            disable_progress_bar=verbose,
            device=device,
        )
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_gpus) as executor:
                for _ in executor.map(
                    wrap_prediction,
                    [str(images_path)] * n_gpus,
                    [str(pred_path)] * n_gpus,
                    [dataset_id] * n_gpus,
                    [config] * n_gpus,
                    [verbose] * n_gpus,
                    [n_gpus] * n_gpus,
                    [p_id for p_id in range(n_gpus)],
                    [torch.device(f"cuda:{i}") for i in range(n_gpus)],
                ):
                    pass
        except BrokenProcessPool as exc:
            raise MemoryError(
                "One of the worker processes died. "
                "This usually happens because you run out of memory. "
                "Try running with less processes."
            ) from exc

    os.makedirs(loop_results_path, exist_ok=True)
    compute_metrics_on_folder2(
        folder_ref=str(labels_path),
        folder_pred=str(pred_path),
        dataset_json_file=str(dataset_json_path),
        plans_file=str(plans_path),
        output_file=str(loop_summary_json),
        num_processes=8,
    )

    # Summarize the cross validation performance as json. Might be interesting to track across loops
    logger.info("Creating a summary of the cross validation results from training...")
    num_folds = config.working_folds
    summary_cross_val_dict = {}

    # first save the individual cross val dicts by simply appending them with key fold_X
    for fold in range(num_folds):
        trained_model_path = get_output_folder(
            dataset_id, config.trainer, config.model_plans, config.model_config, fold
        )

        os.makedirs(loop_results_path, exist_ok=True)
        compute_metrics_on_folder2(
            folder_ref=str(labels_path),
            folder_pred=str(pred_path),
            dataset_json_file=str(dataset_json_path),
            plans_file=str(plans_path),
            output_file=str(loop_summary_json),
            num_processes=8,
        )

        # Summarize the cross validation performance as json. Might be interesting to track across loops
        logger.info(
            "Creating a summary of the cross validation results from training..."
        )
        num_folds = config.working_folds
        summary_cross_val_dict = {}

        # first save the individual cross val dicts by simply appending them with key fold_X
        for fold in range(num_folds):
            trained_model_path = get_output_folder(
                dataset_id,
                config.trainer,
                config.model_plans,
                config.model_config,
                fold,
            )
            summary_json_train = (
                Path(trained_model_path) / "validation" / "summary.json"
            )
            with open(summary_json_train, "r") as f:
                summary_dict_train = json.load(f)
            summary_cross_val_dict[f"fold_{fold}"] = summary_dict_train

        # get foreground mean across folds
        foreground_mean_cv = get_mean_foreground_cv(summary_cross_val_dict, num_folds)
        # get the per class mean across folds
        per_class_mean_cv = get_mean_cv(summary_cross_val_dict, num_folds)
        summary_cross_val_dict["mean"] = {}
        summary_cross_val_dict["mean"]["foreground_mean"] = foreground_mean_cv
        summary_cross_val_dict["mean"]["mean"] = per_class_mean_cv

        # save the cv results
        with open(loop_summary_cross_val_json, "w") as f:
            json.dump(summary_cross_val_dict, f, indent=2)

        if not force:
            state.get_performance = True
            state.save_state()
