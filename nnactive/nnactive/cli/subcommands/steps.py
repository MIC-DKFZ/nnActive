import itertools
import json
import multiprocessing
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterable

import nnunetv2.paths
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subfiles,
)
from loguru import logger
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
from nnunetv2.run.run_training import run_training
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

import wandb
from nnactive import paths
from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.logger import monitor
from nnactive.loops.cross_validation import (
    get_mean_cv,
    get_mean_foreground_cv,
    read_classes_from_dataset_json,
)
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.predict import predict_entry_point
from nnactive.nnunet.preprocessor import nnActivePreprocessor
from nnactive.nnunet.utils import (
    convert_id_to_dataset_name,
    get_preprocessed_path,
    get_raw_path,
    get_results_path,
    read_dataset_json,
)
from nnactive.paths import get_nnActive_results
from nnactive.query_pool import query_pool
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder
from nnactive.update_data import update_data
from nnactive.utils.io import save_json
from nnactive.utils.timer import Timer

nnActive_results = get_nnActive_results()


def wrap_training(
    dataset_id: int,
    config: ActiveConfig,
    folds: Iterable[int],
    device: torch.device,
    wandbgroup: str | None,
    state: State | None = None,
):
    config.set_nnunet_env()

    if wandb.run is None:
        # No active wandb run, create a new one
        wandb_context = monitor.active_run(
            config=config, state=state, state_tag="training"
        )
    elif wandbgroup is None:
        # Do nothing, assuming active wandb context
        wandb_context = nullcontext()
    else:
        # Initialize with wandb group (for multi gpu compatibility)
        wandb_context = monitor.active_run(
            group=wandbgroup, state=state, state_tag="training"
        )

    with wandb_context:
        # ensure that each fold/fork is mapped onto one gpu
        torch.cuda.set_device(device)
        for fold in folds:
            logger.info(
                f"Running training fold '{fold}' in process '{multiprocessing.current_process()}' with device '{device}'"
            )
            run_training(
                str(
                    dataset_id
                ),  # TODO: fix this bug in nnU-Net requiring input to be string.
                config.model_config,
                fold,
                trainer_class_name=config.trainer,
                device=device,
                logger=monitor.get_logger(),
            )


@register_subcommand("step_train")
def step_train(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    force: bool = False,
    raise_on_in_progress: bool = True,
):
    config.set_nnunet_env()

    if continue_id is None:
        state = State.latest(config)
        if not force:
            state.verify()
    else:
        state = State.get_id_state(continue_id, verify=not force)

    # ensure that set_num_interop is not executed twice
    # multithreading in torch doesn't help nnU-Net if run on GPU
    try:
        os.environ["torchset"]
    except KeyError:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["torchset"] = "True"

    num_folds = config.train_folds

    # Custom preprocessor handles deleting of old _seg.npy files
    npp = (
        runtime_config.num_processes
        if runtime_config.n_gpus == 0
        else runtime_config.num_processes * runtime_config.n_gpus
    )
    unpack_dataset(
        folder=str(
            get_preprocessed_path(state.dataset_id)
            / "_".join([config.model_plans, config.model_config]),
        ),
        unpack_segmentation=True,
        overwrite_existing=False,
        num_processes=npp,
        verify=False,
    )

    if raise_on_in_progress and state.in_progress:
        raise RuntimeError(
            f"Training already in progress for experiment {config.name()}. Check the "
            "current trainings or set up a new nnActive experiment."
        )
    state.in_progress = True
    state.save_state()
    try:
        if runtime_config.n_gpus == 0:
            wrap_training(
                dataset_id=state.dataset_id,
                config=config,
                folds=list(range(num_folds)),
                device=torch.device("cuda:0"),
                wandbgroup=None,
                state=state,
            )
        else:
            devices = [torch.device(f"cuda:{i}") for i in range(runtime_config.n_gpus)]
            folds = [
                [fold for fold in range(num_folds) if fold % runtime_config.n_gpus == d]
                for d in range(runtime_config.n_gpus)
            ]
            try:
                with ProcessPoolExecutor(
                    max_workers=runtime_config.n_gpus,
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    for _ in executor.map(
                        wrap_training,
                        [state.dataset_id] * num_folds,
                        [config] * num_folds,
                        folds,
                        devices,
                        [wandb.run.group] * num_folds,
                        [state] * num_folds,
                    ):
                        pass
            except BrokenProcessPool as exc:
                raise MemoryError(
                    "One of the worker processes died. "
                    "This usually happens because you run out of memory. "
                    "Try running with less processes."
                ) from exc

    except Exception as err:
        state.in_progress = False
        state.save_state()
        raise RuntimeError("An error occured in 'step_train'") from err

    state.in_progress = False
    if not force:
        state.training = True
    state.save_state()


def wrap_prediction(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    config: ActiveConfig,
    verbose: bool,
    num_parts: int,
    part_id: int,
    device: torch.device,
    wandb_group: str,
    state: State | None = None,
):
    config.set_nnunet_env()
    with monitor.active_run(group=wandb_group, state=state, state_tag="prediction"):
        logger.info(
            f"Running prediction in process '{multiprocessing.current_process()}' with device '{device}'"
        )
        folds = [fold for fold in range(config.train_folds)]
        torch.cuda.set_device(device)
        predict_entry_point(
            input_folder=input_folder,
            output_folder=output_folder,
            dataset_id=dataset_id,
            train_identifier=config.trainer,
            configuration_identifier=config.model_config,
            folds=folds,
            step_size=config.pred_tile_step_size,
            disable_tta=config.disable_pred_tta,
            verbose=verbose,
            num_parts=num_parts,
            part_id=part_id,
            disable_progress_bar=verbose,
        )


@register_subcommand("step_performance")
def step_performance(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    force: bool = False,
    verbose: bool = False,
):
    config.set_nnunet_env()
    if continue_id is None:
        state = State.latest(config)
        if not force:
            state.verify()
    else:
        state = State.get_id_state(continue_id, verify=not force)
    images_path = get_raw_path(state.dataset_id) / "imagesVal"
    labels_path = get_raw_path(state.dataset_id) / "labelsVal"
    loop_val = len(get_sorted_loop_files(get_raw_path(state.dataset_id))) - 1
    pred_path = get_results_path(state.dataset_id) / "predVal"
    dataset_json_path = get_raw_path(state.dataset_id) / "dataset.json"
    plans_path = get_preprocessed_path(state.dataset_id) / f"{config.model_plans}.json"

    num_folds = config.train_folds
    loop_results_path: Path = (
        get_results_folder(state.dataset_id) / f"loop_{loop_val:03d}"
    )

    loop_summary_json = loop_results_path / "summary.json"
    loop_summary_cross_val_json = loop_results_path / "summary_cross_val.json"

    # TODO: redo add_validation in config!
    if runtime_config.n_gpus == 0:
        device = torch.device("cuda:0")
        predict_entry_point(
            input_folder=str(images_path),
            output_folder=str(pred_path),
            dataset_id=state.dataset_id,
            train_identifier=config.trainer,
            configuration_identifier=config.model_config,
            folds=[i for i in range(num_folds)],
            step_size=config.pred_tile_step_size,
            disable_tta=config.disable_pred_tta,
            verbose=verbose,
            disable_progress_bar=verbose,
            device=device,
        )
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=runtime_config.n_gpus, mp_context=mp.get_context("spawn")
            ) as executor:
                for _ in executor.map(
                    wrap_prediction,
                    [str(images_path)] * runtime_config.n_gpus,
                    [str(pred_path)] * runtime_config.n_gpus,
                    [state.dataset_id] * runtime_config.n_gpus,
                    [config] * runtime_config.n_gpus,
                    [verbose] * runtime_config.n_gpus,
                    [runtime_config.n_gpus] * runtime_config.n_gpus,
                    [p_id for p_id in range(runtime_config.n_gpus)],
                    [torch.device(f"cuda:{i}") for i in range(runtime_config.n_gpus)],
                    [wandb.run.group] * runtime_config.n_gpus,
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
    summary_cross_val_dict = {}

    # first save the individual cross val dicts by simply appending them with key fold_X
    for fold in range(num_folds):
        trained_model_path = get_output_folder(
            state.dataset_id,
            config.trainer,
            config.model_plans,
            config.model_config,
            fold,
        )

        os.makedirs(loop_results_path, exist_ok=True)
        compute_metrics_on_folder2(
            folder_ref=str(labels_path),
            folder_pred=str(pred_path),
            dataset_json_file=str(dataset_json_path),
            plans_file=str(plans_path),
            output_file=str(loop_summary_json),
            num_processes=runtime_config.num_processes,
        )

        # Summarize the cross validation performance as json. Might be interesting to track across loops
        logger.info(
            "Creating a summary of the cross validation results from training..."
        )
        summary_cross_val_dict = {}

        # first save the individual cross val dicts by simply appending them with key fold_X
        for fold in range(num_folds):
            trained_model_path = get_output_folder(
                state.dataset_id,
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


@register_subcommand("step_preprocess")
def preprocess(
    config: ActiveConfig,
    runtime_config: RuntimeConfig,
    continue_id: int | None = None,
    verbose: bool = False,
    do_all: bool = False,
    force: bool = False,
) -> None:
    config.set_nnunet_env()

    if continue_id is None:
        state = State.latest(config)
        if not force:
            state.verify()
    else:
        state = State.get_id_state(continue_id, verify=not force)

    num_processes = [runtime_config.num_processes]
    configurations = [config.model_config]
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f"The list provided with num_processes must either have len 1 or as many elements as there are "
            f"configurations (see --help). Number of configurations: {len(configurations)}, length "
            f"of num_processes: "
            f"{len(num_processes)}"
        )

    dataset_name = convert_id_to_dataset_name(state.dataset_id)
    logger.info(f"Preprocessing dataset {dataset_name}")
    plans_file = join(
        nnunetv2.paths.nnUNet_preprocessed, dataset_name, config.model_plans + ".json"
    )
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        logger.info(f"Configuration: {c}...")
        if c not in plans_manager.available_configurations:
            raise FileNotFoundError(
                f"INFO: Configuration {c} not found in plans file {config.model_plans + '.json'} of "
                f"dataset {dataset_name}. Skipping."
            )
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = nnActivePreprocessor(verbose=verbose)
        preprocessor.run(
            state.dataset_id, c, config.model_plans, num_processes=n, do_all=do_all
        )
    maybe_mkdir_p(
        join(nnunetv2.paths.nnUNet_preprocessed, dataset_name, "gt_segmentations")
    )
    [
        shutil.copy(
            i,
            join(
                join(
                    nnunetv2.paths.nnUNet_preprocessed, dataset_name, "gt_segmentations"
                )
            ),
        )
        for i in subfiles(join(nnunetv2.paths.nnUNet_raw, dataset_name, "labelsTr"))
    ]

    if not force:
        state.preprocess = True
        state.save_state()


@register_subcommand("step_query")
def step_query(
    config: ActiveConfig,
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    verbose: bool = False,
    force: bool = False,
):
    """Run Query with trained models on the dataset pool.

    Args:
        config (ActiveConfig): Carries all revelant information of experiment
        runtime_config (RuntimeConfig, optional): carries n_gpus, processes etc.. Defaults to RuntimeConfig().
        continue_id (int | None, optional): _description_. Defaults to None.
        verbose (bool, optional): Disables progress bars and get more explicit print statements.. Defaults to False.
        force (bool, optional): Set this to force using this command without taking the state.json of the dataset into account. Defaults to False.
    """
    config.set_nnunet_env()
    timer = Timer()

    print(f"{continue_id=}")
    if continue_id is None:
        state = State.latest(config)
        if not force:
            state.verify()
    else:
        state = State.get_id_state(continue_id, verify=not force)

    # If the query step is already done, re-running it has to be forced.
    if state.query and not force:
        raise RuntimeError(
            f"Query step already performed for loop {state.loop + 1}. Use --force=True "
            "to re-run the query step."
        )

    with monitor.active_run(config=config.to_dict(), state=state, state_tag="query"):
        timer.start()
        query_pool(
            config, runtime_config, state.dataset_id, force=force, verbose=verbose
        )
        timer.stop()

    b_times = {}
    b_times["times"] = {"Query Time": timer.average()}
    b_times["config"] = config.to_dict()
    b_times["runtime_config"] = runtime_config.to_dict()
    save_json(
        b_times, get_results_folder(state.dataset_id) / "benchmark_time_query.json"
    )


@register_subcommand("step_update")
def step_update(
    config: ActiveConfig,
    continue_id: int | None = None,
    num_folds: int = 5,
    loop_val: int | None = None,
    annotated: bool = True,
    force: bool = False,
    no_state: bool = False,
    ensure_classes_in_folds: bool = True,
):
    config.set_nnunet_env()
    if continue_id is None:
        state = State.latest(config)
        if not force:
            state.verify()
    else:
        state = State.get_id_state(continue_id)
    data_path = get_raw_path(state.dataset_id)
    save_splits_file = get_preprocessed_path(state.dataset_id) / "splits_final.json"
    target_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(state.dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]
    file_ending = dataset_json["file_ending"]

    additional_label_path = data_path / "addTr"
    if not additional_label_path.is_dir():
        additional_label_path = None

    if annotated:
        base_dir = paths.nnActive_raw / "nnUNet_raw" / config.dataset / "labelsTr"
    else:
        base_dir = get_raw_path(state.dataset_id) / f"annoTr_{loop_val:02}"

    if not no_state:
        state = State.get_id_state(state.dataset_id, verify=not force)

    if ensure_classes_in_folds:
        logger.info("Ensure every class in all train folds.")
        ensure_classes = read_classes_from_dataset_json(dataset_json)

    else:
        logger.info(
            "Standard splits creation. Possibly not every class in all train folds."
        )
        ensure_classes = None

    update_data(
        data_path,
        save_splits_file,
        ignore_label,
        file_ending,
        base_dir,
        target_dir,
        loop_val=loop_val,
        num_folds=num_folds,
        annotated=annotated,
        additional_label_path=additional_label_path,
        ensure_classes=ensure_classes,
    )

    if not force and not no_state:
        state.update_data = True

        results_path = get_results_path(state.dataset_id)
        results_name = "__".join(
            [config.trainer, config.model_plans, config.model_config]
        )
        results_save = f"loop_{state.loop:03d}__" + results_name
        shutil.move(results_path / results_name, results_path / results_save)
        shutil.move(
            results_path / "predVal", results_path / f"loop_{state.loop:03d}__predVal"
        )
        state.new_loop()
        state.save_state()
