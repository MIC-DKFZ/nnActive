import os
import os.path
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger

import nnactive.paths as paths
from nnactive.cli.registry import register_subcommand
from nnactive.cli.subcommands.steps import preprocess, step_update
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.experiments import (
    list_experiments,
    list_finished_experiments,
    list_prepared_experiments,
)
from nnactive.loops.loading import (
    get_loop_patches,
    get_patches_from_loop_files,
    get_sorted_loop_files,
)
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, get_results_path
from nnactive.production import produce_empty_masks
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.utils.io import load_json


# TODO: delete old trainings in nnUNet_results and artifacts in nnActive_results??
@register_subcommand("util_reset_loops")
def util_reset_loops(nnActive_results_folder: str, loop: int = 0, npp: int = 4) -> None:
    """Reset experiment to which nnActive_results_folder belongs to loop {loop}.
    Deletes all loop files which are bigger than {loop}.
    If reset to loop 0, no preprocessing is performed to completely reset the dataset.

    Currently neither old training files in nnUNet_results nor artifacts in nnActive_results are deleted.


    Args:
        nnActive_results_folder (str): folder with config.json
        loop (int, optional): value to which experiment is to be resetted. Defaults to 0.
        npp(int, optional): num processes for preprocessing. Defaults to 4.
    """
    nnActive_results_folder: Path = Path(nnActive_results_folder)
    config = ActiveConfig.from_json(nnActive_results_folder / ActiveConfig.filename())
    config.set_nnunet_env()
    dataset_id = int(nnActive_results_folder.name.split("_")[0][-3:])
    reset_loop_nr = loop
    raw_dataset_path = get_raw_path(dataset_id)
    state = State.get_id_state(dataset_id)
    if reset_loop_nr > state.loop:
        raise AttributeError(
            "Loop number to reset to is higher than the current loop number."
        )

    for file in os.listdir(raw_dataset_path):
        if file.startswith(f"{config.uncertainty}_") and file.endswith(".json"):
            if int(file.split("_")[-1].split(".")[0]) > reset_loop_nr:
                print("Deleting file ", str(raw_dataset_path / file))
                os.remove(raw_dataset_path / file)

    step_update(
        config=config,
        continue_id=dataset_id,
        loop_val=reset_loop_nr,
        annotated=True,
        force=True,
    )

    for loop_file in get_sorted_loop_files(raw_dataset_path)[reset_loop_nr + 1 :]:
        logger.info("Deleting file ", str(raw_dataset_path / loop_file))
        os.remove(raw_dataset_path / loop_file)

    nnactive_results_folder = get_nnactive_results_folder(dataset_id)
    for folder in os.listdir(nnactive_results_folder):
        if folder.startswith("loop"):
            if int(folder.split("_")[-1]) >= reset_loop_nr:
                logger.info("Deleting folder ", str(nnactive_results_folder / folder))
                shutil.rmtree(nnactive_results_folder / folder)

    logger.info("Resetting State file...")
    state.reset()
    if reset_loop_nr > 0:
        state.loop = reset_loop_nr
    state.save_state()

    # Preprocess with do_all is esp. needed for loop larger 0
    if reset_loop_nr > 0:
        preprocess(
            config=config,
            runtime_config=RuntimeConfig(),
            do_all=True,
        )


@register_subcommand("util_verify_data")
def util_verify_data(
    raw_folder: str,
    loop_val: int = None,
    no_state: bool = False,
):
    """Verify that patches in loop files contain no ignore label.

    Args:
        raw_folder (str): folder to raw_experiment
        loop_val (int, optional): up to which loop val to check. Defaults to None.
        no_state (bool, optional): Do not load state. Defaults to False.
    """
    data_path = Path(raw_folder)
    label_dir = data_path / "labelsTr"
    if loop_val is not None:
        pass
    elif loop_val is None and no_state is False:
        state = State.from_json(data_path / State.filename)
        loop_val = state.loop
        if state.query:
            loop_val -= 1
    else:
        raise NotImplementedError()
    patches = get_patches_from_loop_files(data_path, loop_val)
    logger.info(
        f"Veryfing labels for loop {loop_val}.\n Cumulative sum of patches: {len(patches)}"
    )
    dataset_json = load_json(raw_folder / "dataset.json")
    ignore_label = dataset_json["labels"]["ignore"]

    for loop_check in range(loop_val + 1):
        patches = get_loop_patches(data_path, loop_check)
        logger.info(
            f"Verifying labels for loop {loop_check} with {len(patches)} patches."
        )

        unique_files = np.unique([patch.file for patch in patches])
        for file in unique_files:
            patch_file = [patch for patch in patches if patch.file == file]
            file_p = label_dir / file
            seg = sitk.GetArrayFromImage(sitk.ReadImage(file_p)).astype(np.uint8)
            for patch in patch_file:
                slices = []
                for start_index, size in zip(patch.coords, patch.size):
                    slices.append(slice(start_index, start_index + size))
                if np.any(seg[tuple(slices)] == ignore_label):
                    raise RuntimeError(
                        f"For loop {loop_check} patch in file {file} has ignore_label {ignore_label}"
                    )

    add_path = data_path / "addTr"
    if add_path.is_dir():
        logger.info(f"Verifying that labels contain addTr data.")
        for file in label_dir.iterdir():
            if file.name.endswith(dataset_json["file_ending"]):
                seg = sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(np.uint8)
                add_seg = sitk.GetArrayFromImage(
                    sitk.ReadImage(add_path / file.name)
                ).astype(np.uint8)
                equal = seg == add_seg
                if not np.all(equal[add_seg != 255]):
                    raise RuntimeError(
                        f"For file {file.name} in labelsTr the labels from addTr were not added."
                    )


@register_subcommand("util_get_times")
def util_get_times(base_path: str | None = None, filter_times=True):
    """Get the times from the benchmark_times.json files in the base_path."""

    def get_file_dicts(base_path: str):
        # Find all files with name "benchmark_times.json"
        base_path: Path = Path(base_path)
        file_paths = base_path.rglob("benchmark_times.json")

        file_dicts = []
        for file in file_paths:
            file = Path(file)
            # Load the JSON file
            data = load_json(file)

            # Extract the time from the JSON file
            file_dict = {}
            file_dict["Experiment"] = file.parent.name
            file_dict["Trainer"] = data["config"]["trainer"]
            file_dict["n_gpus"] = data["runtime_config"]["n_gpus"]
            file_dict.update(data["times"])
            file_dicts.append(file_dict)
        #
        # Create a DataFrame from the list of dictionaries
        return file_dicts

    if base_path is None:
        base_path = Path(os.getenv("nnActive_data"))
        base_paths = os.listdir(base_path)
        base_paths = [
            (base_path / sub_path / "nnActive_results") for sub_path in base_paths
        ]

    else:
        base_paths = [base_path]

    df_dicts = []
    for base_path in base_paths:
        file_dicts = get_file_dicts(base_path)
        df_dicts.extend(file_dicts)
    df = pd.DataFrame(df_dicts)
    # filter rows
    if len(df) == 0:
        print("No benchmark_times.json files found.")
    else:
        if filter_times:
            df = df.loc[df["Loop Time"].isnull() == False]
        print(df.to_csv())


@register_subcommand("util_list_experiments")
def util_list_experiments():
    """List all configured experiments each experiment in one row."""
    for experiment in sorted(list_experiments()):
        print(experiment)


@register_subcommand("util_list_finished_experiments")
def util_list_finished_experiments(base_id: int):
    """List all finished experiments with dataset rowwise for a specific base_id."""
    for experiment in list_finished_experiments(base_id):
        print(experiment)


@register_subcommand("util_list_unfinished_experiments")
def util_list_unfinished_experiments(base_id: int):
    """List all prepared but not finished experiments with dataset rowwise for a specific base_id."""
    prepared_experiments = list_prepared_experiments(base_id)
    finished_experiments = list_finished_experiments(base_id)
    for experiment in prepared_experiments:
        if experiment not in finished_experiments:
            print(experiment)


@register_subcommand("util_list_prepared_experiments")
def util_list_prepared_experiments(base_id: int):
    """List all prepared experiments with dataset rowwise for a specific base_id."""
    for experiment in sorted(list_prepared_experiments(base_id)):
        print(experiment)


@register_subcommand("util_produce_empty_masks")
def util_produce_empty_masks(
    images_folder: Path,
    output_folder: Path,
    fill_value: int,
    file_ending: str = ".nii.gz",
    additional_label_folder: Path | None = None,
    modality_iden: str = "_0000",
):
    """Create empty labels for all images in images_folder.

    Args:
        images_folder (Path): Folder with images
        output_folder (Path): Folder for labels
        fill_value (int): Label for ignore regions
        file_ending (str): File ending. Defaults to ".nii.gz".
        additional_label_folder (Path, optional): Folder with additional labels. Defaults to None.
        modality_iden (str): _0000 modality string after img_identifier. Defaults to "_0000".
    """
    produce_empty_masks(
        images_folder,
        output_folder,
        fill_value,
        file_ending,
        additional_label_folder,
        modality_iden,
    )


@register_subcommand("util_get_experiment_dirs")
def util_get_experiment_dirs(
    config: ActiveConfig,
    continue_id: int | None = None,
):
    """Get the experiment directories for the current config.

    Args:
        config (ActiveConfig): ActiveConfig object
        continue_id (int, optional): Continue with this id in nnActive Structure. Defaults to None.
    """
    config.set_nnunet_env()
    state = State.latest(config)
    continue_id = state.dataset_id
    print("\n")
    print("Experiment Directories:")
    print("-" * 5)
    out_dict = {}
    out_dict["nnActive_raw"] = str(paths.nnActive_raw / "nnUNet_raw" / config.dataset)
    out_dict["nnActive_results"] = str(get_nnactive_results_folder(continue_id))
    out_dict["nnUNet_raw"] = str(get_raw_path(continue_id))
    out_dict["nnUNet_preprocessed"] = str(get_preprocessed_path(continue_id))
    out_dict["nnUNet_results"] = str(get_results_path(continue_id))
    for key, val in out_dict.items():
        print(f"{key}: '{val}'")
