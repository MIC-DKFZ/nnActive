import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import nnunetv2.paths
import numpy as np
from loguru import logger
from tqdm import tqdm

from nnactive.cli.registry import register_subcommand
from nnactive.loops.loading import (
    get_patches_from_loop_files,
    get_sorted_loop_files,
    save_loop,
)
from nnactive.update_data import update_data
from nnactive.utils import create_mitk_geometry_patch
from nnactive.utils.io import load_json
from nnactive.utils.mitk_integration import get_file_patch_list


# TODO: verify how to give patch_size to the model.
@register_subcommand("human_al_selection_to_loop")
def human_al_manual_selection_to_loop(
    raw_folder: str,
    patch_size: Tuple[int, int, int] = (48, 224, 224),
    loop: int | None = None,
    debug: bool = False,
) -> None:
    """
    Create a loop_XXX file that contains the manually selected patches as a list that should be included for
    training in the next cycle. The manual selected patches are stored as cropped versions of the original images
    in the patches_manual_selected folder in the raw data path.
    If some of the manually selected patches overlap, the loop file will not be created and the user is asked to
    create the patches again without overlap.
    """
    # turn around the patch size list to match the order inside of MITK...
    patch_size = list(patch_size)[::-1]
    patch_size = tuple(patch_size)
    # create an empty dict to store all patches that should be in the loop_XXX.json file
    all_patches_dict = {"patches": []}

    # setup path and get image names
    data_path = Path(raw_folder)
    dataset_json = load_json(data_path / "dataset.json")
    file_ending = dataset_json["file_ending"]
    if loop is None:
        loop = len(get_sorted_loop_files(data_path))

    images_tr_dir = data_path / "imagesTr"
    selected_patch_dir = data_path / f"patches_manual_selected_{loop:02d}"

    image_patch = [
        images_tr_dir / (image.name[: -len("_00.nrrd")] + file_ending)
        for image in selected_patch_dir.iterdir()
        if image.name.endswith(".nrrd")
    ]

    # preliminary is set to true as soon as some patches overlap, which means no loop file is created
    preliminary = False

    # iterate through images and get patch list for the images

    # subtract one value from loop_val as we want to only take the patches from the previous loop into account
    all_patches = get_patches_from_loop_files(data_path, loop - 1)
    logger.info("Found {} patches for loop {}".format(len(all_patches), loop))
    for image in tqdm(image_patch):
        file_patches = [
            patch
            for patch in all_patches
            if patch.file[: -len(file_ending)]
            == image.name[: -(len(file_ending) + len("_0000"))]
        ]
        patches_image_list, preliminary_image = get_file_patch_list(
            original_image_path=image,
            cropped_path=selected_patch_dir,
            patch_size_required=patch_size,
            file_patches=file_patches,
            debug=debug,
        )
        all_patches_dict["patches"].extend(patches_image_list)
        if preliminary_image:
            preliminary = True
    # store loop file if no patches overlap
    if not preliminary:
        save_loop(data_path, all_patches_dict, loop)
        # prelim_patches is the folder where overlapping patches are stored as .mitkgeometry files
        if os.path.isdir(selected_patch_dir / "prelim_patches"):
            shutil.rmtree(selected_patch_dir / "prelim_patches")


# TODO: allow updating of data based on annotations based on paths!
@register_subcommand("human_al_update_data")
def human_al_update_data(
    target_raw_folder: str,
    splits_file_path: str,
    loop_val: int,
    ignore_label: int,
):
    """Update the data based on the loop file.
    Can be executed outside of nnActive folder structure.

    Args:
        target_raw_folder (str): Path to the raw data folder
        splits_file (str): Path to the splits_file file created for current loop
        loop_val (int): Loop number
        debug (bool, optional): Debug mode. Defaults to False.
    """
    data_path = Path(target_raw_folder)
    splits_file_path = Path(splits_file_path)

    label_dir = data_path / f"annoTr_{loop_val:02}"

    additional_label_path = data_path / f"additional_labels"
    additional_label_path = (
        additional_label_path if additional_label_path.is_dir() else None
    )

    update_data(
        data_path=data_path,
        save_splits_file=splits_file_path,
        ignore_label=ignore_label,
        base_dir=label_dir,
        target_dir=data_path / "labelsTr",
        file_ending=".nii.gz",
        loop_val=loop_val,
        additional_label_path=additional_label_path,
        annotated=False,
    )


@register_subcommand("human_al_create_mitk_geometry_file")
def human_al_create_mitk_geometry_file(
    target_raw_folder: str, plans_file: str, patch_size: List[int]
):
    """Creates mitk_geometry file to allow human to select patches.

    Args:
        target_raw_folder (str): Path where patch.mitkgeometry should be created
        plans_file (str): Path to nnUNetPlans.json file
        patch_size (List[int]): Patch Size
    """
    nnunet_plans_path = Path(plans_file)
    target_dir = Path(target_raw_folder)
    assert nnunet_plans_path.exists()  # Please preprocess the dataset before
    with open(nnunet_plans_path, "r") as f:
        nnunet_plans = json.load(f)
    scale_factor = nnunet_plans["original_median_spacing_after_transp"]
    if len(scale_factor) == 3:
        scale_factor.reverse()
    scale_factor = np.array(scale_factor)
    if len(patch_size) == 3:
        patch_size.reverse()
    patch_size = np.array(patch_size)
    create_mitk_geometry_patch.main(
        target_dir / "patch.mitkgeometry",
        tuple(np.multiply(scale_factor, patch_size)),
    )
