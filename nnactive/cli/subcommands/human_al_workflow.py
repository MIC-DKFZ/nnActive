import json
import os
import shutil
from pathlib import Path
from typing import List

import nnunetv2.paths
import numpy as np

from nnactive.cli.registry import register_subcommand
from nnactive.loops.loading import get_sorted_loop_files, save_loop
from nnactive.utils import create_mitk_geometry_patch
from nnactive.utils.io import load_json
from nnactive.utils.mitk_integration import get_file_patch_list


@register_subcommand("human_al_selection_to_loop")
def human_al_manual_selection_to_loop(raw_folder: str, debug: bool = False) -> None:
    """
    Create a loop_XXX file that contains the manually selected patches as a list that should be included for
    training in the next cycle. The manual selected patches are stored as cropped versions of the original images
    in the patches_manual_selected folder in the raw data path.
    If some of the manually selected patches overlap, the loop file will not be created and the user is asked to
    create the patches again without overlap.
    """

    # create an empty dict to store all patches that should be in the loop_XXX.json file
    all_patches_dict = {"patches": []}

    # setup path and get image names
    data_path = Path(raw_folder)
    dataset_json = load_json(data_path / "dataset.json")
    file_ending = dataset_json["file_ending"]

    images_tr_dir = data_path / "imagesTr"
    selected_patch_dir = data_path / "patches_manual_selected"
    os.makedirs(selected_patch_dir, exist_ok=True)
    images = [
        images_tr_dir / image
        for image in os.listdir(images_tr_dir)
        if image.endswith(file_ending)
    ]

    # preliminary is set to true as soon as some patches overlap, which means no loop file is created
    preliminary = False

    # iterate through images and get patch list for the images
    for image in images:
        patches_image_list, preliminary_image = get_file_patch_list(
            original_image_path=image,
            cropped_path=selected_patch_dir,
            data_path=data_path,
            debug=debug,
        )
        all_patches_dict["patches"].extend(patches_image_list)
        if preliminary_image:
            preliminary = True
    # store loop file if no patches overlap
    if not preliminary:
        loop = len(get_sorted_loop_files(data_path))
        save_loop(data_path, all_patches_dict, loop)
        # prelim_patches is the folder where overlapping patches are stored as .mitkgeometry files
        if os.path.isdir(selected_patch_dir / "prelim_patches"):
            shutil.rmtree(selected_patch_dir / "prelim_patches")


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
