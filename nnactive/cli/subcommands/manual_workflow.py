import json
import os
from pathlib import Path

import nnunetv2.paths
import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import (
    get_current_loop,
    get_loop_patches,
    get_patches_from_loop_files,
    get_sorted_loop_files,
)
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.utils.io import load_json, load_label_map
from nnactive.utils.patches import create_patch_mask_for_image


@register_subcommand(
    "manual_crop_pred",
)
def manual_crop_pred(
    raw_folder: str,
    loop: int | None = None,
    labels_folder: str | None = None,
    output_folder: str | None = None,
) -> None:
    """Crop predictions to region requested in loop_xxx.json file.
    Predictions are expected to be in 'predTr_{loop-1}'
    Resulting patches are saved in 'predTr_crop_{loop-1}'

    Args:
        raw_folder (str): Path to folder with raw (containing predTr_{loop-1} and loop_{loop}.json)
        loop (int | None, optional): Set loop file. Defaults to None.
        labels_folder (str | None): Ovewrite default raw_folder/predTr_{loop-1}. Defaults to None.
        output_folder (str | None): Ovewrite default raw_folder/predTr_crop_{loop-1}. Defaults to None.
    """
    raw_folder = Path(raw_folder)

    with open(raw_folder / "dataset.json", "r") as file:
        dataset_json = json.load(file)
    file_ending = dataset_json["file_ending"]

    if loop is None:
        loop = get_current_loop(raw_folder)

    patches = get_loop_patches(raw_folder, loop_val=loop)
    labels_folder = (
        raw_folder / f"predTr_{loop-1:02d}"
        if labels_folder is None
        else Path(labels_folder)
    )

    logger.info(
        f"Creation of cropped predictions for loop {loop} with {len(patches)} Patches"
    )

    img_names = [
        file for file in os.listdir(labels_folder) if file.endswith(file_ending)
    ]
    logger.info(f"Found images {len(img_names)} in {labels_folder}")

    output_folder = (
        raw_folder / f"predTr_crop_{loop-1:02d}"
        if output_folder is None
        else Path(output_folder)
    )
    logger.info(f"Saving images to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    for img_name in img_names:
        img_patches = [patch for patch in patches if patch.file == img_name]
        if len(img_patches) == 0:
            continue
        logger.info("-" * 8)
        logger.info(f"Start Image: {img_name}")
        logger.info("Load label...")
        seg = load_label_map(img_name, labels_folder, "")

        logger.info("Select region...")
        seg_crop = np.zeros_like(seg)

        for i, img_patch in enumerate(img_patches):
            slices = []
            for start_index, size in zip(img_patch.coords, img_patch.size):
                slices.append(slice(start_index, start_index + size))
            seg_crop[tuple(slices)] = seg[tuple(slices)]
        logger.info("Save image...")
        img = sitk.ReadImage(labels_folder / img_name)
        seg_save = sitk.GetImageFromArray(seg_crop)
        seg_save = copy_geometry_sitk(seg_save, img)
        sitk.WriteImage(
            seg_save,
            (output_folder / img_name),
        )


@register_subcommand("manual_query")
def manual_query(
    raw_folder: str, loop: int | None = None, identify_patches: bool = False
) -> None:
    """Prepare the query step giving patches to human annotators in form of masks.

    Args:
        raw_folder (str): path to raw_folder for experiment
        loop (int | None, optional): multiple patches within the same input get different labels for identification. Defaults to None.
        identify_patches (bool, optional): _description_. Defaults to False.
    """

    data_path = Path(raw_folder)
    labels_dir = data_path / "labelsTr"

    dataset_json = load_json(data_path / "dataset.json")
    file_ending = dataset_json["file_ending"]

    if loop is None:
        loop = get_current_loop(data_path)

    patches = get_loop_patches(data_path, loop_val=loop)

    logger.info(f"Found Patches: {len(patches)}")

    img_names = [file for file in os.listdir(labels_dir) if file.endswith(file_ending)]
    logger.info(f"Image Names: {len(img_names)}")
    save_path = data_path / f"masksTr_boundary_{loop:02d}"
    logger.info(f"Saving boundaries of patches to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    for img_name in img_names:
        img_patches = [patch for patch in patches if patch.file == img_name]
        if len(img_patches) == 0:
            continue
        logger.info("-" * 8)
        logger.info(f"Start Image: {img_name}")
        logger.info("Load label...")
        label_shape = load_label_map(
            img_name.replace(file_ending, ""), labels_dir, file_ending
        ).shape
        logger.info("Create Mask...")
        mask = create_patch_mask_for_image(
            img_name, patches, label_shape, identify_patch=identify_patches
        )
        logger.info("Save Image...")
        img = sitk.ReadImage(labels_dir / img_name)
        mask_bound = create_patch_mask_for_image(
            img_name,
            patches,
            label_shape,
            identify_patch=identify_patches,
            size_offset=-1,
        )
        mask_bound = mask - mask_bound
        mask_save = sitk.GetImageFromArray(mask_bound)
        mask_save = copy_geometry_sitk(mask_save, img)
        sitk.WriteImage(
            mask_save,
            (save_path / img_name),
        )


@register_subcommand(
    "manual_vis_annotated",
)
def manual_vis_annotated(
    config: ActiveConfig,
    continue_id: int | None = None,
    data_path: str | None = None,
    output_path: str | None = None,
    loop: int | None = None,
    one_loop: bool = False,
    identify_patch: bool = False,
):
    if data_path is None:
        config.set_nnunet_env()
        print(f"{continue_id=}")
        if continue_id is None:
            state = State.latest(config)
        else:
            state = State.get_id_state(continue_id)

        dataset_id = state.dataset_id
        raw_dataset_path = get_raw_path(dataset_id)
    else:
        raw_dataset_path = Path(data_path)

    labels_dir = raw_dataset_path / "labelsTr"

    dataset_json = load_json(raw_dataset_path / "dataset.json")
    file_ending = dataset_json["file_ending"]

    loop = len(get_sorted_loop_files(raw_dataset_path)) - 1 if loop is None else loop

    save_path = (
        raw_dataset_path / "annotated_regions"
        if output_path is None
        else Path(output_path)
    )
    if loop >= 0:
        os.makedirs(save_path, exist_ok=True)
        if one_loop:
            labeled_patches = get_loop_patches(raw_dataset_path, loop)
        else:
            labeled_patches = get_patches_from_loop_files(raw_dataset_path, loop)

        img_names = [
            file for file in os.listdir(labels_dir) if file.endswith(file_ending)
        ]
        os.makedirs(save_path, exist_ok=True)
        for img_name in img_names:
            img_patches = [patch for patch in labeled_patches if patch.file == img_name]
            if len(img_patches) == 0:
                continue
            label_shape = load_label_map(
                img_name.replace(file_ending, ""),
                raw_dataset_path / "labelsTr",
                file_ending,
            ).shape
            mask = create_patch_mask_for_image(
                img_name, labeled_patches, label_shape, identify_patch=identify_patch
            )
            img = sitk.ReadImage(labels_dir / img_name)
            mask_save = sitk.GetImageFromArray(mask)
            mask_save = copy_geometry_sitk(mask_save, img)
            sitk.WriteImage(
                mask_save,
                (save_path / img_name),
            )
