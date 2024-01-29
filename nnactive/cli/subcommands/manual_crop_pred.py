import json
import os
from argparse import Namespace

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_loop_patches
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.random import create_patch_mask_for_image, load_label_map


# TODO: rename this to manual_query
@register_subcommand(
    "manual_crop_pred",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
        (
            ("-i", "--input_folder"),
            {
                "type": str,
                "required": True,
                "help": "Path to folder containing predictions used for filling areas within patches.",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id

    data_path = get_raw_path(dataset_id)
    labels_dir = data_path / "predTr_00"

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]

    patches = get_loop_patches(data_path)

    logger.info(f"Found Patches: {len(patches)}")

    img_names = [file for file in os.listdir(labels_dir) if file.endswith(file_ending)]
    logger.info(f"Image Names: {len(img_names)}")
    save_path = data_path / "predTr_crop"
    os.makedirs(save_path, exist_ok=True)
    for img_name in img_names:
        img_patches = [patch for patch in patches if patch.file == img_name]
        if len(img_patches) == 0:
            continue
        logger.info("-" * 8)
        logger.info(f"Start Image: {img_name}")
        logger.info("Load label...")
        seg = load_label_map(img_name.replace(file_ending, ""), labels_dir, file_ending)

        logger.info("Select region...")
        seg_crop = np.zeros_like(seg)

        for i, img_patch in enumerate(img_patches):
            slices = []
            for start_index, size in zip(img_patch.coords, img_patch.size):
                slices.append(slice(start_index, start_index + size))
            seg_crop[tuple(slices)] = seg[tuple(slices)]
        logger.info("Save image...")
        img = sitk.ReadImage(labels_dir / img_name)
        seg_save = sitk.GetImageFromArray(seg_crop)
        seg_save = copy_geometry_sitk(seg_save, img)
        sitk.WriteImage(
            seg_save,
            (save_path / img_name),
        )
