import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_current_loop, get_loop_patches
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.random import create_patch_mask_for_image, load_label_map


@register_subcommand(
    "manual_crop_pred",
)
def main(
    data_path: str,
    config: ActiveConfig = ActiveConfig([0, 0, 0]),
    runtime_config: RuntimeConfig = RuntimeConfig(),
    continue_id: int | None = None,
    loop: int | None = None,
) -> None:
    """Crop predictions to region requested in loop_xxx.json file.
    Predictions are expected to be in 'predTr_{loop-1}'
    Resulting patches are saved in 'predTr_crop_{loop-1}'

    Args:
        data_path (str): Path to folder with raw (containing predTr_{loop-1} and loop_{loop}.json)
        config (ActiveConfig, optional): Ignore. Defaults to ActiveConfig([0, 0, 0]).
        runtime_config (RuntimeConfig, optional): Ignore. Defaults to RuntimeConfig().
        continue_id (int | None, optional): Ignore. Defaults to None.
        loop (int | None, optional): Set loop file. Defaults to None.
    """
    data_path = Path(data_path)

    with open(data_path / "dataset.json", "r") as file:
        dataset_json = json.load(file)
    file_ending = dataset_json["file_ending"]

    if loop is None:
        loop = get_current_loop(data_path)

    patches = get_loop_patches(data_path, loop_val=loop)
    labels_dir = data_path / f"predTr_{loop-1:02d}"

    logger.info(
        f"Creation of cropped predictions for loop {loop} with {len(patches)} Patches"
    )

    img_names = [file for file in os.listdir(labels_dir) if file.endswith(file_ending)]
    logger.info(f"Found images {len(img_names)} in {labels_dir}")
    save_path = data_path / f"predTr_crop_{loop-1:02d}"
    logger.info(f"Saving images to: {save_path}")
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
