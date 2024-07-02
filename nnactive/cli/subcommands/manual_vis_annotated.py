import os
from argparse import Namespace
from pathlib import Path

import SimpleITK as sitk

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import (
    get_loop_patches,
    get_patches_from_loop_files,
    get_sorted_loop_files,
)
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.results.state import State
from nnactive.utils.io import load_json, load_label_map
from nnactive.utils.patches import create_patch_mask_for_image


@register_subcommand(
    "manual_vis_annotated",
)
def main(
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
