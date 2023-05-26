import os
from argparse import Namespace

import SimpleITK as sitk

from nnactive.cli.registry import register_subcommand
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_patches_from_loop_files, get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.random import create_patch_mask_for_image
from nnactive.query_patches import get_label_map


@register_subcommand(
    "manual_vis_annotated",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True, "help": "Dataset ID"}),
    ],
)
def main(args: Namespace):
    dataset_id: int = args.dataset_id

    raw_dataset_path = get_raw_path(dataset_id)
    labels_dir = raw_dataset_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]

    loop = len(get_sorted_loop_files(raw_dataset_path))
    save_path = raw_dataset_path / "annotated"
    if loop >= 0:
        os.makedirs(save_path, exist_ok=True)
        labeled_patches = get_patches_from_loop_files(raw_dataset_path, loop - 1)

        img_names = [
            file for file in os.listdir(labels_dir) if file.endswith(file_ending)
        ]
        os.makedirs(save_path, exist_ok=True)
        for img_name in img_names:
            img_patches = [patch for patch in labeled_patches if patch.file == img_name]
            if len(img_patches) == 0:
                continue
            label_shape = get_label_map(
                img_name.replace(file_ending, ""), raw_dataset_path, file_ending
            ).shape
            mask = create_patch_mask_for_image(
                img_name, labeled_patches, label_shape, identify_patch=True
            )
            img = sitk.ReadImage(labels_dir / img_name)
            mask_save = sitk.GetImageFromArray(mask)
            mask_save = copy_geometry_sitk(mask_save, img)
            sitk.WriteImage(
                mask_save,
                (save_path / img_name),
            )
