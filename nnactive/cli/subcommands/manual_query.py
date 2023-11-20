import json
import os
from argparse import Namespace

import SimpleITK as sitk

from nnactive.cli.registry import register_subcommand
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_loop_patches
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.random import create_patch_mask_for_image, load_label_map


@register_subcommand(
    "manual_query",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
        (
            ("--identify_patches"),
            {
                "action": "store_true",
                "help": "multiple patches within the same input get different labels for identification.",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    identify_patches = args.identify_patches

    data_path = get_raw_path(dataset_id)
    labels_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]

    patches = get_loop_patches(data_path)

    print(f"Found Patches: {len(patches)}")

    img_names = [file for file in os.listdir(labels_dir) if file.endswith(file_ending)]
    print(f"Image Names: {len(img_names)}")
    save_path = data_path / "masksTr_boundary"
    os.makedirs(save_path, exist_ok=True)
    for img_name in img_names:
        img_patches = [patch for patch in patches if patch.file == img_name]
        if len(img_patches) == 0:
            continue
        print("-" * 8)
        print(f"Start Image: {img_name}")
        print("Load label...")
        label_shape = load_label_map(
            img_name.replace(file_ending, ""), labels_dir, file_ending
        ).shape
        print("Create Mask...")
        mask = create_patch_mask_for_image(
            img_name, patches, label_shape, identify_patch=identify_patches
        )
        print("Save Image...")
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
