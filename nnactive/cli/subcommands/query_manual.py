import os
from argparse import Namespace

import SimpleITK as sitk

from nnactive.cli.registry import register_subcommand
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.random import create_patch_mask_for_image, get_label_map


@register_subcommand(
    "query_manual", [(("-d", "--dataset_id"), {"type": int, "required": True})]
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id

    data_path = get_raw_path(dataset_id)
    labels_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    file_ending = dataset_json["file_ending"]

    patches = get_patches_from_loop_files(data_path, None)
    print(f"Found Patches: {len(patches)}")

    img_names = [file for file in os.listdir(labels_dir) if file.endswith(file_ending)]
    print(f"Image Names: {len(img_names)}")
    save_path = data_path / "masksTr"
    save_path = data_path / "masksTr_boundary"
    os.makedirs(save_path, exist_ok=True)
    for img_name in img_names:
        label_shape = get_label_map(
            img_name.replace(file_ending, ""), labels_dir, file_ending
        ).shape
        mask = create_patch_mask_for_image(
            img_name, patches, label_shape, identify_patch=True
        )
        img = sitk.ReadImage(labels_dir / img_name)
        mask_save = sitk.GetImageFromArray(mask)
        mask_save = copy_geometry_sitk(mask_save, img)
        sitk.WriteImage(
            mask_save,
            (save_path / img_name),
        )
        mask_bound = create_patch_mask_for_image(
            img_name, patches, label_shape, identify_patch=True, size_offset=-1
        )
        mask_bound = mask - mask_bound
        mask_save = sitk.GetImageFromArray(mask_bound)
        mask_save = copy_geometry_sitk(mask_save, img)
        sitk.WriteImage(
            mask_save,
            (save_path / img_name),
        )
