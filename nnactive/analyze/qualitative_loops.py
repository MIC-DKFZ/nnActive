import os
from pathlib import Path

import SimpleITK as sitk

from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_nested_patches_from_loop_files
from nnactive.query.random import create_patch_mask_for_image
from nnactive.utils.io import load_json


def visualize_query_trajectory(raw_folder: Path, output_folder: Path):
    file_ending = load_json(raw_folder / "dataset.json")["file_ending"]
    save_folder = output_folder
    loop_patches = get_nested_patches_from_loop_files(raw_folder)
    img_names = [
        f.name
        for f in (raw_folder / "labelsTr").iterdir()
        if f.name.endswith(file_ending)
    ]
    for i in range(len(loop_patches)):
        os.makedirs(save_folder / f"loop_{i:03d}", exist_ok=True)

    for img_name in img_names:
        all_img_patches = [x for xs in loop_patches for x in xs]
        if len([p for p in all_img_patches if p.file == img_name]) == 0:
            continue
        img = sitk.ReadImage(raw_folder / "labelsTr" / img_name)
        label_shape = sitk.GetArrayFromImage(img).shape
        for i, l_ps in enumerate(loop_patches):
            img_patches = [patch for patch in l_ps if patch.file == img_name]
            if len(img_patches) == 0:
                continue
            mask = create_patch_mask_for_image(
                img_name, l_ps, label_shape, identify_patch=False
            )
            mask = sitk.GetImageFromArray(mask)
            mask = copy_geometry_sitk(mask, img)
            sitk.WriteImage(
                mask,
                (save_folder / f"loop_{i:03d}" / img_name),
            )


if __name__ == "__main__":
    raw_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347"
    )
    output_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/query__analysis"
    )
    visualize_query_trajectory(raw_folder, output_folder)
