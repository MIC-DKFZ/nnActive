"""_
Script to create empty masks that are set to 'ignore label' value. 
Adds ignore value to dataset.json if not already present
"""

import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.data.utils import copy_geometry_sitk


def read_dataset_json(dataset_name: str) -> dict:
    raw_data_folder: str = os.path.join(nnUNet_raw, dataset_name)
    dataset_json_path: str = os.path.join(raw_data_folder, "dataset.json")
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    return dataset_json


def add_ignore_label_to_dataset_json(dataset_json: dict) -> dict:
    """Add ignore label to the dataset_json dict if not already present.
    ignore label value = max(label value)+1

    Args:
        dataset_json (dict): _description_

    Returns:
        dict: dataset_json
    """
    if "ignore" not in dataset_json["labels"]:
        all_labels = []
        for k, r in dataset_json["labels"].items():
            if isinstance(r, (tuple, list)):
                for ri in r:
                    all_labels.append(int(ri))
            else:
                all_labels.append(int(r))
        all_labels = list(np.unique(all_labels))
        ignore_label_id = int(max(all_labels) + 1)
        dataset_json["labels"]["ignore"] = ignore_label_id
    return dataset_json


def create_empty_mask(
    image_filename: Path,
    ignore_label: int,
    save_filename: Path,
    additional_label_file: Path | None = None,
):
    """Create an empty label mask for a sitk readable image with ignore label.

    Args:
        image_filename (Path): filename of labelmap to be loaded
        ignore_label (int): fill value of new labelmap
        save_filename (Path): path to new filename
        additional_label_file (Path|None): filename of labelmap from which to take values !=1/255
    """
    img_itk = sitk.ReadImage(image_filename)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_npy.fill(ignore_label)
    if additional_label_file is not None:
        new_label = sitk.ReadImage(additional_label_file)
        new_label = sitk.GetArrayFromImage(new_label)
        mask = new_label != 255
        img_npy[mask] = new_label[mask]
    img_itk_new = sitk.GetImageFromArray(img_npy)

    img_itk_new = copy_geometry_sitk(img_itk_new, img_itk)
    sitk.WriteImage(img_itk_new, save_filename)


def create_images_ignore_label(dataset_name: str, dataset_json: dict):
    imagesTr_dir = os.path.join(nnUNet_raw, dataset_name, "imagesTr")
    labelsTr_dir = os.path.join(nnUNet_raw, dataset_name, "labelsTr")
    os.makedirs(labelsTr_dir, exist_ok=True)
    ignore_label = dataset_json["labels"]["ignore"]

    for image_path in os.listdir(imagesTr_dir):
        if image_path.endswith(dataset_json["file_ending"]):
            save_filename = (
                f"{'_'.join(image_path.split('_')[:-1])}{dataset_json['file_ending']}"
            )
            create_empty_mask(
                os.path.join(imagesTr_dir, image_path),
                ignore_label,
                os.path.join(labelsTr_dir, save_filename),
            )


if __name__ == "__main__":
    dataset_name = convert_id_to_dataset_name(4)
    raw_data_folder = os.path.join(nnUNet_raw, dataset_name)
    dataset_json_path = os.path.join(raw_data_folder, "dataset.json")
    dj = read_dataset_json(dataset_name)
    dj = add_ignore_label_to_dataset_json(dj)
    with open(dataset_json_path, "w") as f:
        json.dump(dj, f, indent=2)
    create_images_ignore_label(dataset_name, dj)
