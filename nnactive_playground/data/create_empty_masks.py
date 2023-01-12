"""_
Script to create empty masks that are set to 'ignore label' value. 
Adds ignore value to dataset.json if not already present
"""

import os
import json
import numpy as np
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
import SimpleITK as sitk

def read_dataset_json(dataset_name):
    raw_data_folder = os.path.join(nnUNet_raw, dataset_name)
    dataset_json_path = os.path.join(raw_data_folder, "dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)
    return dataset_json

def add_ignore_label_to_dataset_json(dataset_json, dataset_name):        
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
        raw_data_folder = os.path.join(nnUNet_raw, dataset_name)
        dataset_json_path = os.path.join(raw_data_folder, "dataset.json")
        with open(dataset_json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)
    return dataset_json

def create_empty_mask(image_filename, ignore_label, save_filename):
    """Create an empty label mask for a sitk readable image with ignore label.

    Args:
        image_filename (_type_): _description_
        ignore_label (_type_): _description_
        save_filename (_type_): _description_
    """
    img_itk = sitk.ReadImage(image_filename)
    img_npy = sitk.GetArrayFromImage(img_itk)
    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = np.array(img_itk.GetDirection())
    img_npy.fill(ignore_label)
    img_itk_new = sitk.GetImageFromArray(img_npy)
    img_itk_new.SetSpacing(spacing)
    img_itk_new.SetOrigin(origin)
    img_itk_new.SetDirection(direction)
    sitk.WriteImage(img_itk_new, save_filename)

def create_images_ignore_label(dataset_name, dataset_json):
    imagesTr_dir = os.path.join(nnUNet_raw, dataset_name, "imagesTr")
    labelsTr_dir = os.path.join(nnUNet_raw, dataset_name, "labelsTr")
    os.makedirs(labelsTr_dir, exist_ok=True)
    ignore_label  = dataset_json["labels"]["ignore"]
    
    for image_path in os.listdir(imagesTr_dir):
        if image_path.endswith(dataset_json["file_ending"]):
            save_filename = f"{'_'.join(image_path.split('_')[:-1])}{dataset_json['file_ending']}"
            create_empty_mask(os.path.join(imagesTr_dir, image_path), ignore_label, os.path.join(labelsTr_dir, save_filename))
            
    
        

if __name__ == "__main__":
    dataset_name = convert_id_to_dataset_name(4)
    dj = read_dataset_json(dataset_name)
    dj = add_ignore_label_to_dataset_json(dj, dataset_name)
    create_images_ignore_label(dataset_name, dj)