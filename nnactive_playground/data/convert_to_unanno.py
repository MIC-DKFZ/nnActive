# TODO: Make this script based on argparse arguments
# TODO: functional instead of pipeline workflow
# TODO: Check new script based on function
# TODO: make a script which creates a custom cross-validation file for splits!
import shutil
import os
import json
import numpy as np
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
import SimpleITK as sitk
import random

from create_empty_masks import create_empty_mask,create_images_ignore_label,read_dataset_json, add_ignore_label_to_dataset_json

image_percentage=0.2
patch_percentage=0
min_class=0

base_dataset_id = 4
id_offset=500

random_seed= 1234

random.seed(random_seed)
np_state = np.random.RandomState(random_seed)





def create_images_ignore_label(base_dataset_name:str, target_dataset_name:str, dataset_json:dict):
    """Create a new dataset with partially annotated data from a base_dataset to target_dataset

    Args:
        base_dataset_name (str): _description_
        target_dataset_name (str): _description_
        dataset_json (dict): _description_
    """
    imagesTr_dir = os.path.join(nnUNet_raw, base_dataset_name, "imagesTr")
    labelsTr_dir = os.path.join(nnUNet_raw, base_dataset_name, "labelsTr")
    target_labelsTr_dir = os.path.join(nnUNet_raw, target_dataset_name, "labelsTr")

    with open(os.path.join(nnUNet_raw, target_dataset_name, "dataset.json"), "w") as file:
        json.dump(dataset_json, file)
    
    os.makedirs(labelsTr_dir, exist_ok=True)
    ignore_label  = dataset_json["labels"]["ignore"]

    image_paths = os.listdir(imagesTr_dir)
    seg_paths = os.listdir(labelsTr_dir)
    assert len(image_paths) == len(seg_paths)
    random.shuffle(image_paths)

    num_full_ano = int(len(image_paths) * image_percentage)
    full_ano = [image_paths.pop() for i in range(num_full_ano)]

    # Load labelsTr for full_ano training images
    for image_path in full_ano:
        if image_path.endswith(dataset_json["file_ending"]):
            # Create savename for segmentation
            data_name = "_".join(image_path.split('_')[:-1])
            seg_name = data_name+dataset_json["file_ending"]
            if seg_name in seg_paths:
                shutil.copy(os.path.join(labelsTr_dir, seg_name), os.path.join(target_labelsTr_dir, seg_name))

    # TODO Put here logic for part_ano training images
    
            

    # Create empty masks for the rest of the training images
    for image_path in image_paths:
        if image_path.endswith(dataset_json["file_ending"]):
            save_filename = f"{'_'.join(image_path.split('_')[:-1])}{dataset_json['file_ending']}"
            create_empty_mask(os.path.join(imagesTr_dir, image_path), ignore_label, os.path.join(target_labelsTr_dir, save_filename))


if __name__ == "__main__":
    dataset_name = convert_id_to_dataset_name(base_dataset_id)
    dj = read_dataset_json(dataset_name)
    dj["name"] = "{}-partanno".format(dj["name"])
    dj = add_ignore_label_to_dataset_json(dj, dataset_name)

    target_dataset_name = f"Dataset{id_offset+base_dataset_id:03d}_"+dj["name"]
    shutil.copytree(os.path.join(nnUNet_raw, dataset_name), os.path.join(nnUNet_raw, target_dataset_name))

    target_labelsTr_dir = os.path.join(nnUNet_raw, target_dataset_name, "labelsTr")
    shutil.rmtree(target_labelsTr_dir)
    os.makedirs(target_labelsTr_dir)

    create_images_ignore_label(dataset_name, target_dataset_name,dj)
    
