# TODO: Make this script based on argparse arguments
# TODO: functional instead of pipeline workflow
# TODO: Check new script based on function
# TODO: make a script which creates a custom cross-validation file for splits!
import shutil
import os
from typing import Optional, Union, List
import json
import numpy as np
from argparse import ArgumentParser
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

# import SimpleITK as sitk
import random
from pprint import pprint
from copy import deepcopy
from pathlib import Path

from create_empty_masks import (
    create_empty_mask,
    read_dataset_json,
    add_ignore_label_to_dataset_json,
)

parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset-id",
    type=int,
    help="dataset ID for nnU-Net, needs to be present in $nnUNet_raw",
)
parser.add_argument(
    "-o",
    "--output-id",
    type=int,
    default=None,
    help="target dataset ID for nnU-Net, default base on offset",
)
parser.add_argument(
    "-f",
    "--force-override",
    action="store_true",
    help="Force overriding output dataset",
)
parser.add_argument(
    "--offset", type=int, default=500, help="ouput_id = dataset_id + offset"
)
parser.add_argument("--seed", default=12345)

parser.add_argument(
    "--full-labeled",
    type=str,
    help="0.X = percentage, int = full number of completely annotated images",
)  # how to make float and integers
parser.add_argument(
    "--minimal-example-per-class",
    type=int,
    default=None,
    help="minimal amount of examples per class -- not implemented yet",
)  # int
parser.add_argument(
    "--partial-labeled",
    type=str,
    default=None,
    help="Labeling Scheme for partial annotation -- not implemented yet",
)  # how to make float and integers

NNUNET_RAW = Path(nnUNet_raw)

def placeholder_patch_anno(
    image_names: List[str], patch_kwargs: dict, label_area: List[dict]
):
    return image_names, label_area


def convert_dataset_to_unannotated(
    base_id: int,
    id_offset: int,
    full_images: Union[float, int],
    target_id=None,
    rewrite=False,
    name_suffix: str = "partanno",
    patch_kwargs: Optional[dict] = None,
):
    nnUNet_raw = NNUNET_RAW

    if target_id is None:
        target_id = id_offset + base_id

    # check if target_id already exists
    already_exists = False
    try:
        convert_id_to_dataset_name(target_id)
        print("Dataset with ID {} already exists.".format(target_id))
        already_exists = True
    except:
        pass
    # remove Folder if target_id dataset already exists and rewrite
    if already_exists and rewrite is True:
        target_folder:str = convert_id_to_dataset_name(target_id)
        print("Dataset ID {} already in nnU-Net under name {}".format(target_id, target_folder))
        remove_folder = nnUNet_raw / target_folder
        if remove_folder.exists():
            print("Removing already existing target directory:\n{}".format(remove_folder))
            shutil.rmtree(remove_folder)
        else:
            print("Found no folder: {}".format(remove_folder))
            print("Proceed as if no id conflict")
            already_exists = False
    # logic for creating partially annotated dataset
    if not already_exists or (already_exists and rewrite is True):
        # load base_dataset_json
        base_dataset: str = convert_id_to_dataset_name(base_id)
        base_dataset_json: dict = read_dataset_json(base_dataset)
        base_dir = nnUNet_raw / base_dataset

        # rewrite target_dataset_json and save
        target_dataset_json = deepcopy(base_dataset_json)
        target_dataset_json["name"] = "{}-{}".format(
            base_dataset_json["name"], name_suffix
        )
        target_dataset_json = add_ignore_label_to_dataset_json(
            target_dataset_json, base_dataset
        )
        target_dataset: str = f"Dataset{target_id:03d}_" + target_dataset_json["name"]
        target_dir = nnUNet_raw / target_dataset
        os.makedirs(target_dir)
        # Save target dataset.json
        with open(target_dir / "dataset.json", "w") as file:
            json.dump(target_dataset_json, file)
        assert (
            read_dataset_json(base_dataset) == base_dataset_json
        )  # basedataset/dataset.json is not supposed to change!

        # Copy all data except for labelsTr to target_dir and dataset.json
        copy_folders = ["imagesTr", "imagesTs", "labelsTs"]
        for copy_folder in copy_folders:
            if copy_folder in os.listdir(base_dir):
                shutil.copytree(base_dir / copy_folder, target_dir / copy_folder)
            else:
                print(
                    "Skip Path for copying into target:\n{}".format(
                        base_dir / copy_folder
                    )
                )


        # Create imagesTr for target dataset
        imagesTr_dir = base_dir / "imagesTr"
        base_labelsTr_dir = base_dir / "labelsTr"
        target_labelsTr_dir = target_dir / "labelsTr"

        os.makedirs(target_labelsTr_dir, exist_ok=True)
        ignore_label = target_dataset_json["labels"]["ignore"]

        image_names = os.listdir(imagesTr_dir)
        image_names = [
            image_name
            for image_name in image_names
            if image_name.endswith(target_dataset_json["file_ending"])
        ]

        seg_names = os.listdir(base_labelsTr_dir)
        seg_names = [
            seg_name
            for seg_name in seg_names
            if seg_name.endswith(target_dataset_json["file_ending"])
        ]

        # Current implementation only works if all data has a corresponding lablesTr
        assert len(image_names) == len(seg_names)
        rand_np_state = np.random.RandomState(random_seed)
        rand_np_state.shuffle(image_names)

        if full_images < 1:
            full_images = int(len(image_names) * full_images)
        full_ano = [image_names.pop() for i in range(full_images)]
        label_json = []

        # Copyt labelsTr from base to target for full_ano training images
        for image_name in full_ano:
            # Create savename for segmentation
            data_name = "_".join(image_name.split("_")[:-1])
            seg_name = data_name + target_dataset_json["file_ending"]
            if seg_name in seg_names:
                shutil.copy(
                    base_labelsTr_dir / seg_name, target_labelsTr_dir / seg_name
                )
                label_json.append(
                    {
                        "file": seg_name,
                        "coords": [0, 0, 0],
                        "size": "whole",
                    }
                )

        # TODO Put here logic for part_ano training images
        image_names, label_area = placeholder_patch_anno(
            image_names, patch_kwargs, label_area
        )

        # Create empty masks for the rest of the training images
        for image_name in image_names:
            save_filename = f"{'_'.join(image_name.split('_')[:-1])}{base_dataset_json['file_ending']}"
            create_empty_mask(
                imagesTr_dir / image_name,
                ignore_label,
                target_labelsTr_dir / save_filename,
            )

        with open(target_dir / "label_00.json", "w") as file:
            json.dump(label_json, file)
    else:
        print("No Override")


if __name__ == "__main__":
    args = parser.parse_args()
    # TODO: rewrite arguements
    full_images = args.full_labeled

    partial_labeled = args.partial_labeled
    minimal_example_per_class = args.minimal_example_per_class
    assert partial_labeled is None  # partial labeling not implemented yet
    assert (
        minimal_example_per_class is None
    )  # minimal examples per class not implemented yet

    base_dataset_id = args.dataset_id
    target_id = args.output_id
    id_offset = args.offset
    random_seed = args.seed
    rewrite = args.force_override
    convert_dataset_to_unannotated(
        base_dataset_id, id_offset, full_images, target_id=target_id, rewrite=rewrite
    )
