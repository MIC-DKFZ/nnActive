# TODO: how to test
# TODO: Files
import json
import os
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.data.create_empty_masks import (
    add_ignore_label_to_dataset_json,
    create_empty_mask,
    read_dataset_json,
)
from nnactive.data.prepare_starting_budget import (
    Patch,
    make_patches_from_ground_truth,
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
    type=float,
    default=0.1,
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

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None


def placeholder_patch_anno(
    image_names: list[str], patch_kwargs: dict, label_area: list[dict]
):
    return image_names, label_area


def make_whole_from_ground_truth(patches, gt_path, target_path):
    """Copies over all files in patches from gt_path to target_path
    """
    for patch in patches:
        seg_name = patch["file"]
        shutil.copy(
                    gt_path / seg_name, target_path / seg_name
                )

LOOP_NAME = "loop_000.json"


def convert_dataset_to_partannotated(
    base_id: int,
    target_id: int,
    full_images: Union[float, int],
    patch_func: Callable,
    name_suffix: str = "partanno",
    patch_kwargs: Optional[dict] = None,
    rewrite: bool = False,
    force: bool = False,
):

    # check if target_id already exists
    exists_name = None
    already_exists = False
    try:
        # TODO: Add case for nnU-Net preprocessed etc. already exists!
        exists_name = convert_id_to_dataset_name(target_id)
        print(f"Dataset with ID {target_id} already exists under name {exists_name}.")
        already_exists = True
    except:
        pass
    # remove Folder if target_id dataset already exists and rewrite
    if already_exists:
        remove_folders = [
            folder / exists_name
            for folder in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]
        ]
        for remove_folder in remove_folders:
            if remove_folder.exists():
                print(f"Found folder: {remove_folder}")
                if force:
                    shutil.rmtree(remove_folder)
                if rewrite:
                    val = input("Should this folder be deleted? [y/n]")
                    if val == "y":
                        shutil.rmtree(remove_folder)

            else:
                print(f"Found no folder: {remove_folder}")
                print("Proceed as if no ID conflict")

    # logic for creating partially annotated dataset
    if (
        not already_exists
        or (already_exists and rewrite is True)
        or (already_exists and force)
    ):
        # load base_dataset_json
        base_dataset: str = convert_id_to_dataset_name(base_id)
        base_dataset_json: dict = read_dataset_json(base_dataset)
        base_dir = NNUNET_RAW / base_dataset

        # rewrite target_dataset_json and save
        target_dataset_json = deepcopy(base_dataset_json)
        target_dataset_json["name"] = "{}-{}".format(
            base_dataset_json["name"], name_suffix
        )
        target_dataset_json = add_ignore_label_to_dataset_json(
            target_dataset_json, base_dataset
        )
        target_dataset: str = f"Dataset{target_id:03d}_" + target_dataset_json["name"]
        target_dir = NNUNET_RAW / target_dataset
        os.makedirs(target_dir)
        # Save target dataset.json
        with open(target_dir / "dataset.json", "w") as file:
            json.dump(target_dataset_json, file, indent=4)
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

        # Create labelstTr for target dataset
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
        label_list = []

        # Copy labelsTr from base to target for full_ano training images
        for image_name in full_ano:
            # Create savename for segmentation
            data_name = "_".join(image_name.split("_")[:-1])
            seg_name = data_name + target_dataset_json["file_ending"]
            if seg_name in seg_names:
                shutil.copy(
                    base_labelsTr_dir / seg_name, target_labelsTr_dir / seg_name
                )
                label_list.append(
                    {
                        "file": seg_name,
                        "coords": [0, 0, 0],
                        # TODO: Read out Size?
                        "size": "whole",
                    }
                )

        patches: list[Patch] = patch_func()
        make_patches_from_ground_truth(
            patches=patches,
            gt_path=base_labelsTr_dir,
            target_path=target_labelsTr_dir,
            dataset_cfg=target_dataset_json,
            ignore_label=ignore_label,
        )

        patched_images = set(map(lambda patch: patch.file, patches))

        empty_images = [ image_name for image_name in filter(lambda img: img not in patched_images, image_names)]
        # Create empty masks for the rest of the training images
        for image_name in empty_images:
            save_filename = f"{'_'.join(image_name.split('_')[:-1])}{base_dataset_json['file_ending']}"
            create_empty_mask(
                imagesTr_dir / image_name,
                ignore_label,
                target_labelsTr_dir / save_filename,
            )

        loop_json = {"patches": label_list}
        with open(target_dir / LOOP_NAME, "w") as file:
            json.dump(loop_json, file, indent=4)
    else:
        print("No Override")


def generate_custom_splits_file(target_id: int, label_file: str, num_folds: int = 5):
    """Generates a custom split file in NNUNET_PREPROCESSED folder which only has labeled data.

    Args:
        target_id (int): dataset id
        label_file (str): loop_XXX.json file all smaller are also taken to get list of names
        num_folds (int, optional): Folds vor KFoldCV. Defaults to 5.
    """
    dataset: str = convert_id_to_dataset_name(target_id)
    base_name = label_file.split("_")[0]
    number = int(label_file.split("_")[1].split(".")[0])

    patches = []
    for i in range(number + 1):
        fn = f"{base_name}_{i:03d}.json"
        with open(NNUNET_RAW / dataset / fn, "r") as file:
            data_file: dict = json.load(file)
        patch_file: list = data_file["patches"]
        patches = patches + patch_file

    labeled_images = [patch["file"].split(".")[0] for patch in patches]
    labeled_images = list(set(labeled_images))

    splits_final = kfold_cv(num_folds, labeled_images)

    # Create path if not exists
    if not (NNUNET_PREPROCESSED / dataset).exists():
        os.makedirs(NNUNET_PREPROCESSED / dataset)
    # save splits_file
    with open(NNUNET_PREPROCESSED / dataset / "splits_final.json", "w") as file:
        json.dump(splits_final, file, indent=4)


def kfold_cv(k: int, labeled_images: list[str]):
    """Create K Fold CV splits

    Args:
        k (int): num_folds
        labeled_images (list[str]): _description_

    Returns:
        list[dict]: dict={train:list, val:list}
    """
    folds = [[] for _ in range(k)]
    rand_np_state = np.random.RandomState(random_seed)
    rand_np_state.shuffle(labeled_images)
    for i in range(len(labeled_images)):
        folds[i % k].append(labeled_images.pop())

    for fold in folds:
        assert (
            len(fold) > 0
        )  # no fold is supposed to have a length of zero! set num_folds smaller

    splits_final = []
    for i in range(k):
        train_select = [j for j in range(k) if j != i]
        val_select = [i]
        train_set = []
        for train_fold in train_select:
            train_set = train_set + folds[train_fold]
        val_set = []
        for val_fold in val_select:
            val_set = val_set + folds[val_fold]
        splits_final.append(
            {
                "train": train_set,
                "val": val_set,
            }
        )

    return splits_final


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
    if not isinstance(target_id, int):
        target_id = base_dataset_id + id_offset
    print(f"output_id is {target_id}")
    convert_dataset_to_partannotated(
        base_dataset_id, target_id, full_images, rewrite=rewrite
    )
    generate_custom_splits_file(target_id, LOOP_NAME, 5)
