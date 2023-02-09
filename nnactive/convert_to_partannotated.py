# TODO: how to test
# TODO: Files
import json
import os
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.data import Patch
from nnactive.data.annotate import create_labels_from_patches
from nnactive.data.create_empty_masks import (
    add_ignore_label_to_dataset_json,
    read_dataset_json,
)
from nnactive.loops.cross_validation import kfold_cv_from_patches
from nnactive.loops.loading import get_patches_from_loop_files, save_loop

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
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None


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
        base_labelsTr_dir = base_dir / "labelsTr"
        target_labelsTr_dir = target_dir / "labelsTr"

        ignore_label = target_dataset_json["labels"]["ignore"]
        file_ending = base_dataset_json["file_ending"]

        # create patches list for dataset creation
        patches = get_patches_for_partannotation(
            full_images,
            patch_func,
            file_ending,
            base_labelsTr_dir,
            target_labelsTr_dir,
            patch_func_kwargs=patch_kwargs,
        )

        # Create labels from patches
        create_labels_from_patches(
            patches, ignore_label, file_ending, base_labelsTr_dir, target_labelsTr_dir
        )

        loop_json = {"patches": patches}
        save_loop(target_dir, loop_json, 0)
    else:
        print("No Override")


def get_patches_for_partannotation(
    full_images: Union[int, float],
    patch_func: callable,
    file_ending: str,
    base_labelsTr_dir: Path,
    target_labelsTr_dir: Path,
    patch_func_kwargs: dict = None,
) -> list[Patch]:
    """Creates patches based on annotation strategies.

    Args:
        full_images (Union[int, float]): _description_
        patch_func (callable): _description_
        file_ending (str): _description_
        base_labelsTr_dir (Path): _description_
        target_labelsTr_dir (Path): _description_
        patch_func_kwargs (dict, optional): _description_. Defaults to None.

    Returns:
        list[Patch]: annotated patches
    """
    if patch_func_kwargs is None:
        patch_func_kwargs = {}
    os.makedirs(target_labelsTr_dir, exist_ok=True)

    seg_names = os.listdir(base_labelsTr_dir)
    seg_names = [seg_name for seg_name in seg_names if seg_name.endswith(file_ending)]

    # Current implementation only works if all data has a corresponding lablesTr

    rand_np_state = np.random.RandomState(random_seed)
    rand_np_state.shuffle(seg_names)

    if full_images < 1:
        full_images = int(len(seg_names) * full_images)
    patches = [
        Patch(file=seg_names.pop(), coords=[0, 0, 0], size="whole")
        for i in range(full_images)
    ]

    patches: list[Patch] = (
        patch_func(seg_names=seg_names, **patch_func_kwargs) + patches
    )
    return patches


def generate_custom_splits_file(
    target_id: int, loop_count: Optional[int] = None, num_folds: int = 5
):
    """Generates a custom split file in NNUNET_PREPROCESSED folder which only has labeled data.

    Args:
        target_id (int): dataset id
        loop_count (int): X for loop_XXX.json file all smaller are also taken to get list of names
        num_folds (int, optional): Folds vor KFoldCV. Defaults to 5.
    """
    dataset: str = convert_id_to_dataset_name(target_id)

    patches = get_patches_from_loop_files(NNUNET_RAW / dataset, loop_count)

    splits_final = kfold_cv_from_patches(num_folds, patches)

    # Create path if not exists
    if not (NNUNET_PREPROCESSED / dataset).exists():
        os.makedirs(NNUNET_PREPROCESSED / dataset)
    # save splits_file
    with open(NNUNET_PREPROCESSED / dataset / "splits_final.json", "w") as file:
        json.dump(splits_final, file, indent=4)


def dummy_patch_func(*args, **kwargs) -> list[Patch]:
    return []


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
        base_dataset_id,
        target_id,
        full_images,
        rewrite=rewrite,
        patch_func=dummy_patch_func,
    )
    generate_custom_splits_file(target_id, 0, 5)
