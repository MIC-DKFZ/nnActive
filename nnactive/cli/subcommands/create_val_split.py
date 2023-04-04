import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np

from nnactive.cli.registry import register_subcommand
from nnactive.nnunet.utils import get_raw_path, read_dataset_json

random_seed = 12345


def create_test_datasets(
    base_labelsTr_dir: Path,
    base_imagesTr_dir: Path,
    target_labelsVal_dir: Path,
    target_imagesVal_dir: Path,
    file_ending: str,
    test_size: Union[int, float] = 0.25,
    move: bool = True,
):
    seg_names = os.listdir(base_labelsTr_dir)
    seg_names = [seg_name for seg_name in seg_names if seg_name.endswith(file_ending)]

    def _clean_file_ending(file_names: list[str]):
        file_names = [seg_name[: -len(file_ending)] for seg_name in file_names]
        return file_names

    seg_names = _clean_file_ending(seg_names)

    rng = np.random.default_rng(random_seed)
    rng.shuffle(seg_names)
    if test_size < 1:
        test_size = test_size * len(seg_names)
    test_size = int(test_size)

    val_segs = [seg_names.pop() for _ in range(test_size)]

    image_names = os.listdir(base_imagesTr_dir)
    image_names = [
        image_name for image_name in image_names if image_name.endswith(file_ending)
    ]
    image_names = _clean_file_ending(image_names)

    # deprecated due to naming issues in general.
    # def _return_true_if_string_in_list_set(string: str, list_set: list[str]):
    #     for list_string in list_set:
    #         if string[: len(list_string)] == list_string:
    #             return True
    #     return False

    def _return_true_if_file_in_list_set(string: str, list_set: list[str]) -> bool:
        for list_string in list_set:
            if "_".join(string.split("_")[:-1]) == list_string:
                return True
        return False

    val_images = [
        image_name
        for image_name in image_names
        if _return_true_if_file_in_list_set(image_name, val_segs)
    ]
    print(
        f"Moving {len(val_segs)} out {len(val_segs)+len(seg_names)} Label Maps to Validation Data"
    )
    print(f"Moving images from folder {base_imagesTr_dir} to {target_imagesVal_dir}")
    print(f"Moving labels from folder {base_labelsTr_dir} to {target_labelsVal_dir}")

    if move:
        move_files(base_labelsTr_dir, target_labelsVal_dir, val_segs, file_ending)
        move_files(base_imagesTr_dir, target_imagesVal_dir, val_images, file_ending)
    else:
        copy_files(base_labelsTr_dir, target_labelsVal_dir, val_segs, file_ending)
        copy_files(base_imagesTr_dir, target_imagesVal_dir, val_images, file_ending)


def move_files(
    source_dir: Path, target_dir: Path, file_names: list[str], file_ending: str
):
    os.makedirs(target_dir, exist_ok=False)
    for filename in file_names:
        file_name = filename + file_ending
        shutil.move(source_dir / file_name, target_dir / file_name)


def copy_files(
    source_dir: Path, target_dir: Path, file_names: list[str], file_ending: str
):
    os.makedirs(target_dir, exist_ok=False)
    for filename in file_names:
        file_name = filename + file_ending
        shutil.copy(source_dir / file_name, target_dir / file_name)


@register_subcommand(
    "create_val_split",
    [
        (("-d", "--dataset_id"), {"type": int}),
        ("--test_size", {"default": 0.25, "type": float}),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    test_size = args.test_size
    # TODO: here the config would be useful!
    raw_folder = get_raw_path(dataset_id)
    file_ending = read_dataset_json(dataset_id)["file_ending"]
    imagesTr = raw_folder / "imagesTr"
    imagesVal = raw_folder / "imagesVal"
    labelsTr = raw_folder / "labelsTr"
    labelsVal = raw_folder / "labelsVal"
    # TODO: Create an ignore statement if data already has been split!
    create_test_datasets(
        labelsTr, imagesTr, labelsVal, imagesVal, file_ending, test_size=test_size
    )
