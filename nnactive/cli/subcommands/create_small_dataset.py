import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.utils.io import save_json

random_seed = 12345


def copy_percentage(
    base_images_dir: Path,
    base_labels_dir: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
    file_ending: str,
    retain_size: Union[int, float] = 0.25,
) -> int:
    seg_names = os.listdir(base_labels_dir)
    seg_names = [seg_name for seg_name in seg_names if seg_name.endswith(file_ending)]

    def _clean_file_ending(file_names: list[str]):
        file_names = [seg_name[: -len(file_ending)] for seg_name in file_names]
        return file_names

    seg_names = _clean_file_ending(seg_names)

    rng = np.random.default_rng(random_seed)
    rng.shuffle(seg_names)
    if retain_size < 1:
        retain_size = retain_size * len(seg_names)
    retain_size = int(retain_size)

    copy_segs = [seg_names.pop() for _ in range(retain_size)]

    image_names = os.listdir(base_images_dir)
    image_names = [
        image_name for image_name in image_names if image_name.endswith(file_ending)
    ]
    image_names = _clean_file_ending(image_names)

    def _return_true_if_file_in_list_set(string: str, list_set: list[str]) -> bool:
        for list_string in list_set:
            if "_".join(string.split("_")[:-1]) == list_string:
                return True
        return False

    copy_images = [
        image_name
        for image_name in image_names
        if _return_true_if_file_in_list_set(image_name, copy_segs)
    ]

    logger.info(
        f"Writing {len(copy_segs)} labels from folder {base_labels_dir} to folder {target_labels_dir}"
    )
    if not target_labels_dir.is_dir():
        logger.info(f"Creating folder {target_labels_dir}")
        # os.makedirs(target_labels_dir)
    else:
        raise RuntimeError(f"Target label folder already exists {target_labels_dir}")
    copy_files(base_labels_dir, target_labels_dir, copy_segs, file_ending)

    logger.info(
        f"Writing {len(copy_images)} images from folder {base_images_dir} to folder {target_images_dir}"
    )
    if not target_images_dir.is_dir():
        logger.info(f"Creating folder {target_images_dir}")
    else:
        raise RuntimeError(f"Target image folder already exists {target_images_dir}")
    copy_files(base_images_dir, target_images_dir, copy_images, file_ending)

    return retain_size


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
    "create_small_dataset",
    [
        (("-bd", "--base_dataset_id"), {"type": int}),
        (("-td", "--target_dataset_id"), {"type": int}),
        ("--relative_size", {"default": 0.2, "type": float}),
    ],
)
def main(args: Namespace) -> None:
    base_dataset_id = args.base_dataset_id
    target_dataset_id = args.target_dataset_id
    relative_size = args.relative_size

    dataset_json = read_dataset_json(base_dataset_id)

    base_raw_folder = get_raw_path(base_dataset_id)

    file_ending = dataset_json["file_ending"]
    name = dataset_json["name"]
    target_raw_folder = (
        base_raw_folder.parent
    ) / f"Dataset{target_dataset_id:03d}_{name}_small"
    if target_raw_folder.is_dir():
        raise RuntimeError(f"Target raw folder already exists: {target_raw_folder}")
    else:
        logger.info(f"Creating folder {target_raw_folder}")
        os.makedirs(target_raw_folder)

    base_images = base_raw_folder / "imagesTr"
    base_labels = base_raw_folder / "labelsTr"
    target_images = target_raw_folder / "imagesTr"
    target_labels = target_raw_folder / "labelsTr"

    num_train = copy_percentage(
        base_images,
        base_labels,
        target_images,
        target_labels,
        file_ending=file_ending,
        retain_size=relative_size,
    )

    base_images = base_raw_folder / "imagesVal"
    base_labels = base_raw_folder / "labelsVal"
    target_images = target_raw_folder / "imagesVal"
    target_labels = target_raw_folder / "labelsVal"

    num_val = copy_percentage(
        base_images,
        base_labels,
        target_images,
        target_labels,
        file_ending=file_ending,
        retain_size=relative_size,
    )

    dataset_json["numTraining"] = num_train
    dataset_json["numVal"] = num_val

    save_json(dataset_json, target_raw_folder / "dataset.json")
