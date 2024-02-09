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


def create_test_datasets(
    base_labelsTr_dir: Path,
    base_imagesTr_dir: Path,
    target_labelsVal_dir: Path,
    target_imagesVal_dir: Path,
    file_ending: str,
    test_size: Union[int, float] = 0.25,
    move: bool = True,
    level_seperator: None | str = None,
) -> tuple[int, int]:
    seg_names = os.listdir(base_labelsTr_dir)
    seg_names = [seg_name for seg_name in seg_names if seg_name.endswith(file_ending)]

    def _clean_file_ending(file_names: list[str]):
        file_names = [seg_name[: -len(file_ending)] for seg_name in file_names]
        return file_names

    seg_names = _clean_file_ending(seg_names)
    rng = np.random.default_rng(random_seed)

    if level_seperator is not None:
        levels = [seg_name.split(level_seperator) for seg_name in seg_names]
        num_levels = len(levels[0])
        logger.info(
            f"Creating validation split using {num_levels} levels sepearted by {level_seperator}"
        )
        logger.info(
            f"For the split the first dimension is used and the second ignored. e.g. {levels[0]}"
        )
        for l in levels:
            if len(l) != num_levels:
                raise RuntimeError(
                    f"Number of levels in Dataset seperated by level_separator ({level_seperator})is not conistently {num_levels}."
                )
        if num_levels > 2:
            raise NotImplementedError(
                "More than 2 levels of e.g. patient and frame are currently not supported."
            )
        split_names = [l[0] for l in levels]
        split_names: list[str] = np.unique(split_names).tolist()
        rng.shuffle(split_names)
        if test_size < 1:
            test_size = test_size * len(split_names)
        test_size = int(test_size)
        val_split = split_names[:test_size]

        def _return_true_if_string_startswith_list_set(
            string: str, list_set: list[str]
        ) -> bool:
            for list_string in list_set:
                if string.startswith(list_string):
                    return True
            return False

        val_segs = [
            seg_name
            for seg_name in seg_names
            if _return_true_if_string_startswith_list_set(seg_name, val_split)
        ]

    else:
        rng.shuffle(seg_names)
        if test_size < 1:
            test_size = test_size * len(seg_names)
        test_size = int(test_size)

        val_segs = seg_names[:test_size]

    image_names = os.listdir(base_imagesTr_dir)
    image_names = [
        image_name for image_name in image_names if image_name.endswith(file_ending)
    ]
    image_names = _clean_file_ending(image_names)

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
    logger.info(
        f"Moving {len(val_segs)} out {len(seg_names)} Label Maps to Validation Data"
    )
    logger.info(
        f"Moving images from folder {base_imagesTr_dir} to {target_imagesVal_dir}"
    )
    logger.info(
        f"Moving labels from folder {base_labelsTr_dir} to {target_labelsVal_dir}"
    )

    if move:
        move_files(base_labelsTr_dir, target_labelsVal_dir, val_segs, file_ending)
        move_files(base_imagesTr_dir, target_imagesVal_dir, val_images, file_ending)
    else:
        copy_files(base_labelsTr_dir, target_labelsVal_dir, val_segs, file_ending)
        copy_files(base_imagesTr_dir, target_imagesVal_dir, val_images, file_ending)

    return len(seg_names) - len(val_segs), len(val_segs)


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
        (
            (
                "--level_seperator",
                {
                    "default": None,
                    "type": str,
                    "help": "Sperator by which multiple images coming from the same subgroup can be identified to have no overlap in the split."
                    "E.g. 'patient1_img2' with seperator '_' will be split according to patientX while imgX the images are added according to splits.",
                },
            )
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    test_size = args.test_size
    level_seperator = args.level_seperator
    # TODO: here the config would be useful!
    # TODO: save the split to a split file which can be loaded. (avoid unnecessary seed problems)
    raw_folder = get_raw_path(dataset_id)
    dataset_json = read_dataset_json(dataset_id)

    file_ending = dataset_json["file_ending"]
    imagesTr = raw_folder / "imagesTr"
    imagesVal = raw_folder / "imagesVal"
    labelsTr = raw_folder / "labelsTr"
    labelsVal = raw_folder / "labelsVal"
    if imagesVal.exists() or labelsVal.exists():
        raise RuntimeError(
            f"It seems as if the splits have already been created. Check:\n{labelsTr} \n{labelsVal} "
        )
    num_train, num_val = create_test_datasets(
        labelsTr,
        imagesTr,
        labelsVal,
        imagesVal,
        file_ending,
        test_size=test_size,
        level_seperator=level_seperator,
    )
    dataset_json["numTraining"] = num_train
    dataset_json["numVal"] = num_val

    save_json(dataset_json, raw_folder / "dataset.json")
