import os
import shutil
from pathlib import Path
from typing import Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from loguru import logger
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
    DatasetFingerprintExtractor,
)
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.cli.registry import register_subcommand
from nnactive.nnunet.fingerprint_extractor import NNActiveDatasetFingerprintExtractor
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.paths import set_raw_paths
from nnactive.utils.io import save_json

RANDOM_SEED = 12345


def extract_fingerprint_dataset(
    dataset_id: int,
    fingerprint_extractor_class: Type[
        DatasetFingerprintExtractor
    ] = NNActiveDatasetFingerprintExtractor,
    num_processes: int = default_num_processes,
    check_dataset_integrity: bool = False,
    clean: bool = True,
    verbose: bool = True,
):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    with set_raw_paths():
        dataset_name = convert_id_to_dataset_name(dataset_id)
        print(dataset_name)

        if check_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

        fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)

        return fpe.run(overwrite_existing=clean)


@register_subcommand("init_nnunet_extract_fingerprint")
def init_nnunet_extract_fingerprint(
    dataset_id: int,
    np: int = default_num_processes,
    verify_dataset_integrity: bool = False,
    clean: bool = True,
    verbose: bool = False,
):
    """nnActive wrapper around the nnUNetv2_extract_fingerprint functionality.
    Use this fingerprint extractor after resampling of the original dataset

    Use this as fingerprint extractor to prepare datastes s.a. BraTS where large areas outside of the brain are free annotations.
    Generally also advised to use inside of nnActive_raw/... folder

    Uses Fingerprint Extractor with all functionality from nnU-Net. It saves out data to folder addTr if `use_mask_for_norm` is true in dataset.json.
    dataset.json gets rewritten with `use_mask_for_nrom` value in convert_to_partannotated if plans would use it.

    Args:
        dataset_id (int): dataset id
        np (int, optional): Number of processes used for fingerprint extraction. Defaults to default_num_processes.
        verify_dataset_integrity (bool, optional): set this flag to check the dataset integrity. This is useful and should be done once for "
                "each dataset!. Defaults to False.
        clean (bool, optional): Set this flag to overwrite existing fingerprints. If not set and a fingerprint exists, the extractor won't run. Defaults to True.
        verbose (bool, optional): Set this to print a lot of stuff. Useful for debugging. Disables the progress bar! Recommended for clusters. Defaults to False.
    """

    extract_fingerprint_dataset(
        dataset_id,
        fingerprint_extractor_class=NNActiveDatasetFingerprintExtractor,
        num_processes=np,
        check_dataset_integrity=verify_dataset_integrity,
        clean=clean,
        verbose=verbose,
    )


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
    rng = np.random.default_rng(RANDOM_SEED)

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

    rng = np.random.default_rng(RANDOM_SEED)
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


@register_subcommand("init_create_small_dataset")
def init_create_small_dataset(
    base_dataset_id: int, target_dataset_id: int, relative_size: float = 0.2
) -> None:
    """Create small derivative dataset from large dataset.
    ids are set according to nnActive_raw/nnUNet_raw/Dataset{id}...

    Args:
        base_dataset_id (int): dataset id from which derivative is supposed to be created
        target_dataset_id (int): dataset id for derivative
        relative_size (float, optional): Relative Size derivative to base dataset. Defaults to 0.2.
    """
    with set_raw_paths():
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


@register_subcommand("init_create_val_split")
def init_create_val_split(
    dataset_id: int, test_size: float = 0.25, level_seperator: str | None = None
) -> None:
    """Create train val split for nnActive Training.
    ids are set according to nnActive_raw/nnUNet_raw/Dataset{id}...

    Args:
        dataset_id (int): dataset id in which val split is created
        test_size (float, optional): Relative size of test set to whole dataset. Defaults to 0.25.
        level_seperator (str | None, optional): Sperator by which multiple images coming from the same subgroup can be identified to have no overlap in the split.
                    E.g. 'patient1_img2' with seperator '_' will be split according to patientX while imgX the images are added according to splits.". Defaults to None.
    """
    with set_raw_paths():
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
