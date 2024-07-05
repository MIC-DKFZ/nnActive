import os

from loguru import logger
from nnunetv2.configuration import default_num_processes

from nnactive.cli.registry import register_subcommand
from nnactive.data.splits import copy_percentage, create_test_datasets
from nnactive.data.utils import extract_dataset_fingerprint
from nnactive.nnunet.fingerprint_extractor import NNActiveDatasetFingerprintExtractor
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.paths import set_raw_paths
from nnactive.utils.io import save_json


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
    extract_dataset_fingerprint(
        dataset_id,
        fingerprint_extractor_class=NNActiveDatasetFingerprintExtractor,
        num_processes=np,
        check_dataset_integrity=verify_dataset_integrity,
        clean=clean,
        verbose=verbose,
    )


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
