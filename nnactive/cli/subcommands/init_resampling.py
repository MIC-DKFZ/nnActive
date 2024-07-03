import json
import shutil
from pathlib import Path

import nnunetv2.paths

from nnactive.cli.registry import register_subcommand
from nnactive.data.resampling import resample_dataset
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.paths import set_raw_paths
from nnactive.utils.io import load_json, save_json


def get_target_spacing(path: Path) -> tuple[int]:
    with open(path / "nnUNetPlans.json", "r") as file:
        data = json.load(file)
    target_spacing: tuple[int] = data["3d_fullres"]["spacing"]
    return target_spacing


@register_subcommand("init_resample_from_folder")
def init_resample_from_folder(
    target_preprocessed: str,
    target_raw: str,
    num_processes: int = 4,
    configuration: str = "3d_fullres",
) -> None:
    target_preprocessed = Path(target_preprocessed)
    target_raw = Path(target_raw)
    n_workers = num_processes
    configuration = configuration

    dataset_json = load_json(target_raw / "dataset.json")

    resample_dataset(
        dataset_cfg=dataset_json,
        rs_img_path=target_raw / "imagesTr_original",
        rs_gt_path=target_raw / "labelsTr_original",
        img_path=target_raw / "imagesTr",
        gt_path=target_raw / "labelsTr",
        preprocessed_path=target_preprocessed,
        n_workers=n_workers,
        configuration=configuration,
    )

    resample_dataset(
        dataset_cfg=dataset_json,
        rs_img_path=target_raw / "imagesVal_original",
        rs_gt_path=target_raw / "labelsVal_original",
        img_path=target_raw / "imagesVal",
        gt_path=target_raw / "labelsVal",
        preprocessed_path=target_preprocessed,
        n_workers=n_workers,
        configuration=configuration,
    )

    nnUNet_plans = load_json(target_preprocessed / "nnUNetPlans.json")
    if any(nnUNet_plans["configurations"][configuration]["use_mask_for_norm"]):
        dataset_json["use_mask_for_norm"] = True

    save_json(dataset_json, target_raw / "dataset.json")


@register_subcommand("init_resample_from_id")
def resample_nnunet_dataset(
    dataset_id: int, workers: int, configuration: str = "3d_fullres"
):
    """Resample dataset in nnUNet_raw folder, original images and labels are saved in {name}_original.
    Requires nnUNetPlans in nnActive_raw/nnUNet_preprocessed/Dataset{id}...
    ids are set according to nnActive_raw/nnUNet_raw/Dataset{id}...

    Args:
        dataset_id (int): Dataset id
        workers (int): number of workers used
        configuration (str): configuration for training
    """
    with set_raw_paths():
        dataset_json = read_dataset_json(dataset_id)
        raw_path = get_raw_path(dataset_id)
        preprocessed_path = get_preprocessed_path(dataset_id)
        rs_img_path = raw_path / "imagesTr_original"
        rs_gt_path = raw_path / "labelsTr_original"
        img_path = raw_path / "imagesTr"
        gt_path = raw_path / "labelsTr"
        print("Resampling training images.")
        print(f"Moving images from: {img_path} \n To: {rs_img_path}")
        print(f"Moving labels from: {gt_path} \n To: {rs_gt_path}")
        if rs_img_path.is_dir():
            print("Training images for this dataset are already preprocessed.")
        else:
            print("Starting resampling.")
            shutil.move(img_path, rs_img_path)
            shutil.move(gt_path, rs_gt_path)
            resample_dataset(
                dataset_cfg=dataset_json,
                rs_img_path=rs_img_path,
                rs_gt_path=rs_gt_path,
                img_path=img_path,
                gt_path=gt_path,
                preprocessed_path=preprocessed_path,
                n_workers=workers,
                configuration=configuration,
            )
            nnUNet_plans = load_json(preprocessed_path / "nnUNetPlans.json")
            if any(nnUNet_plans["configurations"][configuration]["use_mask_for_norm"]):
                dataset_json["use_mask_for_norm"] = True

            save_json(dataset_json, raw_path / "dataset.json")

        rs_img_path = raw_path / "imagesVal_original"
        rs_gt_path = raw_path / "labelsVal_original"
        img_path = raw_path / "imagesVal"
        gt_path = raw_path / "labelsVal"

        print("Resampling validation images.")
        print(f"Moving images from: {img_path} \n To: {rs_img_path}")
        print(f"Moving labels from: {gt_path} \n To: {rs_gt_path}")
        if rs_img_path.is_dir():
            print("Validation images for this dataset are already preprocessed.")
        else:
            print("Starting resampling.")
            shutil.move(img_path, rs_img_path)
            shutil.move(gt_path, rs_gt_path)
            resample_dataset(
                dataset_cfg=dataset_json,
                rs_img_path=rs_img_path,
                rs_gt_path=rs_gt_path,
                img_path=img_path,
                gt_path=gt_path,
                preprocessed_path=preprocessed_path,
                n_workers=workers,
                configuration=configuration,
            )
