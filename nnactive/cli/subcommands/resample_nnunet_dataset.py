import shutil
from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.data.resampling import resample_dataset
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.utils.io import load_json, save_json


@register_subcommand(
    "resample_nnunet_dataset",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (("-np", "--num-processes"), {"type": int, "default": 4}),
        (("--configuration"), {"type": str, "default": "3d_fullres"}),
    ],
)
def main(args: Namespace) -> None:
    workers: int = args.num_processes
    dataset_id: int = args.dataset_id
    configuration: str = args.configuration
    resample_nnunet_dataset(dataset_id, workers, configuration)


def resample_nnunet_dataset(dataset_id: int, workers: int, configuration: str):
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
