import json
from argparse import Namespace
from pathlib import Path

from nnactive.cli.registry import register_subcommand
from nnactive.data.resampling import resample_dataset
from nnactive.utils.io import load_json, save_json


def get_target_spacing(path: Path) -> tuple[int]:
    with open(path / "nnUNetPlans.json", "r") as file:
        data = json.load(file)
    target_spacing: tuple[int] = data["3d_fullres"]["spacing"]
    return target_spacing


@register_subcommand(
    "resample",
    [
        ("--target_preprocessed", {"type": str}),
        ("--target_raw", {"type": str}),
        ("--num_processes", {"type": int, "default": 4}),
        ("--configuration", {"type": str, "default": "3d_fullres"}),
    ],
)
def main(args: Namespace) -> None:
    target_preprocessed = Path(args.target_preprocessed)
    target_raw = Path(args.target_raw)
    n_workers = args.num_processes
    configuration: str = args.configuration

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
