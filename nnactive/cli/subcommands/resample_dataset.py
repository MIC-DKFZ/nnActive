import json
from argparse import Namespace
from pathlib import Path

from nnactive.cli.registry import register_subcommand
from nnactive.data.resampling import resample_dataset


def get_target_spacing(path: Path) -> tuple[int]:
    with open(path / "nnUNetPlans.json", "r") as file:
        data = json.load(file)
    target_spacing: tuple[int] = data["3d_fullres"]["spacing"]
    return target_spacing


@register_subcommand(
    "resample",
    [("--target_preprocessed", {"type": str}), ("--target_raw", {"type": str})],
)
def main(args: Namespace) -> None:
    target_preprocessed = Path(args.target_preprocessed)
    target_raw = Path(args.target_raw)

    with open(target_raw / "dataset.json", "r") as file:
        dataset_json = json.load(file)

    resample_dataset(
        dataset_cfg=dataset_json,
        rs_img_path=target_raw / "imagesTr_original",
        rs_gt_path=target_raw / "labelsTr_original",
        img_path=target_raw / "imagesTr",
        gt_path=target_raw / "labelsTr",
        preprocessed_path=target_preprocessed,
        n_workers=4,
    )

    resample_dataset(
        dataset_cfg=dataset_json,
        rs_img_path=target_raw / "imagesVal_original",
        rs_gt_path=target_raw / "labelsVal_original",
        img_path=target_raw / "imagesVal",
        gt_path=target_raw / "labelsVal",
        preprocessed_path=target_preprocessed,
        n_workers=4,
    )
