import json
from argparse import ArgumentParser
from pathlib import Path

# from nnactive.data.resampling_fabian import resample_save
from nnactive.data.resampling import resample_dataset


def get_target_spacing(path: Path) -> tuple[int]:
    with open(path / "nnUNetPlans.json", "r") as file:
        data = json.load(file)
    target_spacing = data["3d_fullres"]["spacing"]
    return target_spacing


def main():
    parser = ArgumentParser()
    parser.add_argument("--target_preprocessed", type=str)
    parser.add_argument("--target_raw", type=str)
    # parser.add_argument("--target_add", type=str)

    args = parser.parse_args()
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
        n_workers=8,
    )


if __name__ == "__main__":
    main()
