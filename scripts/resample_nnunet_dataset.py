from argparse import ArgumentParser

from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.resample_dataset import resample_dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id
    dataset_json = read_dataset_json(dataset_id)
    raw_path = get_raw_path(dataset_id)
    preprocessed_path = get_preprocessed_path(dataset_id)
    resample_dataset(
        dataset_cfg=dataset_json,
        rs_img_path=raw_path / "imagesTr_original",
        rs_gt_path=raw_path / "labelsTr_original",
        img_path=raw_path / "imagesTr",
        gt_path=raw_path / "labelsTr",
        preprocessed_path=preprocessed_path,
        n_workers=8,
    )


if __name__ == "__main__":
    main()
