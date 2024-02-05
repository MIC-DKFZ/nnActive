import json
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger

from nnactive.data.annotate import create_labels_from_patches
from nnactive.loops.cross_validation import kfold_cv_from_patches
from nnactive.loops.loading import get_loop_patches, get_patches_from_loop_files


def update_data(
    data_path: Path,
    save_splits_file: Path,
    ignore_label: int,
    file_ending: str,
    base_dir: Path,
    target_dir: Path,
    loop_val: int = None,
    num_folds: int = 5,
    annotated: bool = True,
    additional_label_path: Path | None = None,
):
    """Update Dataset Raw with a novel splits_file in Preprocessed

    Disclaimer:
        Labels from additional_label_path (if used) are added last and overwrite GT from labelsTr.
        All areas inside images in additional_label_path that are not 255 = -1 will be written to labelsTr.

    Args:
        data_path (Path): raw dataset dir
        save_splits_file (Path): path to save splits file
        ignore_label (int): ignore label
        file_ending (str): desc
        base_dir (Path): path to annotated labels
        target_dir (Path): path to labels to be updated
        loop_val (int, optional): which loop val to use. Defaults to None.
        additional_label_path (Path, optional): path to files with labels to be added to labelsTr. Defaults to None
    """
    all_patches = get_patches_from_loop_files(data_path, loop_val)
    if annotated:
        patches = all_patches
    else:
        patches = get_loop_patches(data_path, loop_val)
    logger.info(
        "Updating Data for loop {} with {} patches".format(
            "max" if loop_val is None else loop_val, len(patches)
        )
    )
    create_labels_from_patches(
        patches,
        ignore_label,
        file_ending,
        base_dir,
        target_dir,
        overwrite=annotated,
        additional_label_path=additional_label_path,
    )
    logger.info(
        "Creating splits file {} with {} patches".format(
            save_splits_file, len(all_patches)
        )
    )
    splits_final = kfold_cv_from_patches(num_folds, all_patches)

    with open(save_splits_file, "w") as file:
        json.dump(splits_final, file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--dataset_path")
    parser.add_argument("-i", "--input_data")
    parser.add_argument("-l", "--loop", default=None, type=int)
    parser.add_argument("--save_splits_file")

    args = parser.parse_args()
    labeled_path = Path(args.input_data)
    data_path = Path(args.dataset_path)
    save_splits_file = Path(args.save_splits_file)
    loop_val = args.loop

    with open(data_path / "dataset.json", "r") as file:
        data_json = json.load(file)
    ignore_label: int = data_json["labels"]["ignore"]
    file_ending = data_json["file_ending"]

    base_dir = labeled_path / "labelsTr"
    target_dir = data_path / "labelsTr"

    update_data(
        data_path,
        save_splits_file,
        loop_val,
        ignore_label,
        file_ending,
        base_dir,
        target_dir,
    )
