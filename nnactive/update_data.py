import json
from argparse import ArgumentParser
from pathlib import Path

from nnactive.data.annotate import create_labels_from_patches
from nnactive.loops.cross_validation import kfold_cv_from_patches
from nnactive.loops.loading import get_patches_from_loop_files

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

    patches = get_patches_from_loop_files(data_path, loop_val)

    with open(data_path / "dataset.json", "r") as file:
        data_json = json.load(file)
    ignore_label: int = data_json["labels"]["ignore"]
    file_ending = data_json["file_ending"]

    base_dir = labeled_path / "labelsTr"
    target_dir = data_path / "labelsTr"

    create_labels_from_patches(patches, ignore_label, file_ending, base_dir, target_dir)

    splits_final = kfold_cv_from_patches(5, patches)

    with open(save_splits_file, "w") as file:
        json.dump(splits_final, file, indent=4)
