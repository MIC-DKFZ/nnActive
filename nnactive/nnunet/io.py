import json
import os
from typing import Optional

from nnactive.loops.cross_validation import kfold_cv_from_patches
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json


def generate_custom_splits_file(
    dataset_id: int,
    loop_count: Optional[int] = None,
    num_folds: int = 5,
    ensure_all_classes: bool = True,
    verify: bool = False,
):
    """Generates a custom split file in NNUNET_PREPROCESSED folder which only has labeled data.

    Args:
        dataset_id (int): dataset id
        loop_count (int): X for loop_XXX.json file all smaller are also taken to get list of names
        num_folds (int, optional): Folds vor KFoldCV. Defaults to 5.
    """

    patches = get_patches_from_loop_files(get_raw_path(dataset_id), loop_count)

    if ensure_all_classes:
        raw_path = get_raw_path(dataset_id)
        labels_path = raw_path / "labelsTr"
        dataset_json = read_dataset_json(dataset_id)
        file_ending = dataset_json["file_ending"]
        dataset_classes = dataset_json["labels"]
        for label in dataset_classes:
            if isinstance(dataset_classes[label], (list, tuple)):
                dataset_classes[label] = dataset_classes[label][0]
        ensure_classes = [
            val for key, val in dataset_classes.items() if key != "ignore"
        ]
    else:
        ensure_classes = None
        labels_path = None
        file_ending = None

    splits_final = kfold_cv_from_patches(
        num_folds,
        patches,
        ensure_classes=ensure_classes,
        labels_path=labels_path,
        file_ending=file_ending,
        verify=verify,
    )

    # Create path if not exists
    prepocessed_path = get_preprocessed_path(dataset_id)
    if not (prepocessed_path).exists():
        os.makedirs(prepocessed_path)
    # save splits_file
    with open(prepocessed_path / "splits_final.json", "w") as file:
        json.dump(splits_final, file, indent=4)
