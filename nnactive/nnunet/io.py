import os
from typing import Optional
import json

from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.loops.cross_validation import kfold_cv_from_patches



def generate_custom_splits_file(
    dataset_id: int, loop_count: Optional[int] = None, num_folds: int = 5
):
    """Generates a custom split file in NNUNET_PREPROCESSED folder which only has labeled data.

    Args:
        dataset_id (int): dataset id
        loop_count (int): X for loop_XXX.json file all smaller are also taken to get list of names
        num_folds (int, optional): Folds vor KFoldCV. Defaults to 5.
    """


    patches = get_patches_from_loop_files(get_raw_path(dataset_id), loop_count)

    splits_final = kfold_cv_from_patches(num_folds, patches)

    

    # Create path if not exists
    prepocessed_path  =get_preprocessed_path(dataset_id)
    if not (prepocessed_path).exists():
        os.makedirs(prepocessed_path)
    # save splits_file
    with open(prepocessed_path / "splits_final.json", "w") as file:
        json.dump(splits_final, file, indent=4)