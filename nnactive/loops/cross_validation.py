import numpy as np

from nnactive.data import Patch


def kfold_cv(
    k: int, labeled_images: list[str], random_seed: int = 12345
) -> list[dict[str, list[str]]]:
    """Create K Fold CV splits

    Args:
        k (int): num_folds
        labeled_images (list[str]): list of named images without suffix
        random_seed (int, optional): _description_. Defaults to 12345.

    Returns:
        dict={train:list, val:list}
    """
    folds = [[] for _ in range(k)]
    rand_np_state = np.random.RandomState(random_seed)
    rand_np_state.shuffle(labeled_images)
    for i in range(len(labeled_images)):
        folds[i % k].append(labeled_images.pop())

    for fold in folds:
        assert (
            len(fold) > 0
        )  # no fold is supposed to have a length of zero! set num_folds smaller

    splits_final = []
    for i in range(k):
        train_select = [j for j in range(k) if j != i]
        val_select = [i]
        train_set = []
        for train_fold in train_select:
            train_set = train_set + folds[train_fold]
        val_set = []
        for val_fold in val_select:
            val_set = val_set + folds[val_fold]
        splits_final.append(
            {
                "train": train_set,
                "val": val_set,
            }
        )
    return splits_final


def kfold_cv_from_patches(
    num_folds: int, patches: list[Patch], random_seed: int = 12345
) -> list[dict[str, list[str]]]:
    """Create K Fold CV splits with patches as inputs

    Args:
        num_folds (int): num_folds
        patches (list[Patch]): list of named images without suffix
        random_seed (int, optional): _description_. Defaults to 12345.

    Returns:
        dict={train:list, val:list}
    """
    labeled_images = [patch.file.split(".")[0] for patch in patches]
    labeled_images = list(set(labeled_images))

    splits_final = kfold_cv(num_folds, labeled_images, random_seed)
    return splits_final
