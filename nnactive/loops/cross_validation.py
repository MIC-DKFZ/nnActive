from pathlib import Path

import numpy as np
from medpy.io import load

from nnactive.data import Patch


def kfold_cv(
    k: int, labeled_images: list[str], random_seed: int = 12345, label_dict: dict = None
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

    if label_dict is not None:
        # in this case we want to make sure that each label is present in at least two folds
        for label, images in label_dict.items():
            if len(images) < 2 and label != -1:
                raise ValueError(
                    f"Label {label} has less than 2 images. Cannot ensure that all train folds contain all classes."
                )
            if label == -1:
                continue
            rand_np_state.shuffle(images)
            # We need to add label here because otherwise the random seed would be the same for all classes
            rng = np.random.default_rng(seed=random_seed + label)
            folds_for_class = rng.choice(k, 2, replace=False)
            for fold_for_class in folds_for_class:
                folds[fold_for_class].append(images.pop())
    rand_np_state.shuffle(labeled_images)

    i = 0
    while len(labeled_images) > 0:
        min_len_fold = np.min([len(fold) for fold in folds])
        max_len_fold = np.max([len(fold) for fold in folds])
        if len(folds[i % k]) < max_len_fold or max_len_fold == min_len_fold:
            folds[i % k].append(labeled_images.pop())
        i += 1

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
    num_folds: int,
    patches: list[Patch],
    random_seed: int = 12345,
    ensure_classes: list[int] = None,
    labels_path: Path = None,
    file_ending: str = None,
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

    if (
        ensure_classes is not None
        and labels_path is not None
        and file_ending is not None
    ):
        label_dict = {}
        for labeled_image in labeled_images:
            label_path = labels_path / f"{labeled_image}{file_ending}"
            label_map, _ = load(str(label_path))
            unique_labels = set(np.unique(label_map))
            for label in unique_labels:
                if label not in ensure_classes:
                    continue
                if label not in label_dict:
                    label_dict[label] = [labeled_image]
                else:
                    label_dict[label].append(labeled_image)

        assert sorted(ensure_classes) == sorted(
            list(label_dict.keys())
        ), "Not all classes are in loaded label maps"

    else:
        label_dict = None

    splits_final = kfold_cv(
        num_folds, labeled_images, random_seed, label_dict=label_dict
    )
    return splits_final
