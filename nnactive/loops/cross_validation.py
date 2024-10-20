from copy import deepcopy
from pathlib import Path

import numpy as np
from loguru import logger

from nnactive.data import Patch
from nnactive.utils.io import load_label_map
from nnactive.utils.pyutils import invert_dict


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

    label_dict_use = deepcopy(label_dict)

    if label_dict_use is not None:
        # in this case we want to make sure that each label is present in at least two folds and folds are filled equally
        # this functionality is test in manual_test/label_in_split_check.py
        while len(label_dict_use) > 0:
            label_use = [(key, len(v)) for key, v in label_dict_use.items()]
            label_use.sort(key=lambda x: x[1])

            label = label_use[0][0]
            images = label_dict_use.pop(label)
            if len(images) < 2 and label != -1:
                logger.warning(
                    f"Label {label} has less than 2 images. Cannot ensure that all train folds contain all classes."
                )
                # this break is necessary so that immediately folds are selected randomly after that.
                # the verification step then ensures that all classes are present in sufficient folds.
                break
            if label == -1:
                continue
            rand_np_state.shuffle(images)

            # assure that folds are equally filled with data for low label regime.
            fold_lens = np.array([len(fold) for fold in folds])
            # folds can only be filled with value max or max-1
            # folds that have value of max-1 are valid
            fold_valid = fold_lens < fold_lens.max()
            if fold_valid.sum() == 0:
                # if all folds have value of max, then all folds are valid
                fold_choice = np.arange(k)
                folds_for_class = rand_np_state.choice(fold_choice, 2, replace=False)
            elif fold_valid.sum() == 1:
                # if only one fold is left with val max-1, then select it by default and draw rest random
                fold_choice = np.arange(k)[~fold_valid]
                folds_for_class = np.concatenate(
                    [
                        np.arange(k)[fold_valid],
                        rand_np_state.choice(fold_choice, 1, replace=False),
                    ]
                )
            else:
                # if more than 1 fold is valid then draw randomly
                fold_choice = np.arange(k)[fold_valid]
                folds_for_class = rand_np_state.choice(fold_choice, 2, replace=False)

            for fold_for_class in folds_for_class:
                fold_class_image = images.pop()
                labeled_images.remove(fold_class_image)
                for label in label_dict_use:
                    if fold_class_image in label_dict_use[label]:
                        label_dict_use[label].remove(fold_class_image)

                folds[fold_for_class].append(fold_class_image)

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
    verify: bool = True,
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
        label_dict: dict[int, list[str]] = {}
        for labeled_image in labeled_images:
            label_map = load_label_map(labeled_image, labels_path, file_ending)
            unique_labels = set(np.unique(label_map))
            logger.info(f"Unique labels for image {labeled_image}: {unique_labels}")
            for label in unique_labels:
                if label not in ensure_classes:
                    continue
                if label not in label_dict:
                    label_dict[label] = [labeled_image]
                else:
                    label_dict[label].append(labeled_image)

        if verify:
            logger.debug(label_dict)

        assert sorted(ensure_classes) == sorted(
            list(label_dict.keys())
        ), "Not all classes are in loaded label maps"

    else:
        label_dict = None

    splits_final = kfold_cv(
        num_folds, labeled_images, random_seed, label_dict=label_dict
    )

    if label_dict is not None and verify:
        image_label_dict = invert_dict(label_dict)

        # counts in many folds each label appears.
        occurences = {key: 0 for key in label_dict}

        for split in splits_final:
            fold = split["val"]
            labels_fold = []
            for file in fold:
                labels_fold += image_label_dict[file]
            labels_fold: list = np.unique(labels_fold).tolist()
            for l in labels_fold:
                occurences[l] += 1

        for l in occurences:
            logger.debug(f"Label {l} occurs in {occurences[l]} folds")
            if occurences[l] < 2:
                raise RuntimeError(f"Label {l} does occur in less than two folds.")

    return splits_final


def get_mean_foreground_cv(summary_cross_val_dict, n_folds):
    """
    Get the mean over the foreground means across folds.
    Each fold has entry "foreground_mean" representing the mean over all foreground classes across images.
    Args:
        summary_cross_val_dict: Dictionary with the individual metrics per fold
        n_folds: number of folds

    Returns:
        Dict: mean dict containing the mean foreground metrics across fold
    """
    all_foreground_mean = []
    for fold in range(n_folds):
        all_foreground_mean.append(
            summary_cross_val_dict[f"fold_{fold}"]["foreground_mean"]
        )
    # Iterate over each metric (e.g. Dice, FN, FP, ...) and take the mean
    mean_dict = {}
    for key in all_foreground_mean[0].keys():
        mean_dict[key] = np.array([d[key] for d in all_foreground_mean]).mean()
    return mean_dict


def get_mean_cv(summary_cross_val_dict, n_folds):
    """
    Get the mean of the individual classes across folds.
    Each fold has entry "mean" representing the mean over the individual classes across images.
    Structure "mean": {'1': Dice: .., FN ..., ... '2': Dice:..., ...}
    Args:
        summary_cross_val_dict: Dictionary with the individual metrics per fold
        n_folds: number of folds

    Returns:
        Dict: per class dict containing the mean metrics per class across folds
    """
    mean_dicts_list = []
    for fold in range(n_folds):
        mean_dicts_list.append(summary_cross_val_dict[f"fold_{fold}"]["mean"])
    class_dicts = {}

    # First iterate over class indices
    for class_idx in mean_dicts_list[0].keys():
        class_dicts[class_idx] = {}
        # Iterate over each metric (e.g. Dice, FN, FP, ...) and take the mean for each class
        for key in mean_dicts_list[0][class_idx].keys():
            class_dicts[class_idx][key] = np.array(
                [d[class_idx][key] for d in mean_dicts_list]
            ).mean()
    return class_dicts


def read_classes_from_dataset_json(dataset_json: dict) -> list[int]:
    dataset_classes = dataset_json["labels"]
    for label in dataset_classes:
        if isinstance(dataset_classes[label], (list, tuple)):
            dataset_classes[label] = dataset_classes[label][0]
    ensure_classes = [val for key, val in dataset_classes.items() if key != "ignore"]
    return ensure_classes
