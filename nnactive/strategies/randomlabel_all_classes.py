import random
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from medpy.io import load
from nnunetv2.utilities.dataset_name_id_conversion import convert_dataset_name_to_id

from nnactive.data import Patch
from nnactive.masking import does_overlap
from nnactive.nnunet.utils import read_dataset_json
from nnactive.strategies.random import Random
from nnactive.strategies.randomlabel import RandomLabel, _obtain_random_patch_from_locs


def query_starting_budget_all_classes(
    raw_labels_path: Path,
    file_ending: str,
    annotated_id: int,
    annotated_patches,
    patch_size,
    rng,
    trials_per_img,
    verbose: bool = False,
):
    dataset_json = read_dataset_json(annotated_id)
    label_dict_dataset_json = dataset_json["labels"]
    label_dict_files = {k: [] for k in label_dict_dataset_json.keys() if k != "ignore"}
    for label_path in raw_labels_path.glob(f"**/*{file_ending}"):
        img, _ = load(str(label_path))
        img_labels = set(np.unique(img))
        for label, files in label_dict_files.items():
            if label_dict_dataset_json[label] in img_labels:
                files.append(label_path)
    labeled_patches = annotated_patches
    patches = []
    for label in label_dict_files.keys():
        if len(label_dict_files[label]) < 2:
            raise RuntimeError(
                f'Label "{label}" does have less than two files. '
                f"This is not enough to ensure all classes are represented in all training folds."
            )
        else:
            samples = random.sample(label_dict_files[label], 2)
        label_per_class_counter = 0
        for sample in samples:
            labeled = False
            num_tries = 0
            while not labeled:
                if verbose:
                    logger.debug(f"Loading Image: {sample.name}")
                label_map, _ = load(str(sample))
                img_size = label_map.shape
                current_patch_list = labeled_patches + patches
                selected_array = [
                    patch for patch in current_patch_list if patch.file == sample.name
                ]
                locs = np.argwhere(label_map == label_dict_dataset_json[label]).tolist()
                (
                    iter_patch_loc,
                    iter_patch_size,
                ) = _obtain_random_patch_from_locs(locs, img_size, patch_size, rng)
                patch = Patch(
                    file=sample.name,
                    coords=iter_patch_loc,
                    size=iter_patch_size,
                )
                # check if patch is valid
                if not does_overlap(patch, selected_array):
                    patches.append(patch)
                    # print(f"Creating Patch with iteration: {num_tries}")
                    labeled = True
                    label_per_class_counter += 1

                # if no new patch could fit inside of img do not consider again
                if num_tries == trials_per_img:
                    logger.info(f"Could not place patch in image {sample.name}")
                    logger.info(f"PatchCount {len(patches)}")
                    logger.info(f"{num_tries=}")
                    break
                num_tries += 1
        if label_per_class_counter < 2:
            raise RuntimeError(f'Could not place 2 patches for class "{label}"')
    # if verbose:
    #     logger.debug(patches)
    return patches


class RandomLabelAllClasses(RandomLabel):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        background_cls: int | None = None,
        additional_label_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            seed,
            trials_per_img,
            file_ending,
            raw_labels_path,
            background_cls,
            additional_label_path,
            **kwargs,
        )
        random.seed(seed)

    def query(self, verbose: bool = False, **kwargs) -> List[Patch]:
        # Do stuff to ensure all lables are represented two times
        annotated_id = convert_dataset_name_to_id(self.raw_labels_path.parent.name)
        selected_patches = query_starting_budget_all_classes(
            self.raw_labels_path,
            self.file_ending,
            annotated_id,
            annotated_patches=self.annotated_patches,
            patch_size=self.patch_size,
            rng=self.rng,
            trials_per_img=self.trials_per_img,
            verbose=verbose,
        )
        return super().query(verbose, selected_patches)


class RandomAllClasses(Random):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        additional_label_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            seed,
            trials_per_img,
            file_ending,
            raw_labels_path,
            additional_label_path,
            **kwargs,
        )
        random.seed(seed)

    def query(self, verbose: bool = False, **kwargs) -> List[Patch]:
        # Do stuff to ensure all lables are represented two times
        annotated_id = convert_dataset_name_to_id(self.raw_labels_path.parent.name)
        selected_patches = query_starting_budget_all_classes(
            self.raw_labels_path,
            self.file_ending,
            annotated_id,
            annotated_patches=self.annotated_patches,
            patch_size=self.patch_size,
            rng=self.rng,
            trials_per_img=self.trials_per_img,
            verbose=verbose,
        )
        return super().query(verbose, selected_patches)
