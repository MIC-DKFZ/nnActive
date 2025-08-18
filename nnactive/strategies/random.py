import random
from pathlib import Path
from typing import Generator, Iterable, List

import numpy as np
from loguru import logger

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.utils import (
    get_infinte_iter,
    obtain_random_patch_for_img,
    query_starting_budget_all_classes,
)
from nnactive.utils.io import load_label_map


@register_strategy("random")
class Random(AbstractQueryMethod):
    def __init__(
        self,
        config: ActiveConfig,
        dataset_id: int,
        loop_val: int,
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        additional_label_path: Path | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset_id=dataset_id,
            config=config,
            loop_val=loop_val,
            file_ending=file_ending,
            additional_label_path=additional_label_path,
            verbose=verbose,
            seed=seed,
        )
        self.trials_per_img = trials_per_img

        self.raw_labels_path = (
            raw_labels_path
            if raw_labels_path is not None
            else get_raw_path(dataset_id) / "labelsTr"
        )

        self.img_names = [
            path.name for path in self.raw_labels_path.glob(f"**/*{self.file_ending}")
        ]
        self.rng.shuffle(self.img_names)

    def query(
        self, verbose: bool = False, n_gpus: int = 0, already_annotated_patches=None
    ) -> list[Patch]:
        """
        Args:
            already_annotated_patches (list[Patch], optional): only used in combination with all_classes. Defaults to None.

        Returns:
            List[Patch]: patches for annotation
        """
        img_generator = get_infinte_iter(self.img_names)
        if already_annotated_patches is None:
            patches = []
        else:
            patches = already_annotated_patches
        logger.info("verbose", verbose)
        for _ in range(self.config.query_size - len(patches)):
            labeled = False
            while True:
                img_name = img_generator.__next__()
                if verbose:
                    logger.debug(f"Loading Image: {img_name}")
                label_map: np.ndarray = load_label_map(
                    img_name.replace(self.file_ending, ""),
                    self.raw_labels_path,
                    self.file_ending,
                )
                current_patch_list = self.annotated_patches + patches
                img_size = label_map.shape
                # only needed for creation of patches in first iteration
                selected_image_patches = [
                    patch for patch in current_patch_list if patch.file == img_name
                ]

                additional_label = None
                if self.additional_label_path is not None:
                    additional_label = load_label_map(
                        img_name.replace(self.file_ending, ""),
                        self.additional_label_path,
                        self.file_ending,
                    )
                    additional_label: np.ndarray = additional_label != 255

                num_tries = 0
                while True:
                    # propose a random patch
                    iter_patch_loc, iter_patch_size = obtain_random_patch_for_img(
                        img_size, self.config.patch_size, self.rng
                    )

                    patch = Patch(
                        file=img_name,
                        coords=iter_patch_loc,
                        size=iter_patch_size,
                    )

                    # check if patch is valid

                    if self.check_overlap(
                        patch, selected_image_patches, additional_label
                    ):
                        patches.append(patch)
                        logger.info(
                            f"Creating Patch in image {img_name} with iteration: {num_tries}"
                        )
                        labeled = True
                        break

                    # if no new patch could fit inside of img do not consider again
                    if num_tries == self.trials_per_img:
                        logger.info(f"Could not place patch in image {img_name}")
                        logger.info(f"PatchCount {len(patches)}")
                        logger.info(f"{num_tries=}")
                        count = 0
                        for item in self.img_names:
                            if item == img_name:
                                break
                            count += 1

                        self.img_names.pop(count)
                        img_generator = get_infinte_iter(self.img_names)
                        break
                    num_tries += 1
                if labeled:
                    break
        return patches


@register_strategy("random-all-classes")
class RandomAllClasses(Random):
    def query(self, verbose: bool = False, n_gpus: int = 0, **kwargs) -> List[Patch]:
        selected_patches = query_starting_budget_all_classes(
            self.raw_labels_path,
            self.file_ending,
            annotated_patches=self.annotated_patches,
            patch_size=self.config.patch_size,
            rng=self.rng,
            trials_per_img=self.trials_per_img,
            additional_label_path=self.additional_label_path,
            additional_overlap=self.config.additional_overlap,
            verbose=verbose,
        )
        return super().query(
            verbose=verbose, already_annotated_patches=selected_patches
        )
