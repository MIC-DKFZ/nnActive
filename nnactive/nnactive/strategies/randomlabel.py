import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Union

import numpy as np
from loguru import logger

import wandb
from nnactive.config import ActiveConfig
from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.paths import set_raw_paths
from nnactive.strategies.random import Random
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.utils import (
    generate_random_patch_from_locs,
    get_infinte_iter,
    get_locs_from_segmentation,
    obtain_random_patch_for_img,
    query_starting_budget_all_classes,
)
from nnactive.utils.io import load_label_map


@register_strategy("random-label")
class RandomLabel(Random):
    def __init__(
        self,
        config: ActiveConfig,
        dataset_id: int,
        loop_val: int,
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        background_cls: int | None = None,
        additional_label_path: Path | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """

        Args:
            dataset_id (int): _description_
            query_size (int): _description_
            patch_size (list[int]): _description_
            seed (int): _description_
            trials_per_img (int, optional): _description_. Defaults to 600.
            file_ending (str, optional): _description_. Defaults to ".nii.gz".
            raw_labels_path (Path | None, optional): Is expected to be path to a fully annotated dataset. Defaults to None.
            background_cls (int | None, optional): _description_. Defaults to None.
        """
        super().__init__(
            dataset_id=dataset_id,
            config=config,
            loop_val=loop_val,
            file_ending=file_ending,
            additional_label_path=additional_label_path,
            verbose=verbose,
            seed=seed,
            # Random
            trials_per_img=trials_per_img,
            raw_labels_path=raw_labels_path,
        )
        self.raw_labels_path = raw_labels_path
        if self.raw_labels_path is None:
            config = ActiveConfig.get_from_id(self.dataset_id)
            with set_raw_paths():
                annotated_id = int(config.dataset.split("_")[0][-3:])
                self.raw_labels_path = get_raw_path(annotated_id) / "labelsTr"

        self.background_cls = background_cls
        if self.background_cls is None:
            config = ActiveConfig.get_from_id(self.dataset_id)
            with set_raw_paths():
                annotated_id = int(config.dataset.split("_")[0][-3:])
                dataset_json = read_dataset_json(annotated_id)
                self.background_cls = dataset_json["labels"].get("background")

    def wrap_query(
        self,
        verbose: bool = False,
        already_annotated_patches: list[Patch] = None,
        n_gpus: int = 1,
        wandb_group: str | None = None,
    ):
        self.config.set_nnunet_env()
        with monitor.active_run(group=wandb_group):
            top_patches = self.query(verbose, already_annotated_patches, n_gpus)
        return top_patches

    def query(
        self,
        verbose: bool = False,
        already_annotated_patches: list[Patch] = None,
        n_gpus: int = 0,
    ) -> List[Patch]:
        """
        Args:
            already_annotated_patches (list[Patch], optional): only used in combination with all_classes. Defaults to None.

        Returns:
            List[Patch]: patches for annotation
        """

        # ensure that all processes are run into subprocesses for n_gpus > 1
        # issues can arise if this is not done.
        if n_gpus == 0:
            logger.info(self.img_names)
            img_generator = get_infinte_iter(self.img_names)
            labeled_patches = self.annotated_patches
            if already_annotated_patches is None:
                patches = []
            else:
                patches = already_annotated_patches
            for i in range(self.config.query_size - len(patches)):
                if verbose:
                    logger.debug("-" * 8)
                    logger.debug("-" * 8)
                    logger.debug(f"Start Creation of Patch {i}")
                labeled = False
                patch_count = 0
                while True:
                    if patch_count > 3 * len(self.img_names):
                        logger.warning(
                            f"Patch {i} could not be created without overlap!"
                        )
                        break
                    img_name = img_generator.__next__()
                    if verbose:
                        logger.debug(f"Loading Image: {img_name}")
                    label_map: np.ndarray = load_label_map(
                        img_name.replace(self.file_ending, ""),
                        self.raw_labels_path,
                        self.file_ending,
                    )
                    current_patch_list = labeled_patches + patches
                    img_size = label_map.shape
                    # only needed for creation of patches in first iteration
                    if verbose:
                        logger.debug(f"Create Mask: {img_name}")
                    selected_image_patches = [
                        patch for patch in current_patch_list if patch.file == img_name
                    ]

                    additional_label = None
                    if self.additional_label_path is not None:
                        if verbose:
                            logger.debug("Create additional label map.")
                        additional_label = load_label_map(
                            img_name.replace(self.file_ending, ""),
                            self.additional_label_path,
                            self.file_ending,
                        )
                        additional_label: np.ndarray = additional_label != 255

                    if verbose:
                        logger.debug("Mask creation succesfull")

                    area = self.get_area()

                    if verbose:
                        logger.debug(
                            f"Start drawing random patch with strategy: {area}"
                        )

                    if area in ["seg", "border"]:
                        if verbose:
                            logger.debug(f"Get Locations for Style: {area}")
                        locs = get_locs_from_segmentation(
                            label_map,
                            area,
                            state=self.rng,
                            background_cls=self.background_cls,
                        )
                        if verbose:
                            logger.debug("Obtaining Locations was succesful.")
                        if len(locs) == 0:
                            continue

                    num_tries = 0
                    while True:
                        # propose a random patch
                        if area in ["seg", "border"]:
                            # if verbose:
                            # print("Draw Random Patch")
                            (
                                iter_patch_loc,
                                iter_patch_size,
                            ) = generate_random_patch_from_locs(
                                locs, img_size, self.config.patch_size, self.rng
                            )
                        if area in ["all"]:
                            (
                                iter_patch_loc,
                                iter_patch_size,
                            ) = obtain_random_patch_for_img(
                                img_size, self.config.patch_size, self.rng
                            )

                        patch = Patch(
                            file=img_name,
                            coords=iter_patch_loc,
                            size=iter_patch_size,
                        )

                        # check if patch is valid
                        if self.check_overlap(
                            patch, selected_image_patches, additional_label, verbose
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
                            # this is change compared to baseline, but samples where no random patch fits.
                            # no patch at all will fit!
                            if area in ["all"]:
                                count = 0
                                for item in self.img_names:
                                    if item == img_name:
                                        break
                                    count += 1

                                # self.img_names.pop(count)
                                # img_generator = _get_infinte_iter(self.img_names)
                            break
                        num_tries += 1
                    if labeled:
                        break
            return patches
        else:
            logger.debug("Execute Query in Subprocess.")
            patch_list = []
            try:
                with ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("spawn")
                ) as executor:
                    for patch_final in executor.map(
                        self.wrap_query, [verbose], [None], [0], [wandb.run.group]
                    ):
                        patch_list.append(patch_final)
                return patch_list[0]

            except BrokenProcessPool as exc:
                raise MemoryError(
                    "One of the worker processes died. "
                    "This usually happens because you run out of memory. "
                    "Try running with less processes."
                ) from exc

    def get_area(self):
        area = self.rng.choice(["all", "seg", "border"])
        return area


@register_strategy("random-label-all-classes")
class RandomLabelAllClasses(RandomLabel):
    def query(self, verbose: bool = False, n_gpus: int = 0, **kwargs) -> List[Patch]:
        # Do stuff to ensure all lables are represented two times
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
