from pathlib import Path
from typing import List, Union

import numpy as np
from loguru import logger

from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.masking import does_overlap, percentage_overlap_array
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.query.get_locs import get_locs_from_segmentation
from nnactive.strategies.random import (
    Random,
    _get_infinte_iter,
    _obtain_random_patch_for_img,
)
from nnactive.utils.io import load_label_map


class RandomLabel(Random):
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
        additional_overlap: float = 0.1,
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
            dataset_id,
            query_size,
            patch_size,
            seed,
            trials_per_img,
            file_ending,
            raw_labels_path,
            additional_label_path,
            additional_overlap,
            verbose=verbose,
        )
        self.raw_labels_path = raw_labels_path
        if self.raw_labels_path is None:
            config = ActiveConfig.get_from_id(self.dataset_id)
            annotated_id = int(config.dataset.split("_")[0][-3:])
            self.raw_labels_path = get_raw_path(annotated_id) / "labelsTr"

        self.background_cls = background_cls
        if self.background_cls is None:
            config = ActiveConfig.get_from_id(self.dataset_id)
            annotated_id = int(config.dataset.split("_")[0][-3:])
            dataset_json = read_dataset_json(annotated_id)
            self.background_cls = dataset_json["labels"].get("background")

    def query(
        self, verbose: bool = False, already_annotated_patches: list[Patch] = None
    ) -> List[Patch]:
        logger.info(self.img_names)
        img_generator = _get_infinte_iter(self.img_names)
        labeled_patches = self.annotated_patches
        if already_annotated_patches is None:
            patches = []
        else:
            patches = already_annotated_patches
        for i in range(self.query_size - len(patches)):
            if verbose:
                logger.debug("-" * 8)
                logger.debug("-" * 8)
                logger.debug(f"Start Creation of Patch {i}")
            labeled = False
            patch_count = 0
            while True:
                if patch_count > 3 * len(self.img_names):
                    logger.warning(f"Patch {i} could not be created without overlap!")
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

                area = self.rng.choice(["all", "seg", "border"])

                if verbose:
                    logger.debug(f"Start drawing random patch with strategy: {area}")

                if area in ["seg", "border"]:
                    if verbose:
                        logger.debug(f"Get Locations for Style: {area}")
                    locs = get_locs_from_segmentation(
                        label_map,
                        area,
                        state=self.rng,
                        background_cls=self.background_cls,
                    ).tolist()
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
                        ) = _obtain_random_patch_from_locs(
                            locs, img_size, self.patch_size, self.rng
                        )
                    if area in ["all"]:
                        iter_patch_loc, iter_patch_size = _obtain_random_patch_for_img(
                            img_size, self.patch_size, self.rng
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


def _obtain_random_patch_from_locs(
    locs: Union[tuple, list],
    img_size: list,
    patch_size: list,
    rng=np.random.default_rng(),
) -> tuple[list[int], list[int]]:
    """Locs describe the center of the area that should be cropped. Can be np.argwhere(img>0)"""
    patch_real_size = []
    # Get correct size of patch
    for dim_img, dim_patch in zip(img_size, patch_size):
        if dim_patch >= dim_img:
            patch_real_size.append(dim_img)
        else:
            patch_real_size.append(dim_patch)

    loc = locs[rng.choice(len(locs))]
    patch_loc = []

    for dim_loc, dim_img, dim_patch in zip(loc, img_size, patch_real_size):
        if dim_patch >= dim_img:
            patch_loc.append(0)
        else:
            # patch fits right into the image
            if dim_loc + dim_patch // 2 <= dim_img and dim_loc - dim_patch // 2 >= 0:
                patch_loc.append(dim_loc - dim_patch // 2)
            # patch overshoots, set to maximal possible value
            elif dim_loc + dim_patch // 2 > dim_img:
                patch_loc.append(dim_img - dim_patch)
            # patch undershoots, set to minimal possible value
            elif dim_loc - dim_patch // 2 < 0:
                patch_loc.append(0)
            else:
                raise NotImplementedError

    return (patch_loc, patch_real_size)
