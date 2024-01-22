from pathlib import Path
from typing import Generator, Iterable

import numpy as np

from nnactive.data import Patch
from nnactive.masking import create_patch_mask_for_image, does_overlap
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.utils.io import load_label_map


class Random(AbstractQueryMethod):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(dataset_id, query_size, patch_size, file_ending)
        self.trials_per_img = trials_per_img
        self.rng = np.random.default_rng(seed)

        self.raw_labels_path = (
            raw_labels_path
            if raw_labels_path is not None
            else get_raw_path(dataset_id) / "labelsTr"
        )

        self.img_names = [
            path.name for path in self.raw_labels_path.glob(f"**/*{self.file_ending}")
        ]
        self.rng.shuffle(self.img_names)

    def query(self, verbose: bool = False) -> list[Patch]:
        img_generator = _get_infinte_iter(self.img_names)
        patches = []
        print("verbose", verbose)
        for _ in range(self.query_size):
            labeled = False
            while True:
                img_name = img_generator.__next__()
                if verbose:
                    print(f"Loading Image: {img_name}")
                label_map: np.ndarray = load_label_map(
                    img_name.replace(self.file_ending, ""),
                    self.raw_labels_path,
                    self.file_ending,
                )
                current_patch_list = self.annotated_patches + patches
                img_size = label_map.shape
                # only needed for creation of patches in first iteration
                selected_array = [
                    patch for patch in current_patch_list if patch.file == img_name
                ]

                num_tries = 0
                while True:
                    # propose a random patch
                    iter_patch_loc, iter_patch_size = _obtain_random_patch_for_img(
                        img_size, self.patch_size, self.rng
                    )

                    patch = Patch(
                        file=img_name,
                        coords=iter_patch_loc,
                        size=iter_patch_size,
                    )

                    # check if patch is valid
                    if not does_overlap(patch, selected_array):
                        patches.append(patch)
                        # print(f"Creating Patch with iteration: {num_tries}")
                        labeled = True
                        print(num_tries)
                        break

                    # if no new patch could fit inside of img do not consider again
                    if num_tries == self.trials_per_img:
                        print(f"Could not place patch in image {img_name}")
                        print(f"PatchCount {len(patches)}")
                        print(num_tries)
                        count = 0
                        for item in self.img_names:
                            if item == img_name:
                                break
                            count += 1

                        self.img_names.pop(count)
                        img_generator = _get_infinte_iter(self.img_names)
                        break
                    num_tries += 1
                if labeled:
                    break
        return patches


def _get_infinte_iter(finite_list: Iterable):
    while True:
        for elt in finite_list:
            yield elt


def _obtain_random_patch_for_img(
    img_size: list, patch_size: list, rng: Generator = np.random.default_rng()
) -> tuple[list[int], list[int]]:
    """Generates a complete random patch fitting inside the image

    Args:
        img_size (list): size of image
        patch_size (list): size of patch
        rng (Generator, optional): generator for seeding. Defaults to np.random.default_rng().

    Returns:
        tuple[list[int], list[int]]: (location, patch_size)
    """
    patch_loc_ranges = []
    patch_real_size = []
    for dim_img, dim_patch in zip(img_size, patch_size):
        if dim_patch >= dim_img:
            patch_loc_ranges.append([0])
            patch_real_size.append(dim_img)
        else:
            patch_loc_ranges.append([i for i in range(dim_img - dim_patch)])
            patch_real_size.append(dim_patch)

    patch_loc = []
    for loc_range in patch_loc_ranges:
        patch_loc.append(rng.choice(loc_range))

    return (patch_loc, patch_real_size)
