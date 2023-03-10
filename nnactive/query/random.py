from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from nnactive.data import Patch


def does_overlap(
    start_indices: tuple[np.ndarray], patch_size: list[int], selected_array: np.ndarray
) -> bool:
    """
    Check if a patch overlaps with an already annotated region
    Args:
        start_indices (Tuple[str]): start indices of the patch
        patch_size (np.ndarray): patch size to determine end indices
        selected_array (np.ndarray): array containing the already annotated regions

    Returns:
        bool: True if the patch overlaps with an already annotated region, False if not
    """
    # Convert the indices to slices, makes indexing of selected_array possible without being dependent on dimensions
    slices = []
    for start_index, size in zip(start_indices, patch_size):
        slices.append(slice(start_index, start_index + size))
    # Areas that are already annotated should be marked with 1 in the selected_array
    if selected_array[tuple(slices)].max() > 0:
        return True
    return False


def get_label_map(image_id: str, raw_labels_dir: Path, file_ending: str) -> np.ndarray:
    image_path = raw_labels_dir / f"{image_id}{file_ending}"
    sitk_image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(sitk_image)


def generate_random_patches(
    file_ending: str,
    raw_labels_path: Path,
    patch_size: list,
    n_patches: int,
    labeled_patches: list[Patch],
    seed: int = None,
    trials_per_img: int = 6000,
) -> list[Patch]:
    """_summary_

    Args:
        file_ending (str): _description_
        raw_labels_path (Path): _description_
        patch_size (list): _description_
        n_patches (int): _description_
        labeled_patches (list[Patch]): _description_
        seed (int, optional): _description_. Defaults to None.
        trials_per_img (int, optional): _description_. Defaults to 6000.

    Returns:
        list[Patch]: _description_

    Yields:
        Iterator[list[Patch]]: _description_
    """
    rng = np.random.default_rng(seed)
    img_names = [path.name for path in raw_labels_path.glob(f"**/*{file_ending}")]
    rng.shuffle(img_names)

    def get_infinte_iter(finite_list):
        while True:
            for elt in finite_list:
                yield elt

    # return infinite list of the images
    img_generator = get_infinte_iter(img_names)

    patches = []
    for i in range(n_patches):
        labeled = False
        while True:
            img_name = img_generator.__next__()
            label_map: np.ndarray = get_label_map(
                img_name.replace(file_ending, ""), raw_labels_path, file_ending
            )
            current_patch_list = labeled_patches + patches
            img_size = label_map.shape
            # only needed for creation of patches in first iteration
            selected_array = create_patch_mask_for_image(
                img_name, current_patch_list, img_size
            )

            # This line is only necessary for first iteration, where they are non-existent
            num_tries = 0
            while True:
                # propose a random patch
                iter_patch_loc, iter_patch_size = _obtain_random_patch(
                    img_size, patch_size
                )

                # check if patch is valid
                if not does_overlap(iter_patch_loc, iter_patch_size, selected_array):
                    patches.append(Patch(img_name, iter_patch_loc, iter_patch_size))
                    print(f"Creating Patch with iteration: {num_tries}")
                    labeled = True
                    break

                # if no new patch could fit inside of img do not consider again
                if num_tries == trials_per_img:
                    print(f"Could not place patch in image {img_name}")
                    print(f"PatchCount {len(patches)}")
                    count = 0
                    for item in img_names:
                        if item == img_name:
                            break
                        count += 1

                    img_names.pop(count)
                    img_generator = get_infinte_iter(img_names)
                    break
                num_tries += 1
            if labeled:
                break
    return patches


def create_patch_mask_for_image(
    img_name: str,
    current_patch_list: list[Patch],
    img_size: Union[list, tuple],
    identify_patch: bool = False,
    size_offset: int = 0,
) -> np.ndarray:
    """Creates a binary mask where patches have a value of 1.

    Args:
        img_name (_type_): _description_
        current_patch_list (_type_): _description_
        img_size (_type_): _description_


    Returns:
        np.ndarray: _description_
    """
    selected_array = np.zeros(img_size, dtype=np.uint8)

    img_patches = [patch for patch in current_patch_list if patch.file == img_name]

    for i, img_patch in enumerate(img_patches):
        slices = []
        # TODO: this could be changed if img_size is changed!
        if img_patch.size == "whole":
            selected_array.fill(1)
        else:
            for start_index, size in zip(img_patch.coords, img_patch.size):
                slices.append(
                    slice(start_index - size_offset, start_index + size + size_offset)
                )
            if identify_patch:
                selected_array[tuple(slices)] = i + 1
            else:
                selected_array[tuple(slices)] = 1
    return selected_array


def _obtain_random_patch(img_size: list, patch_size: list):
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
        patch_loc.append(np.random.choice(loc_range))

    return (patch_loc, patch_real_size)
