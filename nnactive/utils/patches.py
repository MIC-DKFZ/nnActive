from typing import Iterable, List

import numpy as np

from nnactive.data import Patch


def get_slices_for_file_from_patch(
    patches: List[Patch], label_file: str
) -> list[tuple[slice, slice, slice]]:
    patches_file = [patch for patch in patches if patch.file == label_file]
    patch_access = []
    for patch in patches_file:
        slice_x = slice(patch.coords[0], patch.coords[0] + patch.size[0])
        slice_y = slice(patch.coords[1], patch.coords[1] + patch.size[1])
        slice_z = slice(patch.coords[2], patch.coords[2] + patch.size[2])
        patch_access.append((slice_x, slice_y, slice_z))
    return patch_access


# TODO: function is simlar to initialize_selected_array in base.py
def create_patch_mask_for_image(
    img_name: str,
    current_patch_list: list[Patch],
    img_size: list | tuple | Iterable,
    identify_patch: bool = False,
    size_offset: int = 0,
) -> np.ndarray:
    """Creates a binary mask where patches have a value of 1 or [1,..,n] for n patches in image

    Args:
        img_name (str): image name with file_ending
        current_patch_list (list[Patch]): containing current patches
        img_size (list | tuple | Iterable): size of image
        identify_patch (bool, optional): set patches to 1 or [1...n]. Defaults to False.
        size_offset (int, optional): increase or decrease size of mask per dim. Defaults to 0.

    Returns:
        np.ndarray
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
