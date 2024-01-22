from typing import List, Tuple, Union

import numpy as np

from nnactive.data import Patch


def create_patch_mask_for_image(
    img_name: str,
    current_patch_list: list[Patch],
    img_size: Union[list, tuple],
    identify_patch: bool = False,
    size_offset: int = 0,
) -> np.ndarray:
    """Creates a binary mask where patches have a value of 1.

    Args:
        img_name (str): Name of Label file with file_ending
        current_patch_list (list[Patch]): All patches which should be annotated
        img_size (Union[list, tuple]): Size of Image
        identify_patch (bool, optional): If True area with patches are numerated from 1 to X. Defaults to False.
        size_offset (int, optional): positive values increase size of masked area per patch. Defaults to 0.

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


def mark_selected(
    selected_array: np.ndarray,
    coords: Tuple[np.ndarray],
    patch_size: np.ndarray,
    selected_idx: int = 1,
) -> np.ndarray:
    """
    Mark a patch as selected that no area of this patch is queried multiple times
    Args:
        selected_array (np.ndarray): array with already queried regions that should be extended by the patch
        coords (Tuple[np.ndarray]): start coordinated of the patch
        patch_size (np.ndarray): patch size to determine end indices
        selected_idx (int): int which should be used to mark the patch as annotated
        (normally 1, can be changed for visualization)

    Returns:
        np.ndarray: array with queried region including the patch that was passed
    """
    slices = []
    for start_index, size in zip(coords, patch_size):
        slices.append(slice(start_index, start_index + size))
    # Mark the corresponding region
    selected_array[tuple(slices)] = selected_idx
    return selected_array


def mark_already_annotated_patches(
    selected_array: np.ndarray, labeled_array: np.ndarray, ignore_label: int
) -> np.ndarray:
    """Returns array where annotated areas are set to 1 in selected_array

    Args:
        selected_array (np.ndarray): array to simulate selection
        labeled_array (np.ndarray): array with label information
        ignore_label (int): label value signaling unlabeled regions

    Returns:
        np.ndarrary: see description
    """
    selected_array[labeled_array != ignore_label] = 1
    return selected_array


def does_overlap(
    ipatch: Patch,
    patches: list[Patch],
):
    num_dims = len(ipatch.coords)

    for patch in patches:
        if all(
            ipatch.coords[i] < patch.coords[i] + patch.size[i]
            and ipatch.coords[i] + ipatch.size[i] > patch.coords[i]
            for i in range(num_dims)
        ):
            return True
    return False
