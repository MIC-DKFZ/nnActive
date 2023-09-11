from typing import List

from nnactive.data import Patch


def get_slices_for_file_from_patch(patches: List[Patch], label_file: str):
    patches_file = [patch for patch in patches if patch.file == label_file]
    patch_access = []
    for patch in patches_file:
        slice_x = slice(patch.coords[0], patch.coords[0] + patch.size[0])
        slice_y = slice(patch.coords[1], patch.coords[1] + patch.size[1])
        slice_z = slice(patch.coords[2], patch.coords[2] + patch.size[2])
        patch_access.append((slice_x, slice_y, slice_z))
    return patch_access
