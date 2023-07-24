from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk

from nnactive.data import Patch
from nnactive.query.get_locs import get_locs_from_segmentation


def _get_infinte_iter(finite_list):
    while True:
        for elt in finite_list:
            yield elt


def _does_overlap(
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


def generate_random_patches_labels(
    file_ending: str,
    seg_labels_path: Path,
    patch_size: list,
    n_patches: int,
    labeled_patches: list[Patch],
    seed: int = None,
    trials_per_img: int = 6000,
    background_cls: Union[int, None] = None,
    verbose: bool = False,
) -> list[Patch]:
    rng = np.random.default_rng(seed)
    img_names = [path.name for path in seg_labels_path.glob(f"**/*{file_ending}")]
    rng.shuffle(img_names)

    # return infinite list of the images
    img_generator = _get_infinte_iter(img_names)

    patches = []
    for i in range(n_patches):
        if verbose:
            print("-" * 8)
            print("-" * 8)
            print(f"Start Creation of Patch {i}")
        labeled = False
        patch_count = 0
        while True:
            if patch_count > 3 * len(img_names):
                print(f"No more Patches could be Created for Patch {i}!")
                break

            img_name = img_generator.__next__()
            if verbose:
                print("-" * 8)
                print(f"Loading image: {img_name}")
            label_map = get_label_map(
                img_name.replace(file_ending, ""), seg_labels_path, file_ending
            )
            current_patch_list = labeled_patches + patches
            img_size = label_map.shape
            # only needed for creation of patches in first iteration
            if verbose:
                print(f"Create Mask: {img_name}")
            selected_array = create_patch_mask_for_image(
                img_name, current_patch_list, img_size
            )
            if verbose:
                print("Mask creation succesfull")

            area = rng.choice(["all", "seg", "border"])

            # set areas that are already selected to ignore
            # print("Disable selected areas")
            # label_map[selected_array] = 0
            # Let us not do that since for big images it takes ages

            if verbose:
                print(f"Start drawing random patch with strategy: {area}")

            if area in ["seg", "border"]:
                if verbose:
                    print(f"Get Locations for Style: {area}")
                locs = get_locs_from_segmentation(
                    label_map, area, state=rng, background_cls=background_cls
                ).tolist()
                if verbose:
                    print("Obtaining Locations was succesful.")
                if len(locs) == 0:
                    continue

            # This line is only necessary for first iteration, where they are non-existent
            num_tries = 0
            while True:
                # propose a random patch
                if area in ["seg", "border"]:
                    if verbose:
                        print("Draw Random Patch")
                    iter_patch_loc, iter_patch_size = _obtain_random_patch_from_locs(
                        locs, img_size, patch_size, rng
                    )
                if area in ["all"]:
                    iter_patch_loc, iter_patch_size = _obtain_random_patch(
                        img_size, patch_size, rng
                    )

                # check if patch is valid
                if not _does_overlap(iter_patch_loc, iter_patch_size, selected_array):
                    patches.append(Patch(img_name, iter_patch_loc, iter_patch_size))
                    # print(f"Creating Patch with iteration: {num_tries}")
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
                    break
                num_tries += 1
            if labeled:
                break
    return patches


def generate_random_patches(
    file_ending: str,
    raw_labels_path: Path,
    patch_size: list,
    n_patches: int,
    labeled_patches: list[Patch],
    seed: int = None,
    trials_per_img: int = 6000,
    verbose: bool = False,
) -> list[Patch]:
    """Generates random patches based on randomly drawing starting indices fitting inside the dataset.

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
    """
    rng = np.random.default_rng(seed)
    img_names = [path.name for path in raw_labels_path.glob(f"**/*{file_ending}")]
    rng.shuffle(img_names)

    # return infinite list of the images
    img_generator = _get_infinte_iter(img_names)

    patches = []
    for i in range(n_patches):
        labeled = False
        while True:
            img_name = img_generator.__next__()
            if verbose:
                print(f"Loading Image: {img_name}")
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
                    img_size, patch_size, rng
                )

                # check if patch is valid
                if not _does_overlap(iter_patch_loc, iter_patch_size, selected_array):
                    patches.append(Patch(img_name, iter_patch_loc, iter_patch_size))
                    # print(f"Creating Patch with iteration: {num_tries}")
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
                    img_generator = _get_infinte_iter(img_names)
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


def _obtain_random_patch(
    img_size: list, patch_size: list, rng=np.random.default_rng()
) -> tuple[list[int], list[int]]:
    """Generates a complete random patch fitting inside the image

    Args:
        img_size (list): _description_
        patch_size (list): _description_
        rng (_type_, optional): _description_. Defaults to np.random.default_rng().

    Returns:
        _type_: _description_
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
