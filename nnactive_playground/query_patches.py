import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# from nnunetv2.paths import nnUNet_raw


def does_overlap(start_indices, patch_size, selected_array):
    """
    Check if a patch overlaps with an already annotated region
    Args:
        start_indices: start indices of the patch
        patch_size: patch size to determine end indices
        selected_array: array containing the already annotated regions

    Returns:
        True if the patch overlaps with an already annotated region, False if not
    """
    # Convert the indices to slices, makes indexing of selected_array possible without being dependent on dimensions
    slices = []
    for start_index, size in zip(start_indices, patch_size.tolist()):
        slices.append(slice(start_index, start_index + size))
    # Areas that are already annotated should be marked with 1 in the selected_array
    if selected_array[tuple(slices)].max() > 0:
        return True
    return False


def image_id_from_aggregated_name(image_aggregated_name, uncertainty_type):
    return image_aggregated_name.split(f"_{uncertainty_type}")[0]


def get_original_input_image(image_id, raw_images_dir):
    # TODO: how to deal with different modalities
    # TODO: how to infere file ending
    image_path = raw_images_dir / f"{image_id}_0000.nii.gz"
    sitk_image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(sitk_image)


def mark_already_annotated_patches(selected_array, raw_images_dir):
    # TODO: mark the already annotated patches here (set the corresponding areas in selected_array to 1)
    # TODO: is the information about the annoatated patches in a json in raw_images_dir? Or how to infer it?
    return selected_array


def mark_selected(selected_array, coords, patch_size, selected_idx=1):
    """
    Mark a patch as selected that no area of this patch is queried multiple times
    Args:
        selected_array: array with already queried regions that should be extended by the patch
        coords: start coordinated of the patch
        patch_size: patch size to determine end indices
        selected_idx: int which should be used to mark the patch as annotated
        (normally 1, can be changed for visualization)

    Returns:
        array with queried region including the patch that was passed
    """
    slices = []
    for start_index, size in zip(coords, patch_size):
        slices.append(slice(start_index, start_index + size))
    # Mark the corresponding region
    selected_array[tuple(slices)] = selected_idx
    return selected_array


def get_top_n_non_overlapping_patches(
    image_name, n, uncertainty_scores, patch_size, raw_images_dir, uncertainty_type
):
    """
    Get the most n uncertain non-overlapping patches for one image based on the aggregated uncertainty map

    Args:
        image_name: the name of the aggregated uncertainty map (npz file)
        n: number of non-overlapping patches that should be queried at most
        uncertainty_scores: the aggregated uncertainty map
        patch_size: patch size that was used to aggregate the uncertainties
        raw_images_dir: the input directory with the raw image data that is used for training
        uncertainty_type: uncertainty type that should be used for ranking

    Returns:
        the most n uncertain non-overlapping patches for one image
    """
    # Get the image id to find the image in the original data folder of the training images
    image_id = image_id_from_aggregated_name(image_name, uncertainty_type)
    selected_patches = []
    # Selected array is an array of the raw input image size that is used to mark which areas have already been queried
    selected_array = np.zeros_like(get_original_input_image(image_id, raw_images_dir))
    # Mark the patched as annotated that were annotated in previous loops
    selected_array = mark_already_annotated_patches(selected_array, raw_images_dir)
    sorted_uncertainty_scores = np.flip(np.sort(uncertainty_scores.flatten()))
    sorted_uncertainty_indices = np.flip(np.argsort(uncertainty_scores.flatten()))
    # This was just for visualization purposes in MITK
    # selected = 0

    # Iterate over the sorted uncertainty scores and their indices to get the most uncertain
    for uncertainty_score, uncertainty_index in zip(
        sorted_uncertainty_scores, sorted_uncertainty_indices
    ):
        # Get the index as coordinates
        coords = np.unravel_index(uncertainty_index, uncertainty_scores.shape)
        # Check if coordinated overlap with already queried region
        if not does_overlap(coords, patch_size, selected_array):
            # If it is a non-overlapping region, append this patch to be queried
            selected_patches.append(
                {
                    "file": image_name,
                    "coords": coords,
                    "size": patch_size,
                    "score": uncertainty_score,
                }
            )
            # selected += 1
            # Mark region as queried
            selected_array = mark_selected(selected_array, coords, patch_size)
        # Stop if we reach the maximum number of patches to be queried
        if n is not None and len(selected_patches) >= n:
            break
    # This was just for visualization purposes in MITK
    # selected_array = selected_array.astype(np.intc)
    # selected_array_image = sitk.GetImageFromArray(selected_array)
    # sitk.WriteImage(
    #     selected_array_image,
    #     f"/home/kckahl/Documents/PatchCheck/check_{image_id}.nrrd",
    # )
    return selected_patches


def get_most_uncertain_patches(
    aggregated_uncertainty_dir, uncertainty_type, raw_images_dir, number_to_query=None
):
    """
    Get the most uncertain patches across all predicted images.
    Args:
        aggregated_uncertainty_dir: directory containing the aggregated uncertainties for ranking the patches
        uncertainty_type: uncertainty type that should be used for ranking (specified in uncertainty file name)
        raw_images_dir: directory with the raw input images
        number_to_query: number of patches to query. If None, all patches from the input images will be ranked

    Returns:
        Either the top n most uncertain patches if number_to_query is specified
        or all non-overlapping patches sorted by uncertainty
    """
    all_top_patches = []
    for image_name in os.listdir(aggregated_uncertainty_dir):
        if image_name.endswith(".npz") and uncertainty_type in image_name:
            # Load the aggregated uncertainty map
            # (npz file containing information about uncertainty score per patch & patch size)
            image = np.load(aggregated_uncertainty_dir / image_name)
            uncertainty_scores = image["patch_score"]
            patch_size = image["patch_size"]
            # Extend the list of patches that can possibly be queried by number_to_query from the current image
            # TODO: this might be optimized, i.e. we don't need to query n uncertain patches from an image
            #  when we already have more uncertain patches from other images
            all_top_patches.extend(
                get_top_n_non_overlapping_patches(
                    image_name,
                    number_to_query,
                    uncertainty_scores,
                    patch_size,
                    raw_images_dir,
                    uncertainty_type,
                )
            )
    print(len(all_top_patches))
    # Sort the patches across the different images according to their uncertainty score
    all_top_patches = sorted(all_top_patches, key=lambda d: d["score"], reverse=True)
    if number_to_query is not None:
        # Only return the top number_to_query patches
        all_top_patches = all_top_patches[:number_to_query]
    print(len(all_top_patches))
    return all_top_patches


def query_patches():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing the aggregated npz uncertainty files",
    )
    parser.add_argument(
        "-u",
        "--uncertainty_type",
        type=str,
        required=True,
        help="Uncertainty type that should be used as query criterion "
        "[expected_entropy | predictive_entropy | mutual_information]",
    )
    parser.add_argument(
        "-r",
        "--raw_image_folder",
        type=str,
        required=False,
        help="Path with the raw input images that are partially annotated. "
        "If not specified, the nnU-Net raw imagesTr will be used.",
    )
    parser.add_argument(
        "-n",
        "--number_to_query",
        type=int,
        required=False,
        default=None,
        help="Number of patches to query. If None, returns all non-overlapping patches sorted by uncertainty",
    )

    args = parser.parse_args()
    aggregated_uncertainty_dir = Path(args.input_dir)
    uncertainty_type = args.uncertainty_type
    number_to_query = args.number_to_query
    raw_image_folder = args.raw_image_folder

    if uncertainty_type not in [
        "expected_entropy",
        "predictive_entropy",
        "mutual_information",
    ]:
        assert (
            "Unkown uncertainty type. Uncertainty type has to be one of the following: "
            "[expected_entropy | predictive_entropy | mutual_information]"
        )

    # TODO: this is not correct yet (Dataset specification missing).
    #  How to infere the origin of the input images? Store somehow as metadata in the uncertainties?
    # if raw_image_folder is None:
    #     raw_image_folder = Path(nnUNet_raw) / "imagesTr"
    # else:
    #     raw_image_folder = Path(raw_image_folder)

    raw_image_folder = Path(raw_image_folder)

    # TODO: these most uncertain patches can then be stored in a patches.json file in the standard format
    #  which can be used to create the input folders for the next cycle
    most_uncertain_patches = get_most_uncertain_patches(
        aggregated_uncertainty_dir, uncertainty_type, raw_image_folder, number_to_query
    )


if __name__ == "__main__":
    query_patches()
