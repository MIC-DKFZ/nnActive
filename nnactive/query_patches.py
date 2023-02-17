import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import SimpleITK as sitk

from nnactive.data import Patch
from nnactive.loops.loading import save_loop

# from nnunetv2.paths import nnUNet_raw


def does_overlap(
    start_indices: Tuple[np.ndarray], patch_size: np.ndarray, selected_array: np.ndarray
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
    for start_index, size in zip(start_indices, patch_size.tolist()):
        slices.append(slice(start_index, start_index + size))
    # Areas that are already annotated should be marked with 1 in the selected_array
    if selected_array[tuple(slices)].max() > 0:
        return True
    return False


def image_id_from_aggregated_name(
    image_aggregated_name: str, uncertainty_type: str
) -> str:
    return image_aggregated_name.split(f"_{uncertainty_type}")[0]


def get_label_map(image_id: str, raw_dataset_dir: Path, file_ending: str) -> np.ndarray:
    # TODO: get file ending from dataset.json
    image_path = raw_dataset_dir / "labelsTr" / f"{image_id}{file_ending}"
    sitk_image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(sitk_image)


def mark_already_annotated_patches(
    selected_array: np.ndarray, labeled_array: np.ndarray, ignore_label: int
) -> np.ndarray:
    """Returns array where annotated areas are set to in selected_array

    Args:
        selected_array (np.ndarray): array to simulate selection
        labeled_array (np.ndarray): array with label information
        ignore_label (int): label value signaling unlabeled regions

    Returns:
        np.ndarrary: see description
    """
    selected_array[labeled_array != ignore_label] = 1
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


def get_top_n_non_overlapping_patches(
    image_name: str,
    n: int,
    uncertainty_scores: np.ndarray,
    patch_size: np.ndarray,
    selected_array: np.ndarray,
) -> List[Dict]:
    """
    Get the most n uncertain non-overlapping patches for one image based on the aggregated uncertainty map

    Args:
        image_name (str): the name of the aggregated uncertainty map (npz file)
        n (int): number of non-overlapping patches that should be queried at most
        uncertainty_scores (np.ndarray): the aggregated uncertainty map
        patch_size (np.ndarray): patch size that was used to aggregate the uncertainties
        selected_array (np.ndarray): array with already labeled patches

    Returns:
        List[Dict]: the most n uncertain non-overlapping patches for one image
    """
    selected_patches = []
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
    aggregated_uncertainty_dir: Path,
    uncertainty_type: str,
    raw_dataset_dir: Path,
    ignore_label: int,
    file_ending: str,
    number_to_query: int = None,
) -> List[Dict]:
    """
    Get the most uncertain patches across all predicted images.
    Args:
        aggregated_uncertainty_dir (Path): directory containing the aggregated uncertainties for ranking the patches
        uncertainty_type (str): uncertainty type that should be used for ranking (specified in uncertainty file name)
        raw_dataset_dir (Path): directory with the raw input images
        ignore_label (int): label value signaling unlabeled regions
        file_ending (str): file ending of the images
        number_to_query (int): number of patches to query. If None, all patches from the input images will be ranked

    Returns:
        List[Dict]: Either the top n most uncertain patches if number_to_query is specified
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
            # Get the image id to find the image in the original data folder of the training images
            image_id = image_id_from_aggregated_name(image_name, uncertainty_type)
            # Selected array is an array of the raw input image size that is used to mark which areas have already been queried
            labeled_map = get_label_map(image_id, raw_dataset_dir, file_ending)
            selected_array = np.zeros_like(labeled_map)
            # Mark the patched as annotated that were annotated in previous loops
            selected_array = mark_already_annotated_patches(
                selected_array, labeled_map, ignore_label
            )
            # Extend the list of patches that can possibly be queried by number_to_query from the current image
            # TODO: this might be optimized, i.e. we don't need to query n uncertain patches from an image
            #  when we already have more uncertain patches from other images
            all_top_patches.extend(
                get_top_n_non_overlapping_patches(
                    image_name,
                    number_to_query,
                    uncertainty_scores,
                    patch_size,
                    selected_array,
                )
            )
    print(len(all_top_patches))
    # Sort the patches across the different images according to their uncertainty score
    all_top_patches = sorted(all_top_patches, key=lambda d: d["score"], reverse=True)
    if number_to_query is not None:
        # Only return the top number_to_query patches
        all_top_patches = all_top_patches[:number_to_query]
    print(len(all_top_patches))
    # bring all_top_patches in a json write and readable format

    for i in range(len(all_top_patches)):
        all_top_patches[i]["coords"] = [x.item() for x in all_top_patches[i]["coords"]]
        all_top_patches[i]["size"] = all_top_patches[i]["size"].tolist()
    return all_top_patches


def main() -> None:
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
    # Not necessary -o captures this!
    # parser.add_argument(
    #     "-r",
    #     "--raw_image_folder",
    #     type=str,
    #     required=False,
    #     help="Path with the raw input images that are partially annotated. "
    #     "If not specified, the nnU-Net raw imagesTr will be used.",
    # )
    parser.add_argument(
        "-n",
        "--number_to_query",
        type=int,
        required=False,
        default=None,
        help="Number of patches to query. If None, returns all non-overlapping patches sorted by uncertainty",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Output directory for loop_xxx.json and uncertainty_xxx.json file. Use folder in $\{nnUNet_raw\}/task.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        type=int,
        default=1,
        help="XXX specifier for loop_XXX.json of saved files.",
    )

    args = parser.parse_args()
    aggregated_uncertainty_dir = Path(args.input_dir)
    uncertainty_type = args.uncertainty_type
    number_to_query = args.number_to_query
    raw_dataset_dir = Path(args.output_path)
    loop = args.loop

    if uncertainty_type not in [
        "expected_entropy",
        "predictive_entropy",
        "mutual_information",
    ]:
        assert (
            "Unkown uncertainty type. Uncertainty type has to be one of the following: "
            "[expected_entropy | predictive_entropy | mutual_information]"
        )

    raw_dataset_dir = Path(raw_dataset_dir)
    with open(raw_dataset_dir / "dataset.json", "r") as file:
        dataset_json = json.load(file)
    file_ending = dataset_json["file_ending"]
    ignore_label = dataset_json["labels"]["ignore"]

    query_most_uncertain_patches(
        aggregated_uncertainty_dir,
        uncertainty_type,
        number_to_query,
        raw_dataset_dir,
        loop,
        file_ending,
        ignore_label,
    )


def query_most_uncertain_patches(
    aggregated_uncertainty_dir: Path,
    uncertainty_type: str,
    number_to_query: int,
    raw_dataset_dir: Path,
    loop: int,
    file_ending: str,
    ignore_label: int,
) -> None:
    all_top_patches = get_most_uncertain_patches(
        aggregated_uncertainty_dir,
        uncertainty_type,
        raw_dataset_dir,
        ignore_label,
        file_ending,
        number_to_query,
    )

    patches = [
        {
            "file": patch["file"].split(f"_{uncertainty_type}")[0] + file_ending,
            "coords": patch["coords"],
            "size": patch["size"],
        }
        for patch in all_top_patches
    ]
    patches = [Patch(**patch) for patch in patches]

    # Save the queries with uncertainty values
    with open(raw_dataset_dir / f"{uncertainty_type}_{loop:03d}.json", "w") as file:
        json.dump(all_top_patches, file, indent=4)

    # bring into loop_XXX.json format and save!
    loop_json = {"patches": patches}

    save_loop(raw_dataset_dir, loop_json, loop)


if __name__ == "__main__":
    main()
