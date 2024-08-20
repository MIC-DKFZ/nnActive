import glob
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.data import Patch
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_sorted_loop_files, save_loop
from nnactive.masking import does_overlap
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.paths import get_nnActive_results
from nnactive.utils import create_mitk_geometry_patch
from nnactive.utils.patches import create_patch_mask_for_image


def get_correct_patch_size(data_path: Path):
    """
    Infer the correct patch size from the nnActive results config
    Args:
        data_path (Path): the raw dataset path

    Returns:
        Tuple[int, int, int]: the correct patch size
    """
    nnActive_results_base = get_nnActive_results()
    dataset_name = data_path.name
    nnActive_results_path = nnActive_results_base / dataset_name
    config_json_path = nnActive_results_path / "config.json"
    with open(config_json_path, "r") as f:
        config_json = json.load(f)
    patch_size = config_json["patch_size"]
    if len(patch_size) == 3:
        patch_size.reverse()
    return tuple(patch_size)


def crop_to_correct_size(
    patch: sitk.Image,
    original_image: sitk.Image,
    correct_patch_size: Tuple[int, int, int],
):
    """
    Crop the patch to the correct size if it has not the desired size
    Args:
        patch (sitk.Image): original patch of incorrect size
        original_image (sitk.Image): original full size image where the patch is selected from
        correct_patch_size (Tuple[int, int, int]): the desired correct patch size

    Returns:
        sitk.Image: the correctly cropped patch
    """
    # Get the size of the input image
    image_size = original_image.GetSize()

    patch_origin_index = list(
        original_image.TransformPhysicalPointToIndex(patch.GetOrigin())
    )
    # Adjust the ROI index if necessary to ensure that the ROI is fully contained within the image
    for i in range(len(correct_patch_size)):
        if patch_origin_index[i] + correct_patch_size[i] > image_size[i]:
            patch_origin_index[i] = image_size[i] - correct_patch_size[i]

    # Define the region of interest
    roi = sitk.Image(correct_patch_size, original_image.GetPixelID())
    roi.SetOrigin(original_image.TransformIndexToPhysicalPoint(patch_origin_index))

    # Apply the RegionOfInterest filter
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(correct_patch_size)
    roi_filter.SetIndex(patch_origin_index)
    cropped_image = roi_filter.Execute(original_image)
    return cropped_image


def save_mitk_geometry_prelim(
    all_patches_list: List[sitk.Image],
    patch_names_list: List[str],
    cropped_path: Path,
    image_id: str,
):
    """
    Save the overlapping patches preliminary as .mitkgeometry files to correct them
    Args:
        all_patches_list (List[sitk.Image]): List of all patches as sitk images
        patch_names_list (List[str]): List with the file names of the patches
        cropped_path (Path): Path to the manually selected patches
        image_id (str): Image id of the original image
    """
    for patch, name in zip(all_patches_list, patch_names_list):
        save_dir = cropped_path / "prelim_patches" / image_id
        os.makedirs(save_dir, exist_ok=True)
        create_mitk_geometry_patch.main(
            save_dir / f"{name}.mitkgeometry",
            size=patch.GetSize(),
            origin=patch.GetOrigin(),
            direction=patch.GetDirection(),
        )


def save_segmentation(
    patches: List[Patch],
    original_image: sitk.Image,
    image_id: str,
    file_ending: str,
    patch_seg_path: Path,
):
    """
    Save the patches as segmentation map. For debugging purposes only
    Args:
        patches: The patches to save
        original_image: The original image in the raw data
        image_id: The image id
        file_ending: The file ending of the image
        patch_seg_path: The path where to store the segmentation map

    """
    org_image_size = list(original_image.GetSize())
    if len(org_image_size) == 3:
        org_image_size.reverse()

    # Create mask for whole patch
    mask = create_patch_mask_for_image(
        image_id + file_ending,
        patches,
        org_image_size,
        identify_patch=True,
    )
    # Create mask for inner patch without boundary
    mask_bound = create_patch_mask_for_image(
        image_id + file_ending,
        patches,
        org_image_size,
        identify_patch=True,
        size_offset=-1,
    )
    # Subtract inner from outer to only have boundary
    mask_bound = mask - mask_bound
    # Save image with sitk
    mask_save = sitk.GetImageFromArray(mask_bound)
    mask_save = copy_geometry_sitk(mask_save, original_image)
    os.makedirs(patch_seg_path, exist_ok=True)
    sitk.WriteImage(
        mask_save,
        patch_seg_path / f"{image_id}{file_ending}",
    )


def get_file_patch_list(
    original_image_path: Path,
    cropped_path: Path,
    patch_size_required: list[int],
    file_patches: List[Patch],
    debug: bool = False,
):
    """
    Get the patch list for a single image
    Args:
        original_image_path (Path): path of the whole input image
        cropped_path (Path): path of the manually selected, cropped images (patches)
        debug (bool): If true, stores the patches as a segmentation map

    Returns:
        List[Patch]: a list with the selected patches for this image.
                     May be empty if no patch was selected for this image.
        bool: whether patches for this image overlap
    """
    original_image = sitk.ReadImage(original_image_path)
    image_id = original_image_path.stem.split(".")[0]
    # remove channel suffix from image id
    image_id = "_".join(image_id.split("_")[:-1])
    file_ending = "".join(original_image_path.suffixes)
    # numpy array with size of original image that marks the already selected patches in the original
    org_image_size = list(original_image.GetSize())
    # Simple ITK indices are x, y, z while numpy indices are z, y, x
    if len(org_image_size) == 3:
        org_image_size.reverse()
    patch_seg = np.zeros(org_image_size)

    # List with patch objects for the current image
    patches_image_list = []
    # List with simple itk patche for the current image (e.g. to retrieve geometry information)
    patches_sitk_list = []
    # List with the file names of the patches for the current image
    patches_names_list = []
    # Indicate whether patches for the current image overlap. If yes, save the patches as .mitkgeometry files
    # to correct them afterwards
    save_preliminary = False
    # Filter the manual selected patches to contain the image name
    for patch_path in sorted(glob.glob(f"{str(cropped_path)}/{image_id}*")):
        patch = sitk.ReadImage(patch_path)
        # Check if the selected patch has the correct size and crop to correct size if this is not the case
        if patch.GetSize() != patch_size_required:
            logger.info(
                f"Patch did not have correct size {patch_size_required}, but {patch.GetSize()}. "
                f"Cropping the patch using the same origin but the correct size..."
            )
            patch = crop_to_correct_size(
                patch=patch,
                original_image=original_image,
                correct_patch_size=patch_size_required,
            )
        # the origin of the selected patch is specified in world coordinates and needs to be transformed to indices
        patch_location = list(
            original_image.TransformPhysicalPointToIndex(patch.GetOrigin())
        )
        patch_size = list(patch.GetSize())
        if len(patch_location) == 3:
            patch_location.reverse()
            patch_size.reverse()
        slices = []
        patch_location = [int(o) for o in patch_location]
        patch_size = [int(s) for s in patch_size]
        for start_index, size in zip(patch_location, patch_size):
            slices.append(slice(start_index, start_index + size))

        # check if patch overlaps with previous patches
        ipatch = Patch(
            file=image_id + file_ending, coords=patch_location, size=patch_size
        )
        if not does_overlap(ipatch, file_patches):
            file_patches.append(ipatch)
        else:
            logger.error(
                f"Error for file {image_id + file_ending}. Patch does already overlap with a previous patch."
                f"Please annotate this case again."
            )
            save_preliminary = True
        patches_image_list.append(ipatch)
        patches_sitk_list.append(patch)
        patches_names_list.append(Path(patch_path).stem.split(".")[0])
    # If patches overlap for this image, save the .mitkgeometry files to correct the patches afterward
    if save_preliminary:
        save_mitk_geometry_prelim(
            patches_sitk_list,
            patches_names_list,
            cropped_path,
            image_id,
        )

    # Save as segmentation map for debugging purposes
    if not save_preliminary and debug and len(patches_image_list) > 0:
        patch_seg_path = cropped_path / "patches_seg"
        save_segmentation(
            patches_image_list, original_image, image_id, file_ending, patch_seg_path
        )
    return patches_image_list, save_preliminary
