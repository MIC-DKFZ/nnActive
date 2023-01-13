import argparse
import os
import numpy as np
from scipy.signal import convolve
from nnactive_playground.utils.image_reading import read_images_to_numpy


def save_aggregated_uncertainties(
    image_name, uncertainties_folder, patch_size, **kwargs
):
    """
    Save the aggregated uncertainties as .npz file

    Args:
        image_name : image name of the aggregated image. Used as basis for file name that is stored
        uncertainties_folder : input folder with uncertainties to aggregate. Results are stored in <uncertainties_folder>/aggregated_uncertainties
        patch_size : patch size of an input patch that is aggregated. Stored in the .npz file to make patches reconstructable
        **kwargs: Should contain all types of aggregated uncertainties with descriptive variable names (variable names will be keys in .npz file)
    """
    patch_size = np.array(patch_size)

    # create aggregation folder
    aggregation_folder = os.path.join(uncertainties_folder, "aggregated_uncertainties")
    os.makedirs(aggregation_folder, exist_ok=True)

    # save patch size and all aggregation maps passed by kwargs
    np.savez(
        os.path.join(aggregation_folder, f"{image_name.split('.')[0]}_aggregated.npz"),
        patch_size=patch_size,
        **kwargs,
    )


def whole_patch_aggregation(
    np_image, image_name, uncertainties_folder, patch_size, mean=True
):
    """
    Simply sum all values inside a patch as aggregation value (and possibly take the mean)

    Args:
        np_image : Image to aggregate as numpy array
        image_name : File name of the image that is aggregated
        uncertainties_folder : Folder with all the uncertainty images that is iterated
        patch_size : Patch size of an input patch that should be aggregated
        mean (bool, optional): If the mean of the patch should be scored, if False, the sum is stored. Defaults to True.
    """
    # Kernel with ones of patch size corresponds to taking the sum
    kernel = np.ones(patch_size)
    # Convolve to get patch wise score in sliding window fashion
    # Mode 'valid' means that the kernel does not move outside of the image region
    patch_wise_aggragated = convolve(np_image, kernel, mode="valid")
    # Take mean of each patch score if desired
    if mean:
        patch_wise_aggragated = patch_wise_aggragated / (np.prod(patch_size))
    # save the aggregated uncertainties
    save_aggregated_uncertainties(
        image_name, uncertainties_folder, patch_size, patch_score=patch_wise_aggragated
    )


def aggregate_uncertainties():
    """
    Aggregate uncertainty maps and store them as .npz files.
    The .npz files contain the patch size of one input patch that is aggregated and the aggregated uncertainties.
    The aggregated uncertainties are stored with a descriptive key and a numpy array containing the patch uncertainty values.
    The indices in these numpy arrays represent the start coordinates of the aggregated patches.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_json",
        type=str,
        required=True,
        help="Root folder containing the softmax predictions of each fold in subfolders /fold_<0-4>",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Root folder containing the softmax predictions of each fold in subfolders /fold_<0-4>",
    )
    args = parser.parse_args()
    dataset_json_path = args.dataset_json
    uncertainty_input_folder = args.input

    # read images to numpy and execute the aggregation function
    read_images_to_numpy(
        dataset_json_path,
        uncertainty_input_folder,
        whole_patch_aggregation,
        patch_size=(10, 10, 10),
    )


if __name__ == "__main__":
    aggregate_uncertainties()
