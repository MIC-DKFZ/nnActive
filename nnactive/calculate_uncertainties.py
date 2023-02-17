"""
Script for calculating the uncertainties based on the nnU-Net ensemble predictions.
Current assumed folder structure of the predictions:
    /base-path-of-pred/fold_<0-4>/
Creates folder containing uncertainties in:
    /base-path-of-pred/uncertainties/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Set

import numpy as np
import SimpleITK as sitk
import torch


def get_predicted_image_names(pred_path: Path) -> Set[str]:
    """
    Get the names of the predicted images, checking if the image is predicted for all 5 folds

    Args:
        pred_path (Path) : base path of the predictions

    Returns:
        Set[str]: set containing the file names of the predicted images (.npz files with softmax probabilities)
    """
    softmax_files = set()

    # check which predictions exists for first fold
    fold_0_path = os.path.join(pred_path, "fold_0")
    for img_name in os.listdir(fold_0_path):
        if img_name.endswith("npz"):
            softmax_files.add(img_name)

    # check which predictions exist in other folds
    for fold in range(1, 5):
        fold_path = os.path.join(pred_path, f"fold_{fold}")
        softmax_files_fold = set()
        for img_name in os.listdir(fold_path):
            if img_name.endswith("npz"):
                softmax_files_fold.add(img_name)
        # remove all files where there is not a prediction for every fold
        softmax_files = softmax_files.intersection(softmax_files_fold)

    return softmax_files


def load_softmax_predictions(softmax_file_name: str, pred_path: Path) -> torch.Tensor:
    """
    Load the softmax predictions of one image for all folds into one tensor

    Args:
        softmax_file_name (str): name of the softmax prediction file (.npz file)
        pred_path (Path): base path of the predictions

    Returns:
        torch.Tensor: tensor containing the softmax predictions with the shape [n_folds, n_classes, image_shape]
    """
    # use softmax prediction from first fold to determine size of the final tensor
    n_preds = 5
    fold_0_image = torch.from_numpy(
        np.load(os.path.join(pred_path, "fold_0", softmax_file_name))[
            "probabilities.npy"
        ]
    )
    softmax_preds = torch.zeros(n_preds, *fold_0_image.shape)
    softmax_preds[0] = fold_0_image

    # fill the tensor with the softmax predictions from the remaining folds
    for fold in range(1, n_preds):
        fold_image = torch.from_numpy(
            np.load(os.path.join(pred_path, f"fold_{fold}", softmax_file_name))[
                "probabilities.npy"
            ]
        )
        softmax_preds[fold] = fold_image
    return softmax_preds


def get_pred_image_info(
    softmax_file_name: str, pred_path: Path
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Get the image information (origin, spacing, direction) from a predicted segmentation.
    Currently uses the predicted segmentation from the first fold to retrieve the information.
    This information is needed to correctly convert the uncertainty map from an numpy array to an image.

    Args:
        softmax_file_name (str): name of the softmax prediction file (.npz file)
        pred_path (Path): base path of the predictions

    Returns:
        pred_origin, pred_spacing, pred_direction: origin, spacing and direction of the segmentation image
    """
    # TODO: (refactor) the simple itk info is stored in the corresponding .pkl file, probably only use this
    fold_0_path = os.path.join(pred_path, "fold_0")
    with open(os.path.join(fold_0_path, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    file_ending_pred = dataset_json["file_ending"]
    pred_path = os.path.join(
        fold_0_path, f"{softmax_file_name.split('.')[0]}{file_ending_pred}"
    )
    pred_image = sitk.ReadImage(pred_path)
    pred_origin = pred_image.GetOrigin()
    pred_spacing = pred_image.GetSpacing()
    pred_direction = np.array(pred_image.GetDirection())
    return pred_origin, pred_spacing, pred_direction


def calculate_uncertainties(
    softmax_file_names: Set[str], pred_path: Path, target_path: Path
) -> None:
    """
    Calculate the predictive entropy, expected entropy and the mutual information for all predicted images.
    Currently, one uncertainty map per uncertainty type is calculated for each prediction (no class-wise uncertainties)

    Args:
        softmax_file_names (Set[str]): the file names from the images that were predicted
        pred_path (Path): the root path of the predictions
    """
    for image_name in softmax_file_names:
        # load the softmax predictions and calculate the mean
        softmax_preds = load_softmax_predictions(image_name, pred_path)
        mean_softmax = torch.mean(softmax_preds, dim=0)

        # calculate the predictive entropy
        pred_entropy = torch.zeros(*mean_softmax.shape[1:])
        for y in range(mean_softmax.shape[0]):
            pred_entropy += mean_softmax[y] * torch.log(mean_softmax[y])
        pred_entropy *= -1
        pred_entropy_image = sitk.GetImageFromArray(pred_entropy.numpy())

        # calculate the expected entropy
        expected_entropy = torch.zeros(softmax_preds.shape[0], *softmax_preds.shape[2:])
        for pred in range(softmax_preds.shape[0]):
            entropy = torch.zeros(*softmax_preds.shape[2:])
            for y in range(softmax_preds.shape[1]):
                entropy += softmax_preds[pred, y] * torch.log(softmax_preds[pred, y])
            entropy *= -1
            expected_entropy[pred] = entropy
        expected_entropy = torch.mean(expected_entropy, dim=0)
        expected_entropy_image = sitk.GetImageFromArray(expected_entropy.numpy())

        # calculate the mutual information
        mutual_information = pred_entropy - expected_entropy
        mutual_information_image = sitk.GetImageFromArray(mutual_information.numpy())

        # get the information about the predicted segmentation with correct spacing etc.
        pred_origin, pred_spacing, pred_direction = get_pred_image_info(
            image_name, pred_path
        )

        # set the correct image information for the uncertainty maps
        for image in [
            pred_entropy_image,
            expected_entropy_image,
            mutual_information_image,
        ]:
            image.SetOrigin(pred_origin)
            image.SetSpacing(pred_spacing)
            image.SetDirection(pred_direction)

        # make an output directory to store the uncertainties
        os.makedirs(target_path, exist_ok=True)

        # save the uncertainty maps as images
        sitk.WriteImage(
            pred_entropy_image,
            target_path / f"{image_name.split('.')[0]}_pred_entropy.nii.gz",
        )
        sitk.WriteImage(
            expected_entropy_image,
            target_path / f"{image_name.split('.')[0]}_expected_entropy.nii.gz",
        )
        sitk.WriteImage(
            mutual_information_image,
            target_path / f"{image_name.split('.')[0]}_mutual_information.nii.gz",
        )


def calculate_uncertainties_from_softmax_preds() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Root folder containing the softmax predictions of each fold in subfolders /fold_<0-4>",
    )
    args = parser.parse_args()
    pred_folder = Path(args.p)
    target_path = pred_folder / "uncertainties"
    write_uncertainties_from_softmax_preds(pred_folder, target_path)


def write_uncertainties_from_softmax_preds(
    pred_folder: Path, target_path: Path
) -> None:
    image_names = get_predicted_image_names(pred_folder)
    calculate_uncertainties(image_names, pred_folder, target_path)


if __name__ == "__main__":
    calculate_uncertainties_from_softmax_preds()
