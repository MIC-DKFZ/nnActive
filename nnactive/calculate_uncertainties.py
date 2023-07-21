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

UNCERTAINTIES = ["pred_entropy", "mutual_information", "expected_entropy"]


def get_predicted_image_names(pred_path: Path, num_folds: int) -> Set[str]:
    """
    Get the names of the predicted images, checking if all images are predicted for each fold

    Args:
        pred_path (Path) : base path of the predictions

    Returns:
        Set[str]: set containing the file names of the predicted images (.npz files with softmax probabilities)
    """
    lengths = dict()
    softmax_files = dict()
    for fold in range(num_folds):
        fold_path = pred_path / (f"fold_{fold}")
        softmax_files[fold]: list = [
            file.name for file in fold_path.iterdir() if file.name.endswith("npz")
        ]
        lengths[fold]: int = len(softmax_files[fold])

    for fold in range(num_folds):
        if not all([lengths[fold] == lengths[fold_s] for fold_s in range(fold + 1)]):
            print(lengths)
            raise RuntimeError("Every fold must predict every case.")
    return softmax_files[0]


def load_softmax_predictions(
    softmax_file_name: str, pred_path: Path, num_folds: int = 5
) -> torch.Tensor:
    """
    Load the softmax predictions of one image for all folds into one tensor

    Args:
        softmax_file_name (str): name of the softmax prediction file (.npz file)
        pred_path (Path): base path of the predictions
        num_folds (int): amount of folds used for training/prediction

    Returns:
        torch.Tensor: tensor containing the softmax predictions with the shape [n_folds, n_classes, image_shape]
    """
    softmax_preds = []
    for fold in range(num_folds):
        softmax_preds.append(
            torch.from_numpy(
                np.load(pred_path / (f"fold_{fold}") / softmax_file_name)[
                    "probabilities.npy"
                ]
            )
        )
    softmax_preds = torch.stack(softmax_preds, 0)
    return softmax_preds


def get_pred_image_info(
    softmax_file_name: str, pred_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def get_uncertainty_image(
    pred_path: Path, num_folds: int, image_name: str, uncertainty: str
):
    out_dict: dict[str, torch.Tensor] = {}
    softmax_preds = load_softmax_predictions(
        image_name, pred_path, num_folds
    )  # M x C x ...
    if uncertainty in ["pred_entropy", "mutual_information", "expected_entropy", "all"]:
        mean_softmax = torch.mean(softmax_preds, dim=0)
        # calculate the predictive entropy
        pred_entropy = torch.zeros(*mean_softmax.shape[1:])
        pred_entropy = -1 * torch.sum(
            mean_softmax * torch.log(mean_softmax), dim=0
        )  # ...
        out_dict["pred_entropy"] = pred_entropy
        del pred_entropy
        del mean_softmax
        if uncertainty in ["mutual_information", "expected_entropy", "all"]:
            expected_entropy = -1 * torch.mean(
                torch.sum(softmax_preds * torch.log(softmax_preds), dim=1), dim=0
            )  # ...
            mutual_information = out_dict["pred_entropy"] - expected_entropy
            out_dict["expected_entropy"] = expected_entropy
            out_dict["mutual_information"] = mutual_information
            del expected_entropy
            del mutual_information
    del softmax_preds

    if uncertainty != "all":
        pop_keys = [key for key in out_dict if key != uncertainty]
        for key in pop_keys:
            out_dict.pop(key)
    return out_dict


def calculate_uncertainties(
    softmax_file_names: Set[str],
    pred_path: Path,
    target_path: Path,
    num_folds: int,
    uncertainty: str,
) -> None:
    """
    Calculate the predictive entropy, expected entropy and the mutual information for all predicted images.
    Currently, one uncertainty map per uncertainty type is calculated for each prediction (no class-wise uncertainties)

    Args:
        softmax_file_names (Set[str]): the file names from the images that were predicted
        pred_path (Path): the root path of the predictions
        num_folds (int): number of folds used for training/prediction
    """
    if uncertainty not in UNCERTAINTIES:
        raise RuntimeError(
            f"Uncertainty: {uncertainty} is not in allowed in set of {UNCERTAINTIES}"
        )
    # make an output directory to store the uncertainties
    os.makedirs(target_path, exist_ok=True)
    print("Start Calculating Uncertainties")
    for image_name in softmax_file_names:
        print(f"Calculate uncertainties for file: {image_name}")
        # load the softmax predictions and calculate the mean
        uncertainty_dict = get_uncertainty_image(
            pred_path, num_folds, image_name, uncertainty
        )
        pred_origin, pred_spacing, pred_direction = get_pred_image_info(
            image_name, pred_path
        )

        # get the information about the predicted segmentation with correct spacing etc.
        pred_origin, pred_spacing, pred_direction = get_pred_image_info(
            image_name, pred_path
        )

        for key in [_key for _key in uncertainty_dict.keys()]:
            write_image: sitk.Image = sitk.GetImageFromArray(
                uncertainty_dict.pop(key).numpy()
            )
            write_image.SetOrigin(pred_origin)
            write_image.SetSpacing(pred_spacing)
            write_image.SetDirection(pred_direction)
            sitk.WriteImage(
                write_image,
                target_path / f"{image_name.split('.')[0]}_{key}.nii.gz",
            )
        del uncertainty_dict
    print("End Calculating Uncertainties")


def calculate_uncertainties_from_softmax_preds() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Root folder containing the softmax predictions of each fold in subfolders /fold_<0-4>",
    )
    parser.add_argument(
        "-u",
        "--uncertainty",
        type=str,
        required=True,
        help=f"Uncertainty in {UNCERTAINTIES}",
    )
    args = parser.parse_args()
    pred_folder = Path(args.p)
    uncertainty = args.uncertainty
    num_folds = len(
        [
            folder.name
            for folder in pred_folder.iterdir()
            if folder.name.startswith("fold_")
        ]
    )
    target_path = pred_folder / "uncertainties"
    write_uncertainties_from_softmax_preds(
        pred_folder, target_path, num_folds, uncertainty
    )


def write_uncertainties_from_softmax_preds(
    pred_folder: Path, target_path: Path, num_folds: int, uncertainty: str
) -> None:
    image_names = get_predicted_image_names(pred_folder, num_folds)
    calculate_uncertainties(
        image_names, pred_folder, target_path, num_folds, uncertainty
    )


if __name__ == "__main__":
    calculate_uncertainties_from_softmax_preds()
