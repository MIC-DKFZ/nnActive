import json
import os
from pathlib import Path

import nnunetv2.paths
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


def get_preprocessed_path(dataset_id: int) -> Path:
    return Path(nnunetv2.paths.nnUNet_preprocessed) / convert_id_to_dataset_name(
        dataset_id
    )


def get_raw_path(dataset_id: int) -> Path:
    return Path(nnunetv2.paths.nnUNet_raw) / convert_id_to_dataset_name(dataset_id)


def get_results_path(dataset_id: int) -> Path:
    return Path(nnunetv2.paths.nnUNet_results) / convert_id_to_dataset_name(dataset_id)


def read_dataset_json(dataset_id: int):
    with open(
        Path(nnunetv2.paths.nnUNet_raw)
        / convert_id_to_dataset_name(dataset_id)
        / "dataset.json"
    ) as file:
        dataset_json = json.load(file)
    return dataset_json


def get_patch_size(dataset_id: int, config: str = "3d_fullres"):
    with open(
        Path(nnunetv2.paths.nnUNet_preprocessed)
        / convert_id_to_dataset_name(dataset_id)
        / "nnUNetPlans.json"
    ) as file:
        plans = json.load(file)
    return plans["configurations"][config]["patch_size"]
