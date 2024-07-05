import os
from pathlib import Path
from typing import Type

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
    DatasetFingerprintExtractor,
)
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.nnunet.fingerprint_extractor import NNActiveDatasetFingerprintExtractor
from nnactive.paths import nnActive_data, set_raw_paths


def data_path():
    data_path = os.getenv("nnActive_raw")
    if data_path is None:
        raise ValueError("OS variable nnActive_raw is not set.")
    return Path(data_path)


def existing_dsets():
    existing_dsets = [
        folder.name
        for folder in nnActive_data.iterdir()
        if folder.is_dir() and folder.name.startswith("Dataset")
    ]
    return existing_dsets


def extract_dataset_fingerprint(
    dataset_id: int,
    fingerprint_extractor_class: Type[
        DatasetFingerprintExtractor
    ] = NNActiveDatasetFingerprintExtractor,
    num_processes: int = default_num_processes,
    check_dataset_integrity: bool = False,
    clean: bool = True,
    verbose: bool = True,
):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    with set_raw_paths():
        dataset_name = convert_id_to_dataset_name(dataset_id)
        print(dataset_name)

        if check_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

        fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)

        return fpe.run(overwrite_existing=clean)


def copy_geometry_sitk(target: sitk.Image, source: sitk.Image) -> sitk.Image:
    """Returns a version of target with origin, direction and spacing from source."""
    target.SetOrigin(source.GetOrigin())
    target.SetDirection(source.GetDirection())
    target.SetSpacing(source.GetSpacing())
    return target


def get_geometry_sitk(source: sitk.Image):
    out = {
        "origin": source.GetOrigin(),
        "direction": source.GetDirection(),
        "spacing": source.GetSpacing(),
    }
    return out


def set_geometry(target: sitk.Image, origin, direction, spacing):
    target.SetOrigin(origin)
    target.SetDirection(direction)
    target.SetSpacing(spacing)
    return target
