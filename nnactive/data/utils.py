import os
import shutil
from pathlib import Path

import SimpleITK as sitk

from nnactive.paths import nnActive_data


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
