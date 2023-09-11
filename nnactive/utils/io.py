import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from pydantic import dataclasses

from nnactive.utils.pyutils import get_clean_dataclass_dict


def save_json(save_json: Any, save_path: Path):
    with open(save_path, "w") as file:
        json.dump(save_json, file, indent=4)


def load_label_map(image_id: str, data_path: Path, file_ending: str) -> np.ndarray:
    image_path = data_path / f"{image_id}{file_ending}"
    sitk_image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(sitk_image)


def save_dataclass_to_json(data: dataclasses, filepath: Path):
    """Saves the dataclass as json file without __keys__

    Args:
        data (dataclasses): _description_
        filepath (Path): _description_
    """
    with open(filepath, "w") as file:
        datadict = get_clean_dataclass_dict(data)
        json.dump(datadict, file, indent=4)
