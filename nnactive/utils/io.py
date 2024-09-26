import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pydantic import dataclasses

from nnactive.utils.pyutils import get_clean_dataclass_dict


def load_json(save_path: Path) -> Any:
    with open(save_path, "r") as file:
        out = json.load(file)
    return out


def save_json(save_json: Any, save_path: Path):
    with open(save_path, "w") as file:
        json.dump(save_json, file, indent=4)


def load_label_map(image_id: str, data_path: Path, file_ending: str) -> np.ndarray:
    image_path = data_path / f"{image_id}{file_ending}"
    sitk_image = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(sitk_image)


def save_pickle(data: Any, filepath: Path):
    """Saves the data as pickle file as a binary file."""
    with open(filepath, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filepath: Path) -> Any:
    """Loads the data from a pickle file as binary."""
    with open(filepath, "rb") as file:
        return pickle.load(file)


def save_dataclass_to_json(data: dataclasses, filepath: Path):
    """Saves the dataclass as json file without __keys__

    Args:
        data (dataclasses): _description_
        filepath (Path): _description_
    """
    with open(filepath, "w") as file:
        datadict = get_clean_dataclass_dict(data)
        json.dump(datadict, file, indent=4)


def save_df_to_txt(df: pd.DataFrame, filepath: Path):
    """Saves the DataFrame as txt file

    Args:
        df (pd.DataFrame): _description_
        filepath (Path): _description_
    """
    with open(filepath, "w") as file:
        file.write(df.to_string())
