from pathlib import Path
from pprint import pprint

import blosc2
import numpy as np
from matplotlib import pyplot as plt

from nnactive.utils.io import load_pickle


def load_blosc(filepath: Path):
    data = blosc2.open(urlpath=filepath, mode="r")
    return data


def load_preprocessed(name: str, path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    img = load_blosc(path / f"{name}.b2nd")
    seg = load_blosc(path / f"{name}_seg.b2nd")
    data_dict = load_pickle(path / f"{name}.pkl")
    return img, seg, data_dict


def show_preprocessed(name: str, path: Path):
    img, seg, data_dict = load_preprocessed(name, path)
    unique, counts = np.unique(seg, return_counts=True)
    print("Segmentation:")
    print("Segmentation shape:", seg.shape)
    for i in range(len(unique)):
        print(f"Class {unique[i]}: {counts[i]}")
    print("----------------")
    min_val, max_val = np.min(img), np.max(img)
    print("Image:")
    for i in range(img.shape[0]):
        min_val, max_val = np.min(img[i]), np.max(img[i])
        print(f"min/max channel {i}: {min_val}/{max_val}")
    print("Image shape:", img.shape)
    print("----------------")
    print("Data dictionary:")
    pprint(data_dict)
