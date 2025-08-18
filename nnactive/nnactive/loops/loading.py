import json
import os
from pathlib import Path
from typing import Optional

from nnactive.data import Patch
from nnactive.utils.io import get_clean_dataclass_dict

LOOP_PATTERN = "loop_"


def get_patches_from_loop_files(
    data_path: Path, loop_val: Optional[int] = None
) -> list[Patch]:
    """Returns aggregated labeled patches of all loop_xxx.json files within loop_val

    Args:
        data_path (Path): path to datafolder with loop_xxx.json files
        loop_val (Optional[int], optional): int(xxx) to allow until corresponding file. Defaults to None.

    Returns:
        list[Patch]: see description
    """

    nested_patches = get_nested_patches_from_loop_files(data_path, loop_val)
    patches = []
    for patch in nested_patches:
        patches.extend(patch)
    return patches


def get_nested_patches_from_loop_files(
    data_path: Path, loop_val: Optional[int] = None
) -> list[list[Patch]]:
    """Returns list of labeled patches of all loop_xxx.json files with xxx<= loop_val

    Args:
        data_path (Path): path to datafolder with loop_xxx.json files
        loop_val (Optional[int], optional): int(xxx) to allow until corresponding file. Defaults to None.

    Returns:
        list[list[Patch]]: see description
    """

    loop_files = get_sorted_loop_files(data_path)

    # Take only loop_files up to a certain loop_{loop_val}.json
    if loop_val is not None:
        loop_files = [loop_files[i] for i in range(loop_val + 1)]
    # load info
    nested_patches = []
    for loop_file in loop_files:
        with open(data_path / loop_file, "r") as file:
            patches_loop: list[dict] = json.load(file)["patches"]
        nested_patches.append([Patch(**patch) for patch in patches_loop])
    return nested_patches


def get_sorted_loop_files(data_path: Path) -> list[str]:
    """Returns an ascending list of all loop file names in the data_path"""
    loop_files = []
    for file in os.listdir(data_path):
        if file[: len(LOOP_PATTERN)] == LOOP_PATTERN and file.endswith(".json"):
            loop_files.append(file)

    loop_files.sort(key=lambda x: int(x.split(LOOP_PATTERN)[1].split(".json")[0]))
    return loop_files


def get_current_loop(data_path: Path) -> int:
    loop_val = len(get_sorted_loop_files(data_path)) - 1
    assert loop_val >= 0
    return loop_val


def get_loop_patches(data_path: Path, loop_val: int = None) -> list[Patch]:
    """Returns patches in one loop, if loop_val is None, the most recent loop is selected."""
    if loop_val is None:
        loop_val = len(get_sorted_loop_files(data_path)) - 1
    with open(data_path / f"{LOOP_PATTERN}{loop_val:03d}.json", "r") as file:
        patches = json.load(file)["patches"]
    return [Patch(**patch) for patch in patches]


def save_loop(output_path: Path, loop_json: dict, loop_val: int):
    assert isinstance(loop_json["patches"], list)
    if len(loop_json["patches"]) > 0:
        assert isinstance(loop_json["patches"][0], Patch)
    save_json = loop_json.copy()
    save_json["patches"] = [
        get_clean_dataclass_dict(patch) for patch in save_json["patches"]
    ]

    with open(output_path / f"{LOOP_PATTERN}{loop_val:03d}.json", "w") as file:
        json.dump(save_json, file, indent=4)
