import json
import os
from pathlib import Path
from typing import Optional

from nnactive.data import Patch

LOOP_PATTERN = "loop_"


def get_patches_from_loop_files(
    data_path: Path, loop_val: Optional[int] = None
) -> list[Patch]:
    """Returns aggregated labeled patches of all loop_xxx.json files within loop_val

    Args:
        data_path (Path): _description_
        loop_val (Optional[int], optional): _description_. Defaults to None.

    Returns:
        list[Patch]: _description_
    """

    loop_files = get_sorted_loop_files(data_path)

    # Take only loop_files up to a certain loop_{loop_val}.json
    if loop_val is not None:
        loop_files = [loop_files[i] for i in range(loop_val + 1)]
    # load info
    patches = []
    for loop_file in loop_files:
        with open(data_path / loop_file, "r") as file:
            patches_loop: list[dict] = json.load(file)["patches"]
        patches.extend(patches_loop)
    patches = [Patch(**patch) for patch in patches]
    return patches


def get_sorted_loop_files(data_path: Path) -> list[str]:
    """Returns a sorted list of all loop file names in the data_path"""
    loop_files = []
    for file in os.listdir(data_path):
        if file[: len(LOOP_PATTERN)] == LOOP_PATTERN:
            loop_files.append(file)

    loop_files.sort(key=lambda x: int(x.split(LOOP_PATTERN)[1].split(".json")[0]))
    return loop_files


def save_loop(output_path: Path, loop_json: dict, loop_val: int):
    assert isinstance(loop_json["patches"], list)
    if len(loop_json["patches"]) > 0:
        assert isinstance(loop_json["patches"][0], Patch)
    save_json = loop_json.copy()
    # TODO: SAVING DATACLASS Delete pydantic key value pairs here
    save_json["patches"] = [patch.__dict__ for patch in save_json["patches"]]

    with open(output_path / f"{LOOP_PATTERN}{loop_val:03d}.json", "w") as file:
        json.dump(save_json, file, indent=4)
