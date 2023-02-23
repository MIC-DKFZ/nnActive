from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pydantic.dataclasses import dataclass

from nnactive.results.utils import get_results_folder

FILENAME = "config.json"


@dataclass
class ActiveConfig:
    starting_budget: str  # how was starting budget created?
    trainer: str  # e.g. nnUNetDebugTrainer
    model: str  # 3d_fullres
    uncertainty: str  # mutual_information
    aggregation: str  # patch Currently holds no meaning
    query_size: int  # how many samples are queried
    patch_size: Union[tuple[int, int, int], str]  # what is the patch size to query?

    @classmethod
    def from_json(cls, path: Path) -> ActiveConfig:
        with open(path, "r") as file:
            parsed = json.load(file)
        return ActiveConfig(**parsed)

    @classmethod
    def get_from_id(cls, id: int) -> ActiveConfig:
        fn = get_results_folder(id) / FILENAME
        return ActiveConfig.from_json(fn)

    def save_id(self, id: int):
        save_path: Path = get_results_folder(id) / FILENAME
        with open(save_path, "w") as file:
            json.dump(self.__dict__, file)
