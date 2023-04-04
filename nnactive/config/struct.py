from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pydantic.dataclasses import dataclass

from nnactive.results.utils import get_results_folder

FILENAME = "config.json"


@dataclass
class ActiveConfig:
    patch_size: Union[tuple[int, int, int], str]  # what is the patch size to query?
    starting_budget: str = "standard"  # how was starting budget created?
    trainer: str = "nnUNetTrainer_200epochs"  # e.g. nnUNetDebugTrainer
    model_config: str = "3d_fullres"  # 3d_fullres
    uncertainty: str = "random"  # mutual_information
    aggregation: str = "patch"  # patch Currently holds no meaning
    query_size: int = 20  # how many samples are queried
    query_steps: int = 10  # how many query steps are supposed to be made
    seed: int = 12345  # seed to be used for everything random in the experiment
    num_processes: int = 4  # how many processes are used within nnU-Net
    # patch_size: Union[tuple[int, int, int], str]  # what is the patch size to query?

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
        print(f"Saving Config File to {save_path}")
        with open(save_path, "w") as file:
            json.dump(self.__dict__, file)

    @staticmethod
    def filename():
        return FILENAME
