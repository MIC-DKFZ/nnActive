from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from loguru import logger
from pydantic.dataclasses import dataclass

from nnactive.nnunet.utils import convert_id_to_dataset_name
from nnactive.paths import get_nnActive_results
from nnactive.results.utils import get_results_folder
from nnactive.utils.io import save_dataclass_to_json
from nnactive.utils.pyutils import get_clean_dataclass_dict

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
    agg_stride: Union[int, list[int]] = 1  # stride for the aggregation function
    seed: int = 12345  # seed to be used for everything random in the experiment
    num_processes: int = 4  # how many processes are used within nnU-Net
    full_folds: int = 5  # the amount of folds used in the split
    train_folds: Union[int, None] = None  # if specified, use subset of folds
    dataset: str = "Dataset Identifier"
    add_uncertainty: str = ""  # e.g. --disable_tta
    add_validation: str = ""  # e.g. --disable_tta
    # overlap

    @classmethod
    def from_json(cls, path: Path) -> ActiveConfig:
        with open(path, "r") as file:
            parsed = json.load(file)
        return ActiveConfig(**parsed)

    @classmethod
    def get_from_id(cls, dataset_id: int) -> ActiveConfig:
        fn = get_results_folder(dataset_id) / FILENAME
        return ActiveConfig.from_json(fn)

    def to_dict(self):
        return get_clean_dataclass_dict(self)

    def save_id(self, dataset_id: int):
        try:
            save_path: Path = get_results_folder(dataset_id) / FILENAME
        except FileNotFoundError:
            save_path: Path = get_nnActive_results() / convert_id_to_dataset_name(
                dataset_id
            )
            logger.info(f"Creating Path: {save_path}")
            save_path.mkdir()
            save_path = save_path / FILENAME
        logger.info(f"Saving Config File to {save_path}")

        save_dataclass_to_json(self, save_path)

    @staticmethod
    def filename():
        return FILENAME

    @property
    def working_folds(self):
        return self.train_folds if self.train_folds is not None else self.full_folds
