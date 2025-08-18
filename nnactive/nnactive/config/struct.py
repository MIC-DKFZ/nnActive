from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Union

import nnunetv2.paths
from loguru import logger
from pydantic.dataclasses import dataclass

import nnactive.paths
from nnactive.nnunet.utils import convert_id_to_dataset_name
from nnactive.results.utils import get_results_folder
from nnactive.utils.io import save_dataclass_to_json
from nnactive.utils.pyutils import get_clean_dataclass_dict


@dataclass
class ActiveConfig:
    patch_size: Union[tuple[int, int, int], str]  # what is the patch size to query?
    base_id: int = 0
    starting_budget: str = "random"  # how was starting budget created?
    starting_budget_size: int | None = None
    trainer: str = "nnActiveTrainer_200epochs"  # e.g. nnUNetDebugTrainer
    model_plans: str = "nnUNetPlans"
    model_config: str = "3d_fullres"  # 3d_fullres
    uncertainty: str = (
        "random"  # mutual_information TODO: rename this to query_strategy!
    )
    queries_from_experiment: str | None = (
        None  # use pre-computed queries from this experiment
    )
    aggregation: str = "patch"  # patch Currently holds no meaning
    query_size: int = 20  # how many samples are queried
    query_steps: int = 10  # how many query steps are supposed to be made
    agg_stride: int | list[int] = 1  # stride for the aggregation function
    n_patch_per_image: int | None = (
        None  # how many potential queries per image are allowed
    )
    seed: int = 12345  # seed to be used for everything random in the experiment
    full_folds: int = 5  # the amount of folds used in the split
    train_folds: int | None = None  # if specified, use subset of folds
    dataset: str = "Dataset Identifier"  # required!
    pre_suffix: str = ""
    use_mirroring: bool = False  # use mirroring during query prediction
    use_gaussian: bool = True  # use gaussian during query predition
    tile_step_size: float = 0.75  # %of patch step size per dim in query prediction
    patch_overlap: float = 0  # how much overlap is allowed for patchs
    additional_overlap: float = 0.4  # how much overlap is allowed with cost free annotated regions e.g. BraTS air areas
    pred_tile_step_size: float = 0.5  # step size for validation predictions
    disable_pred_tta: bool = False  # disable TTA for validation predictions

    def __post_init__(self):
        if self.n_patch_per_image is None:
            self.n_patch_per_image = self.query_size
        if self.starting_budget_size is None:
            self.starting_budget_size = self.query_size
        if self.train_folds is None:
            self.train_folds = self.full_folds

    def set_nnunet_env(self):
        experiment_path = self.group_dir()
        os.environ["nnUNet_raw"] = str(experiment_path / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(experiment_path / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(experiment_path / "nnUNet_results")
        nnunetv2.paths.set_paths(
            nnUNet_raw=str(experiment_path / "nnUNet_raw"),
            nnUNet_preprocessed=str(experiment_path / "nnUNet_preprocessed"),
            nnUNet_results=str(experiment_path / "nnUNet_results"),
        )
        nnactive.paths.set_paths(
            nnActive_results=self.group_results_dir(),
        )

    def name(self) -> str:
        dataset = self.dataset.replace(f"Dataset{self.base_id:03}_", "")
        return f"{dataset}{self.pre_suffix}__unc-{self.uncertainty}__seed-{self.seed}"

    def group_dir(self) -> Path:
        assert nnactive.paths.nnActive_data is not None
        return nnactive.paths.nnActive_data / self.dataset

    def group_results_dir(self) -> Path:
        assert nnactive.paths.nnActive_results is not None
        return nnactive.paths.base_nnActive_results / self.dataset / "nnActive_results"

    @classmethod
    def from_json(cls, path: Path) -> ActiveConfig:
        with open(path, "r") as file:
            parsed = json.load(file)
        return ActiveConfig(**parsed)

    @classmethod
    def get_from_id(cls, dataset_id: int) -> ActiveConfig:
        fn = get_results_folder(dataset_id) / cls.filename()
        return ActiveConfig.from_json(fn)

    def to_dict(self) -> dict[str, Any]:
        return get_clean_dataclass_dict(self)

    def to_str_dict(self) -> dict[str, str]:
        temp_dict = self.to_dict()
        for key in temp_dict:
            if not isinstance(temp_dict[key], str):
                if isinstance(temp_dict[key], (float, int, complex)):
                    temp_dict[key] = str(temp_dict[key])
                elif isinstance(temp_dict[key], Iterable):
                    temp_dict[key] = "_".join(map(lambda x: str(x), temp_dict[key]))
                elif temp_dict[key] is None:
                    temp_dict[key] = "None"
                else:
                    raise NotImplementedError(
                        f"KV {key}:{temp_dict[key]} is of a not supported type."
                    )
        return temp_dict

    def set_pre_suffix(self, pre_suffix: str):
        """Set config.pre_suffix with a format on all values in config."""
        self.pre_suffix = pre_suffix.format(**self.to_str_dict())

    def save_id(self, dataset_id: int):
        try:
            save_path: Path = get_results_folder(dataset_id) / self.filename()
        except FileNotFoundError:
            save_path: (
                Path
            ) = nnactive.paths.get_nnActive_results() / convert_id_to_dataset_name(
                dataset_id
            )
            logger.info(f"Creating Path: {save_path}")
            save_path.mkdir()
            save_path = save_path / self.filename()
        logger.info(f"Saving Config File to {save_path}")

        save_dataclass_to_json(self, save_path)

    @staticmethod
    def filename() -> str:
        return "config.json"


@dataclass
class RuntimeConfig:
    num_processes: int = 4
    n_gpus: int = 0
    max_ram_pred_query: int | float = 25
    save_probs: bool = False
    max_loops: int | None = (
        None  # how many loops should be run max None means run until query_steps
    )
    # TODO optionally, save predictions (requires efficient computation of mean probs
    #      across folds)
    # save_preds: bool = False

    def to_dict(self) -> dict[str, Any]:
        return get_clean_dataclass_dict(self)


@dataclass
class Final:
    final: bool = True
    finished: bool = True  # only for now
    note: str = ""

    @classmethod
    def from_json(cls, path: Path) -> Final:
        try:
            with open(path, "r") as file:
                parsed = json.load(file)
        except FileNotFoundError:
            # print(f"No file found for: {path}")
            parsed = {}
        return Final(**parsed)

    @classmethod
    def get_from_id(cls, dataset_id: int) -> Final:
        fn = get_results_folder(dataset_id) / cls.filename()
        return Final.from_json(fn)

    def to_dict(self):
        return get_clean_dataclass_dict(self)

    @staticmethod
    def filename():
        return "final.json"
