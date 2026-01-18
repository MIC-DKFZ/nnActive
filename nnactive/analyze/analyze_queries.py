from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type

import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.data.utils import copy_geometry_sitk
from nnactive.nnunet.predict import predict_from_model_folder
from nnactive.nnunet.utils import get_raw_path, get_results_path
from nnactive.results.state import State
from nnactive.strategies.bald import BALD
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.strategies.random import Random
from nnactive.strategies.randomlabel import RandomLabel
from nnactive.utils.io import get_clean_dataclass_dict, load_json, save_json
from nnactive.utils.patches import create_patch_mask_for_image

QUERY_METHODS: list[Type[AbstractQueryMethod]] = [
    BALD,
    # RandomLabel,
    PredictiveEntropy,
]


class AnalyzeQueries:
    def __init__(
        self,
        config: ActiveConfig,
        dataset_id: int | None,
        loop_val: int | None = None,
    ):
        """Class Analyzing Queries using nnActive Structure.
        Can only be executed when experiment info is stored and accessible from Paths.

        Args:
            config (ActiveConfig): Config for experiment
            dataset_id (int | None): dataset id, if None use most recent
            loop_val (int | None, optional): loop to verify, if None use most recent. Defaults to None.
        """
        self.config = config
        self.config.set_nnunet_env()
        self.dataset_id = dataset_id
        self.loop_val = loop_val
        if dataset_id is None:
            state = State.latest(config)
            dataset_id = state.dataset_id

        self.initialize_querymethods(query_methods=QUERY_METHODS)

    def initialize_querymethods(self, query_methods: list[Type[AbstractQueryMethod]]):
        self.query_methods: dict[str, AbstractQueryMethod] = {
            cls_.__name__: cls_.init_from_dataset_id(
                self.config,
                dataset_id=self.dataset_id,
                loop_val=self.loop_val,
                seed=self.loop_val + self.config.seed + 1,
            )
            for cls_ in query_methods
        }

    @property
    def file_ending(self):
        return load_json(self.raw_folder / "dataset.json")["file_ending"]

    @property
    def raw_folder(self) -> Path:
        return get_raw_path(self.dataset_id)

    @property
    def results_folder(self) -> Path:
        return get_results_path(self.dataset_id)

    @property
    def base_folder(self) -> Path:
        return self.raw_folder / "analysis" / f"loop_{self.loop_val:03d}"

    @property
    def probs_folders(self) -> list[Path]:
        return [
            self.base_folder / f"predTr_{f}" for f in range(self.config.train_folds)
        ]

    @property
    def patches_fname(self) -> str:
        return "final_patches.json"

    @property
    def scores_fname(self) -> str:
        return "all_scores.json"

    def get_patches(self) -> dict[str, list[Patch]]:
        fp = self.base_folder / self.patches_fname
        try:
            dict_dict: dict[str, list[dict[str, Any]]] = load_json(fp)
        except:
            raise FileNotFoundError(
                f"No file with name: {fp}. Run query_from_probs before."
            )
        out_dict = {}
        for key in dict_dict:
            out_dict[key] = [Patch(**p_dict) for p_dict in dict_dict[key]]
        return out_dict

    def get_scores(self) -> dict[str, dict[str, Any]]:
        fp = self.base_folder / self.scores_fname
        try:
            scores_dict = load_json(fp)
        except:
            raise FileNotFoundError(
                f"No file with name: {fp}. Run query_from_probs before."
            )
        for key in scores_dict:
            assert isinstance(scores_dict[key], dict)
        return scores_dict

    def probs_to_voxel_uncertainty(
        self, probs: torch.Tensor | list[Path] | np.ndarray, label_file: str
    ) -> dict[str, torch.Tensor]:
        uncertainty_dict = {}
        for qm_name, qm in self.query_methods.items():
            if isinstance(qm, AbstractUncertainQueryMethod):
                img_dict, _ = qm.query_file_from_dict(
                    {"probs": probs}, file_id=label_file
                )
                uncertainty_dict[qm_name] = img_dict["scores"].cpu()
                del _
                torch.cuda.empty_cache()

            elif isinstance(qm, Random):
                pass
            else:
                raise NotImplementedError
        return uncertainty_dict

    def get_final_queries(
        self,
    ) -> tuple[dict[str, list[Patch]], dict[str, list[dict[str, Any]]]]:
        final_query_patches: dict[str, list[Patch]] = {}
        scores_all: dict[str, list[dict[str, Any]]] = {}
        for qm_name, qm in self.query_methods.items():
            if isinstance(qm, AbstractUncertainQueryMethod):
                final_query_patches[qm_name] = qm.compose_query_of_patches()
                scores_all[qm_name] = qm.top_patches
            elif isinstance(qm, Random):
                continue
                final_query_patches[qm_name] = qm.query()
            else:
                raise NotImplementedError
        return final_query_patches, scores_all

    def predict_training_set_fold(
        self,
        folds: int | list[int] | str,
        npp: int = 3,
        nps: int = 3,
        disable_progress_bar: bool = False,
        num_parts: int = 1,
        part_id: int = 0,
        verbose: bool = False,
    ):
        results_folder_name = "__".join(
            [
                f"loop_{self.loop_val:03d}",
                self.config.trainer,
                self.config.model_plans,
                self.config.model_config,
            ]
        )
        model_folder = self.results_folder / results_folder_name
        if isinstance(folds, int):
            out_path = self.probs_folders[folds]
            folds = [folds]
        elif isinstance(folds, list):
            assert len(folds) == self.config.train_folds
            out_path = self.base_folder / "predTr"
        else:
            raise NotImplementedError

        predict_from_model_folder(
            str(self.raw_folder / "imagesTr"),
            str(out_path),
            model_folder=str(model_folder),
            folds=folds,
            step_size=self.config.tile_step_size,
            disable_tta=not self.config.use_mirroring,
            verbose=verbose,
            save_probabilities=True,
            continue_prediction=False,
            npp=npp,
            nps=nps,
            num_parts=num_parts,
            part_id=part_id,
            disable_progress_bar=disable_progress_bar,
        )

    def query_from_probs(self):
        fns = [f.name for f in self.probs_folders[0].iterdir() if f.suffix == ".npz"]
        probs_paths = [[bf / f for bf in self.probs_folders] for f in fns]
        for prob_paths in probs_paths:
            fn = prob_paths[0].name.split(".")[0]
            uncertainty_dict = self.probs_to_voxel_uncertainty(
                prob_paths, label_file=fn
            )
            for u_n in uncertainty_dict:
                nii_name = fn + self.file_ending
                nii_image = sitk.ReadImage(self.raw_folder / "labelsTr" / nii_name)
                save_image = sitk.GetImageFromArray(uncertainty_dict[u_n].numpy())
                save_image = copy_geometry_sitk(save_image, nii_image)
                if not (self.base_folder / u_n).is_dir():
                    os.makedirs(self.base_folder / u_n)
                sitk.WriteImage(save_image, self.base_folder / u_n / nii_name)

        final_query_patches, scores_all = self.get_final_queries()
        final_query_patches_json = {
            k: [get_clean_dataclass_dict(p) for p in final_query_patches[k]]
            for k in final_query_patches
        }
        save_json(
            final_query_patches_json,
            save_path=Path(self.base_folder / self.patches_fname),
        )
        save_json(scores_all, save_path=Path(self.base_folder / self.scores_fname))

    def visualize_from_query(self):
        final_query_patches = self.get_patches()
        img_names = [
            f.name
            for f in (self.raw_folder / "labelsTr").iterdir()
            if f.name.endswith(self.file_ending)
        ]

        for q_n in final_query_patches:
            save_folder = self.base_folder / f"query_{q_n}"
            if not save_folder.is_dir():
                os.makedirs(save_folder)
            logger.info(f"Saving Queries in Folder: {save_folder}")

        for img_name in img_names:
            for q_n, patches in final_query_patches.items():
                img_patches = [patch for patch in patches if patch.file == img_name]
                if len(img_patches) == 0:
                    continue
                save_folder = self.base_folder / f"query_{q_n}"

                img = sitk.ReadImage(self.raw_folder / "labelsTr" / img_name)
                label_shape = sitk.GetArrayFromImage(img).shape
                mask = create_patch_mask_for_image(
                    img_name, patches, label_shape, identify_patch=False
                )
                mask = sitk.GetImageFromArray(mask)
                mask = copy_geometry_sitk(mask, img)
                sitk.WriteImage(
                    mask,
                    (save_folder / img_name),
                )

    @classmethod
    def initialize_from_config_path(
        cls,
        results_path: Path,
        loop_val: int | None = None,
    ) -> AnalyzeQueries:
        results_path = (
            results_path / "config.json"
            if results_path.name != "config.json"
            else results_path
        )
        config = ActiveConfig.from_json(results_path)
        dataset_id = int(results_path.parent.name.split("_")[0][-3:])
        return cls(config, dataset_id, loop_val=loop_val)


def analyze_queries_from_probs(results_folder: Path, loop_val: int | None = None):
    """Simulate and visualize queries from probs.
    Probs are expected to be stored in structure:
    {experiment_raw_folder}/analysis/loop_{loop_val}/predTr_{fold}


    Args:
        results_folder (str): Path to exact experiment results with config.json.
        loop_val (int | None, optional): loop value. Defaults to None.
    """
    analysis = AnalyzeQueries.initialize_from_config_path(
        results_folder, loop_val=loop_val
    )
    analysis.query_from_probs()
    analysis.visualize_from_query()


def predict_trainingset_model(
    results_folder: Path,
    folds: int | list[int],
    loop_val: int | None = None,
    npp: int = 3,
    nps: int = 3,
    disable_progress_bar: bool = False,
    num_parts: int = 1,
    part_id: int = 0,
    verbose: bool = False,
):
    """Predict fold models on training set for specified loop and saves predictions and outputs.
    Outputs will be stored in structure:
    {experiment_raw_folder}/analysis/loop_{loop_val}/predTr_{fold}

    Args:
        results_folder (Path): Path to exact experiment results with config.json.
        folds (int | list[int]): folds for prediction (if list give all)
        loop_val (int | None, optional): loop value. Defaults to None.
        npp (int, optional): num processes preprocessing. Defaults to 3.
        nps (int, optional): num processes postprocessing. Defaults to 3.
        disable_progress_bar (bool, optional): useful on cluster. Defaults to False.
        num_parts (int, optional): splits prediction into multiple parts (filewise). Defaults to 1.
        part_id (int, optional): which part is executed (starts with 0). Defaults to 0.
        verbose (bool, optional): read a lot. Defaults to False.
    """
    analysis = AnalyzeQueries.initialize_from_config_path(
        results_folder, loop_val=loop_val
    )
    analysis.predict_training_set_fold(
        folds,
        npp,
        nps,
        disable_progress_bar,
        num_parts,
        part_id,
        verbose,
    )
