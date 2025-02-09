from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from loguru import logger
from nnunetv2.utilities.file_path_utilities import get_output_folder

from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path, read_dataset_json
from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import Probs


class ClassBalancedEntropy_66FG(AbstractUncertainQueryMethod):
    def __post_init__(self):
        super().__post_init__()
        labels = self.dataset_json["labels"]
        get_label = lambda x: int(x[-1]) if isinstance(x, (tuple, list)) else int(x)
        self.top_patches = {
            get_label(labels[label]): []
            for label in labels
            if label not in ["background", "ignore"]
        }
        self.top_patches[-1] = []
        self.ratio_fg = 0.66

    @property
    def dataset_json(self):
        return read_dataset_json(self.dataset_id)

    def get_uncertainty(
        self,
        probs: list[Path] | torch.Tensor | Probs,
        device: torch.device = torch.device("cuda:0"),
    ) -> torch.Tensor:
        if not isinstance(probs, Probs):
            probs = Probs.create(probs)
        return probs.class_pred_entropy(probs, device)

    def compute_scores_per_cls(
        self, probs: np.ndarray | list[Path], device: torch.device
    ):
        with torch.no_grad():
            logger.debug("Compute uncertaintes...")
            uncertainty = self.get_uncertainty(probs, device=device)
            # uncertainties are (C+1 x H x W x D)
            # last channel is standard uncertainty (not class specific)
            # first C channels are class specific uncertainties
            logger.debug("Aggregate uncertainties...")
            agg_uncertainty_cls = {}
            for cls in self.top_patches:
                agg_uncertainty_cls[cls], kernel_size = self.aggregation.forward(
                    uncertainty[cls]
                )
        return uncertainty, agg_uncertainty_cls[cls], kernel_size

    def query_file_from_dict(
        self,
        query_dicts: list[dict[str, Any]],
        file_id: str,
        device: torch.device = torch.device("cuda:0"),
    ) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
        """Computes potential queries for a single input image and adds best queries to the internal list of queries.

        Args:
            query_dicts (list[dict[str, Any]]): each element in the list stands for one fold.
            file_id (str): name of label file without suffix
            device (_type_, optional): _description_. Defaults to torch.device("cuda:0").

        Returns:
            list[dict[str, Any]]: selection of potential queries for current file
        """
        with (
            monitor.timer("query_from_probs") if monitor.is_active() else nullcontext()
        ):
            annotated_patches = [
                patch
                for patch in self.annotated_patches
                if patch.file == file_id + ".nii.gz"
            ]
            image_dict, value_dicts_per_cls = self.strategy_per_cls(query_dicts, device)

            logger.info("Select patches...")
            classes = [cls for cls in self.top_patches]
            self.rng.shuffle(classes)
            for cls in classes:
                n_per_image = self.get_n_patch_per_image()
                if cls == -1:
                    n_per_image = n_per_image * (1 - self.ratio_fg)
                else:
                    n_per_image = n_per_image * (self.ratio_fg / len(self.top_patches))
                selected_cls_patches: list[dict] = self.select_file_patches(
                    value_dicts_per_cls[cls],
                    annotated_patches=annotated_patches,
                    label_file=file_id,
                    n=max(
                        int(n_per_image),
                        1,
                    ),
                )
                self.top_patches[cls] += selected_cls_patches
                annotated_patches.extend(
                    [
                        Patch(
                            file=patch["file"],
                            coords=patch["coords"],
                            size=patch["size"],
                        )
                        for patch in selected_cls_patches
                    ]
                )

        return image_dict, value_dicts_per_cls

    def strategy_per_cls(self, query_dicts, device):
        probs: list[np.ndarray] | list[Path] = [qd["probs"] for qd in query_dicts]
        if not isinstance(probs[0], (Path, str)):
            probs = torch.stack(probs)

        scores, agg_scores_per_cls, patch_size = self.compute_scores_per_cls(
            probs, device
        )
        out_list_per_cls = {}
        for cls in self.top_patches:
            sorted_uncertainty_indices, sorted_uncertainty_scores = self.get_top_scores(
                agg_scores_per_cls[cls]
            )
            out_list_per_cls[cls] = [
                {
                    "coords": self.aggregation.backward_index(
                        index, agg_scores_per_cls.shape
                    ),
                    "size": patch_size,
                    "score": score,
                }
                for score, index in zip(
                    sorted_uncertainty_scores, sorted_uncertainty_indices
                )
            ]
        file_dict = {"scores": scores}
        return file_dict, out_list_per_cls

    def query_part(
        self,
        part_id: int = 0,
        num_parts: int = 1,
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict]:
        temp_file_handler = self.get_data_handler(
            temp_path=get_raw_path(self.dataset_id) / f"temp_probs_part{part_id}",
            num_folds=self.config.train_folds,
            max_ram=self.max_ram_pred_query,
        )

        torch.cuda.set_device(device)
        # Initialize Predictor
        predictor = self.build_query_predictor(device)
        # Initialize Model for Predictor
        nnunet_plans_identifier = self.config.model_plans
        nnunet_trainer_name = self.config.trainer
        nnunet_config = self.config.model_config
        model_folder = get_output_folder(
            self.dataset_id, nnunet_trainer_name, nnunet_plans_identifier, nnunet_config
        )
        use_folds = tuple(range(self.config.train_folds))
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=use_folds
        )

        # TODO: check whether model_folder is needed here!
        model_folder = get_output_folder(
            self.dataset_id, nnunet_trainer_name, nnunet_plans_identifier, nnunet_config
        )
        source_folder = str(get_raw_path(self.dataset_id) / "imagesTr")
        output_folder = "/".join(model_folder.split("/")[:-1])

        data_iterator = predictor.get_data_iterator_from_folders(
            list_of_lists_or_source_folder=source_folder,
            output_folder_or_list_of_truncated_output_files=output_folder,
            num_processes_preprocessing=self.num_processes_preprocessing,
            part_id=part_id,
            num_parts=num_parts,
        )
        predictor.predict_from_data_iterator(data_iterator, self, temp_file_handler)
        return self.top_patches

    def query(self, n_gpus: int = 0, verbose: bool = False) -> list[Patch]:
        if n_gpus == 0:
            device = torch.device("cuda:0")
            self.query_part(part_id=0, num_parts=1, device=device)
        else:
            devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
            num_parts = [n_gpus] * n_gpus
            parts = [i for i in range(n_gpus)]
            try:
                with ProcessPoolExecutor(
                    max_workers=n_gpus, mp_context=mp.get_context("spawn")
                ) as executor:
                    for top_patch_part in executor.map(
                        self.wrap_query_part,
                        parts,
                        num_parts,
                        devices,
                        [wandb.run.group] * n_gpus,
                    ):
                        for cls in self.top_patches:
                            self.top_patches[cls].extend(top_patch_part[cls])

            except BrokenProcessPool as exc:
                raise MemoryError(
                    "One of the worker processes died. "
                    "This usually happens because you run out of memory. "
                    "Try running with less processes."
                ) from exc

        return self.compose_query_of_patches()

    def _compose_query_of_patches(self) -> dict[str, Any]:
        """Returns the patches that should be queried.

        Returns:
            dict[str, Any]: list of Patch objects.
        """
        sorted_top_patches = {}
        for cls in self.top_patches:
            sorted_top_patches[cls] = sorted(
                self.top_patches[cls], key=lambda d: d["score"], reverse=True
            )
        n_cls = int(self.config.query_size * self.ratio_fg)
        n_full = self.config.query_size - n_cls
        classes = [cls for cls in self.top_patches if cls != -1]
        # reduce number by one due to class unspecific -1 key
        select_per_cls = {cls: n_cls // (len(self.top_patches) - 1) for cls in classes}
        select_per_cls[-1] = n_full
        rand_per_cls = n_cls % len(self.top_patches)
        self.rng.shuffle(classes)
        for cls in classes[:rand_per_cls]:
            select_per_cls[cls] += 1

        patches = []
        for cls in self.top_patches:
            sorted_top_patches[cls] = sorted_top_patches[cls][: select_per_cls[cls]]
            patches.extend(
                [
                    {
                        "file": patch["file"],
                        "coords": patch["coords"],
                        "size": patch["size"],
                    }
                    for patch in sorted_top_patches[cls]
                ]
            )
        return patches


class ClassBalancedEntropy_33FG(ClassBalancedEntropy_66FG):
    def __post_init__(self):
        super().__post_init__()
        self.ratio_fg = 0.33
