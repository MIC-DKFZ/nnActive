import os
from pathlib import Path
from typing import Any, Type

import numpy as np
import SimpleITK as sitk
import torch

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.data.utils import copy_geometry_sitk
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.bald import BALD
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.dice_query import ExpectedDiceQuery
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.strategies.random import Random
from nnactive.strategies.randomlabel import RandomLabel
from nnactive.utils.io import get_clean_dataclass_dict, save_json

QUERY_METHODS: list[Type[AbstractQueryMethod]] = [
    BALD,
    # RandomLabel,
    PredictiveEntropy,
    # ExpectedDiceQuery,
]


class AnalyzeQueries:
    def __init__(
        self,
        config: ActiveConfig,
        dataset_id: int,
        base_path: Path,
        loop_val: int | None = None,
    ):
        self.config = config
        # self.base_path = base_path
        self.dataset_id = dataset_id
        self.query_methods: dict[str, AbstractQueryMethod] = {
            cls_.__name__: cls_.init_from_dataset_id(config, dataset_id=dataset_id)
            for cls_ in QUERY_METHODS
        }
        for q_n in self.query_methods:
            self.query_methods[q_n].annotated_patches = get_patches_from_loop_files(
                get_raw_path(self.dataset_id), loop_val=loop_val
            )

    def probs_to_voxel_uncertainty(self, probs, label_file) -> dict[str, torch.Tensor]:
        uncertainty_dict = {}
        for qm_name, qm in self.query_methods.items():
            if isinstance(qm, AbstractUncertainQueryMethod):
                uncertainty, aggregated = qm.query_from_probs(
                    probs, image_shape=[0, 0, 0], label_file=label_file
                )
                uncertainty_dict[qm_name] = uncertainty.cpu()

            elif isinstance(qm, ExpectedDiceQuery):
                print("No analysis for this class")
                pass
            elif isinstance(qm, Random):
                pass
            else:
                raise NotImplementedError
        return uncertainty_dict

    def get_final_queries(self) -> tuple[dict[str, list[Patch]], dict[str, list[dict[str, Any]]]]:
        final_query_patches: dict[str, list[Patch]] = {}
        scores_all: dict[str, list[dict[str, Any]]] = {}
        for qm_name, qm in self.query_methods.items():
            if isinstance(qm, AbstractUncertainQueryMethod):
                final_query_patches[qm_name] = qm.compose_query_of_patches()
                scores_all[qm_name] = qm.top_patches
            elif isinstance(qm, ExpectedDiceQuery):
                continue
                # final_query_patches[qm_name] = qm.query()
                scores_all[qm_name] = qm.top_patches
            elif isinstance(qm, Random):
                continue
                final_query_patches[qm_name] = qm.query()
            else:
                raise NotImplementedError
        return final_query_patches, scores_all


if __name__ == "__main__":
    model_folder = "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_results/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/nnActiveTrainer_5epochs__nnUNetPlans__3d_fullres"
    raw_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347"
    )
    input_folder = "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/imagesTr"
    output_folder = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/loop_000__analysis"
    )

    nnactive_results_folder = "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnActive_results/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347"
    base_folders = [
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/probTr_0",
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnUNet_raw/Dataset000_Hippocampus__patch-20__qs20__unc-random-label__seed-12347/probTr_1",
    ]

    base_folders = [Path(bf) for bf in base_folders]

    fns = [f.name for f in base_folders[0].iterdir() if f.suffix == ".npz"]

    probs_paths = [
        [bf / f for bf in base_folders] for f in fns
    ]  # for each file the paths

    config = ActiveConfig.from_json(Path(nnactive_results_folder) / "config.json")
    config.set_nnunet_env()
    analysis = AnalyzeQueries(config, dataset_id=0, base_path=base_folders, loop_val=0)
    for prob_paths in probs_paths:
        fn = prob_paths[0].name.split(".")[0]
        uncertainty_dict = analysis.probs_to_voxel_uncertainty(
            prob_paths, label_file=fn
        )
        for u_n in uncertainty_dict:
            nii_name = fn + ".nii.gz"
            nii_image = sitk.ReadImage(raw_folder / "labelsTr" / nii_name)
            save_image = sitk.GetImageFromArray(uncertainty_dict[u_n].numpy())
            save_image = copy_geometry_sitk(save_image, nii_image)
            if not (output_folder / u_n).is_dir():
                os.makedirs(output_folder / u_n)
            sitk.WriteImage(save_image, output_folder / u_n / nii_name)

    final_query_patches, scores_all = analysis.get_final_queries()
    final_query_patches_json = {
        k: [get_clean_dataclass_dict(p) for p in final_query_patches[k]]
        for k in final_query_patches
    }
    save_json(
        final_query_patches_json,
        save_path=Path(output_folder / "final_patches.json"),
    )
    save_json(scores_all, save_path=Path(output_folder / "all_scores.json"))

    for q_n, patches in final_query_patches.items():
        
