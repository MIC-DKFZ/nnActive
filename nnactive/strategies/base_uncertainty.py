from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Union

import numpy as np
import psutil
import torch
import wandb
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from loguru import logger
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_probs import (
    convert_predicted_logits_to_probs_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.masking import does_overlap
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import BasePredictionQuery, BaseQueryPredictor
from nnactive.strategies.utils import RepresentationHandler
from nnactive.utils.io import load_label_map
from nnactive.utils.timer import CudaTimer, Timer


class AbstractUncertainQueryMethod(BasePredictionQuery):
    def __post_init__(self):
        super().__post_init__()
        if (
            self.config.agg_stride == 1
        ):  # TODO: for strides < 8 for large images scipy is still faster. This can be implemented better
            self.aggregation = ConvolveAggScipy(
                self.config.patch_size, stride=self.config.agg_stride
            )
        else:
            self.aggregation = ConvolveAggTorch(
                self.config.patch_size, stride=self.config.agg_stride
            )

        logger.info(
            f"Aggregation is performed using: {self.aggregation.__class__.__name__} with stride {self.config.agg_stride}"
        )

    def compute_scores(
        self, probs: np.ndarray | list[Path], device: torch.device
    ) -> tuple[torch.Tesnor, np.ndarray, Iterable[int]]:
        with torch.no_grad():
            logger.debug("Compute uncertaintes...")
            uncertainty = self.get_uncertainty(probs, device=device)
            logger.debug("Aggregate uncertainties...")
            agg_uncertainty, kernel_size = self.aggregation.forward(uncertainty)
        return uncertainty, agg_uncertainty, kernel_size

    def strategy(
        self,
        query_dicts: list[Dict[str, Any]],
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict[str, Any]]:
        probs: np.ndarray | list[Path] = [qd["probs"] for qd in query_dicts]
        scores, agg_scores, patch_size = self.compute_scores(probs, device)
        sorted_uncertainty_indices, sorted_uncertainty_scores = self.get_top_scores(
            agg_scores
        )
        # TODO: Think how to cleverly obtain uncertainty in a way to use it for other stuff...
        out_list = [
            {
                "coords": self.aggregation.backward_index(index, agg_scores.shape),
                "size": patch_size,
                "score": score,
            }
            for score, index in zip(
                sorted_uncertainty_scores, sorted_uncertainty_indices
            )
        ]
        return out_list

    @abstractmethod
    def get_uncertainty(
        self,
        probs: list[Path] | torch.Tensor,
        device: torch.device = torch.device("cuda:0"),
    ) -> torch.Tensor:
        """Compute uncertainty values from out_probs

        Args:
            probs (list[Path] | torch.Tensor): paths to probability maps for image
            [1 x C x XYZ] per item in list or [M x C x XYZ]

        Returns:
            torch.Tensor: outputs [XYZ] on device
        """

    # def select_top_n_non_overlapping_patches(
    #     self,
    #     patch_size: list[int],
    #     aggregated: np.ndarray,
    #     annotated_patches: list[Patch],
    #     label_file: str,
    #     n: int,
    # ) -> list[dict]:
    #     """Select top-n non overlapping patches in image.

    #     Args:
    #         patch_size (list[int]): size of patch
    #         aggregated (np.ndarray): aggregated uncertainty
    #         annotated_patches (list[Patch]): list of annotated patches per image
    #         label_file (str): name of label_file without ending (.nii.gz)
    #         n (int): number of patches to select

    #     Returns:
    #         list[dict]: dict contains all values required to build a Patch
    #     """
    #     selected_patches = []
    #     # sort only once since this can take a significant amount of time
    #     logger.info("Sort potential queries")
    #     sorted_uncertainty_indices, sorted_uncertainty_scores = self.get_top_scores(
    #         aggregated
    #     )
    #     logger.info("Start finding non-overlapping patches.")
    #     # Iterate over the sorted uncertainty scores and their indices to get the most uncertain

    #     iterator = zip(sorted_uncertainty_scores, sorted_uncertainty_indices)
    #     pbar0 = tqdm(total=n, position=0, desc="Patch Selection", disable=self.verbose)
    #     pbar1 = tqdm(
    #         total=len(sorted_uncertainty_scores),
    #         position=1,
    #         desc="Possible Patch Search",
    #         disable=self.verbose,
    #     )

    #     additional_label = None
    #     if self.additional_label_path is not None:
    #         if self.verbose:
    #             logger.debug("Create additional label map.")
    #         additional_label = load_label_map(
    #             label_file,
    #             self.additional_label_path,
    #             self.file_ending,
    #         )
    #         additional_label: np.ndarray = additional_label != 255

    #     for i, (uncertainty_score, uncertainty_index) in enumerate(iterator):
    #         pbar1.update()
    #         # get coordinates in image space from aggregated indices
    #         coords = self.aggregation.backward_index(
    #             uncertainty_index, aggregated.shape
    #         )
    #         patch = Patch(
    #             file=label_file + ".nii.gz",
    #             coords=coords,
    #             size=patch_size,
    #         )

    #         # Check if coordinated overlap with already queried region
    #         if self.check_overlap(
    #             patch, annotated_patches, additional_label, verbose=self.verbose
    #         ):
    #             # If it is a non-overlapping region, append this patch to be queried
    #             selected_patches.append(
    #                 {
    #                     "file": label_file + ".nii.gz",
    #                     "coords": coords,
    #                     "size": patch_size,
    #                     "score": uncertainty_score,
    #                 }
    #             )
    #             # Mark region as queried
    #             annotated_patches.append(patch)
    #             # Stop if we reach the maximum number of patches to be queried
    #             pbar0.update()
    #         if n is not None and len(selected_patches) >= n:
    #             break
    #     pbar1.close()
    #     pbar0.close()
    #     logger.info(f"Finished patch selection for image {label_file}")
    #     return selected_patches

    def get_top_scores(self, aggregated: np.ndarray) -> tuple[list[int], list[float]]:
        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        sorted_uncertainty_scores: list[float] = np.take_along_axis(
            flat_aggregated_uncertainties, sorted_uncertainty_indices, axis=0
        ).tolist()
        sorted_uncertainty_indices: list[int] = sorted_uncertainty_indices.tolist()

        return sorted_uncertainty_indices, sorted_uncertainty_scores

    def compose_query_of_patches(self):
        with (
            monitor.timer("compose_query_of_patches")
            if monitor.is_active()
            else nullcontext()
        ):
            sorted_top_patches = sorted(
                self.top_patches, key=lambda d: d["score"], reverse=True
            )[: self.config.query_size]
            patches = [
                {
                    "file": patch["file"],
                    "coords": patch["coords"],
                    "size": patch["size"],
                }
                for patch in sorted_top_patches
            ]
            patches = [Patch(**patch) for patch in patches]
            if len(patches) < self.config.query_size:
                raise RuntimeError(
                    f"Not enough patches could be queried, {len(patches)} instead of {self.config.query_size}"
                )
            return patches


def select_top_n_non_overlapping_patches(
    image_name: str,
    n: int,
    uncertainty_scores: np.ndarray,
    patch_size: tuple[int, int, int],
    annotated_patches: list[Patch],
    overlap_test: Callable[[Patch, list[Patch]], bool] = lambda x, y: not does_overlap(
        x, y
    ),
) -> list[dict]:
    """
    Get the most n uncertain non-overlapping patches for one image based on the aggregated uncertainty map

    Args:
        image_name (str): the name of the aggregated uncertainty map (npz file)
        n (int): number of non-overlapping patches that should be queried at most
        uncertainty_scores (np.ndarray): the aggregated uncertainty map
        patch_size (np.ndarray): patch size that was used to aggregate the uncertainties
        selected_array (np.ndarray): array with already labeled patches
        overlap_test: Callable[[Patch, list[Patch]], bool]: returns True if overlap is allowed

    Returns:
        list[dict]: the most n uncertain non-overlapping patches for one image
    """
    selected_patches = []
    sorted_uncertainty_scores = np.flip(np.sort(uncertainty_scores.flatten()))
    sorted_uncertainty_indices = np.flip(np.argsort(uncertainty_scores.flatten()))
    # This was just for visualization purposes in MITK
    # selected = 0

    # Iterate over the sorted uncertainty scores and their indices to get the most uncertain
    for uncertainty_score, uncertainty_index in zip(
        sorted_uncertainty_scores, sorted_uncertainty_indices
    ):
        # Get the index as coordinates
        coords = np.unravel_index(uncertainty_index, uncertainty_scores.shape)
        # Check if coordinated overlap with already queried region
        patch = Patch(
            file=image_name,
            coords=coords,
            size=patch_size,
        )
        if overlap_test(patch, annotated_patches):
            # If it is a non-overlapping region, append this patch to be queried
            selected_patches.append(
                {
                    "file": image_name,
                    "coords": coords,
                    "size": patch_size,
                    "score": uncertainty_score,
                }
            )
            # selected += 1
            # Mark region as queried
            annotated_patches.append(patch)
        # Stop if we reach the maximum number of patches to be queried
        if n is not None and len(selected_patches) >= n:
            break
    return selected_patches
