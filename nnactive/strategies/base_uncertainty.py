from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch
from loguru import logger

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.data import Patch
from nnactive.masking import does_overlap
from nnactive.strategies.base import BasePredictionQuery


class AbstractUncertainQueryMethod(BasePredictionQuery):
    def __post_init__(self):
        super().__post_init__()
        striding = (
            self.config.agg_stride ** len(self.config.patch_size)
            if isinstance(self.config.agg_stride, int)
            else np.prod(self.config.agg_stride)
        )
        if (
            striding == 1
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
    ) -> tuple[torch.Tensor, np.ndarray, Iterable[int]]:
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
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # Combine output probabilities from folds. If tensors were kept in RAM, stack
        # them to a single tensor [M x C x XYZ]
        probs: list[np.ndarray] | list[Path] = [qd["probs"] for qd in query_dicts]
        if not isinstance(probs[0], (Path, str)):
            probs = torch.stack(probs)

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
        file_dict = {"scores": scores}
        return file_dict, out_list

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
        pass

    def get_top_scores(self, aggregated: np.ndarray) -> tuple[list[int], list[float]]:
        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        sorted_uncertainty_scores: list[float] = np.take_along_axis(
            flat_aggregated_uncertainties, sorted_uncertainty_indices, axis=0
        ).tolist()
        sorted_uncertainty_indices: list[int] = sorted_uncertainty_indices.tolist()

        return sorted_uncertainty_indices, sorted_uncertainty_scores


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
            # Mark region as queried
            annotated_patches.append(patch)
        # Stop if we reach the maximum number of patches to be queried
        if n is not None and len(selected_patches) >= n:
            break
    return selected_patches
