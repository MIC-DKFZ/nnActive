from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.uncertainties import Probs
from nnactive.strategies.utils import power_noising


@register_strategy("mutual_information")
class BALD(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(probs, Probs):
            probs = Probs.create(probs)
        return probs.mutual_information(probs, device)


@register_strategy("power_bald")
class PowerBALD(BALD):
    """Compute Power BALD with Gumbel Softmax.
    https://openreview.net/pdf?id=vcHwQyNBjW

    Using beta=1

    We add power samples on the aggregated scores directly instead of each voxel.
    This is because the mean score for each voxel aggregated would be always very close to mu(gumbel(0, beta**-1))
    """

    def __post_init__(self):
        super().__post_init__()
        self.beta = 1

    def compute_scores(self, probs, device):
        uncertainty, agg_uncertainty, kernel_size = super().compute_scores(
            probs, device
        )
        agg_uncertainty = power_noising(agg_uncertainty, beta=self.beta, rng=self.rng)
        return uncertainty, agg_uncertainty, kernel_size


@register_strategy("power_bald_b5")
class PowerBALD_b5(PowerBALD):
    def __post_init__(self):
        super().__post_init__()
        self.beta = 5


@register_strategy("power_bald_b10")
class PowerBALD_b10(PowerBALD):
    def __post_init__(self):
        super().__post_init__()
        self.beta = 10


@register_strategy("power_bald_b20")
class PowerBALD_b20(PowerBALD):
    def __post_init__(self):
        super().__post_init__()
        self.beta = 20


@register_strategy("power_bald_b40")
class PowerBALD_b40(PowerBALD):
    def __post_init__(self):
        super().__post_init__()
        self.beta = 40


@register_strategy("softrank_bald")
class SoftRankBALD(BALD):
    """Compute Softrank Bald with Gumbel Softmax.
    This solely perturbes the rank.

    https://openreview.net/pdf?id=vcHwQyNBjW

    Using beta=1

    We sample more patches per image than usual and perform the softranking at the final score.
    """

    def get_n_patch_per_image(self):
        # increase n_patch_per_image to allow more perturbation in rankings
        return int(self.config.n_patch_per_image * 2)

    # def get_top_scores(self, aggregated: np.ndarray) -> tuple[list[int], list[float]]:
    #     sorted_uncertainty_indices, sorted_uncertainty_scores = super().get_top_scores(
    #         aggregated
    #     )
    #     softrankings = np.arange(len(sorted_uncertainty_indices), dtype=np.float32) + 1
    #     softrankings = -np.log(softrankings) + np.random.gumbel(0, 1)
    #     soft_indices = np.argsort(softrankings)
    #     sorted_uncertainty_indices: list[float] = np.take_along_axis(
    #         np.array(sorted_uncertainty_indices),
    #         soft_indices,
    #         axis=0,
    #     ).tolist()
    #     sorted_uncertainty_scores: list[float] = np.take_along_axis(
    #         np.array(sorted_uncertainty_scores),
    #         soft_indices,
    #         axis=0,
    #     ).tolist()
    #     return sorted_uncertainty_indices, sorted_uncertainty_scores

    def _compose_query_of_patches(self) -> dict[str, Any]:
        pre_sorted_top_patches = sorted(
            self.top_patches, key=lambda d: d["score"], reverse=True
        )
        soft_scores = -np.log(np.arange(len(self.top_patches)) + 1) + self.rng.gumbel(
            0, 1, size=len(self.top_patches)
        )
        soft_rankings = np.argsort(soft_scores)[::-1]
        sorted_top_patches: list[dict] = []
        for soft_rank in soft_rankings[: self.config.query_size]:
            sorted_top_patches.append(pre_sorted_top_patches[soft_rank])

        patches = [
            {
                "file": patch["file"],
                "coords": patch["coords"],
                "size": patch["size"],
            }
            for patch in sorted_top_patches
        ]
        return patches
