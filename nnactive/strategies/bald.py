from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_mutual_information
from nnactive.strategies.utils import power_noising


class BALD(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        return prob_mutual_information(probs, device)


class PowerBALD(BALD):
    """Compute Power BALD with Gumbel Softmax.
    https://openreview.net/pdf?id=vcHwQyNBjW

    Using beta=1

    We add power samples on the aggregated scores directly instead of each voxel.
    This is because the mean score for each voxel aggregated would be always very close to mu(gumbel(0, beta**-1))
    """

    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        agg_stride: int | list[int],
        n_patch_per_image: int,
        file_ending: str = ".nii.gz",
        num_processes_preprocessing: int = 3,
        use_gaussian: bool = False,
        use_mirroring: bool = False,
        tile_step_size: float = 0.75,
        additional_label_path: Path | None = None,
        additional_overlap: float = 0.1,
        verbose: bool = False,
        config: ActiveConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            agg_stride,
            n_patch_per_image,
            file_ending,
            num_processes_preprocessing,
            use_gaussian,
            use_mirroring,
            tile_step_size,
            additional_label_path,
            additional_overlap,
            verbose,
            config,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def compute_scores(self, probs, device):
        uncertainty, agg_uncertainty, kernel_size = super().compute_scores(
            probs, device
        )
        agg_uncertainty = power_noising(agg_uncertainty, beta=1, rng=self.rng)
        return uncertainty, agg_uncertainty, kernel_size


class SoftRankBALD(BALD):
    """Compute Softrank Bald with Gumbel Softmax.
    This solely perturbes the rank.

    https://openreview.net/pdf?id=vcHwQyNBjW

    Using beta=1

    We sample more patches per image than usual and perform the softranking at the final score.
    """

    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        agg_stride: int | list[int],
        n_patch_per_image: int,
        file_ending: str = ".nii.gz",
        num_processes_preprocessing: int = 3,
        use_gaussian: bool = False,
        use_mirroring: bool = False,
        tile_step_size: float = 0.75,
        additional_label_path: Path | None = None,
        additional_overlap: float = 0.1,
        verbose: bool = False,
        config: ActiveConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            agg_stride,
            n_patch_per_image,
            file_ending,
            num_processes_preprocessing,
            use_gaussian,
            use_mirroring,
            tile_step_size,
            additional_label_path,
            additional_overlap,
            verbose,
            config,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def __post_init__(self):
        # increase n_patch_per_image to allow more perturbation in rankings
        self.n_patch_per_image = int(2 * self.n_patch_per_image)

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

    def compose_query_of_patches(self):
        with (
            monitor.timer("compose_query_of_patches")
            if monitor.is_active()
            else nullcontext()
        ):
            pre_sorted_top_patches = sorted(
                self.top_patches, key=lambda d: d["score"], reverse=True
            )
            soft_scores = -np.log(
                np.arange(len(self.top_patches)) + 1
            ) + self.rng.gumbel(0, 1, size=soft_scores.shape)
            soft_rankings = np.argsort(soft_scores)[::-1]
            sorted_top_patches: list[dict] = []
            for soft_rank in soft_rankings[: self.query_size]:
                sorted_top_patches.append(pre_sorted_top_patches[soft_rank])

            patches = [
                {
                    "file": patch["file"],
                    "coords": patch["coords"],
                    "size": patch["size"],
                }
                for patch in sorted_top_patches
            ]
            patches = [Patch(**patch) for patch in patches]
            if len(patches) < self.query_size:
                raise RuntimeError(
                    f"Not enough patches could be queried, {len(patches)} instead of {self.query_size}"
                )
            return patches
