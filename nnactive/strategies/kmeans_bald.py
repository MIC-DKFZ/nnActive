from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import device

from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.strategies.bald import BALD
from nnactive.strategies.base_diversity import BaseDiversityQueryMethod
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.utils import RepresentationHandler


@register_strategy("kmeans_bald")
class KMeansBALD(BaseDiversityQueryMethod, BALD):
    """First select most uncertain patches for each image (2xstandard),
    then select diverse final selection of patches using kmeans++ centers of normalized representations.
    """

    def get_n_patch_per_image(self):
        # increase n_patch_per_image to allow more diverse patches
        return int(self.config.n_patch_per_image * 2)

    def strategy(
        self, query_dicts: list[dict[str, Any]], device: device = ...
    ) -> list[dict[str, Any]]:
        img_unc, potential_patches = BALD.strategy(self, query_dicts, device=device)
        if isinstance(query_dicts[0]["probs"], (Path, str)):
            input_shape = np.load(query_dicts[0]["probs"]).shape[1:]
        elif isinstance(query_dicts[0]["probs"], torch.Tensor):
            input_shape = query_dicts[0]["probs"].shape[1:]
        else:
            raise ValueError(f"probs are of type {type(query_dicts[0]['probs'])}")
        representation = [q_d["repr"] for q_d in query_dicts]
        representation = torch.from_numpy(np.concatenate(representation, axis=0))
        representation = RepresentationHandler.init_from_representation(
            representation, input_shape=input_shape
        )
        # build slicers for potential patches
        slicers = [
            tuple(
                slice(coord, coord + size)
                for coord, size in zip(p_patch["coords"], p_patch["size"])
            )
            for p_patch in potential_patches
        ]
        # obtain representations for potential patches
        representations = [
            representation.map_to_representation(slicer).mean(
                dim=list(range(1, len(input_shape) + 1))
            )
            for slicer in slicers
        ]
        # assign representations to potential patches
        for potential_patch, representation in zip(potential_patches, representations):
            potential_patch["repr"] = representation
        return img_unc, potential_patches

    def compose_query_of_patches(self):
        with (
            monitor.timer("compose_query_of_patches")
            if monitor.is_active()
            else nullcontext()
        ):
            patches = self._compose_query_of_patches()
            patches = [Patch(**patch) for patch in patches]
            if len(patches) < self.config.query_size:
                raise RuntimeError(
                    f"Not enough patches could be queried, {len(patches)} instead of {self.config.query_size}"
                )
            return patches

    def _compose_query_of_patches(
        self, device: torch.DeviceObjType = torch.device("cuda:0")
    ) -> dict[str, Any]:
        X = torch.stack([patch["repr"] for patch in self.top_patches], dim=0)
        _, center_inds = kmeanspp(X.to(device), self.config.query_size, self.rng)
        del _
        patches = [
            {
                "file": self.top_patches[ind]["file"],
                "coords": self.top_patches[ind]["coords"],
                "size": self.top_patches[ind]["size"],
            }
            for ind in center_inds
        ]
        for i in range(len(self.top_patches)):
            self.top_patches[i]["selected"] = False
        for ind in center_inds:
            self.top_patches[ind]["selected"] = True

        return patches


def kmeanspp(
    X: torch.Tensor,
    n_clusters: int,
    rng: np.random.Generator = np.random.default_rng(),
    strategy: str = "maxselect",
) -> torch.Tensor:
    if strategy == "maxselect":
        select = MaxSelect()
    elif strategy == "stochastic":
        select = StochasticSelect()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    X = X.to(torch.float32)
    X /= X.std(dim=0)[None, :] + 1e-6  # normalize
    n_samples, n_features = X.shape

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)
    center_inds = [None] * n_clusters
    # Randomly choose the first center
    center_id = select.select_starting_center(X, rng)
    centers[0] = X[center_id]
    center_inds[0] = center_id
    # Compute the squared distance of each sample to the nearest center
    nearest_dist_sq = torch.linalg.norm(X - centers[0], dim=1) ** 2

    for i in range(1, n_clusters):
        # Choose the next center
        center_id = select.select_next_center(nearest_dist_sq, rng)
        centers[i] = X[center_id]
        center_inds[i] = center_id
        # Update the squared distance of each sample to the nearest center
        dist_sq = torch.linalg.norm(X - centers[i], dim=1) ** 2
        nearest_dist_sq = torch.minimum(nearest_dist_sq, dist_sq)
    return centers, center_inds


class StochasticSelect:
    def select_starting_center(
        self, X: torch.Tensor, rng: np.random.Generator = np.random.default_rng()
    ) -> int:
        n_samples = X.shape[0]
        return rng.integers(n_samples)

    def select_next_center(
        self, dist_sq: torch.Tensor, rng: np.random.Generator = np.random.default_rng()
    ) -> int:
        prob = (dist_sq / dist_sq.sum()).cpu().numpy()
        if prob.sum() != 1:
            prob /= prob.sum()
        return rng.choice(len(prob), p=prob)


class MaxSelect:
    def select_starting_center(
        self, X: torch.Tensor, rng: np.random.Generator = np.random.default_rng()
    ) -> int:
        return int(torch.argmax(torch.linalg.norm(X, dim=1)))

    def select_next_center(
        self, dist_sq: torch.Tensor, rng: np.random.Generator = np.random.default_rng()
    ) -> int:
        return int(torch.argmax(dist_sq))
