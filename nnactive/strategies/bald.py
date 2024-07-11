from pathlib import Path

import torch

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

    def compute_scores(self, probs, device):
        uncertainty, agg_uncertainty, kernel_size = super().compute_scores(
            probs, device
        )
        agg_uncertainty = power_noising(agg_uncertainty, beta=1)
        return uncertainty, agg_uncertainty, kernel_size
