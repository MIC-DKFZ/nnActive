from pathlib import Path

import numpy as np
import torch

from nnactive.data import Patch
from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.uncertainties import Probs
from nnactive.strategies.utils import power_noising


@register_strategy("pred_entropy")
class PredictiveEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(probs, Probs):
            probs = Probs.create(probs)
        return probs.pred_entropy(probs, device)


@register_strategy("power_pe")
class PowerPredictiveEntropy(PredictiveEntropy):
    """Compute Power Predictive Entropy with Gumbel Softmax.
    https://openreview.net/pdf?id=vcHwQyNBjW

    Using beta=1

    We add power samples on the aggregated scores directly instead of each sample.
    This is because the mean score for each voxel aggregated would be always very close to mu(gumbel(0, beta**-1)).
    """

    def compute_scores(self, probs, device):
        uncertainty, agg_uncertainty, kernel_size = super().compute_scores(
            probs, device
        )
        agg_uncertainty = power_noising(agg_uncertainty, beta=1)
        return uncertainty, agg_uncertainty, kernel_size
