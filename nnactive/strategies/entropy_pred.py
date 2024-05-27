from pathlib import Path

import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_pred_entropy


class PredictiveEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        return prob_pred_entropy(probs, device)
