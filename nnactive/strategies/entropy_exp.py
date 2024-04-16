from pathlib import Path

import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_exp_entropy


class ExpectedEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        return prob_exp_entropy(probs, device=device)
