from pathlib import Path

import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import Probs


class ExpectedEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(probs, Probs):
            probs = Probs.create(probs)
        return probs.exp_entropy(probs, device=device)
