from pathlib import Path

import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.uncertainties import Probs


@register_strategy("exp_entropy")
class ExpectedEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(
        self, probs: list[Path] | torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(probs, Probs):
            probs = Probs.create(probs)
        return probs.exp_entropy(probs, device=device)
