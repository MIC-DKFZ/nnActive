import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_mutual_information


class BALD(AbstractUncertainQueryMethod):
    def get_uncertainty(self, out_probs: torch.Tensor) -> torch.Tensor:
        return prob_mutual_information(out_probs)
