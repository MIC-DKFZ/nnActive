import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_exp_entropy


class ExpectedEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(self, num_folds: int) -> torch.Tensor:
        return prob_exp_entropy(num_folds, self.dataset_id)
