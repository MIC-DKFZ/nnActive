import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_pred_entropy


class PredictiveEntropy(AbstractUncertainQueryMethod):
    def get_uncertainty(self, num_folds: int) -> torch.Tensor:
        return prob_pred_entropy(num_folds, self.dataset_id)
