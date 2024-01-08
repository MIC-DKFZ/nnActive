import torch

from nnactive.strategies.base_uncertainty import AbstractUncertainQueryMethod
from nnactive.strategies.uncertainties import prob_mutual_information


class BALD(AbstractUncertainQueryMethod):
    def get_uncertainty(self, num_folds: int) -> torch.Tensor:
        return prob_mutual_information(num_folds, self.dataset_id)
