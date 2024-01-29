import numpy as np
import torch
from loguru import logger

from nnactive.results.utils import get_results_folder as get_nnactive_results_folder

DEVICE = "cuda:0"


def log_entropy(probs: torch.Tensor):
    """Computes logarithm for the entropy function and ignores values equal to 0

    Args:
        probs (torch.Tensor): probability values
    """
    out = torch.log(probs)
    # overwrite values which are 0
    out[out == -torch.inf] = 0
    return out


def prob_pred_entropy(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute predictive entropy.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    logger.info("Calc pred entropy")
    mean_prob = None
    for fold in range(num_folds):
        prob_path = str(
            get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
        )
        if mean_prob is None:
            mean_prob = torch.from_numpy(np.load(prob_path)).to(device)
        else:
            mean_prob += torch.from_numpy(np.load(prob_path)).to(device)
    mean_prob /= num_folds
    return -torch.sum(mean_prob * log_entropy(mean_prob), dim=0)


def prob_exp_entropy(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute expected entropy.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    logger.info("Calc exp entropy")
    ee = None
    for fold in range(num_folds):
        prob_path = str(
            get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
        )
        probs = torch.from_numpy(np.load(prob_path)).to(device)
        if ee is None:
            # TODO: dim=0 is correct here since we only load one prob right?
            ee = torch.sum(probs * log_entropy(probs), dim=0).to(device)
        else:
            ee += torch.sum(probs * log_entropy(probs), dim=0).to(device)
    ee /= num_folds
    logger.info("Calculated exp entropy")
    return ee


def prob_mutual_information(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute mutual information.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    return prob_pred_entropy(num_folds, dataset_id, device=device) - prob_exp_entropy(
        num_folds, dataset_id, device=device
    )
