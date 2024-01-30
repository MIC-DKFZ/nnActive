import numpy as np
import torch
from loguru import logger

from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.utils.torchutils import (
    estimate_free_cuda_memory,
    get_tensor_memory_usage,
    log_cuda_memory_info,
)

DEVICE = "cuda:0"


def log_entropy(probs: torch.Tensor):
    """Computes logarithm for the entropy function and ignores values equal to 0

    Args:
        probs (torch.Tensor): probability values
    """
    out = torch.log(probs)
    logger.debug("Compute log on GPU")
    try:
        mask = out == -torch.inf
        out[mask] = 0
    except RuntimeError as e:
        del out, mask
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        raise e
    return out


def prob_pred_entropy(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute predictive entropy.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    logger.info("Calc pred entropy")

    torch.cuda.reset_peak_memory_stats()
    logger.debug("-" * 8)
    logger.debug("Before Compute of Mean Prob")

    log_cuda_memory_info()

    def _compute_mean_prob(mean_prob: torch.Tensor, num_folds: int, dataset_id: int):
        for fold in range(1, num_folds):
            prob_path = str(
                get_nnactive_results_folder(dataset_id)
                / "temp"
                / f"probs_fold{fold}.npy"
            )
            cur_prob = torch.from_numpy(np.load(prob_path)).to(mean_prob.device)
            if mean_prob is None:
                mean_prob = cur_prob
            else:
                mean_prob += cur_prob
            del cur_prob
        mean_prob /= num_folds
        return mean_prob

    fold = 0
    prob_path = str(
        get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
    )

    compute_val = torch.from_numpy(np.load(prob_path))
    # check if it will fit into GPU
    if get_tensor_memory_usage(compute_val) * 2.1 < estimate_free_cuda_memory():
        try:
            logger.debug(f"Compute entropy on device: {device}")
            compute_val = compute_val.to(device)
            compute_val = _compute_mean_prob(compute_val, num_folds, dataset_id)
            compute_val *= torch.log(compute_val)
            # here we assume that all nans are stemming from -inf after log
            compute_val = compute_val.nan_to_num()
            compute_val = compute_val.sum(dim=0)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain compute_val on CPU.")
            del compute_val
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = prob_pred_entropy(num_folds, dataset_id, "cpu").to(device)
        return compute_val
    else:
        logger.debug(f"Compute entropy on CPU instead of {device}")
        compute_val = _compute_mean_prob(compute_val, num_folds, dataset_id)
        compute_val *= torch.log(compute_val)
        # here we assume that all nans are stemming from -inf after log
        compute_val = compute_val.nan_to_num()
        compute_val = compute_val.sum(dim=0)
        return compute_val.to(device)


def prob_exp_entropy(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute expected entropy.
    Assumes probs for computation to be saved in nnActive_results/Dataset/temp/probs_foldx.npz

    Args:
        num_folds (int): number of folds predicted
        dataset_id (int): dataset_id to find folder
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: expected entropy H x W x D
    """
    logger.info("Calc exp entropy")
    fold = 0
    prob_path = str(
        get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
    )
    compute_val = torch.from_numpy(np.load(prob_path))
    if (
        get_tensor_memory_usage(compute_val) * (2.1 + 1 / compute_val.shape[0])
        < estimate_free_cuda_memory()
    ):
        logger.debug(f"Compute on {device}")
        try:
            compute_val = compute_val.to(device)
            compute_val *= torch.log(compute_val)
            compute_val = compute_val.nan_to_num()
            compute_val = compute_val.sum(dim=0)
            for fold in range(1, num_folds):
                prob_path = str(
                    get_nnactive_results_folder(dataset_id)
                    / "temp"
                    / f"probs_fold{fold}.npy"
                )
                temp_val = torch.from_numpy(np.load(prob_path)).to(device)
                temp_val *= torch.log(temp_val)
                temp_val = temp_val.nan_to_num()
                compute_val += temp_val.sum(dim=0)
        except RuntimeError as e:
            logger.debug("Possible CUDA OOM error, try to obtain compute_val on CPU.")
            logger.debug(e)
            del compute_val, temp_val
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = prob_exp_entropy(num_folds, dataset_id, "cpu").to(device)
        return compute_val
    else:
        logger.debug(f"Compute on CPU and move to {device}")
        compute_val *= torch.log(compute_val)
        compute_val = compute_val.nan_to_num()
        compute_val = compute_val.sum(dim=0)
        for fold in range(1, num_folds):
            prob_path = str(
                get_nnactive_results_folder(dataset_id)
                / "temp"
                / f"probs_fold{fold}.npy"
            )
            temp_val = torch.from_numpy(np.load(prob_path))
            temp_val *= torch.log(temp_val)
            temp_val = temp_val.nan_to_num()
            compute_val += temp_val.sum(dim=0)
        return compute_val.to(device)


def prob_mutual_information(
    num_folds: int, dataset_id: int, device: str = DEVICE
) -> torch.Tensor:
    """Compute Mutual information
    Assumes probs for computation to be saved in nnActive_results/Dataset/temp/probs_foldx.npz


    Args:
        num_folds (int): number of folds predicted
        dataset_id (int): dataset_id to find folder
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: mutual infromation H x W x D
    """
    compute_val = prob_pred_entropy(num_folds, dataset_id, device=device)
    compute_val -= prob_exp_entropy(num_folds, dataset_id, device=device)
    return compute_val
