from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.utils.torchutils import (
    estimate_free_cuda_memory,
    get_tensor_memory_usage,
    log_cuda_memory_info,
)

DEVICE = torch.device("cuda:0")


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
    probs: list[Path] | torch.Tensor, device: torch.device = DEVICE
) -> torch.Tensor:
    """Compute predictive entropyon list of paths saving npy arrays or a tensor.

    Args:
        probs (list[Path] | torch.Tensor): paths to probability maps for image
            [C x XYZ] per item in list or [M x C x XYZ]
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: predictive entropy H x W x D (on device)
    """
    logger.info("Compute pred entropy")

    torch.cuda.reset_peak_memory_stats(device)
    logger.debug("-" * 8)
    logger.debug("Before Compute of Mean Prob")

    log_cuda_memory_info(device)

    def _compute_mean_prob(mean_prob: torch.Tensor, probs: list[Path] | torch.Tensor):
        for fold in range(1, len(probs)):
            if isinstance(probs, list):
                temp_val = torch.from_numpy(np.load(probs[fold])).to(mean_prob.device)
            else:
                temp_val = deepcopy(probs[fold]).to(mean_prob.device)
            mean_prob += temp_val
            del temp_val
        mean_prob /= len(probs)
        return mean_prob

    fold = 0
    if isinstance(probs, list):
        compute_val = torch.from_numpy(np.load(probs[fold])).to(device)
    else:
        compute_val = deepcopy(probs[fold]).to(device)
    # check if it will fit into GPU
    if device.type == "cuda":
        if (get_tensor_memory_usage(compute_val) * 2) * 1.1 < estimate_free_cuda_memory(
            device
        ):
            use_device = device
        else:
            use_device = torch.device("cpu")
            logger.debug(
                f"Computation on {device} not feasible due to VRAM, falling back to {use_device} for computation and then move to {device}"
            )
    else:
        # CPU case
        use_device = device

    try:
        logger.debug(f"Compute entropy on device: {use_device}")
        compute_val = compute_val.to(use_device)
        compute_val = _compute_mean_prob(compute_val, probs)
        compute_val *= torch.log(compute_val)
        # here we assume that all nans are stemming from -inf after log
        compute_val = compute_val.nan_to_num()
        compute_val = compute_val.sum(dim=0)
    except RuntimeError as e:
        logger.debug("Possibly CUDA OOM error, try to obtain compute_val on CPU.")
        del compute_val
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        compute_val = prob_pred_entropy(probs, torch.device("cpu"))
    return compute_val.to(device)


def prob_exp_entropy(
    probs: list[Path] | torch.Tensor, device: torch.device = DEVICE
) -> torch.Tensor:
    """Compute expected entropy on list of paths saving npy arrays or a tensor.

    Args:
        probs (list[Path] | torch.Tensor): paths to probability maps for image
            [ C x XYZ] per item in list or [M x C x XYZ]
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: expected entropy H x W x D (on device)
    """
    logger.info("Compute exp entropy")
    if isinstance(probs, list):
        compute_val = torch.from_numpy(np.load(probs[0]))
    else:
        compute_val = deepcopy(probs[0])
    if device.type == "cuda":
        if get_tensor_memory_usage(compute_val) * (
            2 + (2 / compute_val.shape[0])
        ) * 1.1 < estimate_free_cuda_memory(device):
            use_device = device
        else:
            use_device = torch.device("cpu")
            logger.debug(
                f"Computation on {device} not feasible due to VRAM. Falling back to {use_device} for computation and then move to {device}"
            )
    else:
        # CPU case
        use_device = device

    logger.debug(f"Compute on {device}")
    try:
        compute_val = compute_val.to(device)
        compute_val *= torch.log(compute_val)
        compute_val = compute_val.nan_to_num()
        compute_val = compute_val.sum(dim=0)
        for fold in range(1, len(probs)):
            if isinstance(probs, list):
                temp_val = torch.from_numpy(np.load(probs[fold])).to(use_device)
            else:
                temp_val = deepcopy(probs[fold]).to(use_device)
            temp_val *= torch.log(temp_val)
            # set all nan values (nan, inf, -inf) to 0
            temp_val = temp_val.nan_to_num()
            compute_val += temp_val.sum(dim=0)
    except RuntimeError as e:
        logger.debug("Possible CUDA OOM error, try to obtain compute_val on CPU.")
        logger.debug(e)
        del compute_val
        try:
            del temp_val
        except:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        compute_val = prob_exp_entropy(probs, torch.device("cpu"))
    return compute_val.to(device)


def prob_mutual_information(
    probs: list[Path] | torch.Tensor, device: torch.device = DEVICE
) -> torch.Tensor:
    """Compute mutual information on list of paths saving npy arrays or a tensor.

    Args:
        probs (list[Path] | torch.Tensor): paths to probability maps for image
            [ C x XYZ] per item in list or [M x C x XYZ]
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: expected entropy H x W x D (on device)
    """

    logger.info("Compute mutual information")
    compute_val = prob_pred_entropy(probs, device=device)
    compute_val -= prob_exp_entropy(probs, device=device)
    return compute_val
