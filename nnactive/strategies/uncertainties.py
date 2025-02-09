from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from numpy.lib.npyio import NpzFile

from nnactive.utils.torchutils import (
    estimate_free_cuda_memory,
    get_tensor_memory_usage,
    log_cuda_memory_info,
    move_tensor_check_vram,
    reset_cuda_memory_stats,
)

DEVICE = torch.device("cuda:0")


class Probs:
    def __init__(self, data: torch.Tensor):
        """Internal Class to hold probs and calculate statistics.

        Args:
            data (torch.Tensor): probability maps for images [M x C x XYZ]
        """
        self.data = data
        self.dictname = "probabilities"

    @classmethod
    def create(cls, data: list[Path] | torch.Tensor | np.ndarray):
        if isinstance(data, torch.Tensor):
            return cls(data=data)
        elif isinstance(data, np.ndarray):
            return cls(data=torch.from_numpy(data))
        elif isinstance(data, list):
            return ProbsFromFiles(data=data)
        else:
            raise ValueError(f"Received invalid data type {type(data)}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    @classmethod
    def mutual_information(
        cls, probs: Probs, device: torch.device = DEVICE
    ) -> torch.Tensor:
        """Compute mutual information.

        Args:
            probs (Probs): probability maps
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: expected entropy H x W x D (on device)
        """
        logger.info("Compute mutual information")
        compute_val = cls.pred_entropy(probs, device=device)
        compute_val -= cls.exp_entropy(probs, device=device)
        return compute_val

    @staticmethod
    def pred_entropy(probs: Probs, device: torch.device = DEVICE) -> torch.Tensor:
        """Compute predictive entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (Probs): paths to probability maps for image [C x XYZ] per item
                in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: predictive entropy H x W x D (on device)
        """
        logger.info("Compute pred entropy")
        reset_cuda_memory_stats(device=device)

        try:
            # 1 time is enough because data entire data is already
            # checked.
            factor_req_vram = 1.1
            compute_val = move_tensor_check_vram(
                probs.data, device=device, factor_required_vram=factor_req_vram
            )
            # Average across folds
            compute_val = compute_val.mean(dim=0)
            # Class-wise negative PE
            compute_val *= torch.log(compute_val)
            # Sum across classes
            # here we assume that all nans are stemming from -inf after log
            compute_val = -compute_val.nan_to_num().sum(dim=0)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain pred_entropy on CPU.")
            try:
                del compute_val
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.pred_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)

    @staticmethod
    def class_pred_entropy(probs: Probs, device: torch.device = DEVICE) -> torch.Tensor:
        """Compute predictive entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (Probs): paths to probability maps for image [C x XYZ] per item
                in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: predictive entropy C+1 x H x W x D (on device)
        """
        logger.info("Compute class pred entropy")
        reset_cuda_memory_stats(device=device)

        try:
            factor_req_vram = 1.1
            compute_val = move_tensor_check_vram(
                probs.data, device=device, factor_required_vram=factor_req_vram
            )
            # Average across folds
            compute_val = compute_val.mean(dim=0)
            # Class-wise negative PE
            pred_entropy = compute_val * torch.log(compute_val)
            # Sum across classes
            # here we assume that all nans are stemming from -inf after log
            pred_entropy = -pred_entropy.nan_to_num().sum(dim=0, keepdim=True)
            compute_val = torch.cat([pred_entropy * compute_val, pred_entropy], dim=0)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain pred_entropy on CPU.")
            try:
                del compute_val
            except:
                pass
            try:
                del pred_entropy
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.class_pred_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)

    @staticmethod
    def exp_entropy(probs: Probs, device: torch.device = DEVICE) -> torch.Tensor:
        """Compute predictive entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (Probs): paths to probability maps for image [C x XYZ] per item
                in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: predictive entropy H x W x D (on device)
        """
        logger.info("Compute exp entropy")
        reset_cuda_memory_stats(device=device)

        try:
            factor_req_vram = 2 * 1.1
            compute_val = move_tensor_check_vram(
                probs.data, device=device, factor_required_vram=factor_req_vram
            )
            # Class-wise negative PE
            # Why is the deepcopy necessary again?
            compute_val = deepcopy(compute_val) * torch.log(compute_val)
            # Sum across classes
            # here we assume that all nans are stemming from -inf after log
            compute_val = compute_val.nan_to_num().sum(dim=1)
            # Average across folds
            compute_val = -compute_val.mean(dim=0)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain pred_entropy on CPU.")
            try:
                del compute_val
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.exp_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)


class ProbsFromFiles(Probs):
    def __init__(self, data: list[Path]):
        """Internal Class to hold probs stored in files, provide easy access, and
        calculate statistics on them.

        Args:
            data (list[Path]): Paths to probability maps of M images, each of shape
                [C x XYZ].
        """
        self.data = data
        self.dictname = "probabilities"

    def __getitem__(self, index: int) -> torch.Tensor:
        file = np.load(self.data[index])
        if isinstance(file, np.ndarray):
            return torch.from_numpy(file)
        elif isinstance(file, NpzFile):
            return torch.from_numpy(file[self.dictname])

    @staticmethod
    def mean_prob(probs: ProbsFromFiles, device: torch.device = DEVICE) -> torch.Tensor:
        """Compute average probability map across folds.

        Args:
            probs (ProbsFromFiles): paths to probability maps for image [C x XYZ] per
                item in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: average probability map C x H x W x D (on device)
        """
        logger.info("Compute mean prob")
        if not isinstance(probs, ProbsFromFiles):
            logger.warning(
                "ProbsFromFiles.mean_prob should be called on a 'ProbsFromFiles' "
                f"object. Received object of type '{type(probs)}'."
            )
        reset_cuda_memory_stats(device=device)

        try:
            compute_val = deepcopy(probs[0])
            factor_req_vram = 2 * 1.1
            compute_val = move_tensor_check_vram(
                compute_val, device=device, factor_required_vram=factor_req_vram
            )
            for fold in range(1, len(probs)):
                compute_val += probs[fold].to(compute_val.device)
            compute_val /= len(probs)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain mean_prob on CPU.")
            del compute_val
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.mean_prob(probs, torch.device("cpu"))

        return compute_val.to(device)

    @staticmethod
    def pred_entropy(
        probs: ProbsFromFiles, device: torch.device = DEVICE
    ) -> torch.Tensor:
        """Compute predictive entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (ProbsFromFiles): paths to probability maps for image [C x XYZ] per
                item in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: predictive entropy H x W x D (on device)
        """
        logger.info("Compute pred entropy")
        if not isinstance(probs, ProbsFromFiles):
            logger.warning(
                "ProbsFromFiles.pred_entropy should be called on a 'ProbsFromFiles' "
                f"object. Received object of type '{type(probs)}'."
            )
        reset_cuda_memory_stats(device=device)

        compute_val = ProbsFromFiles.mean_prob(probs, device=device)
        try:
            compute_val *= torch.log(compute_val)
            # here we assume that all nans are stemming from -inf after log
            compute_val = compute_val.nan_to_num()
            compute_val = -1 * compute_val.sum(dim=0)
        except RuntimeError as e:
            logger.debug("Possibly CUDA OOM error, try to obtain pred_entropy on CPU.")
            del compute_val
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.pred_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)

    @staticmethod
    def class_pred_entropy(probs: Probs, device: torch.device = DEVICE) -> torch.Tensor:
        """Compute predictive entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (Probs): paths to probability maps for image [C x XYZ] per item
                in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: predictive entropy C+1 x H x W x D (on device)
        """
        logger.info("Compute  class pred entropy")
        if not isinstance(probs, ProbsFromFiles):
            logger.warning(
                "ProbsFromFiles.class_pred_entropy should be called on a 'ProbsFromFiles' "
                f"object. Received object of type '{type(probs)}'."
            )
        reset_cuda_memory_stats(device=device)

        compute_val = ProbsFromFiles.mean_prob(probs, device=device)

        try:
            factor_req_vram = 2 * 1.1
            compute_val = move_tensor_check_vram(
                compute_val, device=device, factor_required_vram=factor_req_vram
            )
            # Class-wise negative PE
            pred_entropy = compute_val * torch.log(compute_val)
            # Sum across classes
            # here we assume that all nans are stemming from -inf after log
            pred_entropy = -pred_entropy.nan_to_num().sum(dim=0, keepdim=True)
            compute_val = torch.cat([pred_entropy * compute_val, pred_entropy], dim=0)
        except RuntimeError as e:
            logger.debug(
                "Possibly CUDA OOM error, try to obtain class_pred_entropy on CPU."
            )
            try:
                del compute_val
            except:
                pass
            try:
                del pred_entropy
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.class_pred_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)

    @staticmethod
    def exp_entropy(
        probs: ProbsFromFiles, device: torch.device = DEVICE
    ) -> torch.Tensor:
        """Compute expected entropy on list of paths saving npy arrays or a tensor.

        Args:
            probs (ProbsFromFiles): paths to probability maps for image [C x XYZ] per
                item in list or [M x C x XYZ].
            device (torch.device, optional): preferred device for computation. Defaults to DEVICE.

        Returns:
            torch.Tensor: expected entropy H x W x D (on device)
        """
        logger.info("Compute exp entropy")
        if not isinstance(probs, ProbsFromFiles):
            logger.warning(
                "ProbsFromFiles.exp_entropy should be called on a 'ProbsFromFiles' "
                f"object. Received object of type '{type(probs)}'."
            )
        reset_cuda_memory_stats(device=device)

        try:
            compute_val = deepcopy(probs[0])
            factor_req_vram = 2 + (2 / compute_val.shape[0])
            compute_val = move_tensor_check_vram(
                compute_val, device=device, factor_required_vram=factor_req_vram
            )
            compute_val *= torch.log(compute_val)
            compute_val = compute_val.nan_to_num()
            compute_val = (-1 / len(probs)) * compute_val.sum(dim=0)
            for fold in range(1, len(probs)):
                temp_val = deepcopy(probs[fold]).to(compute_val.device)
                temp_val *= torch.log(temp_val)
                # set all nan values (nan, inf, -inf) to 0
                temp_val = temp_val.nan_to_num()
                compute_val += (-1 / len(probs)) * temp_val.sum(dim=0)
        except RuntimeError as e:
            logger.debug("Possible CUDA OOM error, try to obtain exp_entropy on CPU.")
            logger.debug(e)
            del compute_val
            try:
                del temp_val
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            compute_val = probs.exp_entropy(probs, torch.device("cpu"))

        return compute_val.to(device)
