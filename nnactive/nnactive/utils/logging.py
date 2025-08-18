import psutil
import torch
from loguru import logger


def log_memory_usage(tag, gpu: bool = False):
    ram_usage = psutil.Process().memory_info().rss * 1e-9  # Convert to GB
    logger.info(f"{tag} - CPU RAM usage: {ram_usage:.2f} GB")

    if gpu and torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() * 1e-9  # Convert to GB
        gpu_memory_reserved = torch.cuda.memory_reserved() * 1e-9  # Convert to GB
        logger.info(
            f"{tag} - GPU memory allocated: {gpu_memory_allocated:.2f} GB, "
            f"GPU memory reserved: {gpu_memory_reserved:.2f} GB"
        )


def log_type(var, tag: str = None):
    logger.info(f"Type of {var.__name__}: {type(var)}")
