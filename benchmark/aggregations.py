import math
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from nnactive.aggregations.convolution import (  # ConvolveAggTorchFFT,
    ConvolveAggScipy,
    ConvolveAggTorch,
)
from nnactive.utils.torchutils import get_tensor_memory_usage


def timeit(func, *args, **kwargs):
    """
    Wrapper function to measure the execution time of another function.

    Parameters:
    - func: The function to be executed and timed.
    - *args: Positional arguments to be passed to the function.
    - **kwargs: Keyword arguments to be passed to the function.

    Returns:
    - result: The result of the wrapped function.
    - elapsed_time: The time taken for the function to execute.
    """
    torch.cuda.synchronize()
    start_time = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def get_array(shape: Iterable[int] = (524, 524, 524)):
    return torch.arange(math.prod(shape), dtype=torch.float).view(*shape)


image_shapes = [64, 128, 256, 512, 512, 512]
kernel_shapes = [32, 64, 128, 64, 128, 256]
agg_classes = [
    ConvolveAggScipy,
    ConvolveAggTorch,
    # ConvolveAggTorchFFT
]
agg_kwargs = [
    {},
    {"stride": 8},
    # {"stride": 8}
]
num_dims = 3

if __name__ == "__main__":
    with torch.no_grad():
        torch.matmul(
            torch.ones(3000, 3000, device="cuda"), torch.ones(3000, 3000, device="cuda")
        )
        all_results = []
        for kernel_s, image_s in zip(kernel_shapes, image_shapes):
            print(f"Image Shape: {image_s} \t Kernel Shape: {kernel_s}")
            input_image = get_array([image_s] * num_dims).to("cuda:0")
            input_image_size = get_tensor_memory_usage(input_image)
            results_dict = {
                "Image Shape": image_s,
                "Image Size": input_image_size,
                "Kernel Shape": kernel_s,
            }
            torch.cuda.reset_peak_memory_stats()
            for agg_c, agg_kw in zip(agg_classes, agg_kwargs, strict=True):
                aggregation = agg_c([kernel_s] * input_image.dim(), **agg_kw)
                result, elapsed_time = timeit(aggregation.forward, input_image)
                print("\t{}: {:.5f}".format(agg_c.__name__, elapsed_time))
                max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                results_dict[f"Time {agg_c.__name__}"] = elapsed_time
                results_dict[f"Mem {agg_c.__name__}"] = max_memory_allocated
                results_dict[f"Ratio max mem/input {agg_c.__name__}"] = (
                    max_memory_allocated / input_image_size
                )
                torch.cuda.reset_peak_memory_stats()
            all_results.append(results_dict)
    df = pd.DataFrame(all_results)
    print(df.to_string())
