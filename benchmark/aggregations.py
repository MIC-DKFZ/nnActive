import math
import time
from typing import Iterable

import numpy as np
import torch

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch


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
    # torch.cuda.synchronize()
    start_time = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def get_array(shape: Iterable[int] = (524, 524, 524)):
    return torch.arange(math.prod(shape), dtype=torch.float).view(*shape)


image_shapes = [64, 128, 256, 512]
kernel_shapes = [32, 64, 128, 128]
agg_classes = [ConvolveAggScipy, ConvolveAggTorch]
num_dims = 3

if __name__ == "__main__":
    with torch.no_grad():
        torch.matmul(
            torch.ones(3000, 3000, device="cuda"), torch.ones(3000, 3000, device="cuda")
        )
        for kernel_s, image_s in zip(kernel_shapes, image_shapes):
            print(f"Image Shape: {image_s} \t Kernel Shape: {kernel_s}")
            input_image = get_array([image_s] * num_dims).to("cuda:0")
            for agg_c in agg_classes:
                aggregation = agg_c([kernel_s] * input_image.dim())
                result, elapsed_time = timeit(aggregation.forward, input_image)
                print("\t{}: {:.5f}".format(agg_c.__name__, elapsed_time))
