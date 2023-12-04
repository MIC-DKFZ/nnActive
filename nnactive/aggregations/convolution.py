from typing import Iterable

import numpy as np
import torch
from scipy.signal import convolve


class ConvolveAgg:
    def __init__(self, patch_size: list[int]):
        self.patch_size = patch_size

    def forward(self, data: torch.Tensor) -> tuple[np.array, list[int]]:
        print("Started forward")
        kernel_size = [
            min(self.patch_size[i], data.shape[i]) for i in range(len(self.patch_size))
        ]
        print("Created kernel")
        kernel = np.ones(kernel_size)
        data = data.cpu().numpy()
        aggregated = convolve(data, kernel, mode="valid")
        print("Done with convolution")
        return aggregated, kernel_size

    def backward_index(
        self,
        aggregated_index: int,
        aggregated_shape: Iterable[int],
    ) -> tuple[int]:
        """Compute indices from the flattened aggregated images into starting coordinates for patches in image space

        Args:
            aggregated_index (int): 1 flattened index in aggregated space
            aggregated_shape (Iterable[int]): shape of aggregated image

        Returns:
            np.ndarray: corresponding index in image space
        """
        # Get the index as coordinates
        # this works after convolution index_img_start = index_conv_start, index_img_end = index_conv_start+ size_dim

        return tuple(
            [t.item() for t in np.unravel_index(aggregated_index, aggregated_shape)]
        )
