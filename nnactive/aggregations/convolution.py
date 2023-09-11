from typing import Iterable

import numpy as np
import torch


class ConvolveAgg:
    def __init__(self, patch_size: list[int]):
        self.patch_size = patch_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        kernel_size = [
            min(self.patch_size[i], data.shape[i]) for i in range(len(self.patch_size))
        ]
        kernel = torch.ones(size=[1, 1] + kernel_size, device=data.device) / np.prod(
            kernel_size
        )

        if len(data.shape) == 2:
            aggregated = torch.nn.functional.conv2d(
                data.unsqueeze(0).unsqueeze(0),
                weight=kernel,
            )
        elif len(data.shape) == 3:
            aggregated = torch.nn.functional.conv3d(
                data.unsqueeze(0).unsqueeze(0),
                weight=kernel,
            )
        else:
            raise NotImplementedError()
        return aggregated.squeeze(), kernel_size

    def backward_index(
        self,
        aggregated_index: np.ndarray,
        aggregated_shape: Iterable[int],
    ) -> np.ndarray:
        """Compute indices from the aggregated images into starting coordinates for patches in image space

        Args:
            aggregated_index (np.ndarray): index in aggregated space
            aggregated_shape (Iterable[int]): shape of aggregated image

        Returns:
            np.ndarray: corresponding index in image space
        """
        # Get the index as coordinates
        # this works after convolution index_img_start = index_conv_start, index_img_end = index_conv_start+ size_dim
        return np.unravel_index(aggregated_index, aggregated_shape)
