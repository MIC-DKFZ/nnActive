from typing import Iterable, Union

import numpy as np
import torch
from scipy.signal import convolve


class ConvolveAggScipy:
    def __init__(self, patch_size: list[int], stride: Union[int, list[int]] = 1):
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * len(self.patch_size)
        if any([s != 1 for s in self.stride]):
            raise NotImplementedError("This class only supports stride 1")

    def forward(self, data: torch.Tensor) -> tuple[np.array, list[int]]:
        print("Started forward")
        kernel_size = [
            min(self.patch_size[i], data.shape[i]) for i in range(len(self.patch_size))
        ]
        print("Created kernel")
        kernel = np.ones(kernel_size) / np.prod(kernel_size)
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


class ConvolveAggTorch:
    def __init__(self, patch_size: list[int], stride: Union[int, list[int]] = 1):
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * len(self.patch_size)

    def forward(self, data: torch.Tensor) -> tuple[np.array, list[int]]:
        print("Started forward")
        kernel_size = [
            min(self.patch_size[i], data.shape[i]) for i in range(len(self.patch_size))
        ]
        print("Created kernel")
        kernel = torch.ones(size=[1, 1] + kernel_size, device=data.device) / np.prod(
            kernel_size
        )

        if len(data.shape) == 2:
            aggregated = torch.nn.functional.conv2d(
                data.unsqueeze(0).unsqueeze(0), weight=kernel, stride=self.stride
            )
        elif len(data.shape) == 3:
            aggregated = torch.nn.functional.conv3d(
                data.unsqueeze(0).unsqueeze(0), weight=kernel, stride=self.stride
            )
        else:
            raise NotImplementedError()
        return aggregated.squeeze(0).squeeze(0).cpu().numpy(), kernel_size

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
        # when a stride is added the index is just multiplied by the size of the stride.

        return tuple(
            [
                t.item() * self.stride[i]
                for i, t in enumerate(
                    np.unravel_index(aggregated_index, aggregated_shape)
                )
            ]
        )
