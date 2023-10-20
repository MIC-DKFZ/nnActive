from typing import Union

import numpy as np
import torch
import torch.nn as nn
from acvl_utils.morphology.gpu_binary_morphology import gpu_binary_erosion
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op


def maybe_gpu_binary_erosion(
    binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray
) -> Union[np.ndarray, torch.Tensor]:
    if torch.cuda.is_available():
        try:
            return gpu_binary_erosion(binary_array, selem)
        except RuntimeError:
            return gpu_binary_erosion(binary_array, selem)
    else:
        return cpu_binary_erosion(binary_array, selem)


def cpu_binary_erosion(
    binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray
) -> Union[np.ndarray, torch.Tensor]:
    """
    IMPORTANT: ALWAYS benchmark your image and kernel sizes first. Sometimes GPU is actually slower than CPU!
    """
    # cudnn.benchmark True DESTROYS our computation time (like 30X decrease lol). Make sure it's disabled (we set is
    # back below)
    assert all(
        [i % 2 == 1 for i in selem.shape]
    ), f"Only structure elements of uneven shape supported. Shape is {selem.shape}"

    with torch.no_grad():
        # move source array to GPU first. Uses non-blocking (important!) so that copy operation can run in background.
        # Cast to half only on the GPU because that is much faster and because the source array is quicker to
        # transfger the less bytes per element it has.
        is_tensor = isinstance(binary_array, torch.Tensor)
        if not is_tensor:
            binary_array = torch.from_numpy(binary_array).float()
        orig_device = binary_array.device
        binary_array = binary_array.to("cpu", non_blocking=True)

        # initialize conv as half
        conv = convert_dim_to_conv_op(len(binary_array.shape))(
            in_channels=1,
            out_channels=1,
            kernel_size=selem.shape,
            stride=1,
            padding="same",
            bias=False,
        )
        conv.weight = nn.Parameter(torch.from_numpy(selem[None, None]).float())
        conv = conv.to("cpu", non_blocking=True)

        # no need for autocast because everything is half already (I tried and it doesn't improve computation times)
        # again convert to 1 byte per element byte on GPU, then copy
        out = (conv(binary_array[None, None]) == selem.sum()).to(orig_device)[0, 0]

    if not is_tensor:
        out = out.numpy()
    # revert changes to cudnn.benchmark
    return out


if __name__ == "__main__":
    from skimage.morphology import ball

    test = np.zeros([10, 10, 10])
    test[3:7, 3:7, 3:7] = 1

    cpu_ = cpu_binary_erosion(test, ball(2))
    gpu_ = gpu_binary_erosion(test, ball(2))
    assert np.equal(cpu_, gpu_).all()
