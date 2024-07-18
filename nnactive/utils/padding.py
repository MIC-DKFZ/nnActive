from typing import Iterable

import numpy as np


def obtain_center_padding_slicers(
    old_shape: Iterable[int], cur_shape: Iterable[int]
) -> tuple[slice, ...]:
    """Returns the slices which allow to go from shape after padding to shape before padding.
    Padding is assumed to be centered with padding above resolving issues of N%2=1


    Args:
        old_shape (Iterable[int]): (120, 40, 40)
        cur_shape (Iterable[int]): (200, 50, 51)

    Returns:
        tuple[slice, ...]: (slice(40, 160), slice(5, 45), slice(5, 45))
    """
    for i in range(old_shape):
        assert old_shape[i] <= cur_shape[i]
    assert len(old_shape) == len(cur_shape)

    difference = cur_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    pad_list = np.array(pad_list)
    pad_list[:, 1] = np.array(cur_shape.shape) - pad_list[:, 1]
    slicer = tuple(slice(*i) for i in pad_list)
    return slicer
