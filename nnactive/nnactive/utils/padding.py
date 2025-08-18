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
    for o_s, c_s in zip(old_shape, cur_shape):
        assert o_s <= c_s
    assert len(old_shape) == len(cur_shape)

    cur_shape = np.array(cur_shape)
    old_shape = np.array(old_shape)
    difference = cur_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = np.stack([pad_below, cur_shape - pad_above], axis=1)
    slicer = tuple(slice(*i) for i in pad_list)
    return slicer
