# Code by Fabian Isensee

from typing import Union

import numpy as np
from acvl_utils.morphology.gpu_binary_morphology import gpu_binary_erosion
from skimage.morphology import ball
from torch.backends import cudnn


def get_locs_from_segmentation(
    orig_seg: np.ndarray,
    area="seg",
    state: np.random.RandomState = np.random.default_rng(),
    background_cls: Union[int, None] = 0,
):
    unique_cls = np.unique(orig_seg)
    delete_cls = [cl for cl in unique_cls if cl < 0]

    print(f"Ignoring Cls < 0 for Patch Selection: {delete_cls}")
    print(f"Ignoring Background Class for Selection: {background_cls}")
    unique_cls = np.array([cl for cl in unique_cls if cl not in delete_cls])
    counter = 0
    selected_cls = background_cls
    while selected_cls == background_cls:
        if counter == 200:
            raise RuntimeError("There is no non-background class in this image!")
        selected_cls = state.choice(unique_cls, 1).item()
        counter += 1
    print(f"Select Area for Class {selected_cls}")
    use_seg = (orig_seg == selected_cls).astype(np.int8)

    cudnn.deterministic = False
    cudnn.benchmark = False
    if area == "border":
        use_seg_border = use_seg - gpu_binary_erosion(use_seg > 0, ball(1))
        return np.argwhere(use_seg_border > 0)
    elif area == "seg":
        return np.argwhere(use_seg > 0)
    else:
        raise NotImplementedError
