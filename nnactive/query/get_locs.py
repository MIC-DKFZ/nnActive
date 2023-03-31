# Code by Fabian Isensee

import numpy as np
from acvl_utils.morphology.gpu_binary_morphology import gpu_binary_erosion
from skimage.morphology import ball
from torch.backends import cudnn


def get_locs_from_segmentation(orig_seg: np.ndarray, area="seg"):
    # if len(np.unique(orig_seg)) > 2:
    #     raise NotImplementedError(
    #         "Currently only foreground and backround are supported for this functionality"
    #     )
    cudnn.deterministic = False
    cudnn.benchmark = False
    if area == "border":
        orig_seg_border = orig_seg - gpu_binary_erosion(orig_seg > 0, ball(1))
        return np.argwhere(orig_seg_border > 0)
    elif area == "seg":
        return np.argwhere(orig_seg > 0)
    else:
        raise NotImplementedError
