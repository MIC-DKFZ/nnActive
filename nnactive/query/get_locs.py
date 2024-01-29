# Code by Fabian Isensee

from typing import Union

import numpy as np
from loguru import logger
from skimage.morphology import ball
from torch.backends import cudnn

from nnactive.utils.torchutils import maybe_gpu_binary_erosion


def get_locs_from_segmentation(
    orig_seg: np.ndarray,
    area="seg",
    state: np.random.RandomState = np.random.default_rng(),
    background_cls: Union[int, None] = 0,
    verbose: bool = False,
):
    unique_cls = np.unique(orig_seg)
    delete_cls = [cl for cl in unique_cls if cl < 0]
    if len(delete_cls) > 0:
        logger.warning("Ignoring Cls < 0 for Patch Selection: {delete_cls}")

    if verbose:
        logger.debug(f"Ignoring Background Class for Selection: {background_cls}")
    unique_cls = np.array([cl for cl in unique_cls if cl not in delete_cls])
    counter = 0
    selected_cls = background_cls
    while selected_cls == background_cls:
        if counter == 200:
            raise RuntimeError("There is no non-background class in this image!")
        selected_cls = state.choice(unique_cls, 1).item()
        counter += 1
    if verbose:
        logger.debug(f"Select Area for Class {selected_cls}")
    use_seg = (orig_seg == selected_cls).astype(np.int8)

    cudnn.deterministic = False
    cudnn.benchmark = False
    if area == "border":
        use_seg_border = use_seg - maybe_gpu_binary_erosion(use_seg > 0, ball(1))
        return np.argwhere(use_seg_border > 0)
    elif area == "seg":
        return np.argwhere(use_seg > 0)
    else:
        raise NotImplementedError
