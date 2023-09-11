from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.utils.patches import get_slices_for_file_from_patch


class AbstractQueryMethod(ABC):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        file_ending: str = ".nii.gz",
        **kwargs,
    ):
        self.dataset_id = dataset_id
        self.query_size = query_size
        self.patch_size = patch_size
        self.file_ending = file_ending
        self.top_patches: list[dict] = []

    @abstractmethod
    def query(self, verbose=False) -> list[Patch]:
        pass

    @property
    def annotated_patches(self) -> list[Patch]:
        return get_patches_from_loop_files(get_raw_path(self.dataset_id))

    def initialize_selected_array(
        self,
        image_shape: Iterable[int],
        label_file: str,
        annotated_patches: list[Patch],
    ) -> np.ndarray:
        """Initializes the array which simulates which are already selected.

        Args:
            image_shape (Iterable[int]): shape of initial image
            label_file (str): name of label file (with ending e.g. .nii.gz)
            annotated_patches (list[Patch]): list of already annotated patches

        Returns:
            np.ndarray: boolean array with annotated areas having value True
        """
        selected_array = np.zeros(image_shape, dtype=bool)

        # mark patches as selected
        patch_access = get_slices_for_file_from_patch(annotated_patches, label_file)

        for slices in patch_access:
            selected_array[slices] = 1
        return selected_array

    @classmethod
    def init_from_dataset_id(cls, dataset_id: int, **kwargs):
        config = ActiveConfig.get_from_id(dataset_id)
        config_kwargs = config.to_dict()
        config_kwargs.pop("seed")
        return cls(dataset_id=dataset_id, **config_kwargs, **kwargs)
