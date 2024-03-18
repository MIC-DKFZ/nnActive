from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.masking import does_overlap, percentage_overlap, percentage_overlap_array
from nnactive.nnunet.utils import get_raw_path
from nnactive.utils.patches import get_slices_for_file_from_patch


class AbstractQueryMethod(ABC):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        file_ending: str = ".nii.gz",
        additional_label_path: Path | None = None,
        additional_overlap: float = 0.1,
        patch_overlap: float = 0,
        verbose: bool = False,
        **kwargs,
    ):
        self.dataset_id = dataset_id
        self.additional_label_path = additional_label_path
        self.query_size = query_size
        self.patch_size = patch_size
        self.file_ending = file_ending
        self.top_patches: list[dict] = []
        self.additional_overlap = additional_overlap
        self.patch_overlap = patch_overlap
        self.verbose = verbose

    @abstractmethod
    def query(self, verbose=False) -> list[Patch]:
        pass

    @property
    def annotated_patches(self) -> list[Patch]:
        return get_patches_from_loop_files(get_raw_path(self.dataset_id))

    def check_overlap(
        self,
        ipatch: Patch,
        patches: list[Patch],
        additional_label: None | np.ndarray = None,
        verbose: bool = False,
    ) -> bool:
        # start with checking overlap compared to other patches
        allow_patch = False
        if self.patch_overlap > 0:
            patch_overlap = percentage_overlap(ipatch, patches)
            allow_patch = patch_overlap <= self.patch_overlap
            if verbose and allow_patch:
                logger.debug(
                    f"Patch creation succesful with patch overlap: {patch_overlap} <= {self.patch_overlap} overlap with additional labels."
                )
        else:
            allow_patch = not does_overlap(ipatch, patches)

        # check overlap with additional labels
        if additional_label is not None and allow_patch:
            additional_overlap = percentage_overlap_array(ipatch, additional_label)
            if additional_overlap <= self.additional_overlap:
                if verbose:
                    logger.debug(
                        f"Patch creation succesful with additional labels overlap: {additional_overlap} <= {self.additional_overlap} overlap with additional labels."
                    )
                return True
            else:
                allow_patch = False
        return allow_patch

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

        additional_label_path: Path = get_raw_path(dataset_id) / "addTr"
        if not additional_label_path.is_dir():
            additional_label_path = None
        return cls(
            dataset_id=dataset_id,
            additional_label_path=additional_label_path,
            **config_kwargs,
            **kwargs,
        )
