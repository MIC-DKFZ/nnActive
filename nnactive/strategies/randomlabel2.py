import random
from pathlib import Path
from typing import List

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.strategies.randomlabel import RandomLabel
from nnactive.strategies.utils import query_starting_budget_all_classes


class RandomRandomLabel(RandomLabel):
    def get_area(self):
        area = self.rng.choice(["all", "all", "all", "all", "seg", "border"])
        return area


class RandomRandomLabelAllClasses(RandomRandomLabel):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        seed: int,
        trials_per_img: int = 600,
        file_ending: str = ".nii.gz",
        raw_labels_path: Path | None = None,
        background_cls: int | None = None,
        additional_label_path: Path | None = None,
        additional_overlap: float = 0.1,
        verbose: bool = False,
        config: ActiveConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            seed,
            trials_per_img,
            file_ending,
            raw_labels_path,
            background_cls,
            additional_label_path,
            additional_overlap,
            verbose=verbose,
            config=config,
            **kwargs,
        )
        random.seed(seed)

    def query(self, verbose: bool = False, n_gpus: int = 0, **kwargs) -> List[Patch]:
        # Do stuff to ensure all lables are represented two times
        selected_patches = query_starting_budget_all_classes(
            self.raw_labels_path,
            self.file_ending,
            annotated_patches=self.annotated_patches,
            patch_size=self.patch_size,
            rng=self.rng,
            trials_per_img=self.trials_per_img,
            additional_label_path=self.additional_label_path,
            additional_overlap=self.additional_overlap,
            verbose=verbose,
        )
        return super().query(
            verbose=verbose, already_annotated_patches=selected_patches
        )
