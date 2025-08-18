import random
from pathlib import Path
from typing import List

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.strategies.randomlabel import RandomLabel
from nnactive.strategies.registry import register_strategy
from nnactive.strategies.utils import query_starting_budget_all_classes


@register_strategy("random-label2")
class RandomRandomLabel(RandomLabel):
    def get_area(self):
        area = self.rng.choice(["all", "all", "all", "all", "seg", "border"])
        return area


@register_strategy("random-label2-all-classes")
class RandomRandomLabelAllClasses(RandomRandomLabel):
    def query(self, verbose: bool = False, n_gpus: int = 0, **kwargs) -> List[Patch]:
        # Do stuff to ensure all lables are represented two times
        selected_patches = query_starting_budget_all_classes(
            self.raw_labels_path,
            self.file_ending,
            annotated_patches=self.annotated_patches,
            patch_size=self.config.patch_size,
            rng=self.rng,
            trials_per_img=self.trials_per_img,
            additional_label_path=self.additional_label_path,
            additional_overlap=self.config.additional_overlap,
            verbose=verbose,
        )
        return super().query(
            verbose=verbose, already_annotated_patches=selected_patches
        )
