import os
from pathlib import Path

from nnactive.data import Patch
from nnactive.data.starting_budget import (
    make_empty_from_ground_truth,
    make_patches_from_ground_truth,
    make_whole_from_ground_truth,
)


def create_labels_from_patches(
    patches: list[Patch],
    ignore_label: int,
    file_ending: str,
    base_dir: Path,
    target_dir: Path,
):
    whole_label = []
    patch_label = []
    for patch in patches:
        if patch.size == "whole":
            whole_label.append(patch)
        else:
            patch_label.append(patch)

    labeled_files = set([patch.file for patch in (patch_label + whole_label)])
    empty_segs = [file for file in os.listdir(base_dir) if file.endswith(file_ending)]
    empty_segs = [file for file in empty_segs if file not in labeled_files]

    make_whole_from_ground_truth(whole_label, base_dir, target_dir)

    make_patches_from_ground_truth(
        patch_label,
        base_dir,
        target_dir,
        ignore_label,
    )

    make_empty_from_ground_truth(
        empty_segs,
        base_dir,
        target_dir,
        ignore_label,
    )
