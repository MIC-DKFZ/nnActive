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
    overwrite: bool = True,
    additional_label_path: Path = None,
):
    """Overwrites the labels files in target_dir based on labels in
    base_dir.

    Disclaimer:
    Labels from additional_label_path (if used) are added last and overwrite GT from labelsTr.
    All areas inside images in additional_label_path that are not -1 will be written to labelsTr.

    Args:
        patches (list[Patch]): Regions to annotate
        ignore_label (int): Set Regions not mentioned in patches
        file_ending (str): File Ending
        base_dir (Path): Source Directory with labels
        target_dir (Path): Target Directory for labels
        overwrite (bool): If true the whole dataset is overwritten based on loop files and base_dir
        additional_label_path (Path, optional): path to files with labels to be added to labelsTr. Defaults to None
    """
    whole_label = []
    patch_label = []
    for patch in patches:
        if patch.size == "whole":
            whole_label.append(patch)
        else:
            patch_label.append(patch)

    labeled_files = set([patch.file for patch in (patch_label + whole_label)])
    make_whole_from_ground_truth(whole_label, base_dir, target_dir)

    make_patches_from_ground_truth(
        patch_label,
        base_dir,
        target_dir,
        ignore_label,
        overwrite=overwrite,
        additional_label_path=additional_label_path,
    )

    if overwrite:
        empty_segs = [
            file
            for file in os.listdir(base_dir)
            if file.endswith(file_ending) and file not in labeled_files
        ]
        make_empty_from_ground_truth(
            empty_segs,
            base_dir,
            target_dir,
            ignore_label,
            additional_label_path=additional_label_path,
        )
