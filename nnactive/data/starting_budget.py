from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from rich.progress import track

from nnactive.data import Patch
from nnactive.data.create_empty_masks import create_empty_mask


def _patch_ids_to_image_coords(
    patch_ids: list[int] | npt.NDArray[np.int_],
    bins: npt.NDArray[np.int_],
    files: list[str],
    sizes: list[tuple[int, int, int]],
    patch_size: int,
) -> list[Patch]:
    img_ids: list[int] = np.digitize(patch_ids, bins).tolist()
    coords = []
    for patch_id, img_id in zip(patch_ids, img_ids):

        patch_id: int = patch_id - bins[img_id - 1]

        size_x, size_y, _ = sizes[img_id]
        size_x = size_x // patch_size
        size_y = size_y // patch_size

        patch_z = patch_id // (size_x * size_y)
        tmp = patch_id % (size_x * size_y)
        patch_y = tmp // size_x
        patch_x = tmp % size_x

        coords.append(
            Patch(
                file=files[img_id],
                coords=(
                    patch_x * patch_size,
                    patch_y * patch_size,
                    patch_z * patch_size,
                ),
                size=(patch_size, patch_size, patch_size),
            )
        )

    return coords


def _compute_patch_mapping(
    dataset_cfg: dict[str, Any],
    raw_path: Path,
) -> dict[str, tuple[int, int, int]]:
    img_sizes = {}

    if not (raw_path / "img_sizes.json").is_file():
        imgs = list(
            (raw_path / "gt_labelsTr").glob(f"**/*{dataset_cfg['file_ending']}")
        )

        for path in track(imgs):
            img = sitk.ReadImage(path)
            name = path.name.replace(dataset_cfg["file_ending"], "")
            size = img.GetSize()

            print(f"{name}: {size}")
            img_sizes[name] = size

        with open(raw_path / "img_sizes.json", "w") as file:
            json.dump(img_sizes, file)
    else:
        with open(raw_path / "img_sizes.json") as file:
            img_sizes = json.load(file)

    img_sizes = dict(sorted(img_sizes.items()))
    return img_sizes


def generate_patches_random_grid(
    dataset_cfg: dict[str, Any],
    raw_path: Path,
    patch_size: int,
    n_samples: int,
) -> list[Patch]:
    """Generate patches randomly on a grid.

    Args:
        dataset_cfg: nnUNet dataset configuration
        raw_path: nnUNet raw path
        patch_size:
        n_samples:

    Returns:
        list of generated patches
    """
    img_sizes = _compute_patch_mapping(
        dataset_cfg,
        raw_path,
    )
    n_patches: npt.NDArray[np.int_] = np.array(
        list(
            size_x // patch_size * size_y // patch_size * size_z // patch_size
            for (size_x, size_y, size_z) in img_sizes.values()
        )
    )
    bins = np.cumsum(n_patches)
    files = list(img_sizes.keys())
    sizes = list(img_sizes.values())

    n_patches_total = np.sum(n_patches)
    patch_ids = np.random.choice(n_patches_total, n_samples, replace=False)

    patches = _patch_ids_to_image_coords(patch_ids, bins, files, sizes, patch_size)

    return patches


def _random_crop(
    size_x: int,
    size_y: int,
    size_z: int,
    crop_min_size: float = 0,
    crop_max_size: float = 1,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    psize_x = np.random.randint(
        int(crop_min_size * size_x), int(crop_max_size * size_x)
    )
    psize_y = np.random.randint(
        int(crop_min_size * size_y), int(crop_max_size * size_y)
    )
    psize_z = np.random.randint(
        int(crop_min_size * size_z), int(crop_max_size * size_z)
    )

    pstart_x = np.random.randint(0, size_x - psize_x + 1)
    pstart_y = np.random.randint(0, size_y - psize_y + 1)
    pstart_z = np.random.randint(0, size_z - psize_z + 1)

    return (pstart_x, pstart_y, pstart_z), (psize_x, psize_y, psize_z)


def generate_patches_random_crop(
    dataset_cfg: dict[str, Any],
    raw_path: Path,
    crop_min_size: float = 0,
    crop_max_size: float = 1,
) -> list[Patch]:
    """Generate patches from random crops.

    Args:
        dataset_cfg: nnUNet dataset configuration
        raw_path: nnUNet raw path
        crop_min_size: minimum relative crop size, must be between 0 and 1
        crop_max_size: maximum relative crop size, must be between 0 and 1
            and larger than crop_min_size

    Returns:
        list of generated patches
    """
    img_sizes = _compute_patch_mapping(dataset_cfg, raw_path)

    crops: list[Patch] = []
    for file, (size_x, size_y, size_z) in img_sizes.items():
        coord, size = _random_crop(size_x, size_y, size_z, crop_min_size, crop_max_size)
        crops.append(Patch(file=file, coords=coord, size=size))

    return crops


def make_patches_from_ground_truth(
    patches: list[Patch],
    gt_path: Path,
    target_path: Path,
    ignore_label: int,
) -> None:
    """Create label files where only some patches are labeled from ground truth
        and the rest are ignored.

    Args:
        patches: list of patches to label
        gt_path: where the ground truth labels are stored
        target_path: where the patched labels should be stored
        ignore_label: the id for ignored labels
    """
    target_path.mkdir(exist_ok=True)

    for patch in track(patches):
        img_gt = sitk.ReadImage((gt_path / patch.file))
        spacing = img_gt.GetSpacing()
        origin = img_gt.GetOrigin()
        direction = np.array(img_gt.GetDirection())
        img_gt = sitk.GetArrayFromImage(img_gt)

        img_new = np.full_like(img_gt, ignore_label)

        slice_x = slice(patch.coords[0], patch.coords[0] + patch.size[0])
        slice_y = slice(patch.coords[1], patch.coords[1] + patch.size[1])
        slice_z = slice(patch.coords[2], patch.coords[2] + patch.size[2])

        img_new[slice_x, slice_y, slice_z] = img_gt[slice_x, slice_y, slice_z]

        img_new = sitk.GetImageFromArray(img_new)
        img_new.SetSpacing(spacing)
        img_new.SetOrigin(origin)
        img_new.SetDirection(direction)
        sitk.WriteImage(
            img_new,
            (target_path / patch.file),
        )


def make_whole_from_ground_truth(
    patches: list[Patch], gt_path: Path, target_path: Path
):
    """Copies over all files in patches from gt_path to target_path"""
    for patch in patches:
        seg_name = patch.file
        shutil.copy(gt_path / seg_name, target_path / seg_name)


def make_empty_from_ground_truth(
    seg_names: list[str], gt_path: Path, target_path: Path, ignore_label: int
):
    """Creates empty segmentations for all filenames"""
    for seg_name in seg_names:
        create_empty_mask(gt_path / seg_name, ignore_label, target_path / seg_name)
