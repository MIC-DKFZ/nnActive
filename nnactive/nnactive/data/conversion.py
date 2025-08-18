import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger
from nnunetv2 import paths
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.data import Patch
from nnactive.data.annotate import create_labels_from_patches
from nnactive.data.create_empty_masks import (
    add_ignore_label_to_dataset_json,
    read_dataset_json,
)
from nnactive.loops.cross_validation import (
    kfold_cv_from_patches,
    read_classes_from_dataset_json,
)
from nnactive.loops.loading import save_loop
from nnactive.paths import nnActive_data, set_raw_paths
from nnactive.strategies import init_strategy
from nnactive.utils.io import save_json


def convert_dataset_to_partannotated(
    base_id: int,
    target_id: int,
    full_images: Union[float, int],
    num_patches: int,
    patch_size: tuple[int],
    name_suffix: str = "partanno",
    patch_kwargs: Optional[dict] = None,
    strategy: str = "random",
    seed: int = 12345,
    additional_overlap: float = 0.6,
    all_class_splits: bool = True,
):
    """Converts base dataset to partly annotated dataset for AL. Raises a RuntimeError if
    the target dataset folder already exists.
    """
    logger.info("Converting base dataset to partly annotated dataset...")

    # load base_dataset_json
    with set_raw_paths():
        base_dataset: str = convert_id_to_dataset_name(base_id)
        base_dataset_json: dict = read_dataset_json(base_dataset)
        base_dir = Path(paths.nnUNet_raw) / base_dataset

    # rewrite target_dataset_json and save
    target_dataset_json = deepcopy(base_dataset_json)
    target_dataset_json["name"] = "{}{}".format(
        target_dataset_json["name"], name_suffix
    )
    target_dataset_json = add_ignore_label_to_dataset_json(target_dataset_json)
    target_dataset_json["annotated_id"] = base_id
    target_dataset: str = f"Dataset{target_id:03d}_" + target_dataset_json["name"]
    target_dir = nnActive_data / base_dataset / "nnUNet_raw" / target_dataset
    try:
        target_dir.mkdir(parents=True)
    except FileExistsError as err:
        raise RuntimeError(
            f"The folder for experiment '{target_dir}' already exists. {target_id = }"
        ) from err
    # Save target dataset.json
    with open(target_dir / "dataset.json", "w") as file:
        json.dump(target_dataset_json, file, indent=4)
    assert (
        read_dataset_json(base_dataset) == base_dataset_json
    )  # basedataset/dataset.json is not supposed to change!

    # Copy all data except for labelsTr to target_dir and dataset.json
    copy_folders = [
        "imagesTr",
        "imagesTs",
        "labelsTs",
        "imagesVal",
        "labelsVal",
        "addTr",
    ]
    for copy_folder in copy_folders:
        if copy_folder in os.listdir(base_dir):
            (target_dir / copy_folder).symlink_to(
                base_dir / copy_folder, target_is_directory=True
            )
        else:
            logger.info(f"Skip Path for copying into target:\n{base_dir / copy_folder}")

    # Create labelstTr for target dataset
    base_labelsTr_dir = base_dir / "labelsTr"
    target_labelsTr_dir = target_dir / "labelsTr"

    ignore_label = target_dataset_json["labels"]["ignore"]
    background_cls = base_dataset_json["labels"].get("background")
    file_ending = base_dataset_json["file_ending"]

    additional_label_path = None
    if target_dataset_json.get("use_mask_for_norm") is True:
        additional_label_path: Path = target_dir / "addTr"

    # create patches list for dataset creation
    patches = get_patches_for_partannotation(
        full_images,
        num_patches,
        patch_size,
        file_ending,
        base_labelsTr_dir,
        target_labelsTr_dir,
        agg_stride=1,  # random queries do not use this variable
        patch_func_kwargs=patch_kwargs,
        strategy_name=strategy,
        seed=seed,
        background_cls=background_cls,
        dataset_id=target_id,
        additional_label_path=additional_label_path,
        additional_overlap=additional_overlap,
    )

    # Create labels from patches
    create_labels_from_patches(
        patches,
        ignore_label,
        file_ending,
        base_labelsTr_dir,
        target_labelsTr_dir,
        additional_label_path=additional_label_path,
    )

    loop_json = {"patches": patches}
    save_loop(target_dir, loop_json, 0)

    target_preprocessed = Path(paths.nnUNet_preprocessed) / target_dataset
    os.makedirs(nnActive_data / base_dataset / "nnUNet_preprocessed" / target_dataset)
    splits_file = target_preprocessed / "splits_final.json"
    if all_class_splits:
        ensure_classes = read_classes_from_dataset_json(target_dataset_json)
    else:
        ensure_classes = None
    splits = kfold_cv_from_patches(
        5,
        patches,
        ensure_classes=ensure_classes,
        labels_path=target_labelsTr_dir,
        file_ending=file_ending,
    )
    save_json(splits, splits_file)


def get_patches_for_partannotation(
    full_images: Union[int, float],
    num_patches: int,
    patch_size: tuple[int],
    file_ending: str,
    base_labelsTr_dir: Path,
    target_labelsTr_dir: Path,
    dataset_id: int,
    agg_stride: Union[int, list[int]] = 1,
    patch_func_kwargs: dict = None,
    strategy_name: str = "random",
    seed: int = 12345,
    background_cls: Union[int, None] = None,
    additional_label_path: Union[Path, None] = None,
    additional_overlap: float = 0.5,
) -> list[Patch]:
    """Creates patches based on annotation strategies.

    Args:
        full_images (Union[int, float]): _description_
        patch_func (callable): _description_
        file_ending (str): _description_
        base_labelsTr_dir (Path): _description_
        target_labelsTr_dir (Path): _description_
        patch_func_kwargs (dict, optional): _description_. Defaults to None.

    Returns:
        list[Patch]: annotated patches
    """
    if patch_func_kwargs is None:
        patch_func_kwargs = {}
    os.makedirs(target_labelsTr_dir, exist_ok=True)

    seg_names = os.listdir(base_labelsTr_dir)
    seg_names = [seg_name for seg_name in seg_names if seg_name.endswith(file_ending)]

    # Current implementation only works if all data has a corresponding lablesTr

    rand_np_state = np.random.RandomState(seed)
    rand_np_state.shuffle(seg_names)

    if full_images < 1:
        full_images = int(len(seg_names) * full_images)
    patches = [
        Patch(file=seg_names.pop(), coords=[0, 0, 0], size="whole")
        for i in range(full_images)
    ]
    logger.info(f"# whole image patches: {len(patches)}")

    strategy = init_strategy(
        strategy_name,
        dataset_id,
        query_size=num_patches,
        patch_size=patch_size,
        seed=seed,
        agg_stride=agg_stride,
        trials_per_img=6000,
        annotated_labels_path=base_labelsTr_dir,
        background_cls=background_cls,
        raw_labels_path=base_labelsTr_dir,
        additional_label_path=additional_label_path,
        additional_overlap=additional_overlap,
        n_patch_per_image=1,  # this value does not affect Random Queries
    )
    logger.info(f"Finished Initialization with strategy {strategy}")
    patches_partial = strategy.query(verbose=True)
    logger.info(f"#{strategy_name} based patches: {patches_partial}")
    patches = patches_partial + patches

    return patches
