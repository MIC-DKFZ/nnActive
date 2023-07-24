# TODO: Split this into two different files, s.a. in other folders
import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.cli.registry import register_subcommand
from nnactive.data import Patch
from nnactive.data.annotate import create_labels_from_patches
from nnactive.data.create_empty_masks import (
    add_ignore_label_to_dataset_json,
    read_dataset_json,
)
from nnactive.loops.loading import save_loop
from nnactive.nnunet.io import generate_custom_splits_file
from nnactive.nnunet.utils import get_patch_size
from nnactive.query.random import (
    generate_random_patches,
    generate_random_patches_labels,
)
from nnactive.results.utils import (
    convert_id_to_dataset_name as nnactive_id_to_dataset_name,
)
from nnactive.utils.hostutils import get_verbose

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None


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
):
    already_exists = False
    try:
        exists_name = convert_id_to_dataset_name(target_id)
        print(
            f"Dataset with ID {target_id} already exists in nnU-Net under name {exists_name}."
        )
        already_exists = True
    except RuntimeError:
        print("No naming conflict with nnU-Net")
    try:
        exists_name = nnactive_id_to_dataset_name(target_id)
        print(
            f"Dataset with ID {target_id} already exists in nnActive under name {exists_name}."
        )
        already_exists = True
    except FileNotFoundError:
        print("No naming conflict with nnActive")

    if already_exists:
        raise NotImplementedError(
            "Dataset ID already exists, check corresponding folders."
        )
    if not already_exists:
        # load base_dataset_json
        base_dataset: str = convert_id_to_dataset_name(base_id)
        base_dataset_json: dict = read_dataset_json(base_dataset)
        base_dir = NNUNET_RAW / base_dataset

        # rewrite target_dataset_json and save
        target_dataset_json = deepcopy(base_dataset_json)
        target_dataset_json["name"] = "{}{}".format(
            target_dataset_json["name"], name_suffix
        )
        target_dataset_json = add_ignore_label_to_dataset_json(target_dataset_json)
        target_dataset_json["annotated_id"] = base_id
        target_dataset: str = f"Dataset{target_id:03d}_" + target_dataset_json["name"]
        target_dir = NNUNET_RAW / target_dataset
        os.makedirs(target_dir)
        # Save target dataset.json
        with open(target_dir / "dataset.json", "w") as file:
            json.dump(target_dataset_json, file, indent=4)
        assert (
            read_dataset_json(base_dataset) == base_dataset_json
        )  # basedataset/dataset.json is not supposed to change!

        # Copy all data except for labelsTr to target_dir and dataset.json
        copy_folders = ["imagesTr", "imagesTs", "labelsTs", "imagesVal", "labelsVal"]
        for copy_folder in copy_folders:
            if copy_folder in os.listdir(base_dir):
                shutil.copytree(base_dir / copy_folder, target_dir / copy_folder)
            else:
                print(
                    "Skip Path for copying into target:\n{}".format(
                        base_dir / copy_folder
                    )
                )

        # Create labelstTr for target dataset
        base_labelsTr_dir = base_dir / "labelsTr"
        target_labelsTr_dir = target_dir / "labelsTr"

        ignore_label = target_dataset_json["labels"]["ignore"]
        background_cls = base_dataset_json["labels"].get("background")
        file_ending = base_dataset_json["file_ending"]

        # create patches list for dataset creation
        patches = get_patches_for_partannotation(
            full_images,
            num_patches,
            patch_size,
            file_ending,
            base_labelsTr_dir,
            target_labelsTr_dir,
            patch_func_kwargs=patch_kwargs,
            strategy=strategy,
            seed=seed,
            background_cls=background_cls,
        )

        # Create labels from patches
        create_labels_from_patches(
            patches, ignore_label, file_ending, base_labelsTr_dir, target_labelsTr_dir
        )

        loop_json = {"patches": patches}
        save_loop(target_dir, loop_json, 0)


def get_patches_for_partannotation(
    full_images: Union[int, float],
    num_patches: int,
    patch_size: tuple[int],
    file_ending: str,
    base_labelsTr_dir: Path,
    target_labelsTr_dir: Path,
    patch_func_kwargs: dict = None,
    strategy: str = "random",
    seed: int = 12345,
    background_cls: Union[int, None] = None,
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
    print(f"# whole image patches: {len(patches)}")

    if strategy == "random":
        patches_partial = generate_random_patches(
            file_ending,
            base_labelsTr_dir,
            patch_size,
            n_patches=num_patches,
            labeled_patches=patches,
            seed=seed,
        )
    elif strategy == "random-label":
        patches_partial = generate_random_patches_labels(
            file_ending,
            base_labelsTr_dir,
            patch_size,
            n_patches=num_patches,
            labeled_patches=patches,
            seed=seed,
            background_cls=background_cls,
            verbose=get_verbose(None),
        )
    else:
        raise NotImplementedError(f"strategy `{strategy}` is not implemented")
    print(f"#{strategy} based patches: {patches_partial}")
    patches = patches_partial + patches

    return patches


@register_subcommand(
    "convert",
    [
        (
            ("-d", "--dataset-id"),
            {
                "type": int,
                "required": True,
                "help": "dataset ID for nnU-Net, needs to be present in $nnUNet_raw",
            },
        ),
        (
            ("-o", "--output-id"),
            {
                "type": int,
                "default": None,
                "help": "target dataset ID for nnU-Net, default base on offset",
            },
        ),
        (
            "--offset",
            {"type": int, "default": 500, "help": "ouput_id = dataset_id + offset"},
        ),
        (
            "--seed",
            {
                "default": 12345,
                "type": int,
                "help": "Random seed for creation of datasets",
            },
        ),
        (
            "--full-labeled",
            {
                "type": float,
                "default": 0,
                "help": "0.X = percentage, int = full number of completely annotated images",
            },
        ),  # how to make float and integers
        (
            "--strategy",
            {
                "type": str,
                "default": "random",
                "help": "strategy employed to select random patches",
            },
        ),
        (
            "--num-patches",
            {
                "type": int,
                "default": 0,
                "help": "Number of randomly drawn patches",
            },
        ),  # how to make float and integers
        (
            "--patch-size",
            {
                "type": int,
                "nargs": "+",
                "default": None,
                "help": "patch size of the object, default is nnU-Net Patch Size",
            },
        ),
        (
            "--name-suffix",
            {
                "type": str,
                "default": "partanno",
                "help": "Suffix for the name of the output dataset",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    full_images = args.full_labeled

    num_patches = args.num_patches
    strategy = args.strategy

    base_dataset_id = args.dataset_id
    target_id = args.output_id
    id_offset = args.offset
    seed = args.seed
    name_suffx = args.name_suffix

    if args.patch_size is not None:
        if len(args.patch_size) == 1:
            # todo make it possible for other input sizes as well
            patch_size = [args.patch_size] * 3
        else:
            patch_size = args.patch_size
    elif base_dataset_id is not None:
        patch_size = get_patch_size(base_dataset_id)
    else:
        raise AttributeError("Either base_dataset or patch-size has to be set.")

    if not isinstance(target_id, int):
        target_id = base_dataset_id + id_offset
    print(f"output_id is {target_id}")
    print(args)
    convert_dataset_to_partannotated(
        base_dataset_id,
        target_id,
        full_images,
        name_suffix=name_suffx,
        patch_size=patch_size,
        num_patches=num_patches,
        seed=seed,
        strategy=strategy,
    )
    generate_custom_splits_file(target_id, 0, 5)
