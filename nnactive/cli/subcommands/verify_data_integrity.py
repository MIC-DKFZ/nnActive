from argparse import Namespace

import numpy as np
import SimpleITK as sitk
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.loops.loading import get_loop_patches, get_patches_from_loop_files
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.results.state import State
from nnactive.update_data import update_data


@register_subcommand(
    "verify_data",
    [
        (("-d", "--dataset_id"), {"type": int, "required": True}),
        (
            ("-l", "--loop"),
            {
                "type": int,
                "default": None,
                "help": "iteration step to update (which loop_XXX file)",
            },
        ),
        (
            "--no_state",
            {
                "action": "store_true",
                "help": "Does not require internal State.",
            },
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    loop_val = args.loop
    no_state = args.no_state

    verify_data(
        dataset_id,
        loop_val=loop_val,
        no_state=no_state,
    )


def verify_data(
    dataset_id: int,
    loop_val: int = None,
    no_state: bool = False,
):
    data_path = get_raw_path(dataset_id)
    label_dir = data_path / "labelsTr"
    if loop_val is not None:
        pass
    elif loop_val is None and no_state is False:
        state = State.get_id_state(dataset_id)
        loop_val = state.loop
        if state.query:
            loop_val -= 1
    else:
        raise NotImplementedError()
    patches = get_patches_from_loop_files(data_path, loop_val)
    logger.info(
        f"Veryfing labels for loop {loop_val}.\n Cumulative sum of patches: {len(patches)}"
    )
    dataset_json = read_dataset_json(dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]

    for loop_check in range(loop_val + 1):
        patches = get_loop_patches(data_path, loop_check)
        logger.info(
            f"Verifying labels for loop {loop_check} with {len(patches)} patches."
        )

        unique_files = np.unique([patch.file for patch in patches])
        for file in unique_files:
            patch_file = [patch for patch in patches if patch.file == file]
            file_p = label_dir / file
            seg = sitk.GetArrayFromImage(sitk.ReadImage(file_p)).astype(np.uint8)
            for patch in patch_file:
                slices = []
                for start_index, size in zip(patch.coords, patch.size):
                    slices.append(slice(start_index, start_index + size))
                if np.any(seg[tuple(slices)] == ignore_label):
                    raise RuntimeError(
                        f"For loop {loop_check} patch in file {file} has ignore_label {ignore_label}"
                    )

    add_path = data_path / "addTr"
    if add_path.is_dir():
        logger.info(f"Verifying that labels contain addTr data.")
        for file in label_dir.iterdir():
            if file.name.endswith(dataset_json["file_ending"]):
                seg = sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(np.uint8)
                add_seg = sitk.GetArrayFromImage(
                    sitk.ReadImage(add_path / file.name)
                ).astype(np.uint8)
                equal = seg == add_seg
                if not np.all(equal[add_seg != 255]):
                    raise RuntimeError(
                        f"For file {file.name} in labelsTr the labels from addTr were not added."
                    )
