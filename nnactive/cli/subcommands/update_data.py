from argparse import Namespace

import nnunetv2.paths
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder
from nnactive.update_data import update_data
from nnactive import paths


@register_subcommand("update_data")
def update_step(
    config: ActiveConfig,
    continue_id: int | None = None,
    num_folds: int = 5,
    loop_val: int | None = None,
    annotated: bool = True,
    force: bool = False,
    no_state: bool = False,
    ensure_classes_in_folds: bool = True,
):
    config.set_nnunet_env()
    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)
    data_path = get_raw_path(state.dataset_id)
    save_splits_file = get_preprocessed_path(state.dataset_id) / "splits_final.json"
    target_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(state.dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]
    file_ending = dataset_json["file_ending"]

    additional_label_path = data_path / "addTr"
    if not additional_label_path.is_dir():
        additional_label_path = None

    if annotated:
        base_dir = paths.nnActive_raw / "nnUNet_raw" / config.dataset / "labelsTr"
    else:
        base_dir = get_raw_path(state.dataset_id) / f"annoTr_{loop_val:02}"

    if not no_state:
        state = State.get_id_state(state.dataset_id, verify=not force)

    if ensure_classes_in_folds:
        logger.info("Ensure every class in all train folds.")
        file_ending = dataset_json["file_ending"]
        dataset_classes = dataset_json["labels"]
        for label in dataset_classes:
            if isinstance(dataset_classes[label], (list, tuple)):
                dataset_classes[label] = dataset_classes[label][0]
        ensure_classes = [
            val for key, val in dataset_classes.items() if key != "ignore"
        ]

    else:
        logger.info(
            "Standard splits creation. Possibly not every class in all train folds."
        )
        ensure_classes = None

    update_data(
        data_path,
        save_splits_file,
        ignore_label,
        file_ending,
        base_dir,
        target_dir,
        loop_val=loop_val,
        num_folds=num_folds,
        annotated=annotated,
        additional_label_path=additional_label_path,
        ensure_classes=ensure_classes,
    )

    if not force and not no_state:
        state.update_data = True
        state.new_loop()
        state.save_state()
