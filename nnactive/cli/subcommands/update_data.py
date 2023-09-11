from argparse import Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json
from nnactive.results.state import State
from nnactive.update_data import update_data


@register_subcommand(
    "update_data",
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
            "--annotated",
            {
                "dest": "annotated",
                "action": "store_true",
                "help": "If an annotated version of the dataset exists, update with annotated ground truth. "
                "If not specified, uses predTr folder in raw dataset folder.",
            },
        ),
        (
            ("-f", "--force"),
            {"action": "store_true", "help": "Ignores the internal State."},
        ),
    ],
)
def main(args: Namespace) -> None:
    dataset_id = args.dataset_id
    loop_val = args.loop
    force = args.force

    update_step(dataset_id, loop_val=loop_val, annotated=args.annotated, force=force)


def update_step(dataset_id, num_folds=5, loop_val=None, annotated=True, force=False):
    data_path = get_raw_path(dataset_id)
    save_splits_file = get_preprocessed_path(dataset_id) / "splits_final.json"
    target_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]
    file_ending = dataset_json["file_ending"]

    if annotated:
        base_dir = get_raw_path(dataset_json["annotated_id"]) / "labelsTr"
    else:
        base_dir = get_raw_path(dataset_id) / "predTr"

    if not force:
        state = State.get_id_state(dataset_id)

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
    )
    if not force:
        state.update_data = True
        state.new_loop()
        state.save_state()
