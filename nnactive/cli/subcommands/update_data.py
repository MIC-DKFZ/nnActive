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
            {
                "action": "store_true",
                "help": "Ignores the internal State.",
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
    dataset_id: int = args.dataset_id
    loop_val: int | None = args.loop
    force: bool = args.force
    no_state: bool = args.no_state
    annotated: bool = args.annotated

    update_step(
        dataset_id,
        loop_val=loop_val,
        annotated=annotated,
        force=force,
        no_state=no_state,
    )


def update_step(
    dataset_id: int,
    num_folds: int = 5,
    loop_val: int | None = None,
    annotated: bool = True,
    force: bool = False,
    no_state: bool = False,
):
    data_path = get_raw_path(dataset_id)
    save_splits_file = get_preprocessed_path(dataset_id) / "splits_final.json"
    target_dir = data_path / "labelsTr"

    dataset_json = read_dataset_json(dataset_id)
    ignore_label = dataset_json["labels"]["ignore"]
    file_ending = dataset_json["file_ending"]

    additional_label_path = data_path / "addTr"
    if not additional_label_path.is_dir():
        additional_label_path = None

    if annotated:
        base_dir = get_raw_path(dataset_json["annotated_id"]) / "labelsTr"
    else:
        base_dir = get_raw_path(dataset_id) / f"annoTr_{loop_val:02}"

    if not no_state:
        state = State.get_id_state(dataset_id, verify=not force)

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
    )

    if not force and not no_state:
        state.update_data = True
        state.new_loop()
        state.save_state()
